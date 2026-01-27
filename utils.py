import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectKBest,VarianceThreshold, SelectPercentile, SelectFromModel, RFE
from sklearn.feature_selection import f_regression,mutual_info_regression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None  # default='warn'

from scipy.stats import uniform, loguniform, randint

from sklearn.mixture import GaussianMixture
from sdv.single_table import CTGANSynthesizer,GaussianCopulaSynthesizer
from sdv.metadata import Metadata

from mapie.regression import CrossConformalRegressor

from collections import defaultdict

import time
import ast


#TODO add .pkl to save the instance of the trained models for CP
class trainingHR:
    def __init__(self,path_data):
        self.data=pd.read_excel(path_data)
        self.regressors={         #to avoid having the same model instance               
                    "rf": (RandomForestRegressor, {"random_state": 2025}), #loss=MSE
                    "svr": (SVR, {},),
                    "xgb": (XGBRegressor, {"random_state": 2025}), #loss=RMSE             
                    "kr": (KernelRidge, {}),
                    "lgbm": (LGBMRegressor, {"verbose":-1,"random_state": 2025})          
                    }
        self.distribution_params_regressor=[ #for RandomizedSearch
                {'model__n_estimators': randint(200,1000),'model__max_depth' : randint(4,12)},#rf 
                {'model__kernel':['rbf','linear','sigmoid'], "model__gamma": loguniform(1e-4,10), "model__C": loguniform(1e-1,1e1)}, #SVR
                {"model__n_estimators": randint(200,1000), "model__max_depth":randint(4,12), "model__learning_rate": loguniform(1e-3, 1e-1)},#xgboost 
                {'model__kernel':['rbf','linear','sigmoid'], "model__gamma": loguniform(1e-4,10),"model__alpha": loguniform(1e-5,10)},#KernelRidge
                {"model__n_estimators": randint(200,1000), "model__max_depth":randint(4,12), "model__learning_rate": loguniform(1e-3, 1e-1)} #LGBM 
                ]
        
    #for gridSearch
    # def grid_params_regressor(): 
    #     return [
    #               {'model__n_estimators': [200, 300, 400],'model__max_depth' : [4,5,6]},#rf 
    #               {'model__kernel':['linear', 'rbf', 'sigmoid'], 'model__gamma' : ['scale','auto']},#svr
    #               {"model__max_depth":[4, 5, 6], "model__n_estimators": [200, 300, 400], "model__learning_rate": [0.01, 0.015]},#xgboost 
    #               {'model__kernel':['rbf','linear','sigmoid'], "model__alpha": [1e0, 0.1, 1e-2, 1e-3], "model__gamma": np.logspace(-2, 2, 5)},#Ridge 
    #              ]
    
    # def grid_params_full():
    #     grid_params_regressor=trainingHR.grid_params_regressor()
    #     grid_params_full=[]
    #     for dic in grid_params_regressor:
    #         dic['scaling']=[MinMaxScaler(),StandardScaler()]
    #         grid_params_selectors = [
    #                         {
    #                             'feature_selection': [SelectKBest(mutual_info_regression)],
    #                             'feature_selection__k': [10, 15, 20, 25, 30] 
    #                         },
    #                         {
    #                             'feature_selection': [SelectPercentile(mutual_info_regression)],
    #                             'feature_selection__percentile': [10, 15, 20, 25, 30] 
    #                         }
    #         ]
    #         # on pourrait ajouter SelectFromModel(model),RFE(model) mais ça ajoute de la compléxité + faire attention que ça soit la même instance de modèle que celui de prédiction dans la pipeline
    #         # ici on se limite à la sélection de features de manière statistique
    #         grid_params = [sel_grid | dic for sel_grid in grid_params_selectors] #on fusionne toutes les possibilités
    #         grid_params_full.append(grid_params)
    #     return grid_params_full
    
    def distribution_params_full(self): 
        grid_params_regressor=self.distribution_params_regressor
        grid_params_full=[]
        for dic in grid_params_regressor:
            dic['scaling']=[MinMaxScaler(),StandardScaler()]
            #here you can add features selector and if it's not the same params for each selector need to do a list of dict for each regressor
            grid_params_full.append(dic)
        return grid_params_full
    
    #train and test without predefined list of features (+test scaling)
    #use for overall performances
    def predictions_HR(self, target, cv_search, n_iter_search, save_results):
        list_regressors_perf = []
        players=self.data["Code anonyme"].unique()
        for p in range(len(players)): 
            start_time = time.time()
            print(players[p])
            filter_player_games=self.data["Code anonyme"]==players[p]
            data_pl=self.data[filter_player_games]
            observations=data_pl.shape[0]  

            #pre processing
            data_pl_na = data_pl.dropna()    
            print('number of observations = '+str(data_pl_na.shape[0]) )
            feature_values = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) ##remove str, time stamp and HR information
            target_values = data_pl_na[target]
            xtrain, xtest, ytrain, ytest = train_test_split(feature_values.values, target_values.values, test_size=0.2, random_state=2025)
            #ML predictions
            regressors=self.regressors
            full_params_reg=self.distribution_params_full()
            for params_reg,(name_reg,reg) in zip(full_params_reg,regressors.items()):
                pip= Pipeline([('variance_0',VarianceThreshold(threshold=0)),('scaling','passthrough'),("feature_selection","passthrough"),("model",reg[0](**reg[1]))])
                start_time = time.time()
                model = RandomizedSearchCV(pip, params_reg, n_iter=n_iter_search, n_jobs = -1, cv=cv_search)
                pred_reg = model.fit(xtrain, ytrain).predict(xtest)
                #features_variance = model.best_estimator_.named_steps['variance_0']
                # selected_features_variance_indices=features_variance.get_support()
                # selected_features_variance=feature_values.columns[selected_features_variance_indices].to_numpy()
                # features_selector=model.best_estimator_.named_steps['feature_selection']
                # selected_features_indices=features_selector.get_support()
                # selected_features=list(selected_features_variance[selected_features_indices])
                selected_features='all'
                selected_scaling = model.best_estimator_.named_steps['scaling'].__class__.__name__
                # Evaluating the model
                indicateurs_reg=[players[p],observations,name_reg,np.sqrt(mean_squared_error(ytest, pred_reg)),
                                  mean_absolute_error(ytest, pred_reg),r2_score(ytest, pred_reg),
                                  mean_absolute_percentage_error(ytest, pred_reg),selected_features, selected_scaling
                        ]
                list_regressors_perf.append(indicateurs_reg)
                print("\n --- %s secondes for this player pipeline --- " % np.round( (time.time() - start_time),3) )
                       
        if( save_results==True):
            #save predictions results
            regressors_perf=pd.DataFrame(list_regressors_perf,columns = ['Code anonyme','Number of observations','Regressor', 'RMSE','MAE','R^2','MAPE','Features selected', 'Scaling selected'])
            regressors_perf.to_csv('results/07072023_16052025_full_prediction.csv',index=False) 

    def create_features_samples(self,player,target,nb_samples):
        filter_player_games=self.data["Code anonyme"]==player
        data_pl=self.data[filter_player_games]
        data_pl_na = data_pl.dropna() 
        features_df = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source'])##remove str, time stamp and HR information
        # gmm = GaussianMixture(n_components=nb_clusters).fit(feature_df.values) #add nb_clusters as an argument
        # features_synth = gmm.sample(n_samples=nb_samples)[0]
        metadata = Metadata.detect_from_dataframe(data=features_df)
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(features_df)
        features_synth=synthesizer.sample(nb_samples)
        features_synth['deviation']=np.random.normal(10,5,(nb_samples,))
        features_synth.to_csv(f'synthetic_data/{player}_{target}_{nb_samples}.csv',index=False)
        return features_synth

    #conformal prediction: training by player no matters the chronology 
    #use regressor or best_regressor=True. If best_regressor=True, the path_result must be provide (path to the results of prediction_HR). This method allows not to test all the regressors
    #if is_balanced=True, for each player same number of normal test and anomaly test (synthetic). Otherwise, you can indicate nb_anomalies to create (default 100)
    #params_reg must be a dict of dict of the params with they first keys being the str name of the regressor. If regressor: just one key otherwise all the possible regressors
    def split_conformal_prediction(self,threshold_nb_data, test_size, calibration_size, target, dict_params_search, cv_search, n_iter_search, alphas,save_results,best_regressor=False,path_result=None,regressor=None,is_balanced=False,nb_anomalies=100):
        players=self.data["Code anonyme"].unique() 
        nb_observations=[self.data[self.data["Code anonyme"]==player].shape[0] for player in players] #nb of obervations by player
        players_filtered=players[np.array(nb_observations)>=threshold_nb_data] #works with arrray but not list
        dict_results = defaultdict(dict)
        for player in players_filtered:           
            print(player)
            filter_player_games=self.data["Code anonyme"]==player
            data_pl=self.data[filter_player_games]
            data_pl_na = data_pl.dropna()    
            feature_values = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str, time stamp and HR information
            target_values = data_pl_na[target]
            #split the data
            cal_size=calibration_size/(1-test_size)
            xtrain_full, xtest, ytrain_full, ytest = train_test_split(feature_values.values, target_values.values, test_size=test_size, random_state=2025)
            xtrain, xcal, ytrain, ycal = train_test_split(xtrain_full, ytrain_full, test_size=cal_size, random_state=2025)
            nb_cal=xcal.shape[0]
            if best_regressor:
                best_models=resultsAnalysis(path_result).get_best_model()
                regressor_used_str=best_models[player]
                model_class,model_params=self.regressors[regressor_used_str]
                regressor_used=model_class(**model_params)
                params_search_reg=dict_params_search[regressor_used_str]
            else:
                model_class,model_params=self.regressors[regressor]
                regressor_used=model_class(**model_params)
                params_search_reg=dict_params_search[regressor]
            #train + cal
            pip= Pipeline([('variance_0',VarianceThreshold()),('scaling','passthrough'),("feature_selection","passthrough"),("model",regressor_used)])
            model = RandomizedSearchCV(pip, params_search_reg, n_iter=n_iter_search, n_jobs = -1, cv=cv_search)
            model.fit(xtrain, ytrain)
            pred_cal= model.predict(xcal) 
            cal_scores=np.abs(pred_cal-ycal)
            #create the synthetic data
            nb_anomalies_used=xtest.shape[0] if is_balanced else nb_anomalies
            try:
                full_df=pd.read_csv(f'synthetic_data/{player}_{target}_{nb_anomalies_used}.csv')
                xsynth_df=full_df.drop(columns=['deviation'])
            except:
                full_df=self.create_features_samples(player,target,nb_anomalies_used)
                xsynth_df=full_df.drop(columns=['deviation'])
            xsynth=xsynth_df.to_numpy()
            #test
            pred_test=model.predict(np.concatenate((xtest,xsynth))) 
            ysynth=pred_test[len(xtest):]+full_df['deviation'] #we increase the expected HR response
            is_anomaly=[False for _ in range(len(xtest))]+[True for _ in range(nb_anomalies_used)]
            for alpha in alphas:
                q=np.quantile(cal_scores, np.min([1,np.ceil((nb_cal+1)*(1-alpha))/nb_cal]), method='higher')
                lb=pred_test-q
                ub=pred_test+q
                is_anomaly_detected= [(y_test>u_b or y_test<l_b) for (y_test,u_b,l_b) in zip(np.concatenate((ytest,ysynth)),ub,lb)]
                df=pd.DataFrame(np.array([lb,ub,np.concatenate((ytest,ysynth)),is_anomaly,is_anomaly_detected]).T,columns=['Lower bound','Upper bound','Ground truth','Anomaly ?','Anomaly detected ?'])
                dict_results[alpha][player]=df
        name_fold_anomalies= "balanced_nb_anomalies" if is_balanced else f"{nb_anomalies}_anomalies"
        for alpha in alphas:
            final_df=pd.concat(dict_results[alpha],names=['Player'])
            if( save_results==True):
                #save predictions results
                final_df.to_csv(f'results/SCP/{name_fold_anomalies}/07072023_16052025_alpha{alpha}.csv') 
        return final_df

    #params_search_qr_l is just about the model not the scaling or the features selection: need to be a dict of dict (not vital here because lgbm and xgb have the same hyperparameters but for future)
    def conformalized_quantile_regression(self,threshold_nb_data, test_size, calibration_size, target, dict_params_search_qr_l, dict_params_search_qr_u, cv_search, n_iter_search, alphas, save_results,best_regressor=False,path_result=None,regressor=None,is_balanced=False, nb_anomalies=100):
        players=self.data["Code anonyme"].unique() 
        nb_observations=[self.data[self.data["Code anonyme"]==player].shape[0] for player in players] #nb of obervations by player
        players_filtered=players[np.array(nb_observations)>=threshold_nb_data] #marche avec les array mais pas les listes
        for alpha in alphas:
            str_to_reg={                        
                    "xgb": (XGBRegressor, {"objective":"reg:quantileerror","quantile_alpha":alpha/2,"random_state": 2025},{"objective":"reg:quantileerror","quantile_alpha":1-(alpha/2),"random_state": 2025}),                
                    "lgbm": (LGBMRegressor, {"objective":'quantile',"alpha":alpha/2,"random_state": 2025,"verbose":-1},{"objective":'quantile',"alpha":1-(alpha/2),"random_state": 2025,"verbose":-1})          
                    }
            dict_results={}
            for player in players_filtered:          
                print(player)
                filter_player_games=self.data["Code anonyme"]==player
                data_pl=self.data[filter_player_games]
                data_pl_na = data_pl.dropna()    
                feature_values = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str et time stamp
                target_values = data_pl_na[target]
                #split the data
                cal_size=calibration_size/(1-test_size)#si on veut 20% de cal au total il faut ajuster car sinon on prend 20% de 90% 
                xtrain_full, xtest, ytrain_full, ytest = train_test_split(feature_values.values, target_values.values, test_size=test_size, random_state=2025)
                xtrain, xcal, ytrain, ycal = train_test_split(xtrain_full, ytrain_full, test_size=cal_size, random_state=2025)
                nb_cal=xcal.shape[0]
                if best_regressor:
                    best_models=resultsAnalysis(path_result).get_best_model_CQR()
                    regressor_used_str=best_models[player]
                    model_class,qr_l_params,qr_u_params=str_to_reg[regressor_used_str]
                    qr_l_used=model_class(**qr_l_params)
                    qr_u_used=model_class(**qr_u_params)
                    params_search_qr_l=dict_params_search_qr_l[regressor_used_str]
                    params_search_qr_u=dict_params_search_qr_u[regressor_used_str]
                else:
                    model_class,qr_l_params,qr_u_params=str_to_reg[regressor]
                    qr_l_used=model_class(**qr_l_params)
                    qr_u_used=model_class(**qr_u_params)
                    params_search_qr_l=dict_params_search_qr_l[regressor]
                    params_search_qr_u=dict_params_search_qr_u[regressor]
                pip_qr_l= Pipeline([('variance_0',VarianceThreshold()),('scaling','passthrough'),("feature_selection","passthrough"),("model",qr_l_used)])
                model_qr_l = RandomizedSearchCV(pip_qr_l, params_search_qr_l,n_iter=n_iter_search,n_jobs = -1, cv=cv_search)
                model_qr_l.fit(xtrain,ytrain)
                #scaled + selection features: we apply the same transformations as the best ones obtained for qr_l
                variance_threshold = model_qr_l.best_estimator_.named_steps['variance_0'] 
                scaling = model_qr_l.best_estimator_.named_steps['scaling']  
                features_selector = model_qr_l.best_estimator_.named_steps['feature_selection']  
                pip_preprocessing= Pipeline([('variance_0',variance_threshold),('scaling',scaling),("feature_selection",features_selector)])
                xtrain_transform=pip_preprocessing.transform(xtrain)
                xtest_transform=pip_preprocessing.transform(xtest)
                xcal_transform=pip_preprocessing.transform(xcal)      
                #xsynth
                nb_anomalies_used=xtest.shape[0] if is_balanced else nb_anomalies
                try:
                    full_df=pd.read_csv(f'synthetic_data/{player}_{target}_{nb_anomalies_used}.csv')
                    xsynth_df=full_df.drop(columns=['deviation'])
                except:
                    full_df=self.create_features_samples(player,target,nb_anomalies_used)
                    xsynth_df=full_df.drop(columns=['deviation'])
                xsynth=xsynth_df.to_numpy()
                xsynth_transform=pip_preprocessing.transform(xsynth) 
                model_qr_u = RandomizedSearchCV(qr_u_used, params_search_qr_u,n_iter=n_iter_search,n_jobs = -1, cv=cv_search) 
                #fit qr_u
                model_qr_u.fit(xtrain_transform,ytrain)
                #cal
                pred_cal_qr_l=model_qr_l.predict(xcal)
                pred_cal_qr_u=model_qr_u.predict(xcal_transform)
                cal_scores=np.maximum(pred_cal_qr_l-ycal,ycal-pred_cal_qr_u)
                q=np.quantile(cal_scores, np.min([1,np.ceil((nb_cal+1)*(1-alpha))/nb_cal]), method='higher')
                #predict test
                pred_test_qr_l=model_qr_l.predict(np.concatenate((xtest,xsynth)))
                pred_test_qr_u=model_qr_u.predict(np.concatenate((xtest_transform,xsynth_transform)))
                lb=pred_test_qr_l-q
                ub=pred_test_qr_u+q
                ysynth=((ub[len(xtest):]+lb[len(xtest):])//2)+full_df['deviation']#on augmente la FC attendue 
                is_anomaly=[False for _ in range(len(xtest))]+[True for _ in range(nb_anomalies_used)]
                is_anomaly_detected= [(y_test>u_b or y_test<l_b) for (y_test,u_b,l_b) in zip(np.concatenate((ytest,ysynth)),ub,lb)]
                df=pd.DataFrame(np.array([lb,ub,np.concatenate((ytest,ysynth)),is_anomaly,is_anomaly_detected]).T,columns=['Lower bound','Upper bound','Ground truth','Anomaly ?','Anomaly detected ?'])
                dict_results[player]=df
            final_df=pd.concat(dict_results,names=['Player'])
            if( save_results==True):
                #save predictions results
                name_fold_anomalies= "balanced_nb_anomalies" if is_balanced else f"{nb_anomalies}_anomalies"
                final_df.to_csv(f'results/CQR/{name_fold_anomalies}/07072023_16052025_alpha{alpha}.csv') 
        return final_df

    
    #jackknife+ method (to avoid calibration set) or CV+ but computationally expensive (retrain a lot of models). If is_jackknife_plus=True, jackkknife+ is performed. Otherwise CV+ and cv_CV_plus is the number of folds
    def cross_conformal_prediction(self,threshold_nb_data,test_size,target, alphas, dict_params_search,cv_search,n_iter_search,save_results,best_regressor=False,path_result=None,regressor=None,is_jackknife_plus=False,cv_CV_plus=20,is_balanced=False,nb_anomalies=100):
        players=self.data["Code anonyme"].unique() 
        nb_observations=[self.data[self.data["Code anonyme"]==player].shape[0] for player in players] #nb of obervations by player
        players_filtered=players[np.array(nb_observations)>=threshold_nb_data] #marche avec les array mais pas les listes
        dict_results=defaultdict(dict)
        for player in players_filtered:           
            print(player)
            filter_player_games=self.data["Code anonyme"]==player
            data_pl=self.data[filter_player_games]
            data_pl_na = data_pl.dropna()    
            feature_values = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str et time stamp
            target_values = data_pl_na[target]
            #split the data
            xtrain, xtest, ytrain, ytest = train_test_split(feature_values.values, target_values.values, test_size=test_size, random_state=2025)
            if best_regressor:
                best_models=resultsAnalysis(path_result).get_best_model()
                regressor_used_str=best_models[player]
                model_class,model_params=self.regressors[regressor_used_str]
                regressor_used=model_class(**model_params)
                params_search_reg=dict_params_search[regressor_used_str]
            else:
                model_class,model_params=self.regressors[regressor]
                regressor_used=model_class(**model_params)
                params_search_reg=dict_params_search[regressor]
            pip= Pipeline([('variance_0',VarianceThreshold()),('scaling','passthrough'),("feature_selection","passthrough"),("model",regressor_used)])
            model= RandomizedSearchCV(pip, params_search_reg,n_iter=n_iter_search,n_jobs = -1, cv=cv_search) 
            model.fit(xtrain,ytrain)
            variance_threshold = model.best_estimator_.named_steps['variance_0'] 
            scaling = model.best_estimator_.named_steps['scaling']  
            features_selector = model.best_estimator_.named_steps['feature_selection']  
            model_hyperparameters = model.best_estimator_.named_steps['model'].get_params()
            estimator=regressor_used.set_params(**model_hyperparameters) #** to unpack the dict
            pip_opt= Pipeline([('variance_0',variance_threshold),('scaling',scaling),("feature_selection",features_selector)])
            xtrain_transform=pip_opt.transform(xtrain)
            xtest_transform=pip_opt.transform(xtest)
            #xsynth
            nb_anomalies_used=xtest.shape[0] if is_balanced else nb_anomalies
            try:
                full_df=pd.read_csv(f'synthetic_data/{player}_{target}_{nb_anomalies_used}.csv')
                xsynth_df=full_df.drop(columns=['deviation'])
            except:
                full_df=self.create_features_samples(player,target,nb_anomalies_used)
                xsynth_df=full_df.drop(columns=['deviation'])
            xsynth=xsynth_df.to_numpy()
            xsynth_transform=pip_opt.transform(xsynth)
            cv_CV_plus_used=xtrain.shape[0] if is_jackknife_plus else cv_CV_plus
            CCP_regressor = CrossConformalRegressor(estimator=estimator,confidence_level=(1-alphas),cv=cv_CV_plus_used,random_state=2025).fit_conformalize(xtrain_transform, ytrain)
            pred_test_point,pred_test_interval=CCP_regressor.predict_interval(np.concatenate((xtest_transform,xsynth_transform)))
            ysynth=pred_test_point[len(xtest):]+full_df['deviation'] 
            is_anomaly=[False for _ in range(len(xtest))]+[True for _ in range(nb_anomalies_used)]
            for i,alpha in enumerate(alphas):
                lb=pred_test_interval[:,0,i] #the last dimension is if alpha is a list or array
                ub=pred_test_interval[:,1,i]
                is_anomaly_detected= [(y_test>u_b or y_test<l_b) for (y_test,u_b,l_b) in zip(np.concatenate((ytest,ysynth)),ub,lb)]
                df=pd.DataFrame(np.array([lb,ub,np.concatenate((ytest,ysynth)),is_anomaly,is_anomaly_detected]).T,columns=['Lower bound','Upper bound','Ground truth','Anomaly ?','Anomaly detected ?'])
                dict_results[alpha][player]=df
        name_fold_anomalies= "balanced_nb_anomalies" if is_balanced else f"{nb_anomalies}_anomalies"
        for alpha in alphas:
            final_df=pd.concat(dict_results[alpha],names=['Player'])
            if( save_results==True):
                if is_jackknife_plus:
                    final_df.to_csv(f'results/Jackknife+/{name_fold_anomalies}/07072023_16052025_alpha{alpha}.csv')
                else:
                    final_df.to_csv(f'results/CV+/{name_fold_anomalies}/07072023_16052025_cvccp{cv_CV_plus}alpha{alpha}.csv')
        return final_df
    
#to analyse full_pred    
class resultsAnalysis:
    def __init__(self,path_result):
        self.results=pd.read_csv(path_result)
        self.metrics= ["RMSE", "MAE", "R^2", "MAPE"]

    #best scaling overall
    def best_scaling(self):
        selected_scaling_count=self.results['Scaling selected'].value_counts()
        for scaler in selected_scaling_count.index.to_list():
            print(f'{scaler} a été utilisé {selected_scaling_count.loc[scaler]} fois')
        return selected_scaling_count

    #best model overall
    def count_best_model(self):
        players=self.results["Code anonyme"].unique()
        idx_best_reg=[]
        for p in range(len(players)):
            filter_player=self.results["Code anonyme"]==players[p]
            data_pl=self.results[filter_player].reset_index()
            idx_best_reg_player=[data_pl['RMSE'].idxmin(),data_pl['MAE'].idxmin(),data_pl['MAPE'].idxmin(),data_pl['R^2'].idxmax()]
            count_best_reg_player=np.array([idx_best_reg_player.count(i) for i in range(data_pl.shape[0])]) 
            best_reg_player=count_best_reg_player.argmax() #for a given player, the best model is the one which has the best performances (BO4 on RMSE, MAE, MAPE and R^2) among the others
            idx_best_reg.append(best_reg_player)
        best_reg_count=np.array([idx_best_reg.count(i) for i in range(data_pl.shape[0])])
        regressors=data_pl['Regressor'].to_list()
        best_reg=[regressors[i] for i in idx_best_reg]
        dict_best_mod_player=dict(zip(players,best_reg))
        for regressor,count in zip(regressors,best_reg_count):
            print(f'{regressor} a été le meilleur modèle pour {count} joueur(s)')
        return dict_best_mod_player
    
    #get best model for CP methods (except CQR)
    def get_best_model(self):
        players=self.results["Code anonyme"].unique()
        idx_best_reg=[]
        for p in range(len(players)):
            filter_player=self.results["Code anonyme"]==players[p]
            data_pl=self.results[filter_player].reset_index()
            idx_best_reg_player=[data_pl['RMSE'].idxmin(),data_pl['MAE'].idxmin(),data_pl['MAPE'].idxmin(),data_pl['R^2'].idxmax()]
            count_best_reg_player=np.array([idx_best_reg_player.count(i) for i in range(data_pl.shape[0])])
            best_reg_player=count_best_reg_player.argmax() #for a given player, the best model is the one with the best performances (BO4 on RMSE, MAE, MAPE and R^2) among the others
            idx_best_reg.append(best_reg_player)
        regressors=data_pl['Regressor'].to_list()
        best_reg=[regressors[i] for i in idx_best_reg]
        dict_best_mod=dict(zip(players,best_reg))
        return dict_best_mod
    
    #get best model for CQR
    def get_best_model_CQR(self):
        players=self.results["Code anonyme"].unique()
        idx_best_reg=[]
        reg_with_pinball_loss=['xgb','lgbm']
        for p in range(len(players)):
            filter_player=self.results["Code anonyme"]==players[p]
            data_pl=self.results[filter_player].reset_index()
            data_pl_filtered=data_pl[data_pl['Regressor'].isin(reg_with_pinball_loss)]
            idx_best_reg_player=[data_pl_filtered['RMSE'].idxmin(),data_pl_filtered['MAE'].idxmin(),data_pl_filtered['MAPE'].idxmin(),data_pl_filtered['R^2'].idxmax()]
            count_best_reg_player=np.array([idx_best_reg_player.count(i) for i in range(data_pl.shape[0])])
            best_reg_player=count_best_reg_player.argmax() #for a given player, the best model is the one with the best performances (BO4 on RMSE, MAE, MAPE and R^2) among the others
            idx_best_reg.append(best_reg_player)
        regressors=data_pl['Regressor'].to_list()
        best_reg=[regressors[i] for i in idx_best_reg]
        dict_best_mod_CQR=dict(zip(players,best_reg))
        return dict_best_mod_CQR
    
    #best features overall
    def best_features(self):
        selected_features=self.results['Features selected'].apply(ast.literal_eval)
        top_list_features=selected_features.describe()['top']
        filter=self.results['Features selected']==str(top_list_features)
        #print(self.results[filter])
        all_features_selected=[]
        nb_features_selected=[]
        sum=0
        for list in selected_features.to_list():
            sum+=len(list)
            nb_features_selected.append(len(list))
            all_features_selected+=list
        occ_nb_features_selected=dict([(i,nb_features_selected.count(i)) for i in set(nb_features_selected)])
        occ_nb_features_selected_sorted=dict(sorted(occ_nb_features_selected.items(), key=lambda item: item[0]))
        mean_nb_features_selected=sum/len(selected_features.to_list())
        all_features_unique=set(all_features_selected)
        occ_features=dict([(i,all_features_selected.count(i)) for i in set(all_features_unique)])
        occ_features_sorted=dict(sorted(occ_features.items(), key=lambda item: item[1]))
        return top_list_features,occ_features_sorted,mean_nb_features_selected,occ_nb_features_selected_sorted
    
    def mean_std(self):
        summary = self.results.groupby("Regressor")[self.metrics].agg(["mean", "std"])
        print(summary)
    
    def plot_bins(self,metric,nb_bins):
        self.results["obs_bin"] = pd.qcut(self.results["Number of observations"],q=nb_bins,duplicates="drop")
        binned = self.results.groupby(["Regressor", "obs_bin"])[metric].mean().reset_index()
        bins = binned["obs_bin"].astype(str).unique()
        regressors = binned["Regressor"].unique()
        x = range(nb_bins)
        bar_width = 0.8 / len(regressors)
        plt.figure()
        for i,reg in enumerate(regressors):
            subset = binned[binned["Regressor"] == reg]
            plt.bar([x_i+i*bar_width for x_i in x],subset[metric],width=bar_width,label=reg)
        plt.xlabel("Number of observations")
        plt.ylabel(metric)
        plt.xticks([x_i+bar_width*(len(regressors)-1)/2 for x_i in x],bins,rotation=30)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

class CPAnalysis:
    def __init__(self,path_result):
        self.results=pd.read_csv(path_result)

    def coverage_and_anomalies(self):
        normal_sample=self.results[self.results['Anomaly ?']==0]
        false_positive_rate=normal_sample['Anomaly detected ?'].mean()
        coverage=1-false_positive_rate
        anomaly_sample=self.results[self.results['Anomaly ?']==1]
        anomalies_detected=anomaly_sample['Anomaly detected ?'].mean()
        false_positive=normal_sample['Anomaly detected ?'].sum()
        true_negative=normal_sample.shape[0]-false_positive
        true_positive=anomaly_sample['Anomaly detected ?'].sum()
        false_negative=anomaly_sample.shape[0]-true_positive
        ppv=true_positive/(true_positive+false_positive)
        npv=true_negative/(true_negative+false_negative)
        return coverage,anomalies_detected,ppv,npv
    
    def interval_size_analysis(self):
        results_filtered=self.results[self.results['Anomaly ?']==0]
        interval_size=results_filtered['Upper bound']-results_filtered['Lower bound']
        stats=interval_size.agg(['mean','std','median'])
        return stats
    
def subplot_CP(ax, df, title,num_plots_idx):
    y_true_ = df["Ground truth"].values
    lower_ = df["Lower bound"].values
    upper_ = df["Upper bound"].values
    y_true = np.take(y_true_, num_plots_idx)
    lower= np.take(lower_, num_plots_idx)
    upper = np.take(upper_, num_plots_idx)
    inside = (y_true >= lower) & (y_true <= upper) 
    #inside_= (y_true_ >= lower_) & (y_true_ <= upper_) 
    outside = ~inside
    x_range = y_true.max() - y_true.min()
    cap = 0.015 * x_range
    ax.vlines(x=y_true[inside],ymin=lower[inside],ymax=upper[inside],color="tab:blue",linewidth=1.5,alpha=0.8,label="Inside prediction interval")
    ax.hlines(lower[inside], y_true[inside] - cap, y_true[inside] + cap, color="tab:blue")
    ax.hlines(upper[inside], y_true[inside] - cap, y_true[inside] + cap, color="tab:blue")
    ax.vlines(x=y_true[outside],ymin=lower[outside],ymax=upper[outside],color="tab:red",linewidth=1.5,alpha=0.8,label="Outside prediction interval")
    ax.hlines(lower[outside], y_true[outside] - cap, y_true[outside] + cap, color="tab:red")
    ax.hlines(upper[outside], y_true[outside] - cap, y_true[outside] + cap, color="tab:red")
    lims = [120,195]
    ax.plot(lims, lims, "--", color="black", label="x=y")
    ax.set_title(title)
    ax.set_xlabel("True heart rate in bpm")
    ax.set_ylabel("Prediction interval")

def plot_CPs(datasets,titles):
    rng=np.random.default_rng()
    perc_obs_plot = 0.02
    nb_samples=datasets[0].shape[0]
    num_plots_idx = rng.choice(nb_samples, int(perc_obs_plot *nb_samples), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for ax, df, title in zip(axes.flat, datasets,titles):
        subplot_CP(ax, df, title,num_plots_idx)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def get_distrib_HR(path_data, threshold_min,threshold_max):
    data=pd.read_excel(path_data)
    players=data["Code anonyme"].unique()
    for p in range(len(players)):
        filter_player=data["Code anonyme"]==players[p]
        data_pl=data[filter_player]
        if data_pl.shape[0]>threshold_min and data_pl.shape[0]<=threshold_max:
            plt.hist(data_pl['FC_moyenne'])
            plt.xlabel("FC")
            plt.ylabel("Fréquence")
            plt.title(f"{players[p]}")
            plt.show()




