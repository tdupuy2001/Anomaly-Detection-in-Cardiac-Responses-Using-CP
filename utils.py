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
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold

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
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, average_precision_score
pd.options.mode.chained_assignment = None  # default='warn'

from scipy.stats import uniform, loguniform, randint
import joblib

from sklearn.mixture import GaussianMixture
from sdv.single_table import CTGANSynthesizer,GaussianCopulaSynthesizer
from sdv.metadata import Metadata

np.random.seed(2025)

from mapie.regression import CrossConformalRegressor

from collections import defaultdict

import time
import ast

import pickle 
import os

#TODO mettre features selection au max ou l'enlever dans la grid et la distrib (possibilité de mettre mais pas nécessaire)
class trainingHR:
    def __init__(self,path_data):
        self.data=pd.read_excel(path_data)
        self.regressors={         #to avoid having the same model instance               
                    "RF": (RandomForestRegressor, {"random_state": 2025}), #loss=MSE
                    "SVR": (SVR, {},),
                    "XGB": (XGBRegressor, {"random_state": 2025}), #loss=RMSE             
                    "KR": (KernelRidge, {}),
                    "LGBM": (LGBMRegressor, {"verbose":-1,"random_state": 2025})          
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
    def predictions_HR(self, target, cv_search, n_iter_search,train_size,save_performance_results,shuffle=False):
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
            #TODO tester avec FC_moyenne_perc entrainer deux modèles ou autre
            feature_values = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) ##remove str, time stamp and HR information
            target_values = data_pl_na[target]
            xtrain, xtest, ytrain, ytest = train_test_split(feature_values.values, target_values.values, train_size=train_size,shuffle=shuffle, random_state=2025)
            #ML predictions
            regressors=self.regressors
            full_params_reg=self.distribution_params_full()
            for params_reg,(name_reg,reg) in zip(full_params_reg,regressors.items()):
                result_path = f'./results/forecasts/HR/{players[p]}/{name_reg}/'
                os.makedirs(result_path, exist_ok=True)
                file_name=result_path+f'train_size{train_size}_target{target}_cv_search{cv_search}_n_iter_search{n_iter_search}'+'.pkl'
                try:
                    with open(file_name, 'rb') as handle:
                        best_model = pickle.load(handle)
                    print("best model already exists")
                except:
                    pip= Pipeline([('variance_0',VarianceThreshold(threshold=0)),('scaling','passthrough'),("feature_selection","passthrough"),("model",reg[0](**reg[1]))])
                    start_time = time.time()
                    model = RandomizedSearchCV(pip, params_reg, n_iter=n_iter_search, n_jobs = -1, cv=cv_search,random_state=2025)
                    model_trained = model.fit(xtrain, ytrain)
                    #features_variance = model.best_estimator_.named_steps['variance_0']
                    # selected_features_variance_indices=features_variance.get_support()
                    # selected_features_variance=feature_values.columns[selected_features_variance_indices].to_numpy()
                    # features_selector=model.best_estimator_.named_steps['feature_selection']
                    # selected_features_indices=features_selector.get_support()
                    # selected_features=list(selected_features_variance[selected_features_indices])
                    best_model=model_trained.best_estimator_
                    print("\n --- %s secondes for this player pipeline --- " % np.round( (time.time() - start_time),3) )
                    with open(file_name, 'wb') as handle:
                        pickle.dump(best_model,handle)
                pred_reg=best_model.predict(xtest)
                selected_features='all'
                selected_scaling = best_model.named_steps['scaling'].__class__.__name__    
                # Evaluating the model
                indicateurs_reg=[players[p],observations,name_reg,np.sqrt(mean_squared_error(ytest, pred_reg)),
                                mean_absolute_error(ytest, pred_reg),r2_score(ytest, pred_reg),
                                mean_absolute_percentage_error(ytest, pred_reg),selected_features, selected_scaling
                        ]
                list_regressors_perf.append(indicateurs_reg)               
        if( save_performance_results==True):
            #save performance results
            regressors_perf=pd.DataFrame(list_regressors_perf,columns = ['Code anonyme','Number of observations','Regressor', 'RMSE','MAE','R^2','MAPE','Features selected', 'Scaling selected'])
            regressors_perf.to_csv('results/07072023_16052025_full_prediction.csv',index=False) 

    def create_features_samples(self,player,target,nb_samples,regressor_str):
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
        result_path = f'./results/forecasts/HR/{player}/{regressor_str}/'
        file_name=result_path+'train_size0.7_targetFC_moyenne_cv_search5_n_iter_search200'+'.pkl'
        with open(file_name, 'rb') as handle:
            best_model = pickle.load(handle)
        y_pred=best_model.predict(features_synth.to_numpy())
        # np.random.normal(10,5,(nb_samples,))
        data_synth=features_synth
        data_synth['Y']=y_pred
        data_synth.to_csv(f'synthetic_data/full_synthetic/{player}/{target}_{nb_samples}.csv',index=False)
        return data_synth

    #conformal prediction: training by player no matters the chronology 
    #use regressor or best_regressor=True. If best_regressor=True, the path_result must be provide (path to the results of prediction_HR). This method allows not to test all the regressors
    #params_reg must be a dict of dict of the params with their first keys being the str name of the regressor. If regressor: just one key otherwise all the possible regressors
    #if deviation is a int the deviation is considered as a number of bpm otherwise as a percentage of bpm
    #TODO add nb of best regressors to test
    def split_conformal_prediction(self, players_tested, train_size, cal_size, test_size, target, dict_params_search, cv_search, n_iter_search, alphas,save_test_results,deviation,type_deviation,type_anomalie,prevalence=0.5,best_regressor=False,path_result=None,regressor=None):
        # players=self.data["Code anonyme"].unique() 
        #nb_observations=[self.data[self.data["Code anonyme"]==player].shape[0] for player in players] #nb of obervations by player
        # players_filtered=players[np.array(nb_observations)>=threshold_nb_data] #marche avec les array mais pas les listes
        dict_results = defaultdict(dict)
        for player in players_tested:           
            print(player)
            filter_player_games=self.data["Code anonyme"]==player
            data_pl=self.data[filter_player_games]
            data_pl_na = data_pl.dropna()    
            feature_values = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str, time stamp and HR information
            target_values = data_pl_na[target]
            #split the data
            xtrain,xcal,xtest,ytrain,ycal,ytest=train_conformalize_test_split_my(feature_values.values, target_values.values, shuffle=False,train_size=train_size,cal_size=cal_size,test_size=test_size, random_state=2025)
            nb_cal=xcal.shape[0]
            if best_regressor:
                best_models_str=resultsAnalysis(path_result).get_best_model()
                regressor_used_str=best_models_str[player]
                model_class,model_params=self.regressors[regressor_used_str]
                regressor_used=model_class(**model_params)
                params_search_reg=dict_params_search[regressor_used_str]
            else:
                regressor_used_str=regressor
                model_class,model_params=self.regressors[regressor]
                regressor_used=model_class(**model_params)
                params_search_reg=dict_params_search[regressor]
            #train + cal
            result_path = f'./results/forecasts/HR/{player}/{regressor_used_str}/'
            os.makedirs(result_path, exist_ok=True)
            file_name=result_path+f'train_size{train_size}_target{target}_cv_search{cv_search}_n_iter_search{n_iter_search}'+'.pkl'
            try:
                with open(file_name, 'rb') as handle:
                    best_model = pickle.load(handle)
            except:
                pip= Pipeline([('variance_0',VarianceThreshold()),('scaling','passthrough'),("feature_selection","passthrough"),("model",regressor_used)])
                model = RandomizedSearchCV(pip, params_search_reg, n_iter=n_iter_search, n_jobs = -1, cv=cv_search,random_state=2025)
                model_trained=model.fit(xtrain, ytrain)
                best_model=model_trained.best_estimator_
                with open(file_name, 'wb') as handle:
                    pickle.dump(best_model,handle)
            pred_cal= best_model.predict(xcal) 
            cal_scores=np.abs(pred_cal-ycal)
            #create the synthetic data
            nb_anomalies_used=int(xtest.shape[0]*prevalence/(1-prevalence)) 
            os.makedirs(f'synthetic_data/full_synthetic/{player}/', exist_ok=True)
            os.makedirs(f'synthetic_data/from_test/{player}/', exist_ok=True)
            if type_anomalie=='full_synthetic':
                try:
                    data_synth_df=pd.read_csv(f'synthetic_data/full_synthetic/{player}/{target}_{nb_anomalies_used}.csv')
                except:
                    data_synth_df=self.create_features_samples(player,target,nb_anomalies_used,regressor_used_str)
                x_anomaly_df=data_synth_df.drop(columns=['Y'])
                x_anomaly=x_anomaly_df.to_numpy()
                y_for_anomaly=data_synth_df['Y'].to_numpy()
            elif type_anomalie=='from_test':
                try:
                    with open(f'synthetic_data/from_test/{player}/{target}_{nb_anomalies_used}.pkl','rb') as handle:
                        dict_data_test_for_anomaly=pickle.load(handle)
                    x_anomaly=dict_data_test_for_anomaly['x_anomaly']
                    y_for_anomaly=dict_data_test_for_anomaly['y_for_anomaly']
                except:
                    indices = np.random.choice(len(xtest),size=nb_anomalies_used,replace=False)
                    x_anomaly=xtest[indices]
                    y_for_anomaly=ytest[indices]
                    with open(f'synthetic_data/from_test/{player}/{target}_{nb_anomalies_used}.pkl','wb') as handle:
                        pickle.dump({'x_anomaly':x_anomaly,'y_for_anomaly':y_for_anomaly},handle)
            else:
                raise(ValueError("Unknown method to create anomalies"))
            if isinstance(deviation,int):
                str_deviation=''
                deviation_adjusted=deviation
                table_deviation=np.ones((nb_anomalies_used,))*deviation
            else:
                str_deviation='%'
                deviation_adjusted=deviation*100
                table_deviation=y_for_anomaly*deviation
            if type_deviation=='plus':
                str_type_deviation='+'
                y_anomaly=y_for_anomaly+table_deviation
            elif type_deviation=='minus':
                str_type_deviation='-'
                y_anomaly=y_for_anomaly-table_deviation
            else:
                raise(ValueError('type_deviation must be "plus" or "minus"'))  
            #test
            full_pred=best_model.predict(np.concatenate((xtest,x_anomaly)))
            is_anomaly=[False for _ in range(len(xtest))]+[True for _ in range(nb_anomalies_used)]
            #ROC curve
            test_scores=np.abs(full_pred-np.concatenate((ytest,y_anomaly)))
            score_anomaly=np.array([(1+np.sum(cal_scores>=y))/(nb_cal+1) for y in test_scores])
            for alpha in alphas:
                q=np.quantile(cal_scores, np.min([1,np.ceil((nb_cal+1)*(1-alpha))/nb_cal]), method='higher')
                lb=full_pred-q
                ub=full_pred+q
                #decision rule
                is_anomaly_detected= [(y_test>u_b or y_test<l_b) for (y_test,u_b,l_b) in zip(np.concatenate((ytest,y_anomaly)),ub,lb)]
                df=pd.DataFrame(np.array([lb,ub,np.concatenate((ytest,y_anomaly)),score_anomaly,is_anomaly,is_anomaly_detected]).T,columns=['Lower bound','Upper bound','Ground truth','Anomaly score','Anomaly ?','Anomaly detected ?'])
                #print(df["Anomaly score"])
                dict_results[alpha][player]=df
        name_fold_anomalies= f"{prevalence}prevalence"
        os.makedirs(f'results/SCP/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/', exist_ok=True)
        for alpha in alphas:
            final_df=pd.concat(dict_results[alpha],names=['Player'])
            if( save_test_results==True):
                #save predictions results
                final_df.to_csv(f'results/SCP/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/07072023_16052025_alpha{alpha}.csv') 
        return final_df

    def conformalized_quantile_regression(self, players_tested, train_size, cal_size, test_size, target, dict_params_search_qr_l, dict_params_search_qr_u, cv_search, n_iter_search, alphas, save_test_results,deviation,type_deviation,type_anomalie,prevalence=0.5,best_regressor=False,path_result=None,regressor=None):
        for alpha in alphas:
            str_to_reg={                        
                    "XGB": (XGBRegressor, {"objective":"reg:quantileerror","quantile_alpha":alpha/2,"random_state": 2025},{"objective":"reg:quantileerror","quantile_alpha":1-(alpha/2),"random_state": 2025}),                
                    "LGBM": (LGBMRegressor, {"objective":'quantile',"alpha":alpha/2,"random_state": 2025,"verbose":-1},{"objective":'quantile',"alpha":1-(alpha/2),"random_state": 2025,"verbose":-1})          
                    }
            dict_results={}
            for player in players_tested:          
                print(player)
                filter_player_games=self.data["Code anonyme"]==player
                data_pl=self.data[filter_player_games]
                data_pl_na = data_pl.dropna()    
                feature_values = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str et time stamp
                target_values = data_pl_na[target]
                #split the data
                xtrain,xcal,xtest,ytrain,ycal,ytest=train_conformalize_test_split_my(feature_values.values, target_values.values, shuffle=False,train_size=train_size,cal_size=cal_size,test_size=test_size, random_state=2025)
                nb_cal=xcal.shape[0]
                if best_regressor:
                    best_models_str=resultsAnalysis(path_result).get_best_model_CQR()
                    regressor_used_str=best_models_str[player]
                    model_class,qr_l_params,qr_u_params=str_to_reg[regressor_used_str]
                    qr_l_used=model_class(**qr_l_params)
                    qr_u_used=model_class(**qr_u_params)
                    params_search_qr_l=dict_params_search_qr_l[regressor_used_str]
                    params_search_qr_u=dict_params_search_qr_u[regressor_used_str]
                else:
                    regressor_used_str=regressor
                    model_class,qr_l_params,qr_u_params=str_to_reg[regressor]
                    qr_l_used=model_class(**qr_l_params)
                    qr_u_used=model_class(**qr_u_params)
                    params_search_qr_l=dict_params_search_qr_l[regressor]
                    params_search_qr_u=dict_params_search_qr_u[regressor]
                result_path = f'./results/forecasts/HR/{player}/{regressor_used_str}/'
                os.makedirs(result_path, exist_ok=True)
                file_name_qr_lower=result_path+f'train_size{train_size}_target{target}_cv_search{cv_search}_n_iter_search{n_iter_search}_lower'+'.pkl'
                try:
                    with open(file_name_qr_lower, 'rb') as handle:
                        best_model_qr_l = pickle.load(handle)
                except:
                    pip_qr_l= Pipeline([('variance_0',VarianceThreshold()),('scaling','passthrough'),("feature_selection","passthrough"),("model",qr_l_used)])
                    model_qr_l = RandomizedSearchCV(pip_qr_l, params_search_qr_l,n_iter=n_iter_search,n_jobs = -1, cv=cv_search,random_state=2025)
                    model_qr_l_trained=model_qr_l.fit(xtrain,ytrain)
                    best_model_qr_l=model_qr_l_trained.best_estimator_
                    with open(file_name_qr_lower, 'wb') as handle:
                        pickle.dump(best_model_qr_l,handle)
                file_name_qr_upper=result_path+f'train_size{train_size}_target{target}_cv_search{cv_search}_n_iter_search{n_iter_search}_upper'+'.pkl'
                try:
                    with open(file_name_qr_upper, 'rb') as handle:
                        best_model_qr_u = pickle.load(handle)
                except:
                    #scaled + selection features: we apply the same transformations as the best ones obtained for qr_l
                    variance_threshold = best_model_qr_l.named_steps['variance_0'] 
                    scaling = best_model_qr_l.named_steps['scaling']  
                    features_selector = best_model_qr_l.named_steps['feature_selection']  
                    pip_qr_u= Pipeline([('variance_0',variance_threshold),('scaling',scaling),("feature_selection",features_selector),("model",qr_u_used)])
                    model_qr_u = RandomizedSearchCV(pip_qr_u, params_search_qr_u,n_iter=n_iter_search,n_jobs = -1, cv=cv_search,random_state=2025) 
                    model_qr_u_trained=model_qr_u.fit(xtrain,ytrain)
                    best_model_qr_u=model_qr_u_trained.best_estimator_
                    with open(file_name_qr_upper, 'wb') as handle:
                        pickle.dump(best_model_qr_u,handle)   
                #cal
                pred_cal_qr_l=best_model_qr_l.predict(xcal)
                pred_cal_qr_u=best_model_qr_u.predict(xcal)
                cal_scores=np.maximum(pred_cal_qr_l-ycal,ycal-pred_cal_qr_u)
                q=np.quantile(cal_scores, np.min([1,np.ceil((nb_cal+1)*(1-alpha))/nb_cal]), method='higher')  
                #create the synthetic data
                nb_anomalies_used=int(xtest.shape[0]*prevalence/(1-prevalence)) 
                os.makedirs(f'synthetic_data/full_synthetic/{player}/', exist_ok=True)
                os.makedirs(f'synthetic_data/from_test/{player}/', exist_ok=True)
                if type_anomalie=='full_synthetic':
                    try:
                        data_synth_df=pd.read_csv(f'synthetic_data/full_synthetic/{player}/{target}_{nb_anomalies_used}.csv')
                    except:
                        data_synth_df=self.create_features_samples(player,target,nb_anomalies_used,regressor_used_str)
                    x_anomaly_df=data_synth_df.drop(columns=['Y'])
                    x_anomaly=x_anomaly_df.to_numpy()
                    y_for_anomaly=data_synth_df['Y'].to_numpy()
                elif type_anomalie=='from_test':
                    try:
                        with open(f'synthetic_data/from_test/{player}/{target}_{nb_anomalies_used}.pkl','rb') as handle:
                            dict_data_test_for_anomaly=pickle.load(handle)
                        x_anomaly=dict_data_test_for_anomaly['x_anomaly']
                        y_for_anomaly=dict_data_test_for_anomaly['y_for_anomaly']
                    except:
                        indices = np.random.choice(len(xtest),size=nb_anomalies_used,replace=False)
                        x_anomaly=xtest[indices]
                        y_for_anomaly=ytest[indices]
                        with open(f'synthetic_data/from_test/{player}/{target}_{nb_anomalies_used}.pkl','wb') as handle:
                            pickle.dump({'x_anomaly':x_anomaly,'y_for_anomaly':y_for_anomaly},handle)
                else:
                    raise(ValueError("Unknown method to create anomalies"))
                if isinstance(deviation,int):
                    str_deviation=''
                    deviation_adjusted=deviation
                    table_deviation=np.ones((nb_anomalies_used,))*deviation
                else:
                    str_deviation='%'
                    deviation_adjusted=deviation*100
                    table_deviation=y_for_anomaly*deviation
                if type_deviation=='plus':
                    str_type_deviation='+'
                    y_anomaly=y_for_anomaly+table_deviation
                elif type_deviation=='minus':
                    str_type_deviation='-'
                    y_anomaly=y_for_anomaly-table_deviation
                else:
                    raise(ValueError('type_deviation must be "plus" or "minus"'))
                #test
                full_pred_qr_u=best_model_qr_u.predict(np.concatenate((xtest,x_anomaly)))
                full_pred_qr_l=best_model_qr_l.predict(np.concatenate((xtest,x_anomaly)))
                lb=full_pred_qr_l-q
                ub=full_pred_qr_u+q
                is_anomaly=[False for _ in range(len(xtest))]+[True for _ in range(nb_anomalies_used)]
                #decision rule
                is_anomaly_detected= [(y_test>u_b or y_test<l_b) for (y_test,u_b,l_b) in zip(np.concatenate((ytest,y_anomaly)),ub,lb)]
                #ROC curve
                test_scores=np.maximum(full_pred_qr_l-np.concatenate((ytest,y_anomaly)),np.concatenate((ytest,y_anomaly))-full_pred_qr_u)
                score_anomaly=[(1+np.sum(cal_scores>=y))/(nb_cal+1) for y in test_scores]
                df=pd.DataFrame(np.array([lb,ub,np.concatenate((ytest,y_anomaly)),score_anomaly,is_anomaly,is_anomaly_detected]).T,columns=['Lower bound','Upper bound','Ground truth','Anomaly score','Anomaly ?','Anomaly detected ?'])
                dict_results[player]=df
            final_df=pd.concat(dict_results,names=['Player'])
            if( save_test_results==True):
                #save predictions results
                name_fold_anomalies= f"{prevalence}prevalence"
                os.makedirs(f'results/CQR/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/', exist_ok=True)
                final_df.to_csv(f'results/CQR/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/07072023_16052025_alpha{alpha}.csv') 
        return final_df

    
    #jackknife+ method (to avoid calibration set) or CV+ but computationally expensive (retrain a lot of models). If is_jackknife_plus=True, jackkknife+ is performed. Otherwise CV+ and cv_CV_plus is the number of folds
    #D'après la librairie MAPIE, cette méthode devrait être utilisé quand on a moins de 200 données
    def cross_conformal_prediction(self,players_tested,train_size,target, alphas, dict_params_search,cv_search,n_iter_search,save_test_results,deviation,type_deviation,type_anomalie,prevalence=0.5,best_regressor=False,path_result=None,regressor=None,is_jackknife_plus=False,cv_CV_plus=20):
        dict_results=defaultdict(dict)
        for player in players_tested:           
            print(player)
            filter_player_games=self.data["Code anonyme"]==player
            data_pl=self.data[filter_player_games]
            data_pl_na = data_pl.dropna()    
            feature_values = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str et time stamp
            target_values = data_pl_na[target]
            #comment optimiser les hyperparametres du régresseur ? peut etre faire un pré training sur tout le set d'entrainement et garder les hyperparamètres: un peu de data leakage
            #split the data
            xtrain, xtest, ytrain, ytest = train_test_split(feature_values.values, target_values.values, shuffle=False,train_size=train_size, random_state=2025)
            if best_regressor:
                best_models_str=resultsAnalysis(path_result).get_best_model()
                regressor_used_str=best_models_str[player]
                model_class,model_params=self.regressors[regressor_used_str]
                regressor_used=model_class(**model_params)
                params_search_reg=dict_params_search[regressor_used_str]
            else:
                model_class,model_params=self.regressors[regressor]
                regressor_used=model_class(**model_params)
                params_search_reg=dict_params_search[regressor]

            result_path = f'./results/forecasts/HR/{player}/{regressor_used_str}/'
            os.makedirs(result_path, exist_ok=True)
            file_name_model_search=result_path+f'train_size{train_size}_target{target}_cv_search{cv_search}_n_iter_search{n_iter_search}'+'.pkl'
            try:
                with open(file_name_model_search, 'rb') as handle:
                    best_model_full = pickle.load(handle) 
            except:
                pip= Pipeline([('variance_0',VarianceThreshold()),('scaling','passthrough'),("feature_selection","passthrough"),("model",regressor_used)])
                model= RandomizedSearchCV(pip, params_search_reg,n_iter=n_iter_search,n_jobs = -1, cv=cv_search,random_state=2025) 
                model_trained=model.fit(xtrain, ytrain)
                best_model_full=model_trained.best_estimator_
                with open(file_name_model_search, 'wb') as handle:
                    pickle.dump(best_model_full,handle)
            variance_threshold = best_model_full.named_steps['variance_0'] 
            scaling = best_model_full.named_steps['scaling']  
            features_selector = best_model_full.named_steps['feature_selection']  
            model_hyperparameters = best_model_full.named_steps['model'].get_params()
            estimator=regressor_used.set_params(**model_hyperparameters) #** to unpack the dict
            pip_opt= Pipeline([('variance_0',variance_threshold),('scaling',scaling),("feature_selection",features_selector)])
            xtrain_transform=pip_opt.transform(xtrain)
            xtest_transform=pip_opt.transform(xtest)
            #train
            cv_CV_plus_used=xtrain.shape[0] if is_jackknife_plus else cv_CV_plus
            file_name_model_CCP=result_path+f'CCP_train_size{train_size}_target{target}_cv_search{cv_search}_n_iter_search{n_iter_search}_cv_CV_plus{cv_CV_plus_used}'+'.pkl'
            file_name_model_CCP_joblib=result_path+f'CCP_train_size{train_size}_target{target}_cv_search{cv_search}_n_iter_search{n_iter_search}_cv_CV_plus{cv_CV_plus_used}'+'.joblib'
            try:
                with open(file_name_model_CCP, 'rb') as handle:
                    CCP_regressor = pickle.load(handle)
                #CCP_regressor = joblib.load(file_name_model_CCP_joblib,mmap_mode="r")
            except:
                CCP_regressor=CrossConformalRegressor(estimator=estimator,confidence_level=(1-alphas),cv=cv_CV_plus_used,random_state=2025).fit_conformalize(xtrain_transform, ytrain)
                #joblib.dump(CCP_regressor,file_name_model_CCP_joblib)
                with open(file_name_model_CCP, 'wb') as handle:
                    pickle.dump(CCP_regressor,handle)
            #create the synthetic data
            nb_anomalies_used=int(xtest.shape[0]*prevalence/(1-prevalence)) 
            os.makedirs(f'synthetic_data/full_synthetic/{player}/', exist_ok=True)
            os.makedirs(f'synthetic_data/from_test/{player}/', exist_ok=True)
            if type_anomalie=='full_synthetic':
                try:
                    data_synth_df=pd.read_csv(f'synthetic_data/full_synthetic/{player}/{target}_{nb_anomalies_used}.csv')
                except:
                    data_synth_df=self.create_features_samples(player,target,nb_anomalies_used,regressor_used_str)
                x_anomaly_df=data_synth_df.drop(columns=['Y'])
                x_anomaly=x_anomaly_df.to_numpy()
                y_for_anomaly=data_synth_df['Y'].to_numpy()
            elif type_anomalie=='from_test':
                try:
                    with open(f'synthetic_data/from_test/{player}/{target}_{nb_anomalies_used}.pkl','rb') as handle:
                        dict_data_test_for_anomaly=pickle.load(handle)
                    x_anomaly=dict_data_test_for_anomaly['x_anomaly']
                    y_for_anomaly=dict_data_test_for_anomaly['y_for_anomaly']
                except:
                    indices = np.random.choice(len(xtest),size=nb_anomalies_used,replace=False)
                    x_anomaly=xtest[indices]
                    y_for_anomaly=ytest[indices]
                    with open(f'synthetic_data/from_test/{player}/{target}_{nb_anomalies_used}.pkl','wb') as handle:
                        pickle.dump({'x_anomaly':x_anomaly,'y_for_anomaly':y_for_anomaly},handle)
            else:
                raise(ValueError("Unknown method to create anomalies"))
            x_anomaly_transform=pip_opt.transform(x_anomaly)
            if isinstance(deviation,int):
                str_deviation=''
                deviation_adjusted=deviation
                table_deviation=np.ones((nb_anomalies_used,))*deviation
            else:
                str_deviation='%'
                deviation_adjusted=deviation*100
                table_deviation=y_for_anomaly*deviation
            if type_deviation=='plus':
                str_type_deviation='+'
                y_anomaly=y_for_anomaly+table_deviation
            elif type_deviation=='minus':
                str_type_deviation='-'
                y_anomaly=y_for_anomaly-table_deviation
            else:
                raise(ValueError('type_deviation must be "plus" or "minus"'))  
            #test
            _,full_pred_interval=CCP_regressor.predict_interval(np.concatenate((xtest_transform,x_anomaly_transform)))
            is_anomaly=[False for _ in range(len(xtest))]+[True for _ in range(nb_anomalies_used)]
            for i,alpha in enumerate(alphas):
                lb=full_pred_interval[:,0,i] #the last dimension is if alpha is a list or array
                ub=full_pred_interval[:,1,i]
                is_anomaly_detected= [(y_test>u_b or y_test<l_b) for (y_test,u_b,l_b) in zip(np.concatenate((ytest,y_anomaly)),ub,lb)]
                df=pd.DataFrame(np.array([lb,ub,np.concatenate((ytest,y_anomaly)),is_anomaly,is_anomaly_detected]).T,columns=['Lower bound','Upper bound','Ground truth','Anomaly ?','Anomaly detected ?'])
                dict_results[alpha][player]=df
        name_fold_anomalies= f"{prevalence}prevalence"
        os.makedirs(f'results/Jackknife+/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/', exist_ok=True)
        os.makedirs(f'results/CV+/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/', exist_ok=True)
        for alpha in alphas:
            final_df=pd.concat(dict_results[alpha],names=['Player'])
            if( save_test_results==True):
                if is_jackknife_plus:
                    final_df.to_csv(f'results/Jackknife+/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/07072023_16052025_alpha{alpha}.csv')
                else:
                    final_df.to_csv(f'results/CV+/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/07072023_16052025_cvccp{cv_CV_plus}_alpha{alpha}.csv')
        return final_df

    def cross_conformal_prediction_bis(
    self,
    players_tested,
    train_size,
    target,
    alphas,
    dict_params_search,
    cv_search,
    n_iter_search,
    save_test_results,
    deviations,
    type_deviations,
    type_anomalies,
    prevalence=0.5,
    best_regressor=False,
    path_result=None,
    regressor=None,
    is_jackknife_plus=False,
    cv_CV_plus=20,
):
        all_results = {}

        for player in players_tested:
            print(player)

            filter_player_games = self.data["Code anonyme"] == player
            data_pl = self.data[filter_player_games]
            data_pl_na = data_pl.dropna()

            feature_values = data_pl_na.drop(
                columns=[
                    target, 'FC_moyenne_perc', 'HRZ1', 'HRZ2', 'HRZ3',
                    'HRZ4', 'HRZ5', 'HRZ6', 'Date', 'Saison',
                    'Code anonyme', 'Effectif', 'debut_seq_format',
                    'fin_seq_format', 'Timestamp_debut', 'Timestamp_fin',
                    'meteo_source'
                ]
            )
            target_values = data_pl_na[target]

            # Split the data
            xtrain, xtest, ytrain, ytest = train_test_split(
                feature_values.values,
                target_values.values,
                shuffle=False,
                train_size=train_size,
                random_state=2025
            )

            if best_regressor:
                best_models_str = resultsAnalysis(path_result).get_best_model()
                regressor_used_str = best_models_str[player]
                model_class, model_params = self.regressors[regressor_used_str]
                regressor_used = model_class(**model_params)
                params_search_reg = dict_params_search[regressor_used_str]
            else:
                regressor_used_str = regressor
                model_class, model_params = self.regressors[regressor]
                regressor_used = model_class(**model_params)
                params_search_reg = dict_params_search[regressor]

            result_path = f'./results/forecasts/HR/{player}/{regressor_used_str}/'
            os.makedirs(result_path, exist_ok=True)

            file_name_model_search = (
                result_path
                + f'train_size{train_size}_target{target}_cv_search{cv_search}'
                f'_n_iter_search{n_iter_search}.pkl'
            )

            try:
                with open(file_name_model_search, 'rb') as handle:
                    best_model_full = pickle.load(handle)
            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                pip = Pipeline([
                    ('variance_0', VarianceThreshold()),
                    ('scaling', 'passthrough'),
                    ('feature_selection', 'passthrough'),
                    ('model', regressor_used)
                ])
                model = RandomizedSearchCV(
                    pip,
                    params_search_reg,
                    n_iter=n_iter_search,
                    n_jobs=-1,
                    cv=cv_search,
                    random_state=2025
                )
                model_trained = model.fit(xtrain, ytrain)
                best_model_full = model_trained.best_estimator_

                with open(file_name_model_search, 'wb') as handle:
                    pickle.dump(best_model_full, handle)

            variance_threshold = best_model_full.named_steps['variance_0']
            scaling = best_model_full.named_steps['scaling']
            features_selector = best_model_full.named_steps['feature_selection']
            model_hyperparameters = best_model_full.named_steps['model'].get_params()

            estimator = regressor_used.set_params(**model_hyperparameters)

            pip_opt = Pipeline([
                ('variance_0', variance_threshold),
                ('scaling', scaling),
                ('feature_selection', features_selector)
            ])

            xtrain_transform = pip_opt.transform(xtrain)
            xtest_transform = pip_opt.transform(xtest)

            # Fit/load CCP once per player
            cv_CV_plus_used = xtrain.shape[0] if is_jackknife_plus else cv_CV_plus

            file_name_model_CCP = (
                result_path
                + f'CCP_train_size{train_size}_target{target}_cv_search{cv_search}'
                f'_n_iter_search{n_iter_search}_cv_CV_plus{cv_CV_plus_used}.pkl'
            )

            try:
                with open(file_name_model_CCP, 'rb') as handle:
                    CCP_regressor = pickle.load(handle)
            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                CCP_regressor = CrossConformalRegressor(
                    estimator=estimator,
                    confidence_level=(1 - np.asarray(alphas)),
                    cv=cv_CV_plus_used,
                    random_state=2025
                ).fit_conformalize(xtrain_transform, ytrain)

                with open(file_name_model_CCP, 'wb') as handle:
                    pickle.dump(CCP_regressor, handle)

            # Reuse the fitted CCP model for every anomaly configuration
            for type_anomalie in type_anomalies:
                nb_anomalies_used = int(
                    xtest.shape[0] * prevalence / (1 - prevalence)
                )

                os.makedirs(
                    f'synthetic_data/full_synthetic/{player}/',
                    exist_ok=True
                )
                os.makedirs(
                    f'synthetic_data/from_test/{player}/',
                    exist_ok=True
                )

                if type_anomalie == 'full_synthetic':
                    synthetic_file = (
                        f'synthetic_data/full_synthetic/{player}/'
                        f'{target}_{nb_anomalies_used}.csv'
                    )
                    try:
                        data_synth_df = pd.read_csv(synthetic_file)
                    except FileNotFoundError:
                        data_synth_df = self.create_features_samples(
                            player,
                            target,
                            nb_anomalies_used,
                            regressor_used_str
                        )

                    x_anomaly_df = data_synth_df.drop(columns=['Y'])
                    x_anomaly = x_anomaly_df.to_numpy()
                    y_for_anomaly = data_synth_df['Y'].to_numpy()

                elif type_anomalie == 'from_test':
                    synthetic_file = (
                        f'synthetic_data/from_test/{player}/'
                        f'{target}_{nb_anomalies_used}.pkl'
                    )
                    try:
                        with open(synthetic_file, 'rb') as handle:
                            dict_data_test_for_anomaly = pickle.load(handle)

                        x_anomaly = dict_data_test_for_anomaly['x_anomaly']
                        y_for_anomaly = dict_data_test_for_anomaly['y_for_anomaly']

                    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                        indices = np.random.choice(
                            len(xtest),
                            size=nb_anomalies_used,
                            replace=False
                        )
                        x_anomaly = xtest[indices]
                        y_for_anomaly = ytest[indices]

                        with open(synthetic_file, 'wb') as handle:
                            pickle.dump({
                                'x_anomaly': x_anomaly,
                                'y_for_anomaly': y_for_anomaly
                            }, handle)

                else:
                    raise ValueError(
                        "Unknown anomaly method. Use 'full_synthetic' or 'from_test'."
                    )

                x_anomaly_transform = pip_opt.transform(x_anomaly)

                # Intervals depend on X, not on the artificial change applied to y.
                # Therefore calculate them only once for this anomaly dataset.
                x_all_transform = np.concatenate(
                    (xtest_transform, x_anomaly_transform)
                )
                _, full_pred_interval = CCP_regressor.predict_interval(
                    x_all_transform
                )

                is_anomaly = (
                    [False] * len(xtest)
                    + [True] * nb_anomalies_used
                )

                for type_deviation in type_deviations:
                    if type_deviation == 'plus':
                        str_type_deviation = '+'
                    elif type_deviation == 'minus':
                        str_type_deviation = '-'
                    else:
                        raise ValueError(
                            'type_deviation must be "plus" or "minus"'
                        )

                    for deviation in deviations:
                        if isinstance(deviation, (int, np.integer)):
                            str_deviation = ''
                            deviation_adjusted = deviation
                            table_deviation = (
                                np.ones((nb_anomalies_used,)) * deviation
                            )
                        else:
                            str_deviation = '%'
                            deviation_adjusted = deviation * 100
                            table_deviation = y_for_anomaly * deviation

                        if type_deviation == 'plus':
                            y_anomaly = y_for_anomaly + table_deviation
                        else:
                            y_anomaly = y_for_anomaly - table_deviation
                        y_all = np.concatenate((ytest, y_anomaly))
                        estimators = CCP_regressor._mapie_regressor.estimator_.estimators_
                        pred_test_models = np.array([
                            estimator.predict(x_all_transform)
                            for estimator in estimators
                        ])
                        test_scores_models = np.abs(
                            pred_test_models - y_all[None, :]
                        )
                        score_anomaly = np.zeros(len(y_all))
                        if is_jackknife_plus:
                            cal_scores = np.asarray(
                                CCP_regressor._mapie_regressor.conformity_scores_
                            ).ravel()
                            score_anomaly = (
                                1 + np.sum(
                                    cal_scores[:, None] >= test_scores_models,
                                    axis=0
                                )
                            ) / (len(cal_scores) + 1)
                        else:
                            n_splits = CCP_regressor._mapie_regressor.cv
                            assert n_splits==cv_CV_plus
                            splitter = KFold(
                                n_splits=n_splits,
                                shuffle=True,
                                random_state=2025
                            )
                            cal_scores = np.asarray(
                                CCP_regressor._mapie_regressor.conformity_scores_
                            ).ravel()
                            folds = list(splitter.split(xtrain_transform, ytrain))
                            n_cal = len(cal_scores)
                            for k, (_, val_idx) in enumerate(folds):
                                fold_cal_scores = cal_scores[val_idx]
                                score_anomaly += np.sum(
                                    fold_cal_scores[:, None] >=
                                    test_scores_models[k][None, :],
                                    axis=0
                                )
                            score_anomaly = (1 + score_anomaly) / (n_cal + 1)
                        for i, alpha in enumerate(alphas):
                            lb = full_pred_interval[:, 0, i]
                            ub = full_pred_interval[:, 1, i]
                            is_anomaly_detected = [
                                (y_value > upper or y_value < lower)
                                for y_value, upper, lower in zip(y_all, ub, lb)
                            ]
                            df = pd.DataFrame({
                                'Lower bound': lb,
                                'Upper bound': ub,
                                'Ground truth': y_all,
                                'Anomaly score': score_anomaly,
                                'Anomaly ?': is_anomaly,
                                'Anomaly detected ?': is_anomaly_detected
                            })
                            config_key = (
                                type_anomalie,
                                type_deviation,
                                deviation,
                                alpha
                            )
                            all_results.setdefault(config_key, {})
                            all_results[config_key][player] = df

        # Save one result file for each configuration and alpha
        final_results = {}

        for (
            type_anomalie,
            type_deviation,
            deviation,
            alpha
        ), player_results in all_results.items():

            if isinstance(deviation, (int, np.integer)):
                str_deviation = ''
                deviation_adjusted = deviation
            else:
                str_deviation = '%'
                deviation_adjusted = deviation * 100

            str_type_deviation = '+' if type_deviation == 'plus' else '-'
            name_fold_anomalies = f"{prevalence}prevalence"

            method = 'Jackknife+' if is_jackknife_plus else 'CV+'
            result_dir = (
                f'results/{method}/{name_fold_anomalies}/'
                f'{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/'
                f'{type_anomalie}/'
            )
            os.makedirs(result_dir, exist_ok=True)

            final_df = pd.concat(player_results, names=['Player'])

            if save_test_results:
                if is_jackknife_plus:
                    filename = (
                        f'07072023_16052025_alpha{alpha}.csv'
                    )
                else:
                    filename = (
                        f'07072023_16052025_cvccp{cv_CV_plus}_alpha{alpha}.csv'
                    )

                final_df.to_csv(result_dir + filename)

            final_results[
                (type_anomalie, type_deviation, deviation, alpha)
            ] = final_df

        return final_results

    
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
            count_best_reg_player=np.array([idx_best_reg_player.count(i) for i in range(data_pl.shape[0])]) #on compte combien de fois chaque regresseur est le meilleur 
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
            count_best_reg_player=np.array([idx_best_reg_player.count(i) for i in range(data_pl.shape[0])]) #on compte combien de fois chaque regresseur est le meilleur 
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
        reg_with_pinball_loss=['XGB','LGBM']
        for p in range(len(players)):
            filter_player=self.results["Code anonyme"]==players[p]
            data_pl=self.results[filter_player].reset_index()
            data_pl_filtered=data_pl[data_pl['Regressor'].isin(reg_with_pinball_loss)]
            idx_best_reg_player=[data_pl_filtered['RMSE'].idxmin(),data_pl_filtered['MAE'].idxmin(),data_pl_filtered['MAPE'].idxmin(),data_pl_filtered['R^2'].idxmax()]
            count_best_reg_player=np.array([idx_best_reg_player.count(i) for i in range(data_pl.shape[0])]) #on compte combien de fois chaque regresseur est le meilleur 
            best_reg_player=count_best_reg_player.argmax() #for a given player, the best model is the one with the best performances (BO4 on RMSE, MAE, MAPE and R^2) among the others
            idx_best_reg.append(best_reg_player)
        regressors=data_pl['Regressor'].to_list()
        best_reg=[regressors[i] for i in idx_best_reg]
        dict_best_mod_CQR=dict(zip(players,best_reg))
        return dict_best_mod_CQR

    def get_best_player(self):
        dict_best_model=self.get_best_model()
        players=self.results["Code anonyme"].unique()
        dict_mae={}
        dict_rmse={}
        for player in players:
            best_model_pl=dict_best_model[player]
            filter_player=self.results["Code anonyme"]==player
            data_pl=self.results[filter_player]
            data_pl_best_reg=data_pl[data_pl['Regressor']==best_model_pl]
            dict_rmse[player]=data_pl_best_reg['RMSE'].values[0]
            dict_mae[player]=data_pl_best_reg['MAE'].values[0]
        sorted_dict_rmse = dict(sorted(dict_rmse.items(),key=lambda item:item[1]))
        sorted_dict_mae = dict(sorted(dict_mae.items(),key=lambda item:item[1]))
        return(sorted_dict_rmse,sorted_dict_mae)

    #best features overall
    def best_features(self):
        #le nombre de features qui apparait le plus, nb features moyen
        selected_features=self.results['Features selected'].apply(ast.literal_eval) #plus safe que eval() car vérifie, avant d'éxécuter, que c'est un objet python valide
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
    
    def plot_bins(self,metric,nb_bins,colors):
        self.results["obs_bin"] = pd.qcut(self.results["Number of observations"],q=nb_bins,duplicates="drop",precision=0)
        binned = self.results.groupby(["Regressor", "obs_bin"])[metric].mean().reset_index()
        bins = binned["obs_bin"].astype(str).unique()
        regressors = ["KR","SVR","RF","XGB","LGBM"]
        x = range(nb_bins)
        bar_width = 0.8 / len(regressors)
        plt.figure()
        for i,reg in enumerate(regressors):
            subset = binned[binned["Regressor"] == reg]
            plt.bar([x_i+i*bar_width for x_i in x],subset[metric],width=bar_width,label=reg,color=colors[reg])
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

    def interval_size_analysis_full(self):
        interval_size=self.results['Upper bound']-self.results['Lower bound']
        stats=interval_size.agg(['mean','std','median'])
        return stats
    
    def f1(self):
        tnr,tpr,ppv,npv=self.coverage_and_anomalies()
        f1_score=2*tpr*ppv/(tpr+ppv)
        return f1_score

    def coverage_and_anomalies_by_player(self,player):
        results=self.results[self.results['Player']==player]
        normal_sample=results[results['Anomaly ?']==0]
        false_positive_rate=normal_sample['Anomaly detected ?'].mean()
        coverage=1-false_positive_rate
        anomaly_sample=results[results['Anomaly ?']==1]
        anomalies_detected=anomaly_sample['Anomaly detected ?'].mean()
        false_positive=normal_sample['Anomaly detected ?'].sum()
        true_negative=normal_sample.shape[0]-false_positive
        true_positive=anomaly_sample['Anomaly detected ?'].sum()
        false_negative=anomaly_sample.shape[0]-true_positive
        ppv=true_positive/(true_positive+false_positive)
        npv=true_negative/(true_negative+false_negative)
        return coverage,anomalies_detected,ppv,npv
        
    def interval_size_analysis_by_player(self,player):
        results=self.results[self.results['Player']==player]
        results_filtered=results[results['Anomaly ?']==0]
        interval_size=results_filtered['Upper bound']-results_filtered['Lower bound']
        stats=interval_size.agg(['mean','std','median'])
        return stats

    def interval_size_analysis_full_by_player(self,player):
        results=self.results[self.results['Player']==player]
        interval_size=results['Upper bound']-results['Lower bound']
        stats=interval_size.agg(['mean','std','median'])
        return stats
    
    def f1_by_player(self,player):
        tnr,tpr,ppv,npv=self.coverage_and_anomalies_by_player(player)
        f1_score=2*tpr*ppv/(tpr+ppv)
        return f1_score
    
    def AUROC_by_player(self,player):
        results=self.results[self.results['Player']==player]
        score_anomaly=results['Anomaly score']
        is_anomaly=results['Anomaly ?']
        score_anomaly_adjusted=1-score_anomaly.to_numpy()
        auroc=roc_auc_score(is_anomaly, score_anomaly_adjusted)
        return auroc

    def AUCPR_by_player(self,player):
        results=self.results[self.results['Player']==player]
        score_anomaly=results['Anomaly score']
        is_anomaly=results['Anomaly ?']
        score_anomaly_adjusted=1-score_anomaly.to_numpy()
        aucpr=average_precision_score(is_anomaly, score_anomaly_adjusted)
        return aucpr
    

    
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
    # Intervalles corrects
    ax.vlines(x=y_true[inside],ymin=lower[inside],ymax=upper[inside],color="tab:blue",linewidth=1.5,alpha=0.8,label="Inside prediction interval")
    ax.hlines(lower[inside], y_true[inside] - cap, y_true[inside] + cap, color="tab:blue")
    ax.hlines(upper[inside], y_true[inside] - cap, y_true[inside] + cap, color="tab:blue")
    # Intervalles incorrects
    ax.vlines(x=y_true[outside],ymin=lower[outside],ymax=upper[outside],color="tab:red",linewidth=1.5,alpha=0.8,label="Outside prediction interval")
    ax.hlines(lower[outside], y_true[outside] - cap, y_true[outside] + cap, color="tab:red")
    ax.hlines(upper[outside], y_true[outside] - cap, y_true[outside] + cap, color="tab:red")
    # Diagonale y = x
    lims = [120,195]
    ax.plot(lims, lims, "--", color="black", label="x=y")
    # Métriques
    # coverage = inside_.mean()
    # interval_width = (upper_ - lower_).mean()
    # ax.text(
    #     0.05, 0.95,
    #     f"Average coverage: {coverage:.3f}\nAverage interval width: {interval_width:.3f}",
    #     transform=ax.transAxes,
    #     va="top",
    #     bbox=dict(boxstyle="round", facecolor="white")
    # )
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
    # Légende globale
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

def train_conformalize_test_split_my(x,y,train_size,cal_size,test_size,shuffle,random_state):
    x_full_train, xtest, y_full_train, ytest = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
    cal_size_adjusted=cal_size/(1-test_size)
    xtrain, xcal, ytrain, ycal = train_test_split(
        x_full_train,
        y_full_train,
        test_size=cal_size_adjusted,
        random_state=random_state,
        shuffle=shuffle,
    )
    return xtrain,xcal,xtest,ytrain,ycal,ytest


# def LOF(path_data,target,players_tested,train_size,deviation,type_deviation,type_anomalie,prevalence=0.5):
#     data=pd.read_excel(path_data)
#     dict_results={}
#     for player in players_tested:
#         print(player)
#         filter_player_games=data["Code anonyme"]==player
#         data_pl=data[filter_player_games]
#         data_pl_na = data_pl.dropna()    
#         vectors = data_pl_na.drop(columns=['FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str et time stamp
#         vectors_train,vectors_test=train_test_split(vectors.values,train_size=train_size,shuffle=False,random_state=2025)
#         #TODO save .fit
#         lof=LocalOutlierFactor(novelty=True).fit(vectors_train)
#         nb_anomalies_used=int(vectors_test.shape[0]*prevalence/(1-prevalence)) 
#         os.makedirs(f'synthetic_data/full_synthetic/{player}/', exist_ok=True)
#         os.makedirs(f'synthetic_data/from_test/{player}/', exist_ok=True)
#         if type_anomalie=='full_synthetic':
#             data_synth_df=pd.read_csv(f'synthetic_data/full_synthetic/{player}/{target}_{nb_anomalies_used}.csv')
#             x_anomaly_df=data_synth_df.drop(columns=['Y'])
#             x_anomaly=x_anomaly_df.to_numpy()
#             y_for_anomaly=data_synth_df['Y'].to_numpy()
#         elif type_anomalie=='from_test':
#             with open(f'synthetic_data/from_test/{player}/{target}_{nb_anomalies_used}.pkl','rb') as handle:
#                 dict_data_test_for_anomaly=pickle.load(handle)
#             x_anomaly=dict_data_test_for_anomaly['x_anomaly']
#             y_for_anomaly=dict_data_test_for_anomaly['y_for_anomaly']
#         else:
#             raise(ValueError("Unknown method to create anomalies"))
#         if isinstance(deviation,int):
#             str_deviation=''
#             deviation_adjusted=deviation
#             table_deviation=np.ones((nb_anomalies_used,))*deviation
#         else:
#             str_deviation='%'
#             deviation_adjusted=deviation*100
#             table_deviation=y_for_anomaly*deviation
#         if type_deviation=='plus':
#             str_type_deviation='+'
#             y_anomaly=y_for_anomaly+table_deviation
#         elif type_deviation=='minus':
#             str_type_deviation='-'
#             y_anomaly=y_for_anomaly-table_deviation
#         else:
#             raise(ValueError('type_deviation must be "plus" or "minus"'))  
#         vectors_anomaly=np.column_stack((x_anomaly,y_anomaly))
#         full_pred=lof.predict(np.concatenate((vectors_test,vectors_anomaly)))
#         is_anomaly=[False for _ in range(len(vectors_test))]+[True for _ in range(nb_anomalies_used)]
#         is_anomaly_detected= [1 if pred==-1 else 0 for pred in full_pred]
#         df=pd.DataFrame(np.array([np.concatenate((vectors_test[:,-1],y_anomaly)),is_anomaly,is_anomaly_detected]).T,columns=['Ground truth','Anomaly ?','Anomaly detected ?'])
#         dict_results[player]=df
#     name_fold_anomalies= f"{prevalence}prevalence"
#     os.makedirs(f'results/LOF/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/', exist_ok=True)
#     final_df=pd.concat(dict_results,names=['Player'])
#     final_df.to_csv(f'results/LOF/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/07072023_16052025.csv') 
#     return final_df

def one_class_SVM(path_data,target,players_tested,train_size,deviation,type_deviation,type_anomalie,prevalence=0.5):
    data=pd.read_excel(path_data)
    dict_results={}
    for player in players_tested:
        print(player)
        filter_player_games=data["Code anonyme"]==player
        data_pl=data[filter_player_games]
        data_pl_na = data_pl.dropna()    
        vectors=data_pl_na.drop(columns=['FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str et time stamp
        target_val=vectors[target].values
        vectors_train,vectors_test,target_train,target_test=train_test_split(vectors.values,target_val,train_size=train_size,shuffle=False,random_state=2025)
        #TODO save .fit
        lof=OneClassSVM().fit(vectors_train)
        nb_anomalies_used=int(vectors_test.shape[0]*prevalence/(1-prevalence)) 
        if type_anomalie=='full_synthetic':
            data_synth_df=pd.read_csv(f'synthetic_data/full_synthetic/{player}/{target}_{nb_anomalies_used}.csv')
            x_anomaly_df=data_synth_df.drop(columns=['Y'])
            x_anomaly=x_anomaly_df.to_numpy()
            y_for_anomaly=data_synth_df['Y'].to_numpy()
        elif type_anomalie=='from_test':
            with open(f'synthetic_data/from_test/{player}/{target}_{nb_anomalies_used}.pkl','rb') as handle:
                dict_data_test_for_anomaly=pickle.load(handle)
            x_anomaly=dict_data_test_for_anomaly['x_anomaly']
            y_for_anomaly=dict_data_test_for_anomaly['y_for_anomaly']
        else:
            raise(ValueError("Unknown method to create anomalies"))
        if isinstance(deviation,int):
            str_deviation=''
            deviation_adjusted=deviation
            table_deviation=np.ones((nb_anomalies_used,))*deviation
        else:
            str_deviation='%'
            deviation_adjusted=deviation*100
            table_deviation=y_for_anomaly*deviation
        if type_deviation=='plus':
            str_type_deviation='+'
            y_anomaly=y_for_anomaly+table_deviation
        elif type_deviation=='minus':
            str_type_deviation='-'
            y_anomaly=y_for_anomaly-table_deviation
        else:
            raise(ValueError('type_deviation must be "plus" or "minus"'))  
        vectors_anomaly=np.column_stack((x_anomaly,y_anomaly))
        full_pred=lof.predict(np.concatenate((vectors_test,vectors_anomaly)))
        is_anomaly=[False for _ in range(len(vectors_test))]+[True for _ in range(nb_anomalies_used)]
        is_anomaly_detected= [1 if pred==-1 else 0 for pred in full_pred] 
        df=pd.DataFrame(np.array([np.concatenate((target_test,y_anomaly)),is_anomaly,is_anomaly_detected]).T,columns=['Ground truth','Anomaly ?','Anomaly detected ?'])
        dict_results[player]=df
    name_fold_anomalies= f"{prevalence}prevalence"
    os.makedirs(f'results/OCSVM/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/', exist_ok=True)
    final_df=pd.concat(dict_results,names=['Player'])
    final_df.to_csv(f'results/OCSVM/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/07072023_16052025.csv') 
    return final_df

def residual_threshold(path_data,path_result,target,cv_search,n_iter_search,thresholds,players_tested,train_size,deviation,type_deviation,type_anomalie,prevalence=0.5):
    data=pd.read_excel(path_data)
    dict_results = defaultdict(dict)
    for player in players_tested:
        print(player)
        filter_player_games=data["Code anonyme"]==player
        data_pl=data[filter_player_games]
        data_pl_na = data_pl.dropna()    
        feature_values = data_pl_na.drop(columns=[target,'FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str et time stamp
        target_values = data_pl_na[target]
        _, xtest, _, ytest =train_test_split(feature_values.values,target_values.values,train_size=train_size,shuffle=False,random_state=2025)
        best_models_str=resultsAnalysis(path_result).get_best_model()
        regressor_used_str=best_models_str[player]
        result_path = f'./results/forecasts/HR/{player}/{regressor_used_str}/'
        file_name=result_path+f'train_size{train_size}_target{target}_cv_search{cv_search}_n_iter_search{n_iter_search}'+'.pkl'
        with open(file_name, 'rb') as handle:
            best_model = pickle.load(handle)
        nb_anomalies_used=int(xtest.shape[0]*prevalence/(1-prevalence)) 
        if type_anomalie=='full_synthetic':
            data_synth_df=pd.read_csv(f'synthetic_data/full_synthetic/{player}/{target}_{nb_anomalies_used}.csv')
            x_anomaly_df=data_synth_df.drop(columns=['Y'])
            x_anomaly=x_anomaly_df.to_numpy()
            y_for_anomaly=data_synth_df['Y'].to_numpy()
        elif type_anomalie=='from_test':
            with open(f'synthetic_data/from_test/{player}/{target}_{nb_anomalies_used}.pkl','rb') as handle:
                dict_data_test_for_anomaly=pickle.load(handle)
            x_anomaly=dict_data_test_for_anomaly['x_anomaly']
            y_for_anomaly=dict_data_test_for_anomaly['y_for_anomaly']
        else:
            raise(ValueError("Unknown method to create anomalies"))
        if isinstance(deviation,int):
            str_deviation=''
            deviation_adjusted=deviation
            table_deviation=np.ones((nb_anomalies_used,))*deviation
        else:
            str_deviation='%'
            deviation_adjusted=deviation*100
            table_deviation=y_for_anomaly*deviation
        if type_deviation=='plus':
            str_type_deviation='+'
            y_anomaly=y_for_anomaly+table_deviation
        elif type_deviation=='minus':
            str_type_deviation='-'
            y_anomaly=y_for_anomaly-table_deviation
        else:
            raise(ValueError('type_deviation must be "plus" or "minus"'))  
        full_pred=best_model.predict(np.concatenate((xtest,x_anomaly)))
        is_anomaly=[False for _ in range(len(xtest))]+[True for _ in range(nb_anomalies_used)]
        score_anomaly=np.abs(np.concatenate((ytest,y_anomaly))-full_pred)
        for threshold in thresholds:
            is_anomaly_detected= np.abs(np.concatenate((ytest,y_anomaly))-full_pred)>=threshold
            df=pd.DataFrame(np.array([full_pred,np.concatenate((ytest,y_anomaly)),score_anomaly,is_anomaly,is_anomaly_detected]).T,columns=['Prediction','Ground truth','Anomaly score','Anomaly ?','Anomaly detected ?'])
            dict_results[threshold][player]=df
    name_fold_anomalies= f"{prevalence}prevalence"
    os.makedirs(f'results/Res-Threshold/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/', exist_ok=True)
    for threshold in thresholds:
        final_df=pd.concat(dict_results[threshold],names=['Player'])
        final_df.to_csv(f'results/Res-Threshold/{name_fold_anomalies}/{str_type_deviation}{deviation_adjusted}{str_deviation}bpm/{type_anomalie}/07072023_16052025_threshold{threshold}.csv') 
    return final_df

def AUROC_by_player_res(path_result_res,player):
    results_all=pd.read_csv(path_result_res)
    results=results_all[results_all['Player']==player]
    score_anomaly=results['Anomaly score']
    is_anomaly=results['Anomaly ?']
    auroc=roc_auc_score(is_anomaly, score_anomaly)
    return auroc

def AUCPR_by_player_res(path_result_res,player):
    results_all=pd.read_csv(path_result_res)
    results=results_all[results_all['Player']==player]
    score_anomaly=results['Anomaly score']
    is_anomaly=results['Anomaly ?']
    aucpr=average_precision_score(is_anomaly, score_anomaly)
    return aucpr

if __name__=="__main__":
    # best_model=resultsAnalysis('results/07072023_16052025_full_prediction.csv').get_best_model()  
    best_players=['MHSC-71','MHSC-11','MHSC-10','MHSC-29','MHSC-67','MHSC-81','MHSC-45','MHSC-32','MHSC-07','MHSC-51',
              'MHSC-75','MHSC-18','MHSC-73','MHSC-56','MHSC-69','MHSC-19','MHSC-65','MHSC-08','MHSC-23','MHSC-66',
              'MHSC-62']
    # list_=[best_model[player] for player in best_players]
    # regressors=['KR','LGBM','SVR','XGB','RF']
    # count=[list_.count(i) for i in regressors]
    # dict_best_pl_mod=dict(zip(regressors,count))
    # dict_=dict(zip(best_players,list_))
    # print(dict_best_pl_mod)
    # print(dict_)

    #data=pd.read_excel('global_data_7_july_2023_16_may_2025.xlsx')
    # players=dta["Code anonyme"].unique() 
    # filter_player_games=data["Code anonyme"]=='MHSC-10'
    # data_pl=data[filter_player_games]
    # data_pl_na = data_pl.dropna()    
    # feature_values = data_pl_na.drop(columns=['FC_moyenne','FC_moyenne_perc','HRZ1','HRZ2','HRZ3','HRZ4','HRZ5','HRZ6','Date','Saison','Code anonyme','Effectif','debut_seq_format','fin_seq_format','Timestamp_debut','Timestamp_fin','meteo_source']) #remove str et time stamp
    # target_values = data_pl_na['FC_moyenne']
    # #split the data
    # xtrain,xcal,xtest,ytrain,ycal,ytest=train_conformalize_test_split_my(feature_values.values, target_values.values, shuffle=False,train_size=0.7,cal_size=0.1,test_size=0.2, random_state=2025)
    # xtrain_b, xtest_b, ytrain_b, ytest_b = train_test_split(feature_values.values, target_values.values, shuffle=False,train_size=0.8, random_state=2025)
    # print(xtrain.shape[0],xcal.shape[0],xtest.shape[0])
    # print(xtrain_b.shape[0],xtest_b.shape[0])
    # nb_observations=[dta[dta["Code anonyme"]==player].shape[0] for player in players] #nb of obervations by player
    # players_filtered=players[np.array(nb_observations)>=800]
    # print(players_filtered)
    # threshold_nb_data=800
    # test_size=0.1 
    # calibration_size=0.2  
    # target='FC_moyenne'
    # regressor=xgb.XGBRegressor(eval_metric='rmsle')
    # params_grid_scp_err={"scaling":[MinMaxScaler(),StandardScaler()],'feature_selection': [SelectKBest(mutual_info_regression)],
    #                             'feature_selection__k': [5*k for k in range (1,12)],"model__max_depth":[4, 5, 6], "model__n_estimators": [200, 300, 400], "model__learning_rate": [0.01, 0.015]}
    # cv_search=5
    # alpha=0.1
    # a,b,c=trainingHR('global_data_7_july_2023_16_may_2025.xlsx').splitConformalPrediction(threshold_nb_data,test_size,calibration_size,target,regressor,params_grid_scp_err,cv_search,alpha)
    # test=trainingHR('global_data_7_july_2023_16_may_2025.xlsx').create_features_samples('MHSC-45','FC_moyenne',100)
    # print(test[0])
    a=np.array([1,2,2,2,2])
    # b=np.random.normal(20,5,(5,))
    # ysynth=a+b
    # print((a+b)//2,a,b)
    # rng=np.random.default_rng(2025)
    # perc_obs_plot = 0.02
    # nb_samples=1800
    # num_plots_idx = rng.choice(nb_samples, int(perc_obs_plot *nb_samples), replace=False)
    # print(num_plots_idx)
    # print(np.linspace(0.05,0.3,20))
    # a=np.array([1,2,3,4,5,6])
    # n=6
    # alpha=0.3
    # s=np.ceil((n+1)*(1-alpha))/n
    # print(s)
    # print(np.quantile(a,s,method="higher"),np.quantile(a,s,method="lower"))
    # X, y = np.arange(10).reshape((5, 2)), range(5)
    # X_train, X_conformalize, X_test,y_train, y_conformalize, y_test = train_conformalize_test_split(X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=1, shuffle=False)
    # print(X_train,X_conformalize)
    # path_data='global_data_7_july_2023_16_may_2025.xlsx'
    # path_result='results/07072023_16052025_full_prediction.csv'
    # df=residual_threshold(path_data,path_result,'FC_moyenne',5,200,[5],['MHSC-71'],0.8,0.01,'plus','from_test',prevalence=0.5)
    # path_result_anomaly='results/SCP/0.5prevalence/+10bpm/from_test/07072023_16052025_alpha0.1.csv'
    # result=CPAnalysis(path_result_anomaly)
    # print(result.AUROC_by_player('MHSC-71'))
    #df=one_class_SVM(path_data,'FC_moyenne',['MHSC-71'],0.8,0.01,'plus','from_test',prevalence=0.5)
    print(np.sum(a>=2))

