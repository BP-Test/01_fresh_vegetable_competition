# Setting
import os
import sqlite3
from pathlib import Path

# default
import numpy as np
import pandas as pd
from functools import partial

# Manage experiments
import mlflow

# For training
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# model import
from models import Model

# import Optuna
import optuna
from optuna.integration.mlflow import MLflowCallback







class Experiments():
    def __init__(self, EXPERIMENT_NAME, DB_DIR_PATH = '../server/', ARTIFACT_LOCATION = '/data/'):
        # mlflow setting
        self.DB_DIR_PATH = DB_DIR_PATH
        self.DB_PATH = DB_DIR_PATH +'mlruns.db'
        self.ARTIFACT_LOCATION = Path('.').resolve().parents[0].__str__().replace('\\', '/') + ARTIFACT_LOCATION
        self.EXPERIMENT_NAME = EXPERIMENT_NAME
        
        # locate tracking server
        self.TRACKING_URL = f'sqlite:///{self.DB_PATH}'
        
        # make DB
        if 'mlruns.db' not in os.listdir('../server/'):
            os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)  # 親ディレクトリなければ作成
            conn = sqlite3.connect(self.DB_PATH)  # バックエンド用DBを作成
        else:
            pass
    
    def ready_experiment(self, settings = None):
        # Experimentの生成
        self.experiment = mlflow.get_experiment_by_name(self.EXPERIMENT_NAME)
        
        # experiment IDの取得
        if self.experiment is None:
            # 当該Experiment存在しないとき、新たに作成
            self.experiment_id = mlflow.create_experiment(
                name=self.EXPERIMENT_NAME,
                artifact_location=self.ARTIFACT_LOCATION
                )
        else:
            # 当該Experiment存在するとき、IDを取得
            self.experiment_id = self.experiment.experiment_id
            
        
        # Get setting info
        if settings is not None:
            self.settings = settings            
            
            # Add learning Class
            self.learning_process = Learning(settings['model_params'])
    
    
    def optimize(self, X, y):
        # mlflow callbacks
        self.ml_callback = MLflowCallback(
            tracking_uri=self.TRACKING_URL,
            metric_name="RPMSE",
        )
        
        # Conduct optimizing model
        self.learning_process.optimizer(X, y, callbacks = [self.ml_callback])
    
    
    def start_experiment(self, X, y, verification_type = 'CV'):
        
        
        # Connect to tracking server
        mlflow.set_tracking_uri(self.TRACKING_URL)
        
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            
            # record model parameter settings
            mlflow.log_params(
                self.learning_process.param_dict['params']
            )
            
            # add tag info
            mlflow.set_tags(self.settings['tag_info'])
            
            # add data processing info
            mlflow.log_params(
                self.settings['data_processing']
            )
            
            # cross validation learning
            if verification_type == 'CV':
                self.learning_process.cross_validate_learning(X, y, self.settings['CV'])
            
            
        # At the end of experiments
        print('----------------------------------------------------')
        print('command1: cd ' + self.DB_DIR_PATH)
        print('command2: mlflow ui --backend-store-uri sqlite:///mlruns.db')
    
    
    
    def best_model_predict(self, X_test):
        return self.learning_process.model.predict(X_test)










class Learning():
    def __init__(self, param_settings):
        
        # modelのパラメータを記録する
        self.param_dict = param_settings
    
    
    def learning_model(self, X_train, X_valid, y_train, y_valid, trail = None):
        # generate instance
        self.model = Model(self.param_dict)
        

        # Convert data for LightGMB
        lgb_train = self.model.dataset(X_train, y_train)
        lgb_eval = self.model.dataset(X_valid, y_valid)

        # model train
        self.model.train(
            train_data = lgb_train, 
            valid_data = lgb_eval
            )
        
        # 予測
        y_valid_pred = self.model.predict(X_valid)
        
        # score metrics
        self.score = self.model.metrics(y_valid, y_valid_pred)
        
        # return
        return self.model.evaluate(y_valid, y_valid_pred)
    
    
    
    def optimizer(self, X, y, callbacks = None):
        
        X_train, X_valid, y_train, y_valid = train_test_split(X, y)
        
        # create optimizer object
        self.study = optuna.create_study()
        
        # exec optimizer
        self.study.optimize(
            partial(self.learning_model, X_train, X_valid, y_train, y_valid), 
            n_trials=100,
            callbacks = callbacks
            )
    
    
    
    def cross_validate_learning(self, X, Y, CV_setting):
        # k-foldの生成
        self.kf = KFold(
            **CV_setting
            )
        
        # record validation score
        valid_scores = []
        
        # Cross Validation Learning
        for fold, (train_indices, valid_indices) in enumerate(self.kf.split(X)):
            
            
            # split X, y into train data and valid data
            X_train, X_valid = X[train_indices], X[valid_indices]
            y_train, y_valid = Y[train_indices], Y[valid_indices]
            
            
            # learning
            self.learning_model(X_train, X_valid, y_train, y_valid)
            
            
            # record the  metrics
            mlflow.log_metrics(
                self.score,
                step = fold
            )
            print(f'=== fold {fold} MAE: {self.score}')
            valid_scores.append(list(self.score.values()))
        
        # Get mean of model scores
        # TODO: Make mlflow decorator in Environment class 
        cv_score = np.mean(valid_scores)
        mlflow.log_metrics(
            {
                'CV_score' : cv_score
            }
        )
        print(f'=== CV score: {cv_score}')
    
    
    