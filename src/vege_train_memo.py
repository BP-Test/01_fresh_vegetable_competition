# default
import numpy as np
import pandas as pd

# Manage experiments
import mlflow

# For training
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# import model framework
import lightgbm as lgb


class Model(object):
    def __init__(self, params, model_type = 'lightGBM'):
        self.params = params
        self.model_type = model_type
        pass
    
    
    def dataset(self, X, y):
        if self.model_type == 'lightGBM':
            # Convert data for LightGMB
            return lgb.Dataset(X, y)
    
    
    def train(self, train_data, valid_data):
        # lightGBMの場合
        if self.model_type == 'lightGBM':
            
            self.trained_model = lgb.train(
                **self.params,
                train_set=train_data,
                valid_sets=valid_data
                )
    
    def predict(self, *args):
        return self.trained_model.predict(*args)
    
    
    
    def metrics(self, true_y, pred_y):
        return {
            'RMSPE' : self.root_mean_squared_percentage_error(true_y, pred_y)
        }
    
    
    def root_mean_squared_percentage_error(self, true_y, pred_y):
        """metric for modle evaluation"""
        rmspe = np.sqrt(np.mean(((pred_y - true_y) / true_y)**2))*100
        return rmspe










class Experiments():
    def __init__(self, EXPERIMENT_NAME, DB_DIR_PATH = '../server/', ARTIFACT_LOCATION = '../data/'):
        # mlflow setting
        self.DB_DIR_PATH = DB_DIR_PATH
        self.DB_PATH = DB_DIR_PATH +'mlruns.db'
        self.ARTIFACT_LOCATION = '../data/'
        self.EXPERIMENT_NAME = EXPERIMENT_NAME
        
        # トラッキングサーバの（バックエンドの）場所を指定
        TRACKING_URL = f'sqlite:///{self.DB_PATH}'
        mlflow.set_tracking_uri(TRACKING_URL)
        
    
    
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
            
            # modelのパラメータを記録する
            self.param_dict = settings['model_params']
    
    
    
    def start_experiment(self, X, y, verification_type = 'CV'):
        
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            
            # record model parameter settings
            mlflow.log_params(
                self.param_dict
            )
            
            # cross validation learning
            if verification_type == 'CV':
                self.cross_validate_learning(X, y)
            
            
            # add tag info
            mlflow.set_tags(self.settings['tag_info'])
            
            
        # At the end of experiments
        print('----------------------------------------------------')
        print('command1: cd ' + self.DB_DIR_PATH)
        print('command2: mlflow ui --backend-store-uri sqlite:///mlruns.db')
    
    
    
    
    def cross_validate_learning(self, X, Y):
        # k-foldの生成
        self.kf = KFold(
            **self.settings['CV']
            )
        
        # record validation score
        valid_scores = []
        
        # モデルの学習に利用したパラメータを記録
        mlflow.log_params(
            self.settings['CV']
            )
        
        
        # generate instance
        model = Model(self.param_dict)
        
        for fold, (train_indices, valid_indices) in enumerate(self.kf.split(X)):
            
            # split X, y into train data and valid data
            X_train, X_valid = X[train_indices], X[valid_indices]
            y_train, y_valid = Y[train_indices], Y[valid_indices]

            # Convert data for LightGMB
            lgb_train = model.dataset(X_train, y_train)
            lgb_eval = model.dataset(X_valid, y_valid)

            # model train
            model.train(
                train_data = lgb_train, 
                valid_data = lgb_eval
                )
            
            # 予測
            y_valid_pred = model.predict(X_valid)
            
            # score metrics
            score = model.metrics(y_valid, y_valid_pred)
            
            # record the  metrics
            mlflow.log_metrics(
                score,
                step = fold
            )
            print(f'=== fold {fold} MAE: {score}')
            valid_scores.append(list(score.values()))
        
        # Get mean of model scores
        cv_score = np.mean(valid_scores)
        mlflow.log_metrics(
            {
                'CV_score' : cv_score
            }
        )
        print(f'=== CV score: {cv_score}')
    
    
    