import numpy as np

# lgb
import lightgbm as lgb

# pytorch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch



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
    
    
    def evaluate(self, true_y, pred_y):
        """Using a evaluation method to evaluate the model

        Args:
            true_y (_type_): _description_
            pred_y (_type_): _description_
        """
        return self.root_mean_squared_percentage_error(true_y, pred_y)
    
    
    def metrics(self, true_y, pred_y):
        """Get multiple metrics for models

        Args:
            true_y (_type_): _description_
            pred_y (_type_): _description_

        Returns:
            _type_: _description_
        """
        return {
            'RMSPE' : self.root_mean_squared_percentage_error(true_y, pred_y)
        }
    
    
    def root_mean_squared_percentage_error(self, true_y, pred_y):
        """metric for modle evaluation"""
        rmspe = np.sqrt(np.mean(((pred_y - true_y) / true_y)**2))*100
        return rmspe
