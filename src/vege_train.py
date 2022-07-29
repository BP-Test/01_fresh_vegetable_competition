# Essential Libraries
import os
import sys
from pathlib import Path
from dateutil import parser # Refer to https://qiita.com/xza/items/9618e25a8cb08c44cdb0 for details.
from datetime import datetime, timedelta
import pickle
import pandas as pd
import numpy as np

#Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Model Libraries
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# Arguments
sys.dont_write_bytecode=True
sns.set(font="IPAexGothic")
plt.rcParams['font.family'] = "MS Gothic" # Use this font to avoid encoding error

class TrainModel():
    # TODO: This Class will output something that 
    #
    def __init__(self,
                proj_path='',
                #awareness_file='',
                clean_data=True,
                dropdates=True,
                co_file='',
                #smoothdata=False,
                get_monthly=True,
                reg_file='',
                feats_to_remove=[],
                feats_to_keep=[],
                feats_to_leadstring=[],
                target_col= 'mode_price',#TARGET
                impute_data=True,
                norm_data=True,
                synth_data=True,
                save_dir='savefile'
                ):
        print('Start TrainModel')
        projpath = Path(__file__).parents[1] # parent directory
        if proj_path:
            self.projpath = Path(projpath).parent
        else:
            self.projpath = projpath
        self.get_monthly = get_monthly
        self.features = feats_to_keep
        self.feats_keep_leadstring = feats_keep_leadstring
        self.target = target_col
        self.features_to_excl = feats_to_remove
        self.savedir = save_dir
        self.impute_data = impute_data
        self.norm_data = norm_data
        self.awareness_file = awareness_file
        self.dropdates = dropdates
        self.co_file = co_file
        self.reg_file = reg_file
        self.clean_data = clean_data
        self.smoothdata = smoothdata
        self.synth_data = synth_data

        try:
            (self.projpath/'figures'/self.savedir).mkdir(parents=True,exist_ok=False)
        except FileExistsError:
            print("Folder is already there")
        else:
            print("Folder was created")

    def root_mean_squared_percentage_error(self, true_y, pred_y):
        # TODO: Docstring
        rmspe = np.sqrt(np.mean(((pred_y - true_y) / true_y)**2))*100
        return rmpse

    def get_vege_data(self, train=True):
        # TODO: Docstring
        train_df = pd.read_csv(self.projpath / 'data/preprocessed_train.csv')
        test_df = pd.read_csv(self.projpath / 'data/preprocessed_test.csv')
        train_df['year'] = train_df['date']//10000
        test_df['year'] = test_df['date']//10000
        train_df['month'] = train_df['date'].apply(lambda x: int(str(x)[4:6]))
        test_df['month'] = test_df['date'].apply(lambda x: int(str(x)[4:6]))
        if train==True:
            print(train_df.shape)
            train_df.describe()
            # Drop Vegetables that are not in the test dataset
            kinds = test_df['kind'].unique()
            train_df = train_df[train_df['kind'].isin(kinds)]
            print(train_df.shape)
            train_df
            return train_df
        else:
            print(test_df.shape)
            test_df.describe()
            return test_df
    def clean_vege_data(self, _train=True):
        # TODO: Docstring
        train = self.get_vege_data(train = True)
        test = self.get_vege_data(train = False)
        if _train ==True:
            return train
        else:
            return test
    def generate_dummy_data(self, train=True):
        # TODO: Docstring
        max_days = (datetime(2022, 12, 31) - datetime(2005, 1, 1)).days
        dum_data = []

        for i in range(max_days+1):
            date = datetime(2005, 1, 1) + timedelta(days=i)
            y, wn = date.isocalendar()[0], date.isocalendar()[1]
            date = int(date.strftime('%Y%m%d'))
            m = int(str(date)[4:6])
            dum_data.append(['ダミー', date, 0, 0, 'ダミー', y, m])

        dum_df = pd.DataFrame(dum_data, columns=all_df.columns)
        print(dum_df.head())
        return dum_df

    def pivot_data():
        # TODO: Docstring
        data = pd.pivot_table(train_df.query('20210501 <= date <= 20220430'), index='kind', columns='month', values=TARGET, aggfunc='count')
        return data
    def lag_feature(self,):
        # TODO: Docstring

    def light_gbm(self,):
        # TODO: Docstring
    def plot_feature_importance(self):
        # TODO: Docstring
        pass
    def generate_lag_feature(self):
        # TODO: Docstring
        pass




class LightgbmModel(TrainModel):
        # TODO: Docstring

        def train_model():
            return 
        def lag_vote():
            return top_lags
        def