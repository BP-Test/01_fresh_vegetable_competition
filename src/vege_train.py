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
                feats_keep_leadstring=[],
                target_col= 'mode_price',#TARGET
                impute_data=True,
                norm_data=True,
                synth_data=True,
                save_dir=''
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
        #self.awareness_file = awareness_file
        self.dropdates = dropdates
        self.co_file = co_file
        self.reg_file = reg_file
        self.clean_data = clean_data
        #self.smoothdata = smoothdata
        self.synth_data = synth_data
        self.kinds = ['だいこん', 'にんじん', 'キャベツ', 'レタス', 'はくさい', 'こまつな', 'ほうれんそう', 'ねぎ',\
        'きゅうり', 'トマト', 'ピーマン', 'じゃがいも', 'なましいたけ', 'セルリー', 'そらまめ', 'ミニトマト'] # list of unique vegetables to predict

        # create folder to save figures(plots)
        try:
            (self.projpath/'figures'/self.savedir).mkdir(parents=True,exist_ok=False)
        except FileExistsError:
            print("Folder is already there")
        else:
            print("Folder was created")

    def root_mean_squared_percentage_error(self, true_y, pred_y):
        """metric for modle evaluation"""

        rmspe = np.sqrt(np.mean(((pred_y - true_y) / true_y)**2))*100
        return rmpse

    def get_vege_data(self, train=True):
        """get preprocessed data from data/ to work on"""
        train_df = pd.read_csv(self.projpath / 'data/preprocessed_train.csv')
        test_df = pd.read_csv(self.projpath / 'data/preprocessed_test.csv')
        train_df['year'] = train_df['date']//10000
        test_df['year'] = test_df['date']//10000
        train_df['month'] = train_df['date'].apply(lambda x: int(str(x)[4:6]))
        test_df['month'] = test_df['date'].apply(lambda x: int(str(x)[4:6]))
        if train==True:
            #print(train_df.shape)
            train_df.describe()
            # Drop Vegetables that are not in the test dataset
            kinds = test_df['kind'].unique()
            train_df = train_df[train_df['kind'].isin(kinds)]
            #print(train_df.shape)
            train_df
            return train_df
        else:
            #print(test_df.shape)
            test_df.describe()
            return test_df

    def get_vege_all_data(self):
        """concatenates preprocessed train_data and test_data"""
        all_df = pd.concat([self.clean_vege_data(_train=True), self.clean_vege_data(_train=False)]).reset_index(drop=True)
        return all_df

    def get_weather_data(self):
        """read preprocessed weather data"""
        weather_data = pd.read_csv(self.projpath / 'data/preprocessed_weather.csv')
        return weather_data


    def clean_vege_data(self, _train=True):
        """clean preprocessed train and test data, we don't need it at the moment"""
        train = self.get_vege_data(train = True)
        test = self.get_vege_data(train = False)
        if _train ==True:
            return train
        else:
            return test

    def generate_dummy_data(self, train=True):
        """generates a dummy data for dates that market is not open (for specific vegetable)"""
        all_df = self.get_vege_all_data()
        max_days = (datetime(2022, 12, 31) - datetime(2005, 1, 1)).days
        dum_data = []

        for i in range(max_days+1):
            date = datetime(2005, 1, 1) + timedelta(days=i)
            y, wn = date.isocalendar()[0], date.isocalendar()[1]
            date = int(date.strftime('%Y%m%d'))
            m = int(str(date)[4:6])
            dum_data.append(['ダミー', date, 0, 0, 'ダミー', y, m])

        dum_df = pd.DataFrame(dum_data, columns=all_df.columns)
        have_data_combs = [list(i) for i in all_df[['kind','year','month']].drop_duplicates().values]
        #have_data_combs[:5]

        dum_data = []

        for kind in self.kinds:
            for year in range(2005, 2023):
                for month in range(1,13):
                    if year < 2022 or (year == 2022 and month < 5):
                        if [kind, year, month] not in have_data_combs:
                            date = year*10000+month*100+99
                            dum_data.append([kind,date,0,0,'全国',year, month])

        dum_df = pd.DataFrame(dum_data, columns=all_df.columns)
        # print(dum_df.head())
        return dum_df

    def add_weather_feat(self, all_df='', nshift=1):
        """create lag features of weather"""

        all_df = pd.concat([self.get_vege_all_data(), self.generate_dummy_data()]).reset_index(drop=True)
        mer_wea_df = self.get_weather_data()
        mer_wea_df.columns = [f'{i}_{nshift}prev' if i not in ['year','month','area'] else i for i in mer_wea_df.columns]
        mer_wea_df = mer_wea_df.rename(columns={'year':'merge_year','month':'merge_month'})
        #display(mer_wea_df.head())
        data = []

        for year, month in zip(all_df['year'].values, all_df['month'].values):
            month -= nshift
            if month <= 0:
                month += 12
                year -=1
            data.append([year, month])

        tmp_df = pd.DataFrame(data, columns=['merge_year','merge_month'])
        #display(tmp_df.head())
        mer_df = pd.concat([all_df, tmp_df],axis=1)
        #display(mer_df.head())
        mer_df = pd.merge(mer_df, mer_wea_df, on=['merge_year','merge_month','area'], how='left')
        mer_df.drop(['merge_year', 'merge_month'], axis=1, inplace=True)
        return mer_df

    def pivot_data(self):
        """show pivotted data for eda"""
        TARGET = self.target
        train_df = self.get_vege_data(train=True)
        data = pd.pivot_table(train_df.query('20210501 <= date <= 20220430'), index='kind', columns='month', values=TARGET, aggfunc='count')
        return data

    def lag_feature_weather(self,lag_list=[1,2,3,6,9,12]):
        """generate lag features based on the list and returns a dataframe """
        all_df = pd.concat([self.get_vege_all_data(), self.generate_dummy_data()]).reset_index(drop=True)
        mer_df = all_df.copy()
        for lag in lag_list:
            mer_df = pd.merge(mer_df,self.add_weather_feat(nshift = lag),how='left',on=all_df.columns.tolist())
            # print(lag)
            # print(set(mer_df.columns)-set(all_df.columns))

        #print(mer_df.shape)
        return mer_df

    def light_gbm_benchmark(self,_lag_list=[1,2,3,6,9,12],params=''):
        """train a benchmark model, returns a dictionary of vegetables and its models"""
        # モデルベースの予測値

        mer_df = self.lag_feature_weather(lag_list=_lag_list)
        agg_cols = [i for i in mer_df.columns if i not in ['kind','date','year','weekno','area','month','amount']]
        result = []
        model_dict = {}
        pred_df_dict = {}
        feature_imp_dict = {}
        tra_df_dict = {}
        kinds = self.kinds
        TARGET = self.target

        for kind in kinds:

            print(kind)
            ext_df = mer_df[mer_df['kind'] == kind]
            gb_df = ext_df.groupby(['year','month'])[agg_cols].mean().reset_index()
            gb_df[TARGET] = gb_df[TARGET].replace(0,np.nan)

            # 過去の値を特徴量とする
            for i in [1,2,3,6,9,12]:
                gb_df[f'{TARGET}_{i}prev'] = gb_df[TARGET].shift(i)

            test_df = gb_df.query('year == 2022 & month == 5')
            train_df = gb_df.query('~(year == 2022 & month == 5)')
            train_df = train_df.query('year >= 2018') # 2018年以降のデータで学習
            train_df = train_df[train_df[TARGET].notnull()]

            cat_cols = []
            num_cols = [i for i in train_df.columns if i not in [TARGET, 'year', 'month', 'index', 'amount']]
            feat_cols = cat_cols + num_cols

            all_df = pd.concat([train_df, test_df])
            all_df[feat_cols] = all_df[feat_cols].fillna(method='bfill')
            all_df[feat_cols] = all_df[feat_cols].fillna(method='ffill')
            all_df[feat_cols] = all_df[feat_cols].fillna(0)
            train_df = all_df.iloc[:-1,:]
            test_df = all_df.iloc[-1:,:]

            # バリデーションはHold-out法（一定割合で学習データと評価データの2つに分割）で行う

            tra_df = train_df.iloc[:-1]
            val_df = train_df.iloc[-1:] # 2022年4月のデータでvalidation

            tra_x = tra_df[feat_cols]
            tra_y = tra_df[TARGET]
            val_x = val_df[feat_cols]
            val_y = val_df[TARGET]
            test_x = test_df[feat_cols]
            test_y = test_df[TARGET]

            tra_data = lgb.Dataset(tra_x, label=tra_y)
            val_data = lgb.Dataset(val_x, label=val_y)

            model = lgb.train(
                params,
                tra_data,
                categorical_feature = cat_cols,
                valid_names = ['train', 'valid'],
                valid_sets =[tra_data, val_data],
                verbose_eval = 100,
            )

            val_pred = model.predict(val_x, num_iteration=model.best_iteration)

            pred_df = pd.DataFrame(sorted(zip(val_x.index, val_pred, val_y)), columns=['index', 'predict', 'actual'])

            feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(), tra_x.columns)), columns=['importance', 'feature'])

            test_pred = model.predict(test_x, num_iteration=model.best_iteration)

            result.append([kind,2022,5,test_pred[0]])
            model_dict[kind] = model
            pred_df_dict[kind] = pred_df
            feature_imp_dict[kind] = feature_imp
            tra_df_dict[kind] = tra_df
        return model_dict
            
    def plot_feature_importance_lightgbm_benchmark(self,_params):
        """Creates series of plots based on the benchmark model"""
        kinds = self.kinds
        model_dict = self.light_gbm_benchmark(params=_params)
        for kind in kinds:
            model = model_dict[kind]
            lgb.plot_importance(model, figsize=(6,4), max_num_features=10, importance_type='gain', title=kind)
        plt.show()
        plt.close()



class LightgbmModel(TrainModel):
        """This Model will Inherit from TrainModel and create our own ones"""

        def train_model():
            return
        def lag_vote():
            return top_lags