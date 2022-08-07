# Path settings
import sys
from xmlrpc.client import Boolean
sys.path.append('../src/')      # import 用のパスの追加

# default packages
import itertools
import json
import pandas as pd
import numpy as np
from datetime import timedelta

# For data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


class Weather():
    def __init__(self) -> None:
        
        # path to data (defalut)
        self.data_path = '../data/'
        
        # setting for data generating process
        self.setting = {
            'add_variable' : [
                {'roll' : 7,  'lag' : 10, 'agg' : 'mean'}, 
                {'roll' : 14, 'lag' : 14, 'agg' : 'mean'},
                {'roll' : 14,  'lag' : 28, 'agg' : 'mean'}
            ]
        }
        
        
    def read_from_csv(self):
        """データの読み込み

        Args:
            get_faster (bool, optional): _description_. Defaults to True.
        """
        self.data = pd.read_csv(
            self.data_path + 'all_weather.csv', 
            encoding = 'SHIFT-JIS',
            index_col = 0
            )
        
        self.data.index = pd.to_datetime(self.data.index)
    
    
    def add_feature(self, target_area = None):
        
        # 特定地域のみを抽出
        if target_area is not None:
            all_weather = self.data[self.data['area'].isin(target_area)]
        else:
            all_weather = self.data
        
        # 全国情報の追加
        all_area_agg_weather = self.add_global_info(all_weather)
        
        # エリア毎に気象情報を分解
        self.target_weather_info = self.decompose_area_info(all_area_agg_weather)
        
        return pd.concat(
            [
                self.target_weather_info
                ]
            + 
            [
                self.add_lagged_variable(
                    self.target_weather_info, 
                    roll = pattern['roll'], 
                    lag = pattern['lag'], 
                    agg=pattern['agg']
                    )
                for pattern in self.setting['add_variable']
                ],
            axis = 1
            )
    
    
    
    
    def add_global_info(self, all_weather):
        """各地のデータの情報の追加
        """
        average_all_area =  all_weather.reset_index().drop('area', axis = 1).groupby('date').mean()
        average_all_area['area'] = '各地'
        return pd.concat([average_all_area, all_weather])
    
    
    
    
    def decompose_area_info(self, df):
        # 気象情報が存在する全日程
        all_date = set(df.index.drop_duplicates())
        
        # エリアごとの列にグループ化 -> 縦軸：日付、横軸：エリアごとの気候情報
        return pd.concat(
            [
                group.drop('area', axis = 1).rename(columns = lambda x: name + '_' + x)
                for name, group 
                in df.groupby('area')
                ], 
            axis=1
            )
    
    
    
    
    def add_lagged_variable(self, df, roll, lag, agg = 'mean'):
        """
        Args:
            roll (int, optional): 過去集計を行う日数.
            lag (int, optional): ラグ日数.
        """
        if agg == 'mean':
            tmp = df.rolling(roll).mean()
        
        tmp.set_index(df.index + timedelta(days=lag))
        tmp.rename(columns=lambda x:x + '_roll_' + str(roll) + '_lag_' + str(lag), inplace=True)
        tmp = tmp[tmp.index <= max(df.index)]
        return tmp















class Preprocess(Weather):
    def __init__(self):
        super().__init__()
        
        # path to data (defalut)
        self.data_path = '../data/'
        
        # Dict型でデータを保存
        self.preprocessed_data = {}
    
    
    
    
    def read_train_test_from_csv(self, data_path = None):
        
        if data_path is not None:
            self.data_path = data_path
        else:
            pass
        
        # read_csv
        train = pd.read_csv(self.data_path + 'train.csv')
        test = pd.read_csv(self.data_path + 'test.csv')
        
        
        # 共通する野菜の種類を抽出
        self.common_vege_type_set = set(train.kind) & set(test.kind)
        
        # テストデータと訓練データの連結
        self.df = pd.concat([train, test]).sort_values(['kind', 'date']).reset_index(drop=True)
        self.df.date = pd.to_datetime(self.df['date'].astype(str))
    
    
    
    
    def preprocess(self):
        
        # 気象情報の取得
        self.get_wether_info()
        
        
        for vege_name in self.common_vege_type_set:
            sub_df, target_area = self.get_train_test_df(vege_name)
            
            # 変数の追加
            weather_info_df = self.add_feature(target_area)
            
            # 訓練データとテストデータの生成
            train_test_df = pd.concat(
                [
                    sub_df.set_index('date'), weather_info_df
                    ], 
                axis = 1
                ).drop(['kind', 'amount', 'area'], axis = 1)
            
            # 訓練データとテストデータに分離
            train_input = train_test_df.dropna(axis = 0)
            test_input = train_test_df[train_test_df.index > max(train_input.index)]
            
            # 記録
            self.preprocessed_data[vege_name] = {
                'train' : train_input,
                'test' : test_input
            }
    
    
    
    def get_train_test_df(self, vege_name):
        
        # 対象の野菜のデータのみ抽出
        sub_df = self.df[self.df.kind == vege_name]
        
        # 利用する地域情報
        target_area = set(itertools.chain.from_iterable(sub_df.area.drop_duplicates().str.split('_').to_list()))
        
        return sub_df, target_area
    
    
    
    
    def get_wether_info(self):
        # 気象情報の読み込み
        self.read_from_csv()
