# Path settings
import sys
from xmlrpc.client import Boolean
sys.path.append('../src/')      # import 用のパスの追加

# default packages
import itertools
import json
import pandas as pd
import numpy as np

# For data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


class Weather():
    """
    # TODO: add description
    """
    def __init__(self) -> None:
        """
        # TODO: add description
        """
        
        # path to data (defalut)
        self.data_path = '../data/'
        
        # 地名の変換
        # TODO: make a setting file.
        with open('../resource/area_info.json', 'r', encoding="utf-8") as f:
            self.update_area_map = json.load(f) 
        
        # 集計処理の記録
        # TODO: make a json file to record aggregate preprocessing
        self.agg_plan = {
            'monthly' : {
                'scope' : ['area', 'year', 'month'],
                'target_cols' : None,
                'agg_types' : ['mean','max','min']
            }
        }
        
        # テストデータで利用する地理情報に絞り込み
        test_df = pd.read_csv(self.data_path + 'test.csv', usecols = ['area'])
        self.target_area = set(itertools.chain.from_iterable(pd.Series(test_df.area.unique()).str.split('_').to_list()))
    

        
    def read_from_csv(self, get_faster = True):
        """データの読み込み

        Args:
            get_faster (bool, optional): _description_. Defaults to True.
        """
        # read_csv
        if get_faster:
            self.get_faster()
            self.weather = pd.read_csv(self.data_path + 'weather.csv', dtype=self.d_types)
        else:
            self.weather = pd.read_csv(self.data_path + 'weather.csv')
        
        # 日付処理
        self.weather = self.add_variable(self.weather)
        
        # 地名を都道府県名に変換
        #self.weather['area'] = self.weather.area.replace(self.update_area_map)
    
    
    
    
    def get_faster(self):
        """csv読み込み時にfloat, intの方を32で指定
        """
        tmp = pd.read_csv(self.data_path + 'weather.csv', nrows =1)
        self.d_types =  tmp.head(1).dtypes.astype('str').str.replace('64', '32').to_dict()
    
    
    def add_variable(self, df):
        # 日付処理
        date_series = pd.to_datetime(df['date'].astype(str))
        df['year'] = date_series.dt.year
        df['month'] = date_series.dt.month
        df['day'] = date_series.dt.day
        return df
    
    
    
    def add_agg_features(self, df, scope:list = ['area', 'year', 'month'], target_cols:list = None, agg_types:list = None, add_name:Boolean = True):
        """特徴量の追加

        Args:
            df (DataFrame, optional): 編集するデータフレーム. Defaults to None.
            scope (list, optional): 集計に利用するキー. Defaults to ['area', 'year', 'month'].
            target_cols (list, optional): 集計したい列名. Defaults to None.
            agg_types (list, optional): 集計方法. Defaults to None.
            head_name (Boolean, optional): 列名に追加. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if target_cols is None:
            target_cols = df.head(1).select_dtypes(['float', 'float32']).columns.to_list()
        
        if agg_types is None:
            agg_types = ['mean','max','min']
        
        
        # 集計
        tmp_df = df[target_cols + scope].groupby(scope).agg(agg_types).reset_index()
        
        # マルチカラムの解除
        if add_name:
            tmp_df.set_axis([col2 + '_' + col1 if col2 else col1 for col1, col2 in tmp_df.columns], axis=1, inplace = True)
            return tmp_df
        else:
            tmp_df.set_axis([col1 for col1, col2 in tmp_df.columns], axis=1, inplace = True)
            return tmp_df
        
    
    def weather_preprocess(self)-> None:
        """気象データの処理について一連の処理を関数に
        """
        # データの読み込み
        self.read_from_csv()
        
        # 集計作業1：各地の気象情報を月、年単位得で集計
        tmp1 = self.add_agg_features(self.weather)
        
        # 集計作業2：全国の気象情報を月、年単位得で集計
        tmp2 = self.add_agg_features(df = tmp1, scope=['year', 'month'], agg_types=['mean'], add_name = False)
        
        # tmp2にarea情報を追加
        tmp2['area'] = '全国'
        
        # 結合
        return pd.concat([tmp1, tmp2])



class Preprocess(Weather):
    def __init__(self):
        super().__init__()
        
        # path to data (defalut)
        self.data_path = '../data/'
        
        
    def read_train_test_from_csv(self, data_path = None):
        
        if data_path is not None:
            self.data_path = data_path
        else:
            pass
        
        # read_csv
        self.train = pd.read_csv(self.data_path + 'train.csv')
        self.test = pd.read_csv(self.data_path + 'test.csv')
    
    
    def preprocess_train_test(self):
        """ 前処理
        #TODO:メインの処理を書く
        """
        # データ読み込み
        self.read_train_test_from_csv()
        
        # 不要な情報を削除
        #self.drop_unusable_info()
        
        # 時間の情報を追加
        self.train = self.add_variable(self.train)
        self.test = self.add_variable(self.test)
    
    
    def add_variable(self, df):
        #TODO:Weatherクラス継承時に削除
        # 日付処理
        date_series = pd.to_datetime(df['date'].astype(str))
        df['year'] = date_series.dt.year
        df['month'] = date_series.dt.month
        df['day'] = date_series.dt.day
        return df
    
    
    
    def drop_unusable_info(self):
        # テストデータに存在する野菜のみを対象にする。
        self.train = self.train[self.train['kind'].isin(self.test['kind'].unique())]
    
    
    
    def fill_na(self):
        # トレーニングデータの欠損値を埋める
        pass
    
    
    def get_weather_data(self):
        # weather
        self.wea_df = super().weather_preprocess()
    
    
    
    def add_weather_features(self):
        all_df = pd.concat([self.train, self.test])
        
        area_pairs = all_df['area'].unique()
        
        yasai_areas = set()
        
        for area_pair in area_pairs:
            areas = area_pair.split('_')
            yasai_areas |= set(areas)
        
        
        wea_areas = self.wea_df['area'].unique()
        
        area_map = {}

        update_area_map = {
            '岩手':'盛岡','宮城':'仙台','静岡':'浜松','沖縄':'那覇','神奈川':'横浜','愛知':'名古屋','茨城':'水戸','北海道':'帯広','各地':'全国',
            '兵庫':'神戸','香川':'高松','埼玉':'熊谷','国内':'全国','山梨':'甲府','栃木':'宇都宮','群馬':'前橋','愛媛':'松山'
        }

        for yasai_area in yasai_areas:
            if yasai_area not in wea_areas and yasai_area not in update_area_map:
                area_map[yasai_area] = '全国' # 外国の天候は全国にしておく
            else:
                area_map[yasai_area] = yasai_area

        area_map = {**area_map, **update_area_map}
        all_df['area'] = all_df['area'].apply(lambda x: '_'.join([area_map[i] for i in x.split('_')]))
        
        test_df = all_df.iloc[self.train.shape[0]:]
        train_df = all_df.iloc[:self.train.shape[0]]  

        area_pairs = all_df['area'].unique()
        target_cols = [i for i in self.wea_df.columns if i != 'area']
        
        area_pair_dfs = []
        
        for area_pair in area_pairs:
            areas = area_pair.split('_')
            if len(areas) > 0:
                area = areas[0]
                base_tmp_df = self.wea_df[self.wea_df['area'] == area]
                base_tmp_df = base_tmp_df[target_cols].reset_index(drop=True)
                for area in areas[1:]:
                    tmp_df = self.wea_df[self.wea_df['area'] == area]
                    tmp_df = tmp_df[target_cols].reset_index(drop=True)
                    base_tmp_df = base_tmp_df.add(tmp_df)
                base_tmp_df /= len(areas)
                base_tmp_df['area'] = area_pair
                area_pair_dfs.append(base_tmp_df)
        # 結合
        self.wea_df = pd.concat(area_pair_dfs)#.astype({'year' : 'int32', 'month' : 'int32'})
    
    
    def save_weather_df(self):
        self.wea_df.to_save(self.data_path + 'preprocess_weather_data.csv')
    
    
    
