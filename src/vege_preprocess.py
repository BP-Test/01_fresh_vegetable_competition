import pandas as pd
import numpy as np


def preprocess_weather(weather_data):
    #TODO: Convert it to Class Weather()
    """Preprocess of weather

    Args:
        weather_data (pd.DataFrame): Raw weather data from competition

    Returns:
        pd.DataFrame: Preprocessed Data
    """
    # Extract year and month column date : sample 20041106 to year=2004 month= 11
    weather_data['year'] = weather_data['date'].apply(lambda x: int(str(x)[:4]))
    weather_data['month'] = weather_data['date'].apply(lambda x: int(str(x)[4:6]))

    # out of all the columns, pick up aggregation columns[date, mean_temp,max_temp,max_temp_time,min_temp,min_temp_time,sum_rain,sun_time,mean_humid,area]
    agg_cols = ['mean_temp', 'max_temp', 'min_temp', 'sum_rain', 'sun_time', 'mean_humid']
    # create aggregation columns : This will be multiindex
    gb_df = weather_data.groupby(['area', 'year', 'month'])[agg_cols].agg(['mean','max','min']).reset_index()
    # Rename columns to collapse multi index
    new_cols = []
    for col1, col2 in gb_df.columns:
        if col2:
            new_cols.append(col2+'_'+col1)
        else:
            new_cols.append(col1)
    gb_df.columns = new_cols
    # Pick up aggregation columns
    agg_cols = [i for i in gb_df.columns if i not in ['year', 'month', 'area']]
    tmp_df = gb_df.groupby(['year', 'month'])[agg_cols].agg(['mean']).reset_index()

    new_cols = []
    for col1, col2 in tmp_df.columns:
        new_cols.append(col1)

    tmp_df.columns = new_cols
    tmp_df['area'] = '‘S‘'
    tmp_df = tmp_df[gb_df.columns]
    tmp_df
    wea_df = pd.concat([gb_df, tmp_df])
    #TODO Dependencies from preprocess_train_test

    return wea_df
#weather_data = pd.read_csv('../data/weather.csv')
#print(weather_data.shape)
#weather_pre = preprocess_weather(weather_data)

def preprocess_train_test(train_data,test_data, weather_data):
    #TODO: Convert it to Class Train(), Test(Train)
    """Preprocess of train and test

    Args:
        train_data (pd.DataFrame): _description_
        test_data (pd.DataFrame): _description_
        weather_data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    wea_areas = weather_data['area'].unique()
    wea_areas
    all_df = pd.concat([train_data, test_data])
    all_df
    area_pairs = all_df['area'].unique()
    yasai_areas = set()

    for area_pair in area_pairs:
        areas = area_pair.split('_')
        yasai_areas |= set(areas)
    yasai_areas
    area_map = {}

    update_area_map = {
    'Šâè':'·‰ª','‹{é':'å‘ä','Ã‰ª':'•l¼','‰«“ê':'“ß”e','_“Şì':'‰¡•l','ˆ¤’m':'–¼ŒÃ‰®','ˆïé':'…ŒË','–kŠC“¹':'‘ÑL','Še’n':'‘S‘',
    '•ºŒÉ':'_ŒË','ì':'‚¼','é‹Ê':'ŒF’J','‘“à':'‘S‘','R—œ':'b•{','“È–Ø':'‰F“s‹{','ŒQ”n':'‘O‹´','ˆ¤•Q':'¼R'}

    for yasai_area in yasai_areas:
        if yasai_area not in wea_areas and yasai_area not in update_area_map:
            area_map[yasai_area] = '‘S‘' # ŠO‘‚Ì“VŒó‚Í‘S‘‚É‚µ‚Ä‚¨‚­
        else:
            area_map[yasai_area] = yasai_area

    area_map = {**area_map, **update_area_map}
    area_map
    all_df['area'] = all_df['area'].apply(lambda x: '_'.join([area_map[i] for i in x.split('_')]))
    all_df
    train_preprocessed = all_df.iloc[:train_data.shape[0]]
    test_preprocessed = all_df.iloc[train_data.shape[0]:]
    return train_preprocessed, test_preprocessed

# test_data  = pd.read_csv('../data/preprocessed_test.csv')
# train_data= pd.read_csv('../data/preprocessed_train.csv')
# train_pre, test_pre = preprocess_train(train_data,test_data,weather_data)


# # Design Idea
# class Weather:
#     def __init__(self,):
    
#     def aggregation_1(s):
#     def aggregation_2(s):
    
#     def preprocess(data):
#         data = aggregation_1(data)
#         data = aggregation_2(a)


# class Train:

# class Test(Train):
