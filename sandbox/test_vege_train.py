## Run all methods of TrainModel() to see if it works.

import sys
sys.dont_write_bytecode=True
sys.path.append('../src')

from vege_train import TrainModel

vege = TrainModel()
len(vege.kinds)
vege.get_vege_data(train=False)
vege.get_weather_data()
vege.clean_vege_data(_train=True)
vege.clean_vege_data(_train=False)
vege.generate_dummy_data(train=True)
vege.get_vege_all_data()
vege.add_weather_feat(nshift=1)
vege.pivot_data()
vege.lag_feature_weather()
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 42,
    'max_depth': 7,
    "feature_fraction": 0.8,
    'subsample_freq': 1,
    "bagging_fraction": 0.95,
    'minx_data_in_leaf': 2,
    'learning_rate': 0.1,
    "boosting": "gbdt",
    "lambda_l1": 0.1,
    "lambda_l2": 10,
    "verbosity": -1,
    "random_state": 42,
    "num_boost_round": 50000,
    "early_stopping_rounds": 100
}
vege.generate_model_data(file_name='model_input_data')
vege.generate_model_data(file_name='model_input_data',save_as_csv=False) # Does not save as csv

vege.light_gbm_benchmark(params=params)
vege.plot_feature_importance_lightgbm_benchmark(_params=params)