# -*- coding: utf-8 -*-
# hardware stats

# !df -h
# !cat /proc/cpuinfo
# !cat /proc/meminfo

from google.colab import drive
drive.mount('/content/drive')

import os
import gc
import warnings
warnings.filterwarnings('ignore')
import random
import scipy as sp
import numpy as np
import pandas as pd
import joblib
import itertools
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
import itertools
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from itertools import combinations

def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(['customer_ID'])):
        # Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis = 0)
    # Transform to dataframe
    df1 = pd.DataFrame(df1, columns = [col + '_diff1' for col in df[num_features].columns])
    # Add customer id
    df1['customer_ID'] = customer_ids
    return df1

def read_preprocess_data():
    train = pd.read_parquet('/content/drive/MyDrive/train.parquet')
    problematic_values = ['B_29', 'D_103', 'D_139', 'S_9']  # D_103 and D_139 are redundant and B_29, S_9 shows a problematic distribution
    to_drop = ['customer_ID', 'S_2'] + problematic_values

    # Calculating distance between statement dates
    train['S_2'] = pd.to_datetime(train['S_2'])
    train['day'] = train['S_2'].dt.day
    train['month'] = train['S_2'].dt.month
    train['day_distribution'] = train.groupby(['customer_ID', 'day'])['day'].transform('count')
    train['month_distribution'] = train.groupby(['customer_ID', 'month'])['month'].transform('count')
    train['SDist'] = train.groupby("customer_ID")['S_2'].diff() / np.timedelta64(1, 'D')
    train['SDist'].fillna(30.53, inplace=True)

    payments = ['P_2', 'P_3', 'P_4']
    balance = [col for col in train.columns if 'B_' in col]
    spend = [col for col in train.columns if 'S_' in col and col not in ['S_2']]

    train['total_balance'] = train[balance].sum(axis=1)
    train['total_payments'] = train[payments].sum(axis=1)
    train['total_spendings'] = train[spend].sum(axis=1)
    train['to_pay'] = train['total_spendings'] - train['total_balance']
    train['credit_utilization'] = train['total_payments'] / train['total_balance']

    gen_feats = ['total_balance', 'total_payments', 'total_spendings', 'to_pay', 'credit_utilization']
    for col in gen_feats:
        train[f'{col}_ewm'] = train.groupby('customer_ID')[col].transform(lambda x: x.ewm(span=2, adjust=True).mean())

    train.drop(['total_balance', 'total_payments', 'total_spendings', 'to_pay', 'credit_utilization'], axis=1, inplace=True)

    features = train.drop(to_drop, axis = 1).columns.to_list()
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    cat_features = cat_features + ['day', 'month']
    num_features = [col for col in features if col not in cat_features]

    print('Starting train set feature engineering...')
    train_num_agg = train.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'first', 'last'])
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace = True)

    train_cat_agg = train.groupby("customer_ID")[cat_features].agg(['count', 'first', 'last', 'nunique'])
    train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace = True)

    train_labels = pd.read_csv('/content/drive/MyDrive/train_labels.csv')

    # Transform float64 columns to float32
    cols = list(train_num_agg.dtypes[train_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        train_num_agg[col] = train_num_agg[col].astype(np.float32)

    # Transform int64 columns to int32
    cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        train_cat_agg[col] = train_cat_agg[col].astype(np.int32)

    # Get the difference
    train_diff = get_difference(train, num_features)
    train = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_diff, how = 'inner', on = 'customer_ID').merge(train_labels, how = 'inner', on = 'customer_ID')
    del train_num_agg, train_cat_agg, train_diff
    _ = gc.collect()

    # Data preparation on the test set
    test = pd.read_parquet('/content/drive/MyDrive/test.parquet')

    # Calculating distance between statement dates
    test['S_2'] = pd.to_datetime(test['S_2'])
    test['day'] = test['S_2'].dt.day
    test['month'] = test['S_2'].dt.month
    test['day_distribution'] = test.groupby(['customer_ID', 'day'])['day'].transform('count')
    test['month_distribution'] = test.groupby(['customer_ID', 'month'])['month'].transform('count')
    test['SDist'] = test.groupby("customer_ID")['S_2'].diff() / np.timedelta64(1, 'D')
    test['SDist'].fillna(30.53, inplace=True)

    test['total_balance'] = test[balance].sum(axis=1)
    test['total_payments'] = test[payments].sum(axis=1)
    test['total_spendings'] = test[spend].sum(axis=1)
    test['to_pay'] = test['total_spendings'] - test['total_balance']
    test['credit_utilization'] = test['total_payments'] / test['total_balance']

    for col in gen_feats:
        test[f'{col}_ewm'] = test.groupby('customer_ID')[col].transform(lambda x: x.ewm(span=2, adjust=True).mean())

    test.drop(['total_balance', 'total_payments', 'total_spendings', 'to_pay', 'credit_utilization'], axis=1, inplace=True)

    print('Starting test set feature engineering...')
    test_num_agg = test.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'first', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
    test_num_agg.reset_index(inplace = True)

    test_cat_agg = test.groupby("customer_ID")[cat_features].agg(['count', 'first', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
    test_cat_agg.reset_index(inplace = True)

    # Transform float64 columns to float32
    cols = list(test_num_agg.dtypes[test_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        test_num_agg[col] = test_num_agg[col].astype(np.float32)

    # Transform int64 columns to int32
    cols = list(test_cat_agg.dtypes[test_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        test_cat_agg[col] = test_cat_agg[col].astype(np.int32)

    # Get the difference
    test_diff = get_difference(test, num_features)
    test = test_num_agg.merge(test_cat_agg, how = 'inner', on = 'customer_ID').merge(test_diff, how = 'inner', on = 'customer_ID')
    del test_num_agg, test_cat_agg, test_diff
    _ = gc.collect()

    # Save files to disk
    train.to_parquet('train_fe.parquet')
    test.to_parquet('test_fe.parquet')
    print(f'Shape of train: {train.shape}')
    print(f'Shape of test: {test.shape}')

# Read & Preprocess Data
read_preprocess_data()

class CFG:
    #input_dir = '/content/drive/MyDrive/Amex/Data/'
    seed = 42
    n_folds = 5
    target = 'target'
    boosting_type = 'dart'
    metric = 'binary_logloss'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def read_data():
    train = pd.read_parquet('train_fe.parquet')
    test = pd.read_parquet('test_fe.parquet')
    return train, test

def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)

def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True

def train_and_evaluate(train, test):
    # let lgbm handle categoricals on it's own
    cat_features = []
    temp = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
        "day",
        "month"
    ]
    for col in temp:
        cat_features.append(f'{col}_last')
        cat_features.append(f'{col}_first')

    # Rounding last float features to 2 decimal places
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    num_cols = [col for col in num_cols if 'last' in col]
    for col in num_cols:
        train[col + '_round2'] = train[col].round(2)
        test[col + '_round2'] = test[col].round(2)
    
    # Get the difference between last and mean + mean and median
    print('Generating lag features...')
    num_cols = [col for col in train.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    for col in num_cols:
        try:
            train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
            test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
            train[f'{col}_last_first_diff'] = train[f'{col}_last'] - train[f'{col}_first']
            test[f'{col}_last_first_diff'] = test[f'{col}_last'] - test[f'{col}_first']
        except:
            pass
    max_cols = [col[:-4] for col in train.columns if 'max' in col]
    for col in max_cols:
        try:
            train[f'{col}_max_min_ratio'] = train[f'{col}_min'] / train[f'{col}_max']
            test[f'{col}_max_min_ratio'] = test[f'{col}_min'] / test[f'{col}_max']
        except:
            pass

    print('Done with FE...')
    print(f'Dimensions of train: {train.shape}')
    print(f'Dimensions of test: {test.shape}')

    # Transform float64 and float32 to float16
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    for col in tqdm(num_cols):
        train[col] = train[col].astype(np.float16)
        test[col] = test[col].astype(np.float16)

    # Get feature list
    features = [col for col in train.columns if col not in ['customer_ID', CFG.target]]
    params = {
        'objective': 'binary',
        'metric': CFG.metric,
        'boosting': CFG.boosting_type,
        'seed': CFG.seed,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.2,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
        }

    # Create a numpy array to store test predictions
    test_predictions = np.zeros(len(test))

    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(train))

    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} with {len(features)} features...')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[CFG.target].iloc[trn_ind], train[CFG.target].iloc[val_ind]
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature = cat_features)
        lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature = cat_features)
        model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 10500,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 1500,
            verbose_eval = 500,
            categorical_feature = cat_features,
            feval = lgb_amex_metric
            )
        
        # Save best model
        joblib.dump(model, f'/content/drive/MyDrive/Amex/Models/lgbm_roll_{CFG.boosting_type}_fold{fold}_seed{CFG.seed}.pkl')
        # Predict validation
        val_pred = model.predict(x_val)
        # Add to out of folds array
        oof_predictions[val_ind] = val_pred
        # freeing up memory
        del x_train, x_val, y_train, lgb_train, lgb_valid
        _ = gc.collect()
        # Predict the test set
        test_pred = model.predict(test[features])
        test_predictions += test_pred / CFG.n_folds
        # Compute fold metric
        score = amex_metric(y_val, val_pred)
        print(f'Our fold {fold} CV score is {score}')
        del y_val
        _ = gc.collect()

    # Compute out of folds metric
    score = amex_metric(train[CFG.target], oof_predictions)
    print(f'Our out of folds CV score is {score}')
    # Create a dataframe to store out of folds predictions
    oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 'prediction': oof_predictions})
    oof_df.to_csv(f'/content/drive/MyDrive/Amex/OOF/oof_lgbm_roll_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    # Create a dataframe to store test prediction
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
    test_df.to_csv(f'/content/drive/MyDrive/Amex/Predictions/test_lgbm_roll_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    
seed_everything(CFG.seed)
train, test = read_data()
train_and_evaluate(train, test)

del train, test
_ = gc.collect()
