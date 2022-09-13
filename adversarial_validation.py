# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount(
    '/content/drive'
)

from google.colab import output
output.enable_custom_widget_manager()

import os
import gc
import random
import scipy as sp
import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
warnings.filterwarnings('ignore')

# load data from gdrive
test = pd.read_parquet('/content/drive/MyDrive/test.parquet')
test.shape

test = test.drop_duplicates(subset=["customer_ID"], keep="last")
test['S_2'] = pd.to_datetime(test['S_2'])
test['month'] = (test['S_2'].dt.month).astype('int8')
test = test.reset_index(drop=True)

# split into public and private
test['target'] = 0
test.loc[test['month'] == 10,'target'] = 1

class CFG:
    input_dir = '/content/drive/MyDrive/Amex/Data/'
    seed = 42
    n_folds = 5
    target = 'target'
    boosting_type = 'gbdt'
    metric = 'binary_logloss'

params = {
      'objective': 'binary',
      'metric': CFG.metric,
      'boosting': CFG.boosting_type,
      'seed': CFG.seed,
      'num_leaves': 50,
      'learning_rate': 0.1,
      'feature_fraction': 0.20,
      'bagging_freq': 7,
      'bagging_fraction': 0.40,
      'n_jobs': -1,
      'lambda_l2': 2,
      'min_data_in_leaf': 20,
   }

# dropping some columns
drop_cols = ['S_2', 'month', 'customer_ID', 'target']
features = [col for col in test.columns if col not in drop_cols]

# create a numpy array to store test predictions
oof_predictions = np.zeros(len(test))
importance = pd.DataFrame()

kfold = KFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
for fold, (trn_ind, val_ind) in enumerate(kfold.split(test, test[CFG.target])):
    print(' ')
    print('-'*50)
    print(f'Training fold {fold+1} with {len(features)} features...')
    x_train, x_val = test[features].iloc[trn_ind], test[features].iloc[val_ind]
    y_train, y_val = test[CFG.target].iloc[trn_ind], test[CFG.target].iloc[val_ind]
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_valid = lgb.Dataset(x_val, y_val)
    model = lgb.train(
        params = params,
        train_set = lgb_train,
        num_boost_round = 1000,
        valid_sets = [lgb_train, lgb_valid],
        verbose_eval = 500
       )
    
    # predict validation
    val_pred = model.predict(x_val)
    # add to our out of folds array
    oof_predictions[val_ind] = val_pred
    # freeing up memory
    del x_train, x_val, y_train, lgb_train, lgb_valid
    _ = gc.collect()
    # compute roc auc for the fold
    score = roc_auc_score(y_val, val_pred)
    print(f'Our fold {fold+1} ROC AUC score is {score}')
    del y_val
    _ = gc.collect()

    # adding feature importance scores
    importance[f'fold_{fold}'] = model.feature_importance(importance_type = 'gain')

# plotting the feature importances
importance['mean_importance'] = importance.mean(axis=1)
importance['features'] = features
plt.figure(figsize = (10, 10))
sns.barplot(x = "mean_importance", y = "features", data = importance.sort_values(by="mean_importance", ascending=False)[:30])
plt.title('LightGBM Feature Importance')
plt.tight_layout()

print('Number of missing values in public B_29:',test[test['target'] == 0]['B_29'].isnull().sum())
print('Number of missing values in private B_29:',test[test['target'] == 1]['B_29'].isnull().sum())

