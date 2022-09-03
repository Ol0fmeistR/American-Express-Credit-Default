# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm

paths = [
        '/content/drive/MyDrive/Amex/CNN/OOF/keras_CNN_oof.csv',
        '/content/drive/MyDrive/Amex/OOF/oof_lgbm_dart_5fold_seed42.csv'
        ]

oof_df = pd.DataFrame()

for i, path in enumerate(paths):
    # Find filename
    start = path.rfind('/')+1
    end = path.find('.csv')
    fname = path[start:end]  
    print(path)
    temp = pd.read_csv(path)
    # Set column name of oof
    temp.rename(columns={'oof_pred': f'{fname}'}, inplace=True)
    # Drop redundant target column
    if i != 0:
        temp.drop(columns=['target'], inplace=True)
    # Join to main file
    if i == 0:
        oof_df = temp
    else:
        oof_df = pd.merge(oof_df, temp, on="customer_ID", how="left")

oof_df['target'] = pd.read_csv('/content/drive/MyDrive/train_labels.csv').target

"""pearson < 0.95 and ks stat > 0.1 should work well for ensembling"""

def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

oofCols = [col for col in oof_df.columns if 'prediction' in col]

for col in oofCols:
    metric = amex_metric_mod(oof_df['target'], oof_df[col])
    print(f"{col} : {metric}")

# Simple Averaging
oof_preds = []
for col in oofCols:
    oof_preds.append(oof_df[col])

y_avg = np.mean(np.array(oof_preds), axis=0)
sa_metric = amex_metric_mod(oof_df['target'], y_avg)

print(f"Simple Average: {sa_metric}")

# Weighted Average
weights = [1,2,3]

y_wtavg = np.zeros(len(oof_df))

for wt, col in zip(weights, oofCols):
    y_wtavg = y_wtavg + (wt*oof_df[col])

y_wtavg = y_wtavg / sum(weights)
wa_metric = amex_metric_mod(oof_df['target'], y_wtavg)
print(f"Weighted Average: {wa_metric}")

# Rank Average
rankPreds = []

for i, col in enumerate(oofCols):
    globals()[f'y_rank{i+1}'] = oof_df[col].rank().values
    rankPreds.append(globals()[f'y_rank{i+1}'])

y_rankavg = np.mean(np.array(rankPreds), axis=0)
ra_metric = amex_metric_mod(oof_df['target'], y_rankavg)
print(f"Rank Average: {ra_metric}")

# Weighted Rank Average
weights = [1,2,3]

y_wtrankavg = np.zeros(len(oof_df))

for wt, pred in zip(weights, rankPreds):
    y_wtrankavg = y_wtrankavg + (wt*pred)

y_wtrankavg = y_wtrankavg / sum(weights)
wra_metric = amex_metric_mod(oof_df['target'], y_wtrankavg)
print(f"Weighted Rank Average: {wra_metric}")

# Hill Climbing
RES = 20
PATIENCE = 2
DUPLICATES = True


preds = oof_df[oofCols[-1]]

print(f"Starting with {oofCols[-1]}")
hc_metric = amex_metric_mod(oof_df['target'], preds)
print(f"Metric = {hc_metric}")

while True: 
    if PATIENCE == 0:
        break 
    reduce = 1
    
    champ = amex_metric_mod(oof_df['target'], preds)
    wtadd = 0
    newcol = ''
    for col in tqdm(oofCols):
        for wt in range(RES):
            temp = (wt/RES) * preds + (1-wt/RES) * oof_df[col]
            chall = amex_metric_mod(oof_df['target'], temp)
            
            if chall > champ:
                champ = chall
                wtadd = (1-wt/RES)
                newcol = col
                newpred = temp
                reduce = 0
    print(f"Adding {newcol} with weight {wtadd}")
    preds = newpred
    hc_metric = amex_metric_mod(oof_df['target'], preds)
    print(f"Metric = {hc_metric}")
    
    if reduce==1:
        print("Metric has not improved")
        PATIENCE = PATIENCE - reduce

# Weight Optimization
class OptimizeAmex:
    def __init__(self):
        self.coef_ = 0
    
    def _amex(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        amex_score = amex_metric_mod(y, predictions)
        return -1.0 * amex_score
    
    def fit(self, X, y):
        partial_loss = partial(self._amex, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss, init_coef, disp=False)
    
    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions

from sklearn.model_selection import KFold
from functools import partial
from scipy.optimize import fmin

coefs = []

skf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold,(train_idx, valid_idx) in enumerate(skf.split(oof_df, oof_df.target)):
    
    X_train = oof_df.loc[train_idx, oofCols]
    y_train = oof_df.loc[train_idx, 'target']
    
    X_valid = oof_df.loc[valid_idx, oofCols]
    y_valid = oof_df.loc[valid_idx, 'target']
    
    opt = OptimizeAmex()
    opt.fit(X_train, y_train)
    preds = opt.predict(X_valid)
    score = amex_metric_mod(y_valid, preds)
    
    coefs.append(opt.coef_)
    
    print(f"Fold {fold+1} score = {score}")
    print(opt.coef_)
    print()

optcoefs = np.mean(coefs, axis=0)

preds = 0
for wt, col in zip(optcoefs, oofCols):
    preds += wt * oof_df[col]

wopt_metric = amex_metric_mod(oof_df['target'], preds)
print(f"Optimized weighted averaging: {wopt_metric}")

# Final Comparison
print(f"Simple Average: {sa_metric}")
print(f"Weighted Average: {wa_metric}")
print(f"Rank Average: {ra_metric}")
print(f"Weighted Rank Average: {wra_metric}")
print(f"Hill Climbing = {hc_metric}")
print(f"Optimized weighted averaging: {wopt_metric}")

























