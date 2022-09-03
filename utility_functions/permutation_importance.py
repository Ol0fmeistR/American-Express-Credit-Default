class CFG:
    input_dir = '/content/drive/MyDrive/Amex/Data/'
    seed = 42 # 52, 62
    n_folds = 5
    target = 'target'
    boosting_type = 'dart'
    metric = 'binary_logloss'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def read_data():
    train = pd.read_parquet(input_dir + 'train_preprocessed.parquet')
    test = pd.read_parquet(input_dir + 'test_preprocessed.parquet')
    return train, test


def compute_permutation_importance(df):
    seed_everything(CFG.seed)
    compute_perm_importance = True
    one_fold_only = False

    # Transform float64 and float32 to float16
    num_cols = list(df.dtypes[(df.dtypes == 'float32') | (df.dtypes == 'float64')].index)
    for col in tqdm(num_cols):
        df[col] = df[col].astype(np.float16)

    # Get feature list
    features = [col for col in df.columns if col not in ['customer_ID', CFG.target]]
    print(f'Total features used: {len(features)}')

    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(df, df[CFG.target])):
        print(' ')
        print('-'*50)
        print(f'OOF predictions for fold {fold} with {len(features)} features...')
        x_train, x_val = df[features].iloc[trn_ind], df[features].iloc[val_ind]
        y_train, y_val = df[CFG.target].iloc[trn_ind], df[CFG.target].iloc[val_ind]

        if compute_perm_importance:
            results = []
            print(f'Computing permutation feature importance for fold {fold} with {len(features)} features...')
            # Load model for each fold
            lgbm_model = joblib.load(f'/content/drive/MyDrive/Amex/Models/lgbm_dart_{CFG.boosting_type}_fold{fold}_seed{CFG.seed}.pkl')
            oof_preds = lgbm_model.predict(x_val)
            # Computing fold metric
            baseline_score = amex_metric(y_val, oof_preds)
            print(f'Our fold {fold} CV score is {baseline_score}')
            results.append({'feature':'BASELINE','metric':baseline_score})

            for k in tqdm(range(len(features))):
                save_col = x_val.iloc[:,k].copy()
                x_val.iloc[:,k] = np.random.permutation(x_val.iloc[:,k])
                oof_preds = lgbm_model.predict(x_val)
                score = amex_metric(y_val, oof_preds)
                results.append({'feature':features[k],'metric':score})
                x_val.iloc[:,k] = save_col
            
            # Displaying permutation feature importance
            print()
            permutation_df = pd.DataFrame(results)
            permutation_df = permutation_df.sort_values('metric', ascending = False)
            # Saving lgbm_baseline_model permutation feature importance
            df.to_csv(f'/content/drive/MyDrive/Amex/Permutations/permutation_feature_importance_fold{fold}.csv',index=False)

        del x_train, x_val, y_train, y_val, lgbm_model
        _ = gc.collect()

        if one_fold_only:
            break
