from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import itertools
import numpy as np

def read_data(cols):
    print('Reading data...')
    df = pd.read_parquet('/content/drive/MyDrive/train.parquet', columns=cols)
    # simplify customer_id
    unique_cus_ids = df.customer_ID.unique()
    assignment     = dict(zip(unique_cus_ids, list(range(len(unique_cus_ids)))))
    df.customer_ID = df.customer_ID.apply(lambda x: assignment[x]).astype('int32')
    
    print('shape of data:', df.shape)
    return df

# method for Information Value
def iv_woe(data, target, bins=20, show_woe=False):
    # empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    # extract Column Names
    cols = data.columns
    
    # run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        print(ivars)
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>3):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
            
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        # show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

# high information features (selected based on correlation with target variable)
high1_initial = ['P_2',
        'D_48', 'D_44', 'D_61', 'D_39', 'D_41', 'D_55', 'D_58', 'D_62', 'D_75', 'D_74'
        'R_1', 'R_2', 'R_4',
        'S_3', 'S_7',
        'B_7', 'B_23', 'B_9', 'B_2', 'B_1', 'B_3', 'B_4', 'B_11', 'B_16', 'B_18', 'B_19', 'B_20', 'B_22', 'B_33', 'B_37']

high = ['P_2',
        'D_48', 'D_44', 'D_61',
        'R_1',
        'S_7', 'S_3', # both spend variables are only 30 % correlated to target variable
        'B_7', 'B_23', 'B_9',
        ]

# get all possible combination of high information features
all_pairs = []
for i in range(len(high) -1):
    all_pairs.extend(list(itertools.product([high[i]], high[i+1:])))

# read only high-information features
data  = read_data(['customer_ID'] + high)
# read targets
target = pd.read_csv('/content/drive/MyDrive/train_labels.csv', usecols=['target'])

def woe_brute_force(info_cutoff):
    all_features = pd.DataFrame()

    for pair in all_pairs:
        # perform basic aggregations
        group_a = data[['customer_ID', pair[0]]].groupby('customer_ID').agg(['mean', 'std', 'min', 'max', 'first', 'last'])
        group_b = data[['customer_ID', pair[1]]].groupby('customer_ID').agg(['mean', 'std', 'min', 'max', 'first', 'last'])
        group_a.columns = [x[1] + '_' for x in group_a.columns]
        group_b.columns = [x[1] + '_' for x in group_b.columns]

        # combinations
        new_features = pd.DataFrame()

        # generating a bunch of new features based on weight of evidence
        new_features[f'{pair[0]}_last_t_{pair[1]}_std']  = group_a.last_ * group_b.std_
        new_features[f'{pair[0]}_last_d_{pair[1]}_mean'] = group_a.last_ / group_b.mean_
        new_features[f'{pair[0]}_last_p_{pair[1]}_max']  = group_a.last_ + group_b.max_
        new_features[f'{pair[0]}_last_m_{pair[1]}_min']  = group_a.last_ - group_b.min_
        new_features[f'{pair[0]}_last_t_{pair[1]}_first']  = group_a.last_ * group_b.first_
        new_features[f'{pair[0]}_last_t_{pair[1]}_last']  = group_a.last_ * group_b.last_

        new_features[f'{pair[0]}_mean_t_{pair[1]}_std']  = group_a.mean_ * group_b.std_
        new_features[f'{pair[0]}_mean_d_{pair[1]}_mean'] = group_a.mean_ / group_b.mean_
        new_features[f'{pair[0]}_mean_p_{pair[1]}_max']  = group_a.mean_ + group_b.max_
        new_features[f'{pair[0]}_mean_m_{pair[1]}_min']  = group_a.mean_ - group_b.min_
        new_features[f'{pair[0]}_mean_t_{pair[1]}_first']  = group_a.mean_ * group_b.first_
        new_features[f'{pair[0]}_mean_t_{pair[1]}_last']  = group_a.mean_ * group_b.last_

        # clean possible inf's
        new_features.replace([np.inf, -np.inf], -999, inplace=True)

        # get Information Value
        new_features['target'] = target.target
        a, b = iv_woe(new_features, 'target')

        # select only new features with high information!
        good_ones = a.loc[a.IV > info_cutoff].Variable.values
        print(good_ones)

        # save new high-information features
        all_features[good_ones] = new_features[good_ones]
        print('\n', all_features.shape, '\n')
    return all_features
        
new_features = woe_brute_force(2.0)

# new features generated
cols = [col for col in new_features.columns]
print(cols)
