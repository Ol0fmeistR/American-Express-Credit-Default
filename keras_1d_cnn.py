# -*- coding: utf-8 -*-

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

from google.colab import output
output.enable_custom_widget_manager()

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install tensorflow==2.8.2

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install adabelief-tf --no-cache-dir 
# from adabelief_tf import AdaBeliefOptimizer

import pandas as pd
import numpy as np
import dill as pickle   
from matplotlib import pyplot as plt
import random
import datetime
import math
import gc
import os
import warnings
import seaborn as sns
import itertools
import multiprocessing
import joblib
warnings.simplefilter(action='ignore', category=FutureWarning)

from matplotlib.ticker import MaxNLocator
from colorama import Fore, Back, Style
from tqdm import tqdm
import h5py

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder, PowerTransformer
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight 
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(4)
from tensorflow import keras
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Concatenate, Dropout, BatchNormalization, Conv1D, Reshape, Flatten, AveragePooling1D, MaxPool1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
tf.__version__

def amex_metric(y_true, y_pred):
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
    
best_score = 0
class MyCustomMetricCallback(tf.keras.callbacks.Callback):

    def __init__(self, save_path, train=None, validation=None, best_score = 0):
        super(MyCustomMetricCallback, self).__init__()
        self.train = train
        self.validation = validation
        self.best_score = best_score
        self.save_path =save_path
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if self.train:
            pass

        if self.validation:
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid).reshape( (len(X_valid), )) 
            val_score = amex_metric(y_valid, y_pred)
            logs['my_metric_val'] = val_score
            if val_score>self.best_score:
                self.best_score = val_score
                self.best_epoch = epoch
                self.model.save(f"{self.save_path}.h5")
                print('best_val_score: ', self.best_score)
            elif self.best_epoch==0:
                if epoch-self.best_epoch > 40:
                    self.model.stop_training = True
            elif epoch-self.best_epoch > 12:
                self.model.stop_training = True
                
            del X_valid, y_valid, y_pred, val_score
            gc.collect()

ALPHA= 5
GAMMA = 2
def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    return focal_loss

# Using Mish activation function (more robust to noisy data)
from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.layers import Activation

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})

# Defining the model architecture
def my_model(n_inputs):
    """Sequential neural network with a skip connection.
    
    Returns a compiled instance of tensorflow.keras.models.Model.
    """
    activation = 'Mish'
    inputs = Input(shape=(n_inputs, ))
    x = Reshape((n_inputs, 1))(inputs)
    # 15 agg features per main feature, size = 15, step = 15.
    x = Conv1D(24,15,strides=15, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Conv1D(12,1, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Conv1D(4,1, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.33)(x)
    x = Dense(32, activation = activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation = activation)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    gc.collect()
    return Model(inputs, outputs)
model = my_model(190*15)
model.summary()
del model

VERBOSE = 0
CYCLES = 1
EPOCHS = 400
BATCH_SIZE = 2048
FOLDS = 10
SEED = 4   # 0, 1, 3, 4

def fit_model(seed, fold):
    train = joblib.load('/content/drive/MyDrive/Amex/CNN/train_agg_extra2_scaled.pkl').astype('float16')
    # Turn 2D feature array of shape (190, 15) into 1D array
    train = np.array([i.flatten('C') for i in train])
    gc.collect()
    
    target = pd.read_csv(f'/content/drive/MyDrive/train_labels.csv').target.astype('float32')
    idx_tr, idx_va = list(StratifiedKFold(n_splits=FOLDS, shuffle= True, random_state=SEED).split(target,target))[fold]
    X_va = train[idx_va]
    X_tr = train[idx_tr]
    y_tr, y_va = target[idx_tr], target[idx_va]
    del train, target, idx_tr
    gc.collect()
    lr = ReduceLROnPlateau(monitor="val_loss", 
                           factor=0.3, 
                           patience=5, 
                           mode = 'min', 
                           verbose=VERBOSE)
    es = EarlyStopping(monitor="val_loss",
                       patience=7, 
                       min_delta=0.00001,
                       verbose=VERBOSE,
                       mode="min", 
                       restore_best_weights=True)
    best_score= 0
    for seed1 in range(10):
        print('seed: ',seed1)
        np.random.seed(seed1)
        random.seed(seed1)
        tf.random.set_seed(seed1)
        custom = MyCustomMetricCallback(save_path = f'/content/drive/MyDrive/Amex/CNN/Models/model_fold{fold}_seed{seed}', validation=(X_va, y_va), best_score=best_score)
                    
        callbacks = [lr, 
                     es, 
                     tf.keras.callbacks.TerminateOnNaN(), 
                     custom,
                     ]
        model = my_model(X_tr.shape[1])
        model.compile(optimizer=AdaBeliefOptimizer(learning_rate=0.02,
                                                   weight_decay = 1e-5,
                                                   epsilon = 1e-7,
                                                   print_change_log = False,
            ),
            loss=FocalLoss,
            )
        gc.collect()
        model.fit(X_tr, y_tr, 
                validation_data=(X_va, y_va),
                epochs=EPOCHS,
                verbose=VERBOSE,
                batch_size=BATCH_SIZE,
                shuffle=True,
                callbacks=callbacks)
        best_score = custom.best_score
        gc.collect()
        K.clear_session()
        
def fit_train_models(current_fold):
    print(f'StratifiedKFold for Fold {current_fold} with SEED {SEED}:')
    fit_model(SEED, current_fold)
    gc.collect()

for CURRENT_FOLD in range(10):
    fit_train_models(CURRENT_FOLD)

# Get OOF predictions
train = joblib.load('/content/drive/MyDrive/Amex/CNN/train_agg_extra2_scaled.pkl').astype('float16')
train = np.array([i.flatten('C') for i in train])
_ = gc.collect()
target = pd.read_csv(f'/content/drive/MyDrive/train_labels.csv').target.astype('float32')

oof_predictions = np.zeros(len(train))

folds = [i for i in range(10)]
seeds = [0, 1, 3, 4]

_ = gc.collect()
for seed in tqdm(seeds):
    oof_pred = np.zeros(len(train))
    for fold, (idx_tr, idx_va) in enumerate(StratifiedKFold(n_splits=10, shuffle= True, random_state= seed).split(target,target)):
        model = tf.keras.models.load_model(f'/content/drive/MyDrive/Amex/CNN/Models/model_fold{fold}_seed{seed}.h5', 
                                           custom_objects={"mish": mish, 
                                                        "AdaBeliefOptimizer": AdaBeliefOptimizer(learning_rate=0.02,weight_decay = 1e-5,epsilon = 1e-7,print_change_log = False,), 
                                                        "FocalLoss": FocalLoss})
        oof = model.predict(train[idx_va]).reshape((len(idx_va),) ) 
        print(amex_metric(target[idx_va], oof))
        oof_pred[idx_va] += oof
        _ = gc.collect()
        del model, oof
        _ = gc.collect()
        K.clear_session()  
    print(amex_metric(target, oof_pred))
    oof_predictions += oof_pred/ len(seeds)

print(amex_metric(target, oof_predictions))
sub = pd.read_csv('/content/drive/MyDrive/train_labels.csv').drop('target', axis=1)
sub['prediction'] = oof_predictions
sub.to_csv('/content/drive/MyDrive/Amex/CNN/OOF/keras_CNN_oof.csv')
del train, target, oof_predictions, sub

# Get test predictions
test_pred = None
for pt in tqdm([1, 2]):
    test = joblib.load(f'/content/drive/MyDrive/Amex/CNN/test_agg_extra2_pt{pt}_scaled.pkl')
    test = np.array([i.flatten('C') for i in test])
    test_predictions = np.zeros(len(test))
    for seed, fold in tqdm(itertools.product(seeds, folds)):
        model = tf.keras.models.load_model(f'/content/drive/MyDrive/Amex/CNN/Models/model_fold{fold}_seed{seed}.h5', 
                                           custom_objects={"mish": mish, "AdaBeliefOptimizer": AdaBeliefOptimizer, "FocalLoss": FocalLoss})
        def split(a, n):
            k, m = divmod(len(a), n)
            return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
        split_ids = split(range(len(test)), 2)
        for (j, ids) in enumerate(split_ids):
            test_predictions[ids] += model.predict(test[ids]).reshape((len(ids),) ) / len(folds) / len(seeds)
            _ = gc.collect()
        del model, split_ids
        _ = gc.collect()
        K.clear_session()
    if test_pred is None:
        test_pred = test_predictions
    else:
        test_pred = np.concatenate([test_pred, test_predictions])
sub = pd.read_csv('/content/drive/MyDrive/amex_sample_submission.csv')
sub['prediction'] = test_pred
sub.to_csv('/content/drive/MyDrive/Amex/CNN/OOF/keras_CNN_test_predictions.csv')

del test, test_pred, test_predictions
_ = gc.collect()

# Added a blank column by mistake
sub = pd.read_csv('/content/drive/MyDrive/Amex/CNN/OOF/keras_CNN_test_predictions.csv').drop('Unnamed: 0', axis=1)
sub.to_csv('/content/drive/MyDrive/Amex/CNN/OOF/keras_CNN_test_predictions.csv', index = False)
sub.head()

