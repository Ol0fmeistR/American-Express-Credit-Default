# -*- coding: utf-8 -*-

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from warnings import filterwarnings
filterwarnings('ignore')

def read_preprocess_data():
    train = pd.read_parquet('/content/drive/MyDrive/train.parquet')
    problematic_variables = ['B_29', 'D_103', 'D_139', 'S_9']  
    to_drop = ['customer_ID', 'S_2'] + problematic_variables

    # Calculating distance between statement dates
    train['S_2'] = pd.to_datetime(train['S_2'])
    train['SDist'] = train.groupby("customer_ID")['S_2'].diff() / np.timedelta64(1, 'D')
    train['SDist'].fillna(30.53, inplace=True)

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

    # Merging everything together
    train = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_labels, how = 'inner', on = 'customer_ID')
    del train_num_agg, train_cat_agg
    _ = gc.collect()

    # Save files to disk
    train.to_parquet('train_fe.parquet')
    print(f'Shape of train: {train.shape}')

# Read & Preprocess Data
read_preprocess_data()

class CFG:
    seed = 42
    n_folds = 5
    target = 'target'
    metric = 'binary_logloss'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Load data files
train = pd.read_parquet('train_fe.parquet')
train.fillna(value = -1, inplace=True)

target = train[CFG.target]
train.drop(['target'], axis=1, inplace=True)

# Train/test split
features = [col for col in train.columns if col not in ['customer_ID', 'target']]
X_train, X_val, y_train, y_val = train_test_split(train[features], target, test_size=0.25, random_state=1)

del train
_ = gc.collect()

# Defining callbacks
lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.5, 
    patience=10, 
    verbose=True
)

es = keras.callbacks.EarlyStopping(
    monitor="val_loss", 
    patience=15, 
    verbose=True, 
    mode="min", 
    restore_best_weights=True
)

# Adding noise to the data
class SwapRowNoise:
    def __init__(self, proba):
        self.proba = proba
    
    def apply(self, X):
        random_idx = np.random.randint(low=0, high=X.shape[0], size=1)[0]
        swap_matrix = K.random_bernoulli(shape=X.shape, p=self.proba) * tf.ones(shape=X.shape)    
        corrupted = tf.where(swap_matrix==1, X.iloc[random_idx], X)
        return corrupted.numpy()

# Custom Mish activation function 
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

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

# Create autoencoder
class EncodingLayer(layers.Layer):
    def __init__(self, encoding_dim, activation='Mish'):
        super().__init__()
        self.enc1 = layers.Dense(encoding_dim, activation)
        self.enc2 = layers.Dense(encoding_dim, activation)
        self.enc3 = layers.Dense(encoding_dim, activation)
        self.concat = layers.Concatenate()
    
    def call(self, inputs):
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        merge = self.concat([enc1, enc2, enc3])
        return merge

class DecodingLayer(layers.Layer):
    def __init__(self, num_outputs, activation='linear'):
        super().__init__()
        self.dec = layers.Dense(num_outputs, activation)
    
    def call(self, inputs):
        return self.dec(inputs)
    
class AutoEncoder(keras.Model):
    def __init__(self, encoding_dim, num_outputs, activation='Mish'):
        super().__init__()
        self.encoder = EncodingLayer(encoding_dim, activation,)
        self.decoder = DecodingLayer(num_outputs)
    
    def call(self, inputs):
        encoder = self.encoder(inputs)
        decoder = self.decoder(encoder)
        return decoder
    
    def get_encoder(self):
        return self.encoder

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    tf_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except:
    tf_strategy = tf.distribute.get_strategy()
    print(f"Running on {tf_strategy.num_replicas_in_sync} replicas")
    print("Number of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Training and inference

# Creating a noisy version of our train set
noise_maker = SwapRowNoise(0.10)
noisy_train = noise_maker.apply(X_train)
noisy_valid = noise_maker.apply(X_val)

# Applying standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
noisy_train = scaler.transform(noisy_train)
noisy_valid = scaler.transform(noisy_valid)

# Training the autoencoder
with tf_strategy.scope():
    ae = AutoEncoder(
        encoding_dim=2*X_train.shape[-1],
        num_outputs=X_train.shape[-1],
        activation='Mish'
    )
    ae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
        loss=keras.losses.MeanSquaredError()
    )

print('_'*50)
print(f"Training Autoencoder...")
print('_'*50)

history_ae = ae.fit(
    noisy_train, X_train,
    validation_data=(noisy_valid, X_val),
    epochs=500,
    batch_size=64,
    shuffle=True,
    verbose=False,
    callbacks=[lr,es]
)

scores_ae = history_ae.history
print('_'*50)
print(f"AE Min Val Loss: {scores_ae}")
print('_'*50)

del X_train, X_val, noisy_train, noisy_valid
_ = gc.collect()

# Retrieving the encoder part
encoder = ae.get_encoder()

