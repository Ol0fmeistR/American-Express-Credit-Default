# define target encoding class

#### samples -> minimum number of samples to be considered for target encoding ####
#### smoothing_factor -> smoothing factor, higher smoothing gives more priority to overall mean ####
#### noise -> start with small values like 0.01/0.001 ####

from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoding(BaseEstimator, TransformerMixin):
    # class initialization
    def __init__(self, categories='auto', samples=1, smoothing_factor=1, noise=0, random_state=None):
        if type(categories)==str and categories!='auto':
            self.categories = [categories]
        else:
            self.categories = categories
        self.samples = samples
        self.smoothing_factor = smoothing_factor
        self.noise = noise
        self.encodings = dict()
        self.prior = None
        self.random_state = random_state
    
    # add some noise for regularization
    def add_noise(self, series, noise):
        return series * (1 + noise * np.random.randn(len(series)))
        
    def fit(self, X, y=None):
        if type(self.categories)=='auto':
            self.categories = np.where(X.dtypes == type(object()))[0]
        
        temp = X.loc[:, self.categories].copy()
        temp['target'] = y
        self.prior = np.mean(y)
        for variable in self.categories:
            avg = (temp.groupby(by=variable)['target'].agg(['mean', 'count']))
            # compute smoothing 
            smoothing = (1 / (1 + np.exp(-(avg['count'] - self.samples) / self.smoothing_factor)))
            # higher the number of instances, lesser the effect of overall mean
            self.encodings[variable] = dict(self.prior * (1 - smoothing) + avg['mean'] * smoothing)      
        return self
    
    def transform(self, X):
        X_transform = X.copy()
        for variable in self.categories:
            X_transform[variable].replace(self.encodings[variable], inplace=True)
            unknown_value = {value:self.prior for value in X[variable].unique() if value not in self.encodings[variable].keys()}
            if len(unknown_value) > 0:
                X_transform[variable].replace(unknown_value, inplace=True)
            X_transform[variable] = X_transform[variable].astype(float)
            if self.noise > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                X_transform[variable] = self.add_noise(X_transform[variable], self.noise)
        return X_transform
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
