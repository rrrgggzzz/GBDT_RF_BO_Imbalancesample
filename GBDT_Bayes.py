import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from pandas import DataFrame
from hyperopt import fmin,tpe,hp,partial

test = pd.read_csv('F://test.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC', 'class'])

train = pd.read_csv('F://train.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC', 'class'])
target = 'class'
data = 'fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan', 'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'

x_columns_train = [x for x in train.columns if x not in [target]]
x_train = train[x_columns_train]
y_train = train['class']

x_columns_test = [x for x in test.columns if x not in [target]]
x_test = test[x_columns_test]
y_test = test['class']

dataset = pd.read_csv('F:\\inputdata.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC', 'class'])

x_columns_1 = [x for x in dataset.columns if x not in [target]]
x_1 = dataset[x_columns_1]
y_1 = dataset['class']

clf = GradientBoostingClassifier()
clf.fit(x_train, y_train)

def percept(args):
    global x_train,y_train,y_test
    ppn = GradientBoostingClassifier(n_estimators = int(args["n_estimators"]),learning_rate = args['learning_rate'],
                                     subsample = args['subsample'], max_depth=int(args['max_depth']),
                                 min_samples_leaf=int(args["min_samples_leaf"]),max_leaf_nodes= int(args["max_leaf_nodes"]),random_state= 0)
    ppn.fit(x_train, y_train)
    y_pred = ppn.predict(x_test)
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial
space = dict(n_estimators=hp.quniform("n_estimators", 50, 500, 1), learning_rate=hp.uniform('learning_rate', 0.1, 1.0),
             subsample=hp.uniform('subsample', 0.5, 0.8),max_depth=hp.quniform('max_depth', 1,1000,1),
             min_samples_leaf=hp.quniform("min_samples_leaf",1,100,1),
             max_leaf_nodes= hp.quniform(" max_leaf_nodes",2,17,1),random_state= 0)

algo = partial(tpe.suggest,n_startup_jobs=100)
best = fmin(percept,space,algo = algo,max_evals=500)
print (best)
print (percept(best))


