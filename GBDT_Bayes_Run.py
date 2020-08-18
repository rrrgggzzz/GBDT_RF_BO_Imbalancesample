import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from pandas import DataFrame

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

clf=GradientBoostingClassifier(n_estimators=310,max_depth=47,learning_rate=0.30346426751714195,subsample=0.9547479923789509)
clf.fit(x_train, y_train)
predict_target = clf.predict(x_test)
expected = y_train
predicted = clf.predict(x_test)
probability_1 = clf.predict_proba(x_test)
exp = DataFrame(probability_1)
exp.to_csv("F://GBDT_Bayes.csv")
export = DataFrame(y_test)
export.to_csv("F://GBDT_Bayes_predicted.csv")
print(probability_1)

probability = clf.predict_proba(x_1)
exp_1 = DataFrame(probability)
exp_1.to_csv("F://GBDT_Bayes_ALL.csv")
print(probability)

print(metrics.classification_report(y_test, predict_target))
print(metrics.confusion_matrix(y_test, predict_target))