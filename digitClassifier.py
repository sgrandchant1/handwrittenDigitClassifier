import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = '/Users/santiagodegrandchant/Desktop/AI_Projects/handwrittenDigitClassifier/train.csv'
test = '/Users/santiagodegrandchant/Desktop/AI_Projects/handwrittenDigitClassifier/test.csv'
data_train = pd.read_csv(train)
data_test = pd.read_csv(test)

data_train.head()
data_test.head()

data_train = np.array(data_train)
data_test = np.array(data_test)

np.random.shuffle(data_train)
np.random.shuffle(data_test)

data_train = data_train.T
data_test = data_test.T

Y_train = data_train[0]
X_train = data_train[1:]

Y_test = data_test[0]
X_test = data_test[1:]

