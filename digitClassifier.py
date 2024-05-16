# I will be using a database that has already convirted hundreds of handwritten numbers 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filePath = '/Users/santiagodegrandchant/Desktop/AI_Projects/handwrittenDigitClassifier/train.csv'
data = pd.read_csv(filePath)
#print(data.head())

data = np.array(data)

