from cProfile import label

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

input_data = load_breast_cancer()

dataset = pd.DataFrame(input_data["data"], columns=input_data['feature_names'])
print(dataset)

# x.train,x.test,y.train,y.test = train_test_split(x,y,train_size=0.8, random_state=1)
"""Scatter Plot"""
sns.scatterplot(data=dataset, x='mean radius', y='mean fractal dimension', hue=input_data['target'],palette={0:'Red',1:'Green'})


