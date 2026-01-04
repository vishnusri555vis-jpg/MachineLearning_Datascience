import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
import category_encoders as ce

col_names = {"Buying price","Maintainance cost","Door num"}
df = pd.readcsv("car_evaluation.csv", names=col_names,header=None)


