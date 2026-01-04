import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris #When used default dataset - system know which is x(data), y(target)
import pandas as pd

load_iris_input = load_iris() # load iris is a classifying dataset
# print(load_iris_input)
df = pd.DataFrame(
    load_iris_input.data,
    columns=load_iris_input.feature_names
)
df['target'] = load_iris_input.target
# print(df)

target_names = load_iris_input.target_names
# print(target_names) #printing the target-values

x = load_iris_input.data
y = load_iris_input.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=1)

model = DecisionTreeClassifier()
# model_1 = DecisionTreeClassifier(criterion="Entropy")
model.fit(x_train,y_train)
prediction = model.predict(x_test) #Prediction
print("Prediction: ",prediction)

metric_evaluation = accuracy_score(y_test, prediction)
print("Accuracy: ", metric_evaluation)

"""Decision Tree Plot"""
plt.figure(figsize=(10,10))
plot_tree(model,feature_names=load_iris_input.feature_names,class_names=load_iris_input.target_names,filled=True)
plt.show()