import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

y_train_predict = dt.predict(X_train)
y_test_predict = dt.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_predict)
test_accuracy = accuracy_score(y_test, y_test_predict)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

#Entropy measures the amount of uncertainty in the class at a node.
#a good split will choose the node that has higher information which decreases entropy.
#the training accuracy shows us a perfect score of 1.0, while test accuracy drops to around 0.91.
#this suggests overfitting, since the model excels on training data but falls off on unseen data.