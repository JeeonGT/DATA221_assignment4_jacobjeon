import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

decision_tree_model_constrained = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=42)
decision_tree_model_constrained.fit(X_train, y_train)

y_train_predict = decision_tree_model_constrained.predict(X_train)
y_test_predict = decision_tree_model_constrained.predict(X_test)

train_accuracy_constrained = decision_tree_model_constrained.score(X_train, y_train)
test_accuracy_constrained = decision_tree_model_constrained.score(X_test, y_test)

feature_importance = pd.Series(decision_tree_model_constrained.feature_importances_,index=X.columns)
top5_important_features = feature_importance.sort_values(ascending=False).head(5)

print("Top 5 Important Features: ")
print(top5_important_features)

print("Training Accuracy Constrained:", train_accuracy_constrained)
print("Test Accuracy Constrained:", test_accuracy_constrained)

#controlling model complexity like max depth allows us to "simplify" the model to prevent
#the model from memorizing every noise data, preventing overfitting. The training accuracy might become
#lower, however it is a more accurate represenation of the models performance.
#Feature importance shows us which features are high information value, and shows us what
#contributes to the decision making of the model. This contributes to interpretability because
#it shows us which variables are more likely to cause cancer.