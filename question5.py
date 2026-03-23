import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#decision tree constrained
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

print("Training Accuracy Constrained:", train_accuracy_constrained)
print("Test Accuracy Constrained:", test_accuracy_constrained)

#neural network
scaler = StandardScaler()
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

neural_network_model = MLPClassifier(hidden_layer_sizes=(10,),activation='logistic',max_iter=1000,random_state=42)

neural_network_model.fit(X_train_scaled, y_train)

train_acc_neural_network_model = neural_network_model.score(X_train_scaled, y_train)
test_acc_neural_network_model = neural_network_model.score(X_test_scaled, y_test)

y_pred_neural_network_model = neural_network_model.predict(X_test_scaled)

print("Training Accuracy:", train_acc_neural_network_model)
print("Test Accuracy:", test_acc_neural_network_model)

confusion_matrix_decision_tree_constrained = confusion_matrix(y_test, y_test_predict)
confusion_matrix_neural_network = confusion_matrix(y_test, y_pred_neural_network_model)

print("Confusion Matrix : Constrained Decision Tree:")
print(confusion_matrix_decision_tree_constrained)

print("Confusion Matrix : Neural Network:")
print(confusion_matrix_neural_network)

#I would prefer the model with a lower false negative rate, which is the neural network as that is the most
#dangerous mistake the model can make. If someone has cancer (malignant) but is predicted to be
#uncancerous (benign) then the patient is mistakenly considered healthy. This can be deadly for the patient
#if the model incorrectly predicts as so.

#Decision tree advantage: Easy to interpret, we can see which features contribute the most.
#Decision tree disadvantage: notorious for overfitting.

#Neural Network advantage: Much more complex and may be able to predict more accurately.
#Neural Network disadvantage: is a black box, less interpretable and we do not know how it works on the inside.