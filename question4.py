from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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

print("Training Accuracy:", train_acc_neural_network_model)
print("Test Accuracy:", test_acc_neural_network_model)

#feature scaling is necessary for neural networks because it uses gradient descent, and if the
#features are not normalized before hand each step becomes messy and the model becomes useless.
#an epoch is a full training session on all the training data. The neural network will run one epoch
#then repeat it so that it adjusts the weights each time and continues to learn.