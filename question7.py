import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

convoluted_neural_network_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

convoluted_neural_network_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#convoluted_neural_network_model.fit(X_train,y_train,epochs=15)

test_loss, test_accuracy = convoluted_neural_network_model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

y_prob = convoluted_neural_network_model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Fashion MNIST CNN")
plt.show()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

misclassified_idx = np.where(y_pred != y_test)[0]
print("Number of misclassified images:", len(misclassified_idx))

plt.figure(figsize=(10, 4))
for i in range(3):
    idx = misclassified_idx[i]
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    plt.title(
        f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

#One pattern in the misclassifications is that clothes with sleeves are often confused and the model
#struggles to distinguish between types of long sleeve clothes. For example, a long sleeve shirt was
#misclassified as a coat/jacket.

#a way to increase CNN performance is to add another convolution layer so that the model requires
#more learning to analyse and conclude the minor details that distinguish similar categories from one another.