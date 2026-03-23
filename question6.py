import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

convoluted_neural_network_model.fit(X_train,y_train,epochs=15)

test_loss, test_accuracy = convoluted_neural_network_model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

#CNNs are generally preferred over fully connected networks for image data because
#they can mimic human vision and is able to capture details such as edges

#In this task, the convolution layer is learning visual patterns from the
#clothing images, such as fabric curves, corners, and textures that distinguish
#clothing categories from one another.