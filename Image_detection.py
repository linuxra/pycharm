import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


class ImageClassifier:
    """
    A Convolutional Neural Network model for image classification.

    Attributes:
        model: The TensorFlow Sequential model.
    """

    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        """
        Initialize ImageClassifier with network architecture.

        Args:
            input_shape (tuple): The shape of an input image. Default is (64, 64, 3).
            num_classes (int): The number of output classes. Default is 10.
        """
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(num_classes, activation='softmax'))

    def compile_model(self):
        """
        Compile the model with Adam optimizer and sparse categorical cross-entropy loss function.
        """
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self, x_train, y_train, batch_size=32, epochs=10, validation_split=0.2):
        """
        Train the model on the provided data.

        Args:
            x_train: Training images.
            y_train: Training labels.
            batch_size (int): Number of samples per gradient update. Default is 32.
            epochs (int): Number of epochs to train the model. Default is 10.
            validation_split (float): Fraction of the training data to be used as validation data. Default is 0.2.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                       callbacks=[early_stopping])

    def evaluate_model(self, x_test, y_test):
        """
        Evaluate the model's performance on the provided test data.

        Args:
            x_test: Test images.
            y_test: Test labels.

        Returns:
            Test loss and test accuracy.
        """
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        """
        Use the trained model to predict the labels of the provided data.

        Args:
            x: Input data for which to predict labels.

        Returns:
            Predicted labels for the input data.
        """
        return self.model.predict(x)
