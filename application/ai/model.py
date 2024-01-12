from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def get_cnn_model():
    model = Sequential()

    # Convolutional layer with 32 filters, kernel size of 3x3
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))

    # Another convolutional layer with 64 filters and kernel size of 3x3. Adding more filters can help the model to learn more complex patterns
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Max pooling layer to reduce spatial dimensions
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Regularization layer using dropout
    model.add(Dropout(0.25))

    # Flatten layer to reshape the tensor output from previous layer to fit fully connected layer
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))

    # Another dropout layer for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(NUM_DIGITS, activation='softmax'))

    # Compilation of the model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
