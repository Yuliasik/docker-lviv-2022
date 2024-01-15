from tensorflow import keras
from sklearn import datasets

NUM_DIGITS = 10

class NeuralNetConfig:
    def __init__(self):
        self.digits = NUM_DIGITS
        self.image_height = 8
        self.image_width = 8
        self.channels = 1  # Grayscale images
        self.conv_layer_params = [
            (32, (3, 3), 'relu', 'same'),
            (64, (3, 3), 'relu', 'same'),
            (64, (3, 3), 'relu', 'same')
        ]
        self.dense_units = 128
        self.optimizer = 'sgd'
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']

def get_model(config):
    # Input layer
    input_layer = keras.layers.Input(shape=(config.image_height, config.image_width, config.channels))
    
    # Convolutional layers
    x = input_layer
    for filters, kernel_size, activation, padding in config.conv_layer_params:
        x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten layer
    x = keras.layers.Flatten()(x)

    # Dense layer
    x = keras.layers.Dense(units=config.dense_units, activation='relu')(x)

    # Output layer
    output_layer = keras.layers.Dense(units=config.digits, activation='softmax')(x)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)

    return model

# Usage:
# instantiate your configuration class
config = NeuralNetConfig()

# load your dataset using the class attributes
digit_features, digit_classes = datasets.load_digits(n_class=NUM_DIGITS, return_X_y=True)

# Now you can input your config into your get_model function to create your model
model = get_model(config)
