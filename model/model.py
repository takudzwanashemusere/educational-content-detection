from tensorflow import keras
from tensorflow.keras import layers

def build_video_classifier(frame_count=10, frame_height=112, frame_width=112): #changed
    """
    Build CNN-RNN model for educational video classification
    """
    inputs = keras.Input(shape=(frame_count, frame_height, frame_width, 3))

    # CNN over frames
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation='relu', padding='same')
    )(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation='relu', padding='same') #changed
    )(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), activation='relu', padding='same') #changed
    )(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    # Flatten spatial features
    x = layers.TimeDistributed(layers.Flatten())(x)

    # GRU over time
    x = layers.GRU(64, return_sequences=True)(x) #changed
    x = layers.Dropout(0.5)(x)
    x = layers.GRU(32)(x)   #chageed
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # 1 = educational / not

    model = keras.Model(inputs, outputs, name='educational_video_classifier')
    return model


def get_compiled_model():
    """
    Create and compile the model ready for training
    """
    model = build_video_classifier()

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model

