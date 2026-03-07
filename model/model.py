from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from preprocessing import AUDIO_N_MELS, AUDIO_TIME_STEPS, MAX_FRAMES


def build_video_classifier(
    frame_count=MAX_FRAMES,
    frame_height=112,
    frame_width=112,
    mel_bands=AUDIO_N_MELS,
    mel_time_steps=AUDIO_TIME_STEPS
):
    """
    Dual-branch Audio-Visual classifier.

    Branch 1 (video_input): MobileNetV2 frozen CNN + GRU
    Branch 2 (audio_input): Small CNN on mel spectrogram
    Merged: Dense layers -> sigmoid output
    """

    # ── Branch 1: Video ──────────────────────────────────────
    video_input = keras.Input(
        shape=(frame_count, frame_height, frame_width, 3),
        name='video_input'
    )
    base_cnn = tf.keras.applications.MobileNetV2(
        input_shape=(frame_height, frame_width, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_cnn.trainable = False

    x = layers.TimeDistributed(base_cnn)(video_input)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(32)(x)
    x = layers.Dropout(0.3)(x)
    video_features = layers.Dense(64, activation='relu', name='video_features')(x)

    # ── Branch 2: Audio ──────────────────────────────────────
    audio_input = keras.Input(
        shape=(mel_bands, mel_time_steps, 1),
        name='audio_input'
    )
    a = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(audio_input)
    a = layers.MaxPooling2D((2, 2))(a)
    a = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(a)
    a = layers.MaxPooling2D((2, 2))(a)
    a = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(a)
    a = layers.GlobalAveragePooling2D()(a)
    a = layers.Dropout(0.3)(a)
    audio_features = layers.Dense(64, activation='relu', name='audio_features')(a)

    # ── Merge ────────────────────────────────────────────────
    merged = layers.Concatenate()([video_features, audio_features])
    merged = layers.Dense(64, activation='relu')(merged)
    merged = layers.Dropout(0.3)(merged)
    output = layers.Dense(1, activation='sigmoid', name='output')(merged)

    model = keras.Model(
        inputs=[video_input, audio_input],
        outputs=output,
        name='educational_av_classifier'
    )
    return model


def get_compiled_model():
    model = build_video_classifier()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model