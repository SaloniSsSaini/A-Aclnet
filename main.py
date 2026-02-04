# =====================================================
# IMPORTS
# =====================================================

import kagglehub
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# =====================================================
# CONFIGURATION
# =====================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2


# =====================================================
# DATASET DOWNLOAD & PREPARATION (ADDED)
# =====================================================

def prepare_dataset():
    print("Downloading dataset using kagglehub...")

    path = kagglehub.dataset_download(
        "paultimothymooney/chest-xray-pneumonia"
    )

    print("Dataset downloaded at:", path)

    source_base = os.path.join(path, "chest_xray")
    target_base = "data"

    splits = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]

    for split in splits:
        for cls in classes:
            src = os.path.join(source_base, split, cls)
            dst = os.path.join(target_base, split, cls)

            os.makedirs(dst, exist_ok=True)

            for file in os.listdir(src):
                shutil.copy(
                    os.path.join(src, file),
                    os.path.join(dst, file)
                )

    print("Dataset successfully copied into project data folder!")


# =====================================================
# DATA LOADING & PREPROCESSING
# =====================================================

def load_data():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        "data/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_data = test_gen.flow_from_directory(
        "data/val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = test_gen.flow_from_directory(
        "data/test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_data, val_data, test_data


# =====================================================
# FOCAL LOSS
# =====================================================

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * ce)
    return loss


# =====================================================
# ATTENTION LAYER
# =====================================================

class Attention(tf.keras.layers.Layer):
    def call(self, inputs):
        weights = tf.nn.softmax(inputs, axis=1)
        return tf.reduce_sum(weights * inputs, axis=1)


# =====================================================
# MODEL ARCHITECTURE
# =====================================================

def build_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = base_model(inputs)

    reshaped = Reshape((1, features.shape[1]))(features)
    lstm_out = LSTM(128, return_sequences=True)(reshaped)

    attention_out = Attention()(lstm_out)

    x = Dense(64, activation="relu")(attention_out)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model


# =====================================================
# TRAINING
# =====================================================

def train_model(train_data, val_data):
    model = build_model()

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=focal_loss(),
        metrics=["accuracy"]
    )

    print(model.summary())

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    return model, history


# =====================================================
# EVALUATION
# =====================================================

def evaluate_model(model, test_data):
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)

    print("\nClassification Report:")
    print(classification_report(test_data.classes, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_data.classes, y_pred))

    fpr, tpr, _ = roc_curve(test_data.classes, predictions[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    # 1️⃣ Dataset auto-download & setup
    prepare_dataset()

    # 2️⃣ Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data()

    # 3️⃣ Train model
    print("Training model...")
    model, history = train_model(train_data, val_data)

    # 4️⃣ Evaluate model
    print("Evaluating model...")
    evaluate_model(model, test_data)

    # 5️⃣ Save model
    os.makedirs("outputs/models", exist_ok=True)
    model.save("outputs/models/a_aclnet.h5")

    print("Model saved successfully!")
