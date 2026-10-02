import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load Dataset
df = pd.read_csv("cellula_toxic_data.csv")

print("Dataset shape:", df.shape)

# Create the text
df["text"] = df["query"] + " " + df["image descriptions"]

X = df["text"]
y = df["Toxic Category"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Tokenizer
tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Padding
MAX_LEN = 50

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

# Encode labels
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Validation split
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_padded,
    y_train_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_train_encoded
)

# Class weights
classes = np.unique(y_train_final)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_final
)

class_weights = dict(
    zip(classes, class_weights_array)
)

# Build the LSTM
vocab_size = len(tokenizer.word_index) + 1
num_classes = len(label_encoder.classes_)

model = Sequential([
    Input(shape=(MAX_LEN,)),

    Embedding(
        input_dim=vocab_size,
        output_dim=128,
        mask_zero=True
    ),

    LSTM(64),

    Dropout(0.3),

    Dense(32, activation="relu"),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_final,
    y_train_final,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Evaluate
y_pred_prob = model.predict(X_test_padded)
y_pred = np.argmax(y_pred_prob, axis=1)

f1_weighted = f1_score(
    y_test_encoded,
    y_pred,
    average="weighted"
)

f1_macro = f1_score(
    y_test_encoded,
    y_pred,
    average="macro"
)

print("Weighted F1 Score:", f1_weighted)
print("Macro F1 Score:", f1_macro)

print(
    classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# Save the model & preprocessing files
import pickle

model.save("lstm_model.keras")

with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

print("Model and preprocessing files saved successfully.")