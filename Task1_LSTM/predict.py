import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the trained model
model = load_model("lstm_model.keras")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Load the label encoder
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# Classify text
def classify_text(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=50,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_class = np.argmax(prediction, axis=1)[0]

    result = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    return result