import torch
import pickle

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import PeftModel


# Model settings
BASE_MODEL = "distilbert-base-uncased"

LORA_MODEL = "./distilbert_lora_model"

TOKENIZER_PATH = "./distilbert_tokenizer"

LABEL_ENCODER_PATH = "distilbert_label_encoder.pkl"


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH
)


# Load label encoder
with open(
    LABEL_ENCODER_PATH,
    "rb"
) as file:
    label_encoder = pickle.load(file)


# Load base DistilBERT model
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(label_encoder.classes_)
)


# Load LoRA model
model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL
)


# Move model to device
model.to(device)

model.eval()


# Text classification function
def classify_text(text):

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # Make prediction
    with torch.no_grad():

        outputs = model(**inputs)


    predicted_class = torch.argmax(
        outputs.logits,
        dim=1
    ).item()


    # Convert class number to class name
    predicted_label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]


    return predicted_label