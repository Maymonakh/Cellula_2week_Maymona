import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)


# Load Dataset
df = pd.read_csv("cellula_toxic_data.csv")

print("Dataset shape:", df.shape)


# Create the text
df["text"] = (
    df["query"].fillna("")
    + " "
    + df["image descriptions"].fillna("")
)


# Encode Labels
label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["Toxic Category"]
)

num_classes = len(label_encoder.classes_)

print("\nClasses:")
print(label_encoder.classes_)

print("\nNumber of classes:", num_classes)


# Split the Data
X = df["text"]
y = df["label"]


# First split: Training + Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Second split: Training + Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)


print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


# Prepare the Datasets
train_data = Dataset.from_dict({
    "text": X_train.tolist(),
    "label": y_train.tolist()
})

val_data = Dataset.from_dict({
    "text": X_val.tolist(),
    "label": y_val.tolist()
})

test_data = Dataset.from_dict({
    "text": X_test.tolist(),
    "label": y_test.tolist()
})


# Tokenize the Data
def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


train_data = train_data.map(
    tokenize_function,
    batched=True
)

val_data = val_data.map(
    tokenize_function,
    batched=True
)

test_data = test_data.map(
    tokenize_function,
    batched=True
)


# Remove the original text
train_data = train_data.remove_columns(["text"])
val_data = val_data.remove_columns(["text"])
test_data = test_data.remove_columns(["text"])


# Set dataset format
train_data.set_format("torch")
val_data.set_format("torch")
test_data.set_format("torch")


# Load DistilBERT
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_classes
)


# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]
)


# Apply LoRA
model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# Calculate F1 Score
def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "f1": weighted_f1
    }


# Training Settings
training_args = TrainingArguments(
    output_dir="./distilbert_lora_results",

    num_train_epochs=5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",
    greater_is_better=True,

    report_to="none"
)


# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_data,
    eval_dataset=val_data,

    compute_metrics=compute_metrics
)

trainer.train()


# Evaluate on Test Set
predictions = trainer.predict(test_data)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=1
)

true_labels = predictions.label_ids


# Calculate Evaluation Metrics
accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

weighted_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted"
)

macro_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro"
)

print("Final Test Results")

print("Accuracy:", accuracy)
print("Weighted F1:", weighted_f1)
print("Macro F1:", macro_f1)


# Classification Report
print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=label_encoder.classes_
    )
)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(
    figsize=(12, 10)
)

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "DistilBERT + LoRA Confusion Matrix"
)

plt.tight_layout()


# Save the Label Encoder
with open(
    "distilbert_label_encoder.pkl",
    "wb"
) as file:

    pickle.dump(
        label_encoder,
        file
    )


# Save the Tokenizer
tokenizer.save_pretrained(
    "./distilbert_tokenizer"
)


# Save the Best LoRA Model
trainer.save_model(
    "./distilbert_lora_model"
)


print(
    "\nModel, tokenizer, and label encoder "
    "saved successfully."
)

plt.show()