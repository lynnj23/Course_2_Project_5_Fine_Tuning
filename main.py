"""This is an assessment for the Udacity Gen AI for developers nano-degree. \
The aim is to take an LLM, attach a classification head, train the LLM to requirements\
then output a table of learned results."""

# Importing areas
import torch
from datasets import load_dataset, concatenate_datasets
# Step 2 import the tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer as att

# import pandas as pd

# Constants
SPLIT_TRAINING = "1%"
TEST_DATA = "1%"

# Import the chosen model; in this instance I have taken recommendation to stick with LoRa


# Import the chosen labeled dataset from Huggingface.

train_dataset = load_dataset("cirimus/super-emotion",
                             split=f"train[:{SPLIT_TRAINING}]")  # for testing purposes load first 1%. \
# Stream or slice later dependent on the accuracy of the training.
test_dataset = load_dataset("cirimus/super-emotion", split=f"test[:{TEST_DATA}]")  # this loads the test dataset

train_dataset = train_dataset.add_column("split", ["train"] * len(train_dataset))
test_dataset = test_dataset.add_column("split", ["test"] * len(test_dataset))

combined_dataset = concatenate_datasets([train_dataset, test_dataset])

# print(f"test#4 This is the train set: {train_dataset}")
# print(f"test#5 This is the train set: {test_dataset}")

df = combined_dataset.to_pandas()
# print the first 5 rows for visual check
print(df.head(5))

for entry in combined_dataset.select(range(3)):
    text = entry["text"]
    label = entry["label"]
    source = entry["source"]
    split = entry["split"]
    print(f"Test#1 message = {text}, Label = {label},source = {source}, action = {split}\n")

# Following previous examples to make labels easier to understand

subset = combined_dataset[0:3]
print(f"Test#2 This is subset: {subset}\n")

id2label = {0: "NEUTRAL",
            1: "SURPRISE",
            2: "FEAR",
            3: "SADNESS",
            4: "JOY",
            5: "ANGER",
            6: "LOVE"
            }

label2id = {"NEUTRAL": 0,
            "SURPRISE": 1,
            "FEAR": 2,
            "SADNESS": 3,
            "JOY": 4,
            "ANGER": 5,
            "LOVE": 6
            }
for text, label_id in zip(subset["text"][:3], subset["label"][:3]):
    label_id = entry["label"]
    print(f"sample={split},label={id2label[label_id]}, text={text}")

# Tokenize the data for use in the model

tokenizer = att.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    token_dataset = []
    for txt, s_label in zip(examples["text"], examples["split"]):
        if s_label == "test":
            tokenized = tokenizer(txt, truncation=True, padding="max_length")
            token_dataset.append(tokenized["input_ids"])
        else:
            # Tokenising non-test data here. This is a bit ugly, but I have got myself in a twist here at a moment
            tokenized = tokenizer(txt, truncation=True, padding="max_length")
            token_dataset.append(tokenized["input_ids"])
    return {"input_ids": token_dataset}


tokenized_dataset = combined_dataset.map(preprocess_function, batched=True)

# Print tokenized output for the first few examples for verification.
print("Test#7 Tokenized 'text' field for the first few examples:")
for entry in tokenized_dataset.select(range(1)):
    print(entry["input_ids"])

# Load the base transformer model before moving to wrap a classifier
my_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                           num_labels=7, id2label={0: "NEUTRAL",
                                                                                   1: "SURPRISE",
                                                                                   2: "FEAR",
                                                                                   3: "SADNESS",
                                                                                   4: "JOY",
                                                                                   5: "ANGER",
                                                                                   6: "LOVE"
                                                                                   }, label2id={"NEUTRAL": 0,
                                                                                                "SURPRISE": 1,
                                                                                                "FEAR": 2,
                                                                                                "SADNESS": 3,
                                                                                                "JOY": 4,
                                                                                                "ANGER": 5,
                                                                                                "LOVE": 6
                                                                                                })

# Freeze all the parameters of the base model
for param in my_model.base_model.parameters():
    param.requires_grad = False

my_model.classifier
print(f"Test#8 My model output: {my_model}")

# Load the trained model


# The Output text classification results in table format comparing before and after outputs.
