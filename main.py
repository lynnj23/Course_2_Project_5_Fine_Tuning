"""This is an assessment for the Udacity Gen AI for developers nano-degree. \
The aim is to take an LLM, attach a classification head, train the LLM to requirements\
then output a table of learned results."""

# Importing areas
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer as att, DataCollatorWithPadding, Trainer, \
    TrainingArguments
import pandas as pd
import numpy as np

# Constants
SPLIT_TRAINING = "2%"
SPLIT_EVAL = "2%"
LEARN_RATE = 2e-3
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
TRAIN_EPOCHS = 1
WGT_DECAY = 0.01
MY_EVAL_STRAT = "epoch"
S_STRAT = "epoch"
BEST_MODEL = True

# Import the chosen model; in this instance I have taken the recommendation to stick with LoRa


# Import the chosen labelled dataset from Huggingface.
# REMEMBER train_dataset is a huggingface dataset and NOT a dictionary
train_dataset = load_dataset("cirimus/super-emotion",
                             split=f"train[:{SPLIT_TRAINING}]")  # for testing purposes load first 1%. \
# Stream or slice later dependent on the accuracy of the training.
eval_dataset = load_dataset("cirimus/super-emotion", split=f"test[:{SPLIT_EVAL}]")  # this loads the eval dataset

print(f"Test#1 This is train_dataset: {type(train_dataset)}; content: {train_dataset}")
print(f"Test#2 This is eval_dataset: {type(eval_dataset)}; content: {eval_dataset}")

# Add a column to differentiate between the two datasets train and eval
train_dataset = train_dataset.add_column("split", ["train"] * len(train_dataset))
eval_dataset = eval_dataset.add_column("split", ["eval"] * len(eval_dataset))

# concatenate the datafiles together into a
combined_dataset = concatenate_datasets([train_dataset, eval_dataset])
df = combined_dataset.to_pandas()
print(f"Test#5 {df.head(5)}")
print(f"Test#5.1 This is combined dataset: {type(combined_dataset)}; content: {combined_dataset}")

# set up tokenizer and run it through the combined dataset

tokenizer = att.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    # Tokenize the text field
    tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length")
    # Retain the 'split' metadata from the original dataset
    if "split" in examples:
        tokenized_output["split"] = examples["split"]
    return tokenized_output


tokenized_dataset = combined_dataset.map(preprocess_function, batched=True)

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
# When we fine tune the param will be unfrozen
for param in my_model.base_model.parameters():
    param.requires_grad = False

var = my_model.classifier
print(f"Test#8 My model output: {my_model}")

# Time to train my_model and initial evaluation

def compute_metrics(eval_pred):
    model_predictions, true_labels = eval_pred

    # converting model_prediction logits into class indices
    predicted_labels = np.argmax(model_predictions, axis=1)

    print(f"Test#9 These are predicted labels {predicted_labels}")

    model_accuracy: object = (predicted_labels == true_labels).mean()

    print(f"Test#10 This model accuracy {model_accuracy}")

    #return the model accuracy

    return {"accuracy": model_accuracy}


# filter the combined dataset by the split column
train_tokenized_dataset = tokenized_dataset.filter(lambda x: x["split"] == "train")
eval_tokenized_dataset = tokenized_dataset.filter(lambda x: x["split"] == "eval")

my_trainer = Trainer(
    model=my_model,
    args=TrainingArguments(
        output_dir="./data/sentiment_analysis",
        learning_rate=LEARN_RATE,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
        weight_decay=WGT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=BEST_MODEL,
    ),
    # ****************dataset amendment***********************
    train_dataset=train_tokenized_dataset,
    eval_dataset=eval_tokenized_dataset,
    # ********************************************************
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

my_trainer.train()
# now evaluate the model
eval_results = my_trainer.evaluate()

# ------------------ Tabularisation of Results ------------------
# Create a table to display evaluation metrics in a user-friendly format.
df_eval = pd.DataFrame([{
    "Eval Loss": eval_results.get("eval_loss"),
    "Eval Accuracy": eval_results.get("eval_accuracy"),
    "Eval Runtime (s)": eval_results.get("eval_runtime"),
    "Eval Samples/sec": eval_results.get("eval_samples_per_second"),
    "Eval Steps/sec": eval_results.get("eval_steps_per_second"),
    "Epoch": eval_results.get("epoch")
}])

print("\nTabular Evaluation Metrics:")
print(df_eval.to_string(index=False))
# ---------------------------------------------------------------
