"""Having created the model, this file will run the inference on the verification data"""

# *******************Imports*************************
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer as att, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from peft import PeftModelForSequenceClassification
import numpy as np

#import pandas as pd
#from openpyxl.workbook import Workbook
from PEFT import model_save_dir

'PEFT.model_save_dir'

# ******************Constants************************

INFER_VERIFICATION = "1%"
VERI_BATCH_SIZE = 4

# ******************Data prep****************************

verification_dataset = load_dataset("cirimus/super-emotion",
                                    split=f"train[:{INFER_VERIFICATION}]")

print(f"Test#1 This is verification_dataset: {type(verification_dataset)}; content: {verification_dataset}")
tokenizer = att.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length")
    return tokenized_output


tokenized_verification_dataset = verification_dataset.map(preprocess_function, batched=True)

print(tokenized_verification_dataset[:1])

# *************Load trained Model and run inference******************

base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=7,
                                                                id2label={0: "NEUTRAL",
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

loaded_model = PeftModelForSequenceClassification.from_pretrained(base_model, str(model_save_dir))
tokenizer_loaded = att.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer)

my_trainer = Trainer(
    model=loaded_model,
    args=TrainingArguments(output_dir="./inference_output",
                           per_device_eval_batch_size=VERI_BATCH_SIZE,  # Adjust the batch size as required.
                           ),
    eval_dataset=tokenized_verification_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Apply the inference across a sample of the verification dataset

predictions_output = my_trainer.predict(tokenized_verification_dataset)
predicted_labels: object = np.argmax(predictions_output.predictions, axis=1)
print("Predicted Labels:", predicted_labels)
