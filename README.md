# Course_2_Project_5_Fine_Tuning
Project Brief/

Project 5.0 Apply a Lightweight Fine-tuning model to a Foundation Model
Project Introduction
Lightweight fine-tuning is one of the most important techniques for adapting foundation models, because it allows you to modify foundation models for your needs without needing substantial computational resources.

In this project, you will apply parameter-efficient fine-tuning using the Hugging Face peft library.

Project Summary\n
In this project, you will bring together all of the essential components of a PyTorch + Hugging Face training and inference process. Specifically, you will:

Load a pre-trained model and evaluate its performance
Perform parameter-efficient fine tuning using the pre-trained model
Perform inference using the fine-tuned model and compare its performance to the original model

Emotion Classification with BERT & PEFT (LoRA)
Project Overview\n
This project explores building a text classification model using Hugging Face Transformers. The objective was to classify emotional content in text using a pre-trained BERT model, enhanced with parameter-efficient fine-tuning (PEFT).
Model Choice
Model: AutoModelForSequenceClassification

Rationale:\n
•	Familiarity from prior learning and documentation.
•	Proven performance for sequence classification tasks.
•	Seamless integration with Hugging Face ecosystem.

Dataset:\n
Source: cirimus/super-emotion
Composition: Pre-split into Train, Evaluation, and Validation sets.

Challenges:\n
•	Dataset Size: Extremely large, which constrained training speed and hardware usage.
•	Label Complexity: 6 emotion classes created a high-dimensional classification task.
Due to resource limitations, only small representative samples were used from each dataset split.

Training Constraints & Observations\n
•	Hardware limitations meant:
o	Incomplete training cycles.
o	Limited ability to tune batch size, learning rate, and epochs.
•	Initial training with all parameters frozen over 4 epochs yielded accuracy improvement from 0.57 → 0.59.
•	With parameters unfrozen, a learning rate of 5e-5 and batch size of 16 achieved accuracy of 0.64.
•	Incorporating PEFT (LoRA) led to modest gains in training efficiency and slight improvements in accuracy—but the complexity and size of the dataset ultimately capped performance.
•	A full breakdown of results can be referenced in the file 24_03_2025_Learning and finetuning.docx on Github

Outcome\n
•	A working BERT-based text classification model was created.
•	Performance was constrained by dataset scale, model complexity, and hardware availability particularly on the PC platform, though this was less the case on Jupyter.
•	PEFT fine-tuning demonstrated potential but couldn't fully overcome dataset challenges and efficiency issues.
•	Inference worked on the PC platform producing “predicted Labels”.  However, the Jupyter platform presented issues with regard the adapter file location which at time of submission I could not solve.  This was a frustrating result as the 1st run of the inference produced results.

Key Learnings\n
•	Choose datasets aligned with available compute resources.
•	Label complexity can significantly impact model convergence and evaluation.
•	PEFT approaches (e.g., LoRA) offer solid tradeoffs in low-resource environments.
