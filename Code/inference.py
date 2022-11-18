import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from scipy.special import softmax
import re

task='sentiment'
MODEL = "cffl/bert-base-styleclassification-subjective-neutral"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
OUTPUT_DIR = 'model_all/' # Change it to model DIRECTORY
model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")


test_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    do_train = False,
    do_predict = True,
    dataloader_drop_last = False    
)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
# init trainer
trainer = Trainer(
              model = model, 
              args = test_args, 
              compute_metrics = compute_metrics)
eval = pd.read_csv('processed_patients.csv') # change to data to infer
print(f'check CUDA is avaiable; CUDA version: {torch.version.cuda}')


def cleanup_text(texts):
    text = texts
    if (len(texts) < 20):
        return np.nan
    # Lower case
    text = text.lower()
    # remove newline
    text = re.sub(r'\n', ' ', text)    # remove multiple spaces
    text = re.sub(r' +', ' ', text)
    return text
eval = eval.dropna() # Drop na first
eval = eval.rename(columns={"TEXT": "text"}) # Whatever the text column is convert name to text
eval['text'] = eval['text'].apply(cleanup_text)
eval = eval.dropna() # Drop na which means drop sentence less than 20 char

eval_dataset = Dataset.from_pandas(eval['text'].to_frame().sample(100)) # No need to sample in reality
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

outputs = trainer.predict(tokenized_eval,)

probabilities = softmax(outputs.predictions, axis=1)[:,1]
eval['subjective_prob'] = probabilities
print(eval)
eval.to_csv('result.csv',index = False)