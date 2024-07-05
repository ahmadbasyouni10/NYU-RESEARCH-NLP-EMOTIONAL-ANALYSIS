from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import urllib.request
import csv

# Load the pretrained model and tokenizer
MODEL = "Karim-Gamal/BERT-base-finetuned-emojis-IID-Fed"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

text = "Hello world"
text = preprocess(text)

# Tokenize the text (returns pytorch tensors including attention mask, which tells model to ignore padded values)
encoded_input = tokenizer(text, return_tensors='pt')

# Get model predictions, unpacks the input dictionary {input_ids, attention_mask} into the model
output = model(**encoded_input)
scores = output[0][0].detach().numpy()

# Download label mapping
labels = []
mapping_link = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emoji/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

# Get top k predictions
k = 3  # number of top predictions to show
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(k):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")