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

# Download label mapping
labels = []
mapping_link = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emoji/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

# Load the dataset
input_file = "../../data/emotion_dataset.csv"  # Adjusted the file path
dataset = []
with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row if there is one
    for row in reader:
        if len(row) >= 2:
            dataset.append((row[0], row[1]))

# Prepare output CSV file
output_file = "emoji_predictions.csv"
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    header = ["Text", "True Label", "Top1 Prediction", "Top1 Correct", "Top2 Prediction", "Top2 Correct", "Top3 Prediction", "Top3 Correct", "Top4 Prediction", "Top4 Correct", "Top5 Prediction", "Top5 Correct"]
    writer.writerow(header)
    
    # Process each sentence in the dataset
    for text, true_label in dataset:
        preprocessed_text = preprocess(text)
        encoded_input = tokenizer(preprocessed_text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        
        # Get top k predictions
        k = 5
        ranking = np.argsort(scores)[::-1]
        
        # Initialize an empty list to hold the cumulative top N predictions
        cumulative_predictions = []
        
        # Loop through the top 5 predictions
        for i in range(1, k+1):
            # Get the top i predictions based on the ranking
            top_i_predictions = [labels[ranking[j]] for j in range(i)]
            # Add the top i predictions to the cumulative list
            cumulative_predictions.append(top_i_predictions)
        
        row = [text, true_label]
        for i, predictions in enumerate(cumulative_predictions, start=1):
            # Check if the true label is in the current set of predictions
            correct_flag = true_label.strip() in [p.strip() for p in predictions]
            # Append the predictions and the correctness flag to the row
            row.extend([", ".join(predictions), "True" if correct_flag else "False"])
        
        writer.writerow(row)

print(f"Results saved to {output_file}")
