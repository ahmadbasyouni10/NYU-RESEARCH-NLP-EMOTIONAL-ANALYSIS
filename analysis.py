from transformers import pipeline
import pandas as pd

def analyze_emotions(dataset_path):

    # Load the dataset
    dataset = pd.read_csv(dataset_path)
    

    # Initialize pipeline with the model
    BERT = pipeline("text-classification", model="Karim-Gamal/BERT-base-finetuned-emojis-IID-Fed")

    results = []
    
    # Iterate through dataset 40 examples
    for index, row in dataset.iterrows():
        text = row['text']
        label = row['label']

        # Get the prediction from model
        result = BERT(text)
        prediction = result[0]['label']

        # Store the results for analysis
        results.append({'text': text,
                        'predicted': prediction,
                        'true': label})
        # Print the result
        print(f"Text: {text} - Predicted: {prediction} - True: {label}")

if __name__ == '__main__':
    analyze_emotions('data/emotion_dataset.csv')
    print("Analysis completed!")