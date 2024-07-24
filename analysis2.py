from openai import OpenAI
import pandas as pd
import csv

# Initialize OpenAI client
client = OpenAI(api_key="YOUR_API_KEY_HERE")

def chat_gpt(prompt, top_k):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
        logprobs=True,
        top_logprobs=top_k
    )
    return chat_completion.choices[0].logprobs.top_logprobs

def get_emojis(text):
    prompt = f"Return an emoji that describes the emotion of the person saying: '{text}'"
    logprobs = chat_gpt(prompt, top_k=5)
    
    emojis = []
    for logprob in logprobs:
        emoji = logprob.get("emoji", "")
        if emoji:
            emojis.append(emoji)
    
    return emojis[:5]  # Ensure only top 5 are returned

def analyze_emotions_gpt(dataset_path):
    dataset = pd.read_csv(dataset_path)

    # Prepare output CSV file
    output_file = "gpt_emoji_predictions.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ["Text", "True Label", "Top1 Prediction", "Top1 Correct", "Top2 Prediction", "Top2 Correct", "Top3 Prediction", "Top3 Correct", "Top4 Prediction", "Top4 Correct", "Top5 Prediction", "Top5 Correct"]
        writer.writerow(header)
        
        # Process each sentence in the dataset
        for index, row in dataset.iterrows():
            text = row['text']
            true_label = row['label']
            predictions = get_emojis(text)
            
            # Check if the true label is in the predictions
            correct_flags = ["True" if true_label.strip() in p.strip() else "False" for p in predictions]
            
            # Prepare row for CSV
            row_data = [text, true_label]
            for i, (pred, correct) in enumerate(zip(predictions, correct_flags), start=1):
                row_data.extend([pred, correct])
            
            writer.writerow(row_data)

    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    analyze_emotions_gpt('data/emotion_dataset.csv')
    print("GPT Analysis completed!")


from openai import OpenAI
import csv

client = OpenAI(api_key="sk-proj-Q1IBHrPvY1tGCRjFHQ67T3BlbkFJxvfpPQhMVyI3Qx3jGxxZ")

def chat_gpt(prompt, top_k):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo",
        logprobs=True,
        top_logprobs=top_k
    )
    return [c.token for c in chat_completion.choices[0].logprobs.content[0].top_logprobs]

def get_emojis(text):
    prompt = f"Return an emoji that describes the emotion of the person saying: '{text}'"
    emojis = chat_gpt(prompt, top_k=5)
    return emojis

def analyze_emotions_gpt(dataset_path):
    input_file = dataset_path
    output_file = "gpt_predictions.csv"

    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        dataset = [(row[0], row[1]) for row in reader if len(row) >= 2]

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ["Text", "True Label", "Top1 Prediction", "Top1 Correct", "Top2 Prediction", "Top2 Correct", "Top3 Prediction", "Top3 Correct", "Top4 Prediction", "Top4 Correct", "Top5 Prediction", "Top5 Correct"]
        writer.writerow(header)

        for text, true_label in dataset:
            emojis = get_emojis(text)
            row = [text, true_label]
            for i, emoji in enumerate(emojis, start=1):
                correct_flag = true_label.strip() in emoji
                row.extend([emoji, "True" if correct_flag else "False"])
            writer.writerow(row)

    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    analyze_emotions_gpt('data/emotion_dataset.csv')
    print("GPT Analysis completed!")
