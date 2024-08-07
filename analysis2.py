import csv
from openai import OpenAI
import os

client = OpenAI(api_key="")  # Replace with your actual API key

def chat_gpt(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo",
        max_tokens=5,
        temperature=0.7,
        n=5
    )
    return [choice.message.content.strip() for choice in chat_completion.choices]

def get_emojis(text):
    prompt = f"Return a single emoji that best describes the emotion in this text: '{text}'. Only return the emoji, nothing else."
    return chat_gpt(prompt)

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
            for i in range(1, 6):
                predictions = " ".join(emojis[:i])
                correct_flag = true_label.strip() in [e.strip() for e in emojis[:i]]
                row.extend([predictions, "True" if correct_flag else "False"])
            writer.writerow(row)

    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    analyze_emotions_gpt('data/emotion_dataset.csv')
    print("GPT Analysis completed!")