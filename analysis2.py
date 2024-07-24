from openai import OpenAI
import csv
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
