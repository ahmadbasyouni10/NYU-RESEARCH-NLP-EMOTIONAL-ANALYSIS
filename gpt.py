import openai
import csv

# Set up your OpenAI API key
openai.api_key = ""

def chat_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Load the dataset
input_file = "data/emotion_dataset.csv"  # Adjust the file path if necessary
dataset = []
with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row if there is one
    for row in reader:
        if len(row) >= 2:
            dataset.append((row[0], row[1]))

# Function to call the OpenAI API for emojis
def get_emojis(text):
    prompt = f"Return an emoji that describes the emotion of the person saying: '{text}'"
    emojis = [chat_gpt(prompt) for _ in range(5)]  # Get 5 responses
    return emojis

# Prepare output CSV file 
output_file = "gpt_emoji_predictions.csv"
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    header = ["Text", "True Label", "Top1 Prediction", "Top1 Correct", "Top2 Prediction", "Top2 Correct", "Top3 Prediction", "Top3 Correct", "Top4 Prediction", "Top4 Correct", "Top5 Prediction", "Top5 Correct"]
    writer.writerow(header)
    
    # Process each sentence in the dataset
    for text, true_label in dataset:
        emojis = get_emojis(text)
        row = [text, true_label]
        
        for i, emoji in enumerate(emojis):
            correct = true_label == emoji
            row.append(emoji)
            row.append("True" if correct else "False")
        
        writer.writerow(row)

print(f"Results saved to {output_file}")