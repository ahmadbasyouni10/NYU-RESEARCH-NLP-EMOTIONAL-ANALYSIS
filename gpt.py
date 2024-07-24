from openai import OpenAI
import csv

client = OpenAI(
    api_key="sk-proj-Q1IBHrPvY1tGCRjFHQ67T3BlbkFJxvfpPQhMVyI3Qx3jGxxZ"
)

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