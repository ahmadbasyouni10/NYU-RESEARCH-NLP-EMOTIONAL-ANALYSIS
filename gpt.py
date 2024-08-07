from openai import OpenAI
import csv
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def chat_gpt(prompt, top_k):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo",
        logprobs=True,
        top_logprobs=top_k
    )
    return [(c.token, c.logprob) for c in chat_completion.choices[0].logprobs.content[0].top_logprobs]

def get_emojis(text):
    prompt = f"Return an emoji that describes the emotion of the person saying: '{text}'"
    emojis_with_logprobs = chat_gpt(prompt, top_k=5)
    sorted_emojis = sorted(emojis_with_logprobs, key=lambda x: x[1], reverse=True)
    return [emoji for emoji, _ in sorted_emojis[:5]]