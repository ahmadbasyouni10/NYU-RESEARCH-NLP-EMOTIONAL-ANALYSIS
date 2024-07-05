from transformers import pipeline

def load_model():
    BERT = pipeline("text-classification", model="Karim-Gamal/BERT-base-finetuned-emojis-IID-Fed")
    return BERT

if __name__ == '__main__':
    BERT = load_model()
    print("Model loaded successfully!")