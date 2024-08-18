import pandas as pd
import re
import string
import requests

# Your Hugging Face API key
api_key = "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"

# The model you want to use
model_name = "cardiffnlp/twitter-roberta-base-sentiment"

# The API endpoint for the model
api_url = f"https://api-inference.huggingface.co/models/{model_name}"

# The headers for the request, including the API key
headers = {
    "Authorization": f"Bearer {api_key}"
}

# The input text you want to analyze

class Sentiment:
    def __init__(self):
        pass
    def clean_and_truncate_text(self,text, max_length=500):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        return text[:512]
    def analyze_sentiment_berttweet(self,txt):
        try:
            text = txt
            data = {
                "inputs": text
            }
            response = requests.post(api_url, headers=headers, json=data)
            label_mapping = {
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive"
            }

            # Get the response data
            result = response.json()[0]

            # Map the labels to their corresponding sentiment types
            maxx = 0
            cd = "neutral"
            for item in result:
                sentiment = label_mapping[item['label']]
                score = item['score']
                if(score>maxx):
                    maxx = score  
                    cd = sentiment
            return cd 
        except:
            return None

