import pandas as pd
import numpy as np
from transformers import pipeline

class MentalHealthAgent:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.responses = {
            'depression': "It's important to talk to a professional. Consider reaching out to a therapist or counselor.",
            'anxiety': "Deep breathing exercises can help. Try the 4-7-8 technique: breathe in 4s, hold 7s, exhale 8s.",
            'stress': "Regular exercise and mindfulness meditation can reduce stress levels significantly."
        }
    
    def analyze_text(self, text):
        sentiment = self.sentiment_analyzer(text)[0]
        return {
            'sentiment': sentiment,
            'response': self.generate_response(text)
        }
    
    def generate_response(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ['sad', 'depressed', 'hopeless']):
            return self.responses['depression']
        elif any(word in text_lower for word in ['anxious', 'worried', 'nervous']):
            return self.responses['anxiety']
        elif any(word in text_lower for word in ['stress', 'overwhelmed', 'pressure']):
            return self.responses['stress']
        else:
            return "Thank you for sharing. Remember, it's okay to seek professional help when needed."
