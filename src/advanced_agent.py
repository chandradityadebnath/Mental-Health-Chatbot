import pandas as pd
import numpy as np
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import torch
import torch.nn as nn
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class AdvancedMentalHealthAgent:
    def __init__(self):
        self.setup_models()
        self.setup_knowledge_base()
        self.conversation_history = []
        self.user_profile = {}
        
    def setup_models(self):
        """Initialize all ML models and pipelines"""
        print("Loading AI models...")
        
        # Sentiment analysis models
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # Emergency detection model
        self.toxicity_analyzer = pipeline(
            "text-classification",
            model="unitary/unbiased-toxic-robust-english"
        )
        
        self.sia = SentimentIntensityAnalyzer()
        
        # TF-IDF for similarity matching
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        print("All models loaded successfully!")
    
    def setup_knowledge_base(self):
        """Comprehensive mental health knowledge base"""
        self.crisis_keywords = {
            'suicide': ['kill myself', 'end it all', 'suicide', 'want to die', 'no reason to live'],
            'self_harm': ['cut myself', 'self harm', 'hurt myself', 'bleeding'],
            'emergency': ['emergency', 'urgent', 'help now', 'immediate help']
        }
        
        self.therapeutic_techniques = {
            'cbt': {
                'name': 'Cognitive Behavioral Therapy',
                'techniques': [
                    "Let's challenge that thought. What evidence supports this belief?",
                    "Consider alternative perspectives to this situation.",
                    "Practice thought recording to identify cognitive distortions."
                ]
            },
            'mindfulness': {
                'name': 'Mindfulness Practice',
                'techniques': [
                    "Try the 5-4-3-2-1 grounding technique: notice 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, 1 thing you taste.",
                    "Practice deep breathing: inhale for 4 seconds, hold for 7, exhale for 8.",
                    "Try a body scan meditation, noticing sensations from head to toe."
                ]
            },
            'dbt': {
                'name': 'Dialectical Behavior Therapy',
                'techniques': [
                    "Use distress tolerance skills like ACCEPTS (Activities, Contributing, Comparisons, Emotions, Pushing away, Thoughts, Sensations).",
                    "Practice radical acceptance of the current moment.",
                    "Try opposite action to change emotional responses."
                ]
            }
        }
        
        self.resource_database = {
            'depression': {
                'symptoms': ['sadness', 'hopelessness', 'loss of interest', 'fatigue', 'appetite changes'],
                'resources': [
                    "National Depression Helpline: 1-800-273-8255",
                    "Try behavioral activation: schedule pleasant activities",
                    "Consider consulting a psychiatrist about treatment options"
                ],
                'coping_strategies': [
                    "Establish a daily routine",
                    "Practice gratitude journaling",
                    "Engage in physical activity",
                    "Connect with supportive people"
                ]
            },
            'anxiety': {
                'symptoms': ['worry', 'nervousness', 'panic', 'restlessness', 'sleep issues'],
                'resources': [
                    "Anxiety and Depression Association of America: 1-240-485-1001",
                    "Practice progressive muscle relaxation",
                    "Download anxiety management apps like Calm or Headspace"
                ],
                'coping_strategies': [
                    "Practice diaphragmatic breathing",
                    "Use worry time technique",
                    "Challenge catastrophic thinking",
                    "Gradual exposure to feared situations"
                ]
            },
            'stress': {
                'symptoms': ['overwhelmed', 'irritable', 'headaches', 'muscle tension'],
                'resources': [
                    "American Psychological Association Stress Resources",
                    "Try time management techniques",
                    "Practice mindfulness-based stress reduction"
                ],
                'coping_strategies': [
                    "Prioritize tasks using Eisenhower Matrix",
                    "Take regular breaks using Pomodoro technique",
                    "Practice assertive communication",
                    "Ensure adequate sleep and nutrition"
                ]
            }
        }
        
        # Pre-fit TF-IDF on knowledge base texts
        all_texts = []
        for category, data in self.resource_database.items():
            all_texts.extend(data['symptoms'])
            all_texts.extend([tech for tech in data['coping_strategies']])
        self.vectorizer.fit(all_texts)
    
    def analyze_mental_state(self, text):
        """Comprehensive mental state analysis"""
        # Multi-level sentiment analysis
        basic_sentiment = self.sentiment_analyzer(text)[0]
        emotion_scores = self.emotion_analyzer(text)[0]
        vader_scores = self.sia.polarity_scores(text)
        
        # Risk assessment
        risk_level = self.assess_risk_level(text)
        
        # Symptom detection
        detected_symptoms = self.detect_symptoms(text)
        
        # Therapeutic approach recommendation
        recommended_approach = self.recommend_therapeutic_approach(text, emotion_scores)
        
        return {
            'basic_sentiment': basic_sentiment,
            'emotion_breakdown': emotion_scores,
            'vader_scores': vader_scores,
            'risk_assessment': risk_level,
            'detected_symptoms': detected_symptoms,
            'recommended_approach': recommended_approach,
            'timestamp': datetime.now().isoformat()
        }
    
    def assess_risk_level(self, text):
        """Assess potential mental health risk level"""
        text_lower = text.lower()
        risk_score = 0
        
        # Crisis detection
        for crisis_type, keywords in self.crisis_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                risk_score += 3
        
        # Toxicity detection
        toxicity_result = self.toxicity_analyzer(text)[0]
        if toxicity_result['label'] in ['toxicity', 'severe_toxicity'] and toxicity_result['score'] > 0.7:
            risk_score += 2
        
        # Emotional intensity detection
        emotion_scores = self.emotion_analyzer(text)[0]
        high_intensity_emotions = ['anger', 'fear', 'sadness']
        for emotion in emotion_scores:
            if emotion['label'] in high_intensity_emotions and emotion['score'] > 0.8:
                risk_score += 1
        
        if risk_score >= 3:
            return {
                'level': 'CRISIS',
                'action': 'IMMEDIATE_PROFESSIONAL_HELP',
                'message': 'Please contact emergency services or crisis hotline immediately',
                'hotlines': [
                    'National Suicide Prevention Lifeline: 1-800-273-8255',
                    'Crisis Text Line: Text HOME to 741741',
                    'Emergency: 911'
                ]
            }
        elif risk_score >= 2:
            return {
                'level': 'HIGH',
                'action': 'URGENT_PROFESSIONAL_CONSULTATION',
                'message': 'Please consult with a mental health professional soon',
                'resources': [
                    'Therapy: Psychology Today therapist directory',
                    'Crisis support available 24/7'
                ]
            }
        else:
            return {
                'level': 'MONITOR',
                'action': 'SELF_CARE_AND_MONITORING',
                'message': 'Continue with self-care and monitor your symptoms'
            }
    
    def detect_symptoms(self, text):
        """Detect specific mental health symptoms"""
        text_lower = text.lower()
        detected = []
        
        for condition, data in self.resource_database.items():
            symptom_matches = [symptom for symptom in data['symptoms'] if symptom in text_lower]
            if symptom_matches:
                detected.append({
                    'condition': condition,
                    'matched_symptoms': symptom_matches,
                    'confidence': len(symptom_matches) / len(data['symptoms'])
                })
        
        return detected
    
    def recommend_therapeutic_approach(self, text, emotion_scores):
        """Recommend specific therapeutic approaches"""
        approaches = []
        
        # Analyze emotional patterns
        primary_emotion = max(emotion_scores, key=lambda x: x['score'])
        
        if primary_emotion['label'] in ['sadness', 'disgust']:
            approaches.append(self.therapeutic_techniques['cbt'])
        
        if primary_emotion['label'] in ['fear', 'surprise']:
            approaches.append(self.therapeutic_techniques['mindfulness'])
        
        if primary_emotion['label'] in ['anger', 'disgust']:
            approaches.append(self.therapeutic_techniques['dbt'])
        
        # Add general mindfulness for all cases
        if 'mindfulness' not in [app['name'] for app in approaches]:
            approaches.append(self.therapeutic_techniques['mindfulness'])
        
        return approaches
    
    def generate_personalized_response(self, text, analysis_results):
        """Generate personalized response based on comprehensive analysis"""
        risk_level = analysis_results['risk_assessment']['level']
        
        if risk_level in ['CRISIS', 'HIGH']:
            return self.generate_crisis_response(analysis_results)
        
        # Personalized coping strategies
        coping_strategies = []
        for symptom in analysis_results['detected_symptoms']:
            condition = symptom['condition']
            if condition in self.resource_database:
                coping_strategies.extend(
                    self.resource_database[condition]['coping_strategies'][:2]
                )
        
        # Therapeutic techniques
        therapeutic_techniques = []
        for approach in analysis_results['recommended_approach']:
            therapeutic_techniques.extend(approach['techniques'][:1])
        
        # Build comprehensive response
        response_parts = [
            "Thank you for sharing your feelings with me. I've analyzed what you're going through:",
            f"**Primary emotional state**: {analysis_results['emotion_breakdown'][0]['label']}",
            f"**Risk level**: {risk_level}",
            "",
            "**Recommended coping strategies:**",
            *[f"• {strategy}" for strategy in coping_strategies[:3]],
            "",
            "**Therapeutic techniques to try:**",
            *[f"• {technique}" for technique in therapeutic_techniques[:2]],
            "",
            "**Remember**: I'm an AI assistant. For professional help, consider:",
            "• Therapy or counseling services",
            "• Psychiatry consultation if needed",
            "• Support groups in your area"
        ]
        
        return "\n".join(response_parts)
    
    def generate_crisis_response(self, analysis_results):
        """Generate response for crisis situations"""
        risk_info = analysis_results['risk_assessment']
        
        crisis_response = [
            "🚨 **URGENT SUPPORT NEEDED** 🚨",
            "",
            "Based on what you've shared, I'm concerned about your immediate safety.",
            f"Risk level: {risk_info['level']}",
            "",
            "**IMMEDIATE ACTION REQUIRED:**",
            *[f"• {hotline}" for hotline in risk_info.get('hotlines', [])],
            "",
            "**Please:**",
            "• Contact emergency services (911) if you're in immediate danger",
            "• Reach out to someone you trust immediately",
            "• Don't hesitate to use these crisis resources",
            "",
            "You are not alone, and professional help is available 24/7."
        ]
        
        return "\n".join(crisis_response)
    
    def chat(self, user_input):
        """Main chat interface"""
        # Store conversation
        self.conversation_history.append({
            'user': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Comprehensive analysis
        analysis = self.analyze_mental_state(user_input)
        
        # Generate response
        response = self.generate_personalized_response(user_input, analysis)
        
        # Store agent response
        self.conversation_history.append({
            'agent': response,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'response': response,
            'analysis': analysis,
            'conversation_id': len(self.conversation_history) // 2
        }
    
    def get_conversation_summary(self):
        """Generate conversation summary"""
        if not self.conversation_history:
            return "No conversation history"
        
        user_messages = [entry['user'] for entry in self.conversation_history if 'user' in entry]
        
        summary = {
            'total_exchanges': len(user_messages),
            'risk_trend': self.analyze_risk_trend(),
            'common_themes': self.extract_themes(user_messages),
            'recommended_follow_up': self.generate_follow_up_plan()
        }
        
        return summary
    
    def analyze_risk_trend(self):
        """Analyze risk level trend over conversation"""
        risk_levels = []
        for entry in self.conversation_history:
            if 'analysis' in entry:
                risk_levels.append(entry['analysis']['risk_assessment']['level'])
        
        if not risk_levels:
            return "Insufficient data"
        
        return f"Risk trend: {', '.join(risk_levels)}"
    
    def extract_themes(self, messages):
        """Extract common themes from conversation"""
        all_text = ' '.join(messages)
        themes = []
        
        for condition in self.resource_database.keys():
            if condition in all_text.lower():
                themes.append(condition)
        
        return themes if themes else ['general mental wellness']
    
    def generate_follow_up_plan(self):
        """Generate follow-up care plan"""
        return {
            'immediate': [
                "Practice recommended coping strategies daily",
                "Monitor mood changes",
                "Reach out to support system"
            ],
            'short_term': [
                "Consider professional consultation",
                "Establish self-care routine",
                "Track progress in mood journal"
            ],
            'long_term': [
                "Develop sustainable mental health practices",
                "Build resilience through continued learning",
                "Maintain support network"
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    agent = AdvancedMentalHealthAgent()
    
    # Test cases
    test_cases = [
        "I've been feeling really down and hopeless lately, nothing brings me joy",
        "I'm so anxious about my job interview tomorrow I can't sleep",
        "Everything is so overwhelming, I don't know how to cope with all this stress",
        "I just want this pain to end, I can't take it anymore"
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i+1}: {test_case}")
        print(f"{'='*50}")
        
        result = agent.chat(test_case)
        print("RESPONSE:")
        print(result['response'])
        print(f"\nRisk Level: {result['analysis']['risk_assessment']['level']}")
