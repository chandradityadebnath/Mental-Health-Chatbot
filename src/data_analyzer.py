import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MentalHealthDataAnalyzer:
    def __init__(self):
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    def analyze_conversation_patterns(self, conversation_history):
        """Analyze patterns in conversation data"""
        user_messages = [entry['user'] for entry in conversation_history if 'user' in entry]
        
        analysis = {
            'message_count': len(user_messages),
            'average_length': np.mean([len(msg) for msg in user_messages]),
            'word_frequency': self.analyze_word_frequency(user_messages),
            'sentiment_trend': self.analyze_sentiment_trend(conversation_history),
            'risk_progression': self.analyze_risk_progression(conversation_history)
        }
        
        return analysis
    
    def analyze_word_frequency(self, messages):
        """Analyze most frequent words"""
        all_text = ' '.join(messages).lower()
        words = re.findall(r'\b[a-z]{3,15}\b', all_text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        
        return Counter(filtered_words).most_common(20)
    
    def create_visualization_dashboard(self, conversation_history, agent):
        """Create comprehensive visualization dashboard"""
        analysis = self.analyze_conversation_patterns(conversation_history)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Word Frequency Cloud',
                'Risk Level Progression',
                'Emotional Pattern Analysis',
                'Conversation Metrics'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "domain"}, {"type": "xy"}]
            ]
        )
        
        # Word frequency bar chart
        words, counts = zip(*analysis['word_frequency'][:10])
        fig.add_trace(
            go.Bar(x=words, y=counts, name="Word Frequency", marker_color='#4ECDC4'),
            row=1, col=1
        )
        
        # Risk progression
        if analysis['risk_progression']:
            risks = analysis['risk_progression']
            fig.add_trace(
                go.Scatter(x=list(range(len(risks))), y=risks, 
                          mode='lines+markers', name="Risk Trend", line=dict(color='#FF6B6B')),
                row=1, col=2
            )
        
        # Emotional analysis pie chart
        emotion_data = self.extract_emotion_data(conversation_history)
        if emotion_data:
            emotions, scores = zip(*emotion_data.items())
            fig.add_trace(
                go.Pie(labels=emotions, values=scores, name="Emotional Distribution"),
                row=2, col=1
            )
        
        # Conversation metrics
        metrics = ['Messages', 'Avg Length', 'Unique Words']
        values = [analysis['message_count'], 
                 analysis['average_length'], 
                 len(analysis['word_frequency'])]
        
        fig.add_trace(
            go.Bar(x=metrics, y=values, name="Metrics", marker_color='#45B7D1'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Mental Health Conversation Analysis Dashboard")
        return fig
    
    def extract_emotion_data(self, conversation_history):
        """Extract emotion data from analysis"""
        emotion_scores = {}
        
        for entry in conversation_history:
            if 'analysis' in entry and 'emotion_breakdown' in entry['analysis']:
                for emotion in entry['analysis']['emotion_breakdown']:
                    label = emotion['label']
                    score = emotion['score']
                    if label in emotion_scores:
                        emotion_scores[label] += score
                    else:
                        emotion_scores[label] = score
        
        return emotion_scores
    
    def analyze_sentiment_trend(self, conversation_history):
        """Analyze sentiment trend over time"""
        sentiments = []
        
        for entry in conversation_history:
            if 'analysis' in entry and 'basic_sentiment' in entry['analysis']:
                sentiment = entry['analysis']['basic_sentiment']
                score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
                sentiments.append(score)
        
        return sentiments
    
    def analyze_risk_progression(self, conversation_history):
        """Analyze risk level progression"""
        risk_mapping = {'MONITOR': 1, 'HIGH': 2, 'CRISIS': 3}
        risks = []
        
        for entry in conversation_history:
            if 'analysis' in entry and 'risk_assessment' in entry['analysis']:
                risk_level = entry['analysis']['risk_assessment']['level']
                risks.append(risk_mapping.get(risk_level, 0))
        
        return risks

    def generate_insight_report(self, conversation_history, agent):
        """Generate comprehensive insight report"""
        analysis = self.analyze_conversation_patterns(conversation_history)
        summary = agent.get_conversation_summary()
        
        report = f"""
        MENTAL HEALTH CONVERSATION INSIGHT REPORT
        {'='*50}
        
        Conversation Overview:
        • Total exchanges: {analysis['message_count']}
        • Average message length: {analysis['average_length']:.1f} characters
        • Primary themes: {', '.join(summary['common_themes'])}
        
        Risk Assessment:
        • Overall trend: {summary['risk_trend']}
        • Highest risk level detected: {max(analysis['risk_progression']) if analysis['risk_progression'] else 'N/A'}
        
        Key Insights:
        • Most frequent concerns: {', '.join([word[0] for word in analysis['word_frequency'][:5]])}
        • Emotional patterns: {self.describe_emotional_patterns(analysis)}
        
        Recommendations:
        {self.format_recommendations(summary['recommended_follow_up'])}
        
        Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report
    
    def describe_emotional_patterns(self, analysis):
        """Describe emotional patterns from analysis"""
        if not analysis['sentiment_trend']:
            return "Insufficient data for pattern analysis"
        
        avg_sentiment = np.mean(analysis['sentiment_trend'])
        if avg_sentiment > 0.3:
            return "Generally positive emotional tone"
        elif avg_sentiment < -0.3:
            return "Predominantly negative emotional tone"
        else:
            return "Mixed or neutral emotional patterns"
    
    def format_recommendations(self, follow_up_plan):
        """Format recommendations for report"""
        formatted = []
        for timeframe, recommendations in follow_up_plan.items():
            formatted.append(f"{timeframe.replace('_', ' ').title()}:")
            for rec in recommendations:
                formatted.append(f"  • {rec}")
            formatted.append("")
        
        return "\n".join(formatted)
