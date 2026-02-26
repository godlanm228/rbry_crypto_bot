import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.ai_analytics import ai_analyzer

def test_ai_news_positive():
    text = "Bitcoin adoption by institutions is growing rapidly"
    result = ai_analyzer.analyze_news(text)
    assert isinstance(result, str)
    assert len(result) > 5