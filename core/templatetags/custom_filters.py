from django import template
import json

register = template.Library()

@register.filter(name='split')
def split(value, delimiter=' '):
    """
    Split a string by delimiter
    Usage: {{ text|split:" " }}
    """
    if value:
        return str(value).split(delimiter)
    return []

@register.filter(name='parse_json')
def parse_json(value):
    """Parse JSON string to Python list"""
    if value:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    return []

@register.filter(name='get_tokens')
def get_tokens(preprocessed_tweet):
    """
    Extract tokens from PreprocessedTweet object
    Works with both cleaned_text field and tokens JSON field
    """
    if not preprocessed_tweet:
        return []
    
    # Try cleaned_text field (space-separated tokens)
    if hasattr(preprocessed_tweet, 'cleaned_text') and preprocessed_tweet.cleaned_text:
        tokens = preprocessed_tweet.cleaned_text.split()
        return [t for t in tokens if t.strip()]  # Remove empty strings
    
    # Try tokens field (JSON array)
    if hasattr(preprocessed_tweet, 'tokens') and preprocessed_tweet.tokens:
        try:
            if isinstance(preprocessed_tweet.tokens, str):
                return json.loads(preprocessed_tweet.tokens)
            elif isinstance(preprocessed_tweet.tokens, list):
                return preprocessed_tweet.tokens
        except (json.JSONDecodeError, TypeError):
            pass
    
    return []