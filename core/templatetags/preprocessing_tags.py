from django import template
from django.utils.html import format_html, escape
from django.utils.safestring import mark_safe

register = template.Library()


@register.filter(name='highlight_diff')
def highlight_diff(after_text, before_text):
    """
    Highlight changed words (for punctuation, case, clean, stem steps)
    """
    if not before_text or not after_text:
        return after_text or ''
    
    before_words = str(before_text).split()
    after_words = str(after_text).split()
    
    result = []
    for i, after_word in enumerate(after_words):
        if i < len(before_words) and before_words[i] != after_word:
            # Use format_html to properly escape individual parts
            result.append(format_html('<span class="highlight-change">{}</span>', after_word))
        else:
            result.append(escape(after_word))
    
    return mark_safe(' '.join(result))


@register.filter(name='highlight_stopword_removal')
def highlight_stopword_removal(before_text, after_text):
    """
    Show removed stopwords with strikethrough
    """
    if not before_text or not after_text:
        return before_text or ''
    
    before_words = str(before_text).split()
    after_words = str(after_text).split()
    
    # Create set of after words (lowercase for comparison)
    after_set = set(word.lower() for word in after_words)
    
    result = []
    for word in before_words:
        word_lower = word.lower()
        
        # Check if word was removed
        if word_lower not in after_set:
            # Word removed - show with strikethrough
            # Use format_html to properly escape
            result.append(format_html('<span class="stopword-removed" title="Stopword removed">{}</span>', word))
        else:
            # Word kept - show normally (escaped)
            result.append(escape(word))
    
    return mark_safe(' '.join(result))