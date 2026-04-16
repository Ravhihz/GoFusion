from django.contrib import admin
from .models import (
    Dataset, Tweet, PreprocessedTweet, SentimentDictionary, 
    ExtendedDictionary, Label, FeatureVector, SVMModel, 
    Prediction, Topic, TrainingSession, EvaluationMetrics
)


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'total_tweets', 'relevant_tweets', 'not_relevant_tweets', 
                    'percentage_relevant', 'sampled_tweets', 'uploaded_at', 'is_active']
    list_filter = ['is_active', 'uploaded_at']
    search_fields = ['name', 'description']
    readonly_fields = ['uploaded_at']
    list_per_page = 50  # ✅ Added pagination
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'description', 'file_path', 'is_active')
        }),
        ('Statistics', {
            'fields': ('total_tweets', 'relevant_tweets', 'not_relevant_tweets', 
                      'percentage_relevant', 'sampled_tweets')
        }),
        ('Timestamps', {
            'fields': ('uploaded_at',)
        }),
    )


@admin.register(Tweet)
class TweetAdmin(admin.ModelAdmin):
    list_display = ['tweet_id', 'username', 'text_preview', 'is_relevant', 
                    'relevance_checked', 'is_sampled', 'created_at']
    list_filter = ['is_relevant', 'relevance_checked', 'is_sampled', 'dataset']
    search_fields = ['tweet_id', 'username', 'text']
    readonly_fields = ['created_at', 'checked_at', 'sampled_at']
    list_per_page = 100  # ✅ Added pagination
    
    fieldsets = (
        ('Tweet Info', {
            'fields': ('dataset', 'tweet_id', 'username', 'text', 'created_at')
        }),
        ('Relevance Check', {
            'fields': ('is_relevant', 'relevance_checked', 'relevance_note', 
                      'checked_by', 'checked_at')
        }),
        ('Sampling', {
            'fields': ('is_sampled', 'sampled_at')
        }),
    )
    
    def text_preview(self, obj):
        return obj.text[:50] + '...' if len(obj.text) > 50 else obj.text
    text_preview.short_description = 'Text'


@admin.register(PreprocessedTweet)
class PreprocessedTweetAdmin(admin.ModelAdmin):
    list_display = ['tweet', 'token_count', 'processed_at']
    list_filter = ['processed_at']
    search_fields = ['tweet__tweet_id', 'original_text']
    readonly_fields = ['processed_at']
    list_per_page = 100  # ✅ Added pagination
    
    def token_count(self, obj):
        return len(obj.tokens) if obj.tokens else 0
    token_count.short_description = 'Tokens'


@admin.register(SentimentDictionary)
class SentimentDictionaryAdmin(admin.ModelAdmin):
    list_display = ['word', 'weight', 'polarity', 'source', 'added_at']
    list_filter = ['polarity', 'source']
    search_fields = ['word']
    readonly_fields = ['added_at']
    list_per_page = 100  # ✅ Added pagination


@admin.register(ExtendedDictionary)
class ExtendedDictionaryAdmin(admin.ModelAdmin):
    list_display = ['word', 'reference_word', 'similarity_score', 'polarity', 'added_at']
    list_filter = ['polarity']
    search_fields = ['word', 'reference_word']
    readonly_fields = ['added_at']
    ordering = ['-similarity_score']
    list_per_page = 100  # ✅ Added pagination


@admin.register(Label)
class LabelAdmin(admin.ModelAdmin):
    list_display = ['id', 'tweet_preview', 'sentiment', 'dataset_split', 'confidence', 'labeled_by', 'labeled_at']
    list_filter = ['sentiment', 'dataset_split', 'labeled_by']
    search_fields = ['tweet__tweet_id', 'tweet__text']
    readonly_fields = ['labeled_at']
    list_per_page = 100  # ✅ Added pagination - IMPORTANT!
    
    actions = ['delete_selected', 'reset_sentiment', 'mark_as_positive', 'mark_as_negative', 'mark_as_neutral']  # ✅ Bulk actions
    
    def tweet_preview(self, obj):
        """Show tweet text preview"""
        return obj.tweet.text[:50] + '...' if len(obj.tweet.text) > 50 else obj.tweet.text
    tweet_preview.short_description = 'Tweet'
    
    # ✅ Custom bulk actions
    def reset_sentiment(self, request, queryset):
        """Reset sentiment to empty"""
        count = queryset.update(sentiment='', labeled_by='', labeled_at=None)
        self.message_user(request, f"Successfully reset {count} labels.")
    reset_sentiment.short_description = "Reset sentiment (empty)"
    
    def mark_as_positive(self, request, queryset):
        """Mark selected as positive"""
        from django.utils import timezone
        count = queryset.update(sentiment='positive', labeled_by='admin_bulk', labeled_at=timezone.now())
        self.message_user(request, f"Marked {count} labels as positive.")
    mark_as_positive.short_description = "Mark as Positive"
    
    def mark_as_negative(self, request, queryset):
        """Mark selected as negative"""
        from django.utils import timezone
        count = queryset.update(sentiment='negative', labeled_by='admin_bulk', labeled_at=timezone.now())
        self.message_user(request, f"Marked {count} labels as negative.")
    mark_as_negative.short_description = "Mark as Negative"
    
    def mark_as_neutral(self, request, queryset):
        """Mark selected as neutral"""
        from django.utils import timezone
        count = queryset.update(sentiment='neutral', labeled_by='admin_bulk', labeled_at=timezone.now())
        self.message_user(request, f"Marked {count} labels as neutral.")
    mark_as_neutral.short_description = "Mark as Neutral"


@admin.register(FeatureVector)
class FeatureVectorAdmin(admin.ModelAdmin):
    list_display = ['tweet', 'sentiment_score', 'positive_score', 'negative_score', 
                    'polarity', 'has_embedding', 'updated_at']
    list_filter = ['polarity']
    search_fields = ['tweet__tweet_id']
    readonly_fields = ['created_at', 'updated_at']
    list_per_page = 100  # ✅ Added pagination
    
    def has_embedding(self, obj):
        return bool(obj.word_embedding and 'vector' in obj.word_embedding)
    has_embedding.boolean = True
    has_embedding.short_description = 'Has Embedding'


@admin.register(SVMModel)
class SVMModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'train_accuracy', 'test_accuracy', 'trained_at', 'is_active']
    list_filter = ['is_active', 'trained_at']
    search_fields = ['name', 'description']
    readonly_fields = ['trained_at']
    list_per_page = 50  # ✅ Added pagination


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['tweet', 'predicted_sentiment', 'confidence_score', 'model', 'predicted_at']
    list_filter = ['predicted_sentiment', 'model']
    search_fields = ['tweet__tweet_id']
    readonly_fields = ['predicted_at']
    list_per_page = 100  # ✅ Added pagination


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ['name', 'sentiment_type', 'word_count', 'created_at']
    list_filter = ['sentiment_type']
    search_fields = ['name']
    readonly_fields = ['created_at']
    list_per_page = 50  # ✅ Added pagination
    
    def word_count(self, obj):
        return len(obj.top_words) if obj.top_words else 0
    word_count.short_description = 'Words'


@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset', 'train_size', 'test_size', 'status', 'started_at']
    list_filter = ['status', 'started_at']
    search_fields = ['name']
    readonly_fields = ['started_at', 'completed_at']
    list_per_page = 50  # ✅ Added pagination


@admin.register(EvaluationMetrics)
class EvaluationMetricsAdmin(admin.ModelAdmin):
    list_display = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'created_at']
    list_filter = ['created_at']
    search_fields = ['model__name']
    readonly_fields = ['created_at']
    list_per_page = 50  # ✅ Added pagination