from django.db import models
from django.utils import timezone


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file_path = models.CharField(max_length=500)
    total_tweets = models.IntegerField(default=0)
    relevant_tweets = models.IntegerField(default=0)
    not_relevant_tweets = models.IntegerField(default=0)
    percentage_relevant = models.FloatField(default=0.0)
    sampled_tweets = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'datasets'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.name
    
    def update_statistics(self):
        """Update statistics after preliminary check or sampling"""
        self.total_tweets = self.tweets.count() # type: ignore
        self.relevant_tweets = self.tweets.filter(is_relevant=True, relevance_checked=True).count() # type: ignore
        self.not_relevant_tweets = self.tweets.filter(is_relevant=False, relevance_checked=True).count() # type: ignore
        self.percentage_relevant = (self.relevant_tweets / self.total_tweets * 100) if self.total_tweets > 0 else 0
        self.sampled_tweets = self.tweets.filter(is_sampled=True).count() # type: ignore
        self.save()


class Tweet(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='tweets')
    tweet_id = models.CharField(max_length=100)
    text = models.TextField()
    created_at = models.DateTimeField()
    username = models.CharField(max_length=255, blank=True)
    
    # Preliminary analysis fields
    is_relevant = models.BooleanField(default=True)
    relevance_checked = models.BooleanField(default=False)
    relevance_note = models.TextField(blank=True)
    checked_by = models.CharField(max_length=100, blank=True)
    checked_at = models.DateTimeField(null=True, blank=True)
    
    # Sampling fields
    is_sampled = models.BooleanField(default=False)
    sampled_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'tweets'
        ordering = ['-created_at']
        unique_together = ['dataset', 'tweet_id']
        indexes = [
            models.Index(fields=['dataset', 'tweet_id']),
        ]
    
    def __str__(self):
        return f"{self.username}: {self.text[:50]}..."


class PreprocessedTweet(models.Model):
    tweet = models.OneToOneField(Tweet, on_delete=models.CASCADE, related_name='preprocessed')
    original_text = models.TextField()
    after_remove_punctuation = models.TextField(blank=True)
    after_case_folding = models.TextField(blank=True)
    after_cleaning = models.TextField(blank=True)
    after_normalization = models.TextField(blank=True)  # ← TAMBAHKAN INI
    after_stopword = models.TextField(blank=True)
    after_stemming = models.TextField(blank=True)
    tokens = models.JSONField(default=list)
    word_vectors = models.JSONField(default=dict)
    processed_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'preprocessed_tweets'
    
    def __str__(self):
        return f"Preprocessed: {self.tweet.tweet_id}"


class SentimentDictionary(models.Model):
    POLARITY_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral'),
    ]
    
    word = models.CharField(max_length=255, unique=True)
    weight = models.FloatField()
    polarity = models.CharField(max_length=10, choices=POLARITY_CHOICES)
    source = models.CharField(max_length=50, default='base')
    added_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'sentiment_dictionary'
        ordering = ['-weight']
    
    def __str__(self):
        return f"{self.word} ({self.weight})"


class ExtendedDictionary(models.Model):
    word = models.CharField(max_length=255, unique=True)
    similarity_score = models.FloatField()
    reference_word = models.CharField(max_length=255)
    sentiment_value = models.FloatField()
    polarity = models.CharField(max_length=10)
    added_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'extended_dictionary'
        ordering = ['-similarity_score']
    
    def __str__(self):
        return f"{self.word} -> {self.reference_word} ({self.similarity_score:.2f})"


class Label(models.Model):

    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral'),
    ]

    SPLIT_CHOICES = [
        ('train', 'Training'),
        ('test', 'Testing'),
    ]

    # =========================
    # RELATION
    # =========================
    tweet = models.OneToOneField(
        'Tweet',
        on_delete=models.CASCADE,
        related_name='label'
    )

    # =========================
    # GROUND TRUTH (MANUAL)
    # =========================
    sentiment = models.CharField(
        max_length=10,
        choices=SENTIMENT_CHOICES,
        help_text="Manual ground truth label (used for training)"
    )

    # =========================
    # MODEL PREDICTION (TEST)
    # =========================
    predicted_sentiment = models.CharField(
        max_length=10,
        choices=SENTIMENT_CHOICES,
        null=True,
        blank=True,
        help_text="Predicted label by trained model (used for testing)"
    )

    confidence = models.FloatField(
        default=1.0,
        help_text="Prediction confidence (0–1)"
    )

    # =========================
    # METADATA
    # =========================
    labeled_by = models.CharField(
        max_length=50,
        default='manual',
        help_text="manual / svm / model"
    )

    labeled_at = models.DateTimeField(
        default=timezone.now
    )

    dataset_split = models.CharField(
        max_length=10,
        choices=SPLIT_CHOICES,
        default='train'
    )

    class Meta:
        db_table = 'labels'
        verbose_name = 'Label'
        verbose_name_plural = 'Labels'

    def __str__(self):
        if self.dataset_split == 'test' and self.predicted_sentiment:
            return f"{self.tweet.tweet_id}: {self.predicted_sentiment} (predicted)"
        return f"{self.tweet.tweet_id}: {self.sentiment} ({self.dataset_split})"



class FeatureVector(models.Model):
    tweet = models.OneToOneField(Tweet, on_delete=models.CASCADE, related_name='feature_vector')
    
    # Sentiment scores
    sentiment_score = models.FloatField(default=0.0)
    positive_score = models.FloatField(default=0.0)
    negative_score = models.FloatField(default=0.0)
    polarity = models.CharField(max_length=10, default='neutral')
    
    # Feature vectors (stored as JSON)
    tf_idf_vector = models.JSONField(default=dict, blank=True)
    word_embedding = models.JSONField(default=dict, blank=True)
    additional_features = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'feature_vectors'
    
    def __str__(self):
        return f"Features for Tweet {self.tweet.id} - {self.polarity} ({self.sentiment_score:.2f})" # type: ignore


class SVMModel(models.Model):
    name = models.CharField(max_length=255)
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="svm_models",
        null=True,
        blank=True
    )
    description = models.TextField(blank=True)
    hyperparameters = models.JSONField(default=dict)
    support_vectors = models.JSONField(default=list)
    coefficients = models.JSONField(default=list)
    intercept = models.FloatField(default=0.0)
    train_accuracy = models.FloatField(default=0.0)
    test_accuracy = models.FloatField(default=0.0)
    trained_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'svm_models'
        ordering = ['-trained_at']
    
    def __str__(self):
        return f"{self.name} (Acc: {self.test_accuracy:.2f}%)"


class Prediction(models.Model):
    tweet = models.OneToOneField(Tweet, on_delete=models.CASCADE, related_name='prediction')
    model = models.ForeignKey(SVMModel, on_delete=models.CASCADE, related_name='predictions')
    predicted_sentiment = models.CharField(max_length=10)
    confidence_score = models.FloatField()
    decision_value = models.FloatField()
    predicted_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'predictions'
    
    def __str__(self):
        return f"{self.tweet.tweet_id}: {self.predicted_sentiment}"


class Topic(models.Model):
    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('all', 'All'),
    ]
    
    name = models.CharField(max_length=255)
    sentiment_type = models.CharField(max_length=10, choices=SENTIMENT_CHOICES)
    top_words = models.JSONField(default=list)
    word_weights = models.JSONField(default=dict)
    document_distribution = models.JSONField(default=list)
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'topics'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.sentiment_type})"

class TrainingStepLog(models.Model):
    dataset = models.ForeignKey(
        'Dataset',                # pakai string biar aman dari circular import
        on_delete=models.CASCADE,
        related_name='training_step_logs'
    )

    step = models.PositiveSmallIntegerField(
        help_text='Nomor step training (3, 4, 4.5, 5, 6, 7)'
    )

    executed_at = models.DateTimeField(
        auto_now_add=True,
        help_text='Waktu eksekusi step'
    )

    class Meta:
        verbose_name = 'Training Step Log'
        verbose_name_plural = 'Training Step Logs'
        unique_together = ('dataset', 'step')   # 1 dataset = 1 record per step
        ordering = ['step']

    def __str__(self):
        return f"{self.dataset.name} - Step {self.step}"


class TrainingSession(models.Model):
    name = models.CharField(max_length=255)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    train_size = models.IntegerField()
    test_size = models.IntegerField()
    train_ratio = models.FloatField()
    test_ratio = models.FloatField()
    started_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, default='pending')
    
    class Meta:
        db_table = 'training_sessions'
        ordering = ['-started_at']
    
    def __str__(self):
        return f"{self.name} - {self.status}"


class EvaluationMetrics(models.Model):
    model = models.OneToOneField(SVMModel, on_delete=models.CASCADE, related_name='metrics')
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    confusion_matrix = models.JSONField(default=dict)
    classification_report = models.JSONField(default=dict)
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'evaluation_metrics'
    
    def __str__(self):
        return f"Metrics for {self.model.name}"