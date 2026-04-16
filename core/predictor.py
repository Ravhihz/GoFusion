import numpy as np
import re
import pickle
import os
from django.conf import settings


def preprocess_text_simple(text):
    """
    Simple preprocessing for prediction
    Mirrors the preprocessing logic from preprocessing.py
    """
    try:
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove RT
        text = re.sub(r'\brt\b', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize (split by space, filter length >= 3)
        tokens = [word for word in text.split() if len(word) >= 3]
        
        return tokens
        
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {str(e)}")
        return []


def calculate_sentiment_scores_for_tokens(tokens, use_extended=False):
    """
    Calculate sentiment scores using dictionary (3 features)
    
    Args:
        tokens: List of tokens
        use_extended: Use extended dictionary
        
    Returns:
        dict: {sentiment_score, positive_score, negative_score}
    """
    try:
        from .models import SentimentDictionary, ExtendedDictionary
        from collections import Counter
        
        # Load dictionaries
        base_dict = {}
        for entry in SentimentDictionary.objects.filter(source='base'):
            polarity_value = entry.weight if entry.polarity == 'positive' else -entry.weight
            base_dict[entry.word] = polarity_value
        
        extended_dict = {}
        if use_extended:
            for entry in ExtendedDictionary.objects.all():
                polarity_value = entry.sentiment_value if entry.polarity == 'positive' else -entry.sentiment_value
                extended_dict[entry.word] = polarity_value
        
        # Count frequencies
        token_freq = Counter(tokens)
        
        # Calculate scores
        sentiment_sum = 0.0
        positive_sum = 0.0
        negative_sum = 0.0
        
        for token, freq in token_freq.items():
            # Priority: base -> extended
            if token in base_dict:
                value = base_dict[token]
            elif use_extended and token in extended_dict:
                value = extended_dict[token]
            else:
                continue
            
            sentiment_sum += value * freq
            
            if value > 0:
                positive_sum += value * freq
            else:
                negative_sum += abs(value) * freq
        
        # Normalize
        total = len(tokens)
        
        return {
            'sentiment_score': sentiment_sum / total if total > 0 else 0.0,
            'positive_score': positive_sum / total if total > 0 else 0.0,
            'negative_score': negative_sum / total if total > 0 else 0.0
        }
        
    except Exception as e:
        print(f"[ERROR] Sentiment calculation failed: {str(e)}")
        return {
            'sentiment_score': 0.0,
            'positive_score': 0.0,
            'negative_score': 0.0
        }


def calculate_weighted_embedding(tokens, use_extended=False):
    """
    Calculate weighted word embedding (100 dimensions)
    Formula 2 & 3 from paper
    
    Args:
        tokens: List of tokens
        use_extended: Use extended dictionary
        
    Returns:
        np.array: Weighted embedding vector (100 dims)
    """
    try:
        from .models import SentimentDictionary, ExtendedDictionary, PreprocessedTweet
        from gensim.models import FastText
        
        # Load sentiment dictionary
        sentiment_dict = {}
        
        for entry in SentimentDictionary.objects.filter(source='base'):
            polarity_value = entry.weight if entry.polarity == 'positive' else -entry.weight
            sentiment_dict[entry.word] = polarity_value
        
        if use_extended:
            for entry in ExtendedDictionary.objects.all():
                polarity_value = entry.sentiment_value if entry.polarity == 'positive' else -entry.sentiment_value
                sentiment_dict[entry.word] = polarity_value
        
        # Calculate s_max
        if sentiment_dict:
            s_max = max(abs(v) for v in sentiment_dict.values())
        else:
            s_max = 1.0
        
        # Train FastText on existing data (cached)
        if not hasattr(calculate_weighted_embedding, 'fasttext_model'):
            print("[INFO] Training FastText model for prediction...")
            
            all_sentences = []
            for tweet in PreprocessedTweet.objects.all()[:5000]:  # Limit for speed
                if tweet.tokens and len(tweet.tokens) > 0:
                    all_sentences.append(tweet.tokens)
            
            if len(all_sentences) < 50:
                print("[WARNING] Not enough training data for FastText")
                return np.zeros(100)
            
            model = FastText(
                sentences=all_sentences,
                vector_size=100,
                window=5,
                min_count=2,
                workers=4,
                sg=1,
                epochs=10,
                min_n=3,
                max_n=6,
                word_ngrams=1,
                bucket=2000000
            )
            
            calculate_weighted_embedding.fasttext_model = model
            print("[INFO] FastText model cached")
        
        model = calculate_weighted_embedding.fasttext_model
        
        # Calculate weighted average
        weighted_sum = np.zeros(100)
        m = len(tokens)
        
        for token in tokens:
            if token not in model.wv:
                continue
            
            v_di = model.wv[token]
            
            # Calculate weight using Formula 3
            if token in sentiment_dict:
                s_i = sentiment_dict[token]
                wi = (2.0 / (1.0 + np.exp(-5.0 * abs(s_i / s_max)))) - 1.0
            else:
                wi = 0.01
            
            weighted_sum += wi * v_di
        
        # Average
        if m > 0:
            weighted_avg = weighted_sum / m
        else:
            weighted_avg = np.zeros(100)
        
        return weighted_avg
        
    except Exception as e:
        print(f"[ERROR] Weighted embedding failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.zeros(100)


def predict_sentiment(text, model):
    """
    Predict sentiment for single text using trained SVM model
    
    Args:
        text (str): Raw tweet text
        model (SVMModel): Trained SVM model instance from database
        
    Returns:
        dict: Prediction results with scores and confidence
    """
    try:
        print(f"\n[PREDICT] Input: {text[:50]}...")
        
        # Step 1: Preprocess text
        tokens = preprocess_text_simple(text)
        
        if not tokens:
            return {
                'predicted_sentiment': 'neutral',
                'confidence': 0.0,
                'sentiment_score': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'preprocessed_tokens': [],
                'error': 'No valid tokens after preprocessing'
            }
        
        print(f"[INFO] Tokens: {tokens[:10]}...")
        
        # Step 2: Check if model uses extended dictionary
        use_extended = 'extended' in model.name.lower()
        print(f"[INFO] Using extended dictionary: {use_extended}")
        
        # Step 3: Calculate sentiment scores (3 features)
        sentiment_result = calculate_sentiment_scores_for_tokens(tokens, use_extended)
        sentiment_score = sentiment_result['sentiment_score']
        positive_score = sentiment_result['positive_score']
        negative_score = sentiment_result['negative_score']
        
        print(f"[INFO] Sentiment: {sentiment_score:.4f}, pos={positive_score:.4f}, neg={negative_score:.4f}")
        
        # Step 4: Calculate weighted embedding (100 features)
        embedding = calculate_weighted_embedding(tokens, use_extended)
        
        if len(embedding) != 100:
            embedding = np.zeros(100)
        
        print(f"[INFO] Embedding: {len(embedding)} dims")
        
        # Step 5: Combine features (3 + 100 = 103 dimensions)
        features = [sentiment_score, positive_score, negative_score]
        features.extend(embedding.tolist())
        
        X = np.array([features])  # Shape: (1, 103)
        print(f"[INFO] Feature vector: {X.shape}")
        
        # Step 6: Apply PCA if model was trained with PCA
        use_pca = model.hyperparameters.get('use_pca', False)
        print(f"[INFO] Model uses PCA: {use_pca}")
        
        if use_pca:
            # Load PCA transformation
            pca_file = os.path.join(settings.MODELS_DIR, f'pca_{model.dataset.name}.pkl')
            
            if os.path.exists(pca_file):
                with open(pca_file, 'rb') as f:
                    pca_data = pickle.load(f)
                
                X_mean = np.array(pca_data['mean'])
                U = np.array(pca_data['components'])
                
                # Apply transformation
                X_centered = X - X_mean
                X = np.dot(X_centered, U)
                
                print(f"[INFO] After PCA: {X.shape}")
            else:
                print(f"[WARNING] PCA file not found: {pca_file}")
        
        # Step 7: Load SVM model from pickle
        model_file = model.hyperparameters.get('model_file')
        if not model_file:
            return {
                'predicted_sentiment': 'neutral',
                'confidence': 0.0,
                'error': 'Model file not specified'
            }
        
        model_path = os.path.join(settings.MODELS_DIR, model_file)
        if not os.path.exists(model_path):
            return {
                'predicted_sentiment': 'neutral',
                'confidence': 0.0,
                'error': f'Model file not found: {model_file}'
            }
        
        with open(model_path, 'rb') as f:
            svm_model = pickle.load(f)
        
        print(f"[INFO] SVM model loaded from: {model_path}")
        
        # Step 8: Predict
        prediction = svm_model.predict(X)[0]
        
        # Get confidence from decision scores
        try:
            scores = svm_model.decision_function(X)
            if len(scores.shape) > 1:
                # Multiclass: softmax
                exp_scores = np.exp(scores[0] - np.max(scores[0]))
                probs = exp_scores / np.sum(exp_scores)
                confidence = float(np.max(probs))
            else:
                # Binary: sigmoid
                confidence = float(1 / (1 + np.exp(-np.abs(scores[0]))))
        except:
            confidence = 0.5
        
        print(f"[PREDICT] Result: {prediction}, confidence: {confidence:.4f}")
        
        return {
            'predicted_sentiment': prediction,
            'confidence': round(confidence * 100, 2),
            'sentiment_score': round(sentiment_score, 4),
            'positive_score': round(positive_score, 4),
            'negative_score': round(negative_score, 4),
            'preprocessed_tokens': tokens[:10],
            'polarity': 'positive' if sentiment_score > 0 else ('negative' if sentiment_score < 0 else 'neutral')
        }
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Prediction failed: {str(e)}")
        traceback.print_exc()
        
        return {
            'predicted_sentiment': 'neutral',
            'confidence': 0.0,
            'sentiment_score': 0.0,
            'positive_score': 0.0,
            'negative_score': 0.0,
            'preprocessed_tokens': [],
            'error': str(e)
        }


def predict_batch(texts, model):
    """
    Predict sentiment for multiple texts
    
    Args:
        texts (list): List of tweet texts
        model (SVMModel): Trained SVM model
        
    Returns:
        list: List of prediction results
    """
    results = []
    
    for text in texts:
        result = predict_sentiment(text, model)
        result['original_text'] = text
        results.append(result)
    
    return results


def predict_dataset_tweets(dataset_id, model):
    """
    Predict sentiment for all tweets in a dataset
    
    Args:
        dataset_id (int): Dataset ID
        model (SVMModel): Trained SVM model
        
    Returns:
        dict: Prediction statistics and results
    """
    from .models import Tweet, Prediction
    from django.utils import timezone
    
    try:
        # Get all tweets from dataset
        tweets = Tweet.objects.filter(dataset_id=dataset_id)
        
        total_tweets = tweets.count()
        predictions = []
        
        print(f"[INFO] Predicting {total_tweets} tweets...")
        
        # Predict each tweet
        for i, tweet in enumerate(tweets):
            if i % 100 == 0:
                print(f"[INFO] Progress: {i}/{total_tweets}")
            
            result = predict_sentiment(tweet.text, model)
            
            # Save prediction to database
            prediction, created = Prediction.objects.update_or_create(
                tweet=tweet,
                model=model,  # ← Add to lookup fields
                defaults={
                    'predicted_sentiment': result['predicted_sentiment'],
                    'confidence_score': result['confidence'],
                    'decision_value': 0.0,  # ← Add this field
                    'predicted_at': timezone.now()
                }
            )
            
            predictions.append({
                'tweet_id': tweet.id, # type: ignore
                'text': tweet.text,
                'prediction': result['predicted_sentiment'],
                'confidence': result['confidence']
            })
        
        # Calculate statistics
        positive_count = sum(1 for p in predictions if p['prediction'] == 'positive')
        neutral_count = sum(1 for p in predictions if p['prediction'] == 'neutral')
        negative_count = sum(1 for p in predictions if p['prediction'] == 'negative')
        
        print(f"[INFO] Prediction complete!")
        print(f"[INFO] Positive: {positive_count}, Neutral: {neutral_count}, Negative: {negative_count}")
        
        return {
            'success': True,
            'total_predicted': total_tweets,
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'predictions': predictions[:100]  # Return first 100 for display
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }