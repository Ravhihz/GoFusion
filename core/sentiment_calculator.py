from collections import Counter
from .models import Label, PreprocessedTweet, SentimentDictionary, ExtendedDictionary, FeatureVector, Tweet


def calculate_sentiment_scores(dataset_id=None, use_extended=False):
    """
    Calculate sentiment scores for preprocessed tweets
    
    Args:
        dataset_id: ID dataset yang mau diproses
        use_extended: True = pakai base + extended dict, False = base only
    
    Returns:
        tuple: (success_count, error_count, message)
    """
    try:
        if not dataset_id:
            return (0, 0, "Dataset ID is required")
        
        # Get ONLY SAMPLED tweets (yang sudah di-sample untuk training/testing)
        sampled_tweet_ids = Tweet.objects.filter(
            dataset_id=dataset_id,
            is_sampled=True
        ).values_list('id', flat=True)
        
        if not sampled_tweet_ids:
            return (0, 0, "No sampled tweets found. Please sample tweets first.")
        
        # Get preprocessed tweets dari sampled tweets
        preprocessed_tweets = PreprocessedTweet.objects.filter(
            tweet_id__in=sampled_tweet_ids
        )
        
        if not preprocessed_tweets.exists():
            return (0, 0, "No preprocessed tweets found for sampled tweets")
        
        # Load base dictionary into memory for fast lookup
        base_dict = {}
        for word_entry in SentimentDictionary.objects.all():
            polarity_value = 1.0 if word_entry.polarity == 'positive' else -1.0
            base_dict[word_entry.word] = {
                'polarity': polarity_value,
                'weight': word_entry.weight
            }
        
        if not base_dict:
            return (0, 0, "Base dictionary is empty. Please upload dictionary files first.")
        
        # Load extended dictionary (jika diminta)
        extended_dict = {}
        if use_extended:
            for word_entry in ExtendedDictionary.objects.all():
                polarity_value = 1.0 if word_entry.polarity == 'positive' else -1.0
                extended_dict[word_entry.word] = {
                    'polarity': polarity_value,
                    'weight': word_entry.similarity_score
                }
        
        success_count = 0
        error_count = 0
        
        print(f"[DEBUG] Calculating sentiment scores for {preprocessed_tweets.count()} tweets")
        print(f"[DEBUG] Using extended dictionary: {use_extended}")
        print(f"[DEBUG] Base dict size: {len(base_dict)}, Extended dict size: {len(extended_dict)}")
        
        # Process each tweet
        for preprocessed in preprocessed_tweets:
            try:
                tokens = preprocessed.tokens
                
                if not tokens or len(tokens) == 0:
                    error_count += 1
                    continue
                
                # Count token frequencies
                token_freq = Counter(tokens)
                total_tokens = len(tokens)
                
                # Calculate scores
                positive_score = 0.0
                negative_score = 0.0
                matched_tokens = 0
                
                positive_words = []
                negative_words = []
                
                for token, freq in token_freq.items():
                    # Check in base dictionary first (higher priority)
                    if token in base_dict:
                        entry = base_dict[token]
                        weighted_score = entry['polarity'] * entry['weight'] * freq
                        
                        if entry['polarity'] > 0:
                            positive_score += weighted_score
                            positive_words.append(token)
                        else:
                            negative_score += abs(weighted_score)
                            negative_words.append(token)
                        
                        matched_tokens += freq
                    
                    # Check in extended dictionary (lower priority, only if use_extended=True)
                    elif use_extended and token in extended_dict:
                        entry = extended_dict[token]
                        weighted_score = entry['polarity'] * entry['weight'] * freq
                        
                        if entry['polarity'] > 0:
                            positive_score += weighted_score
                            positive_words.append(token)
                        else:
                            negative_score += abs(weighted_score)
                            negative_words.append(token)
                        
                        matched_tokens += freq
                
                # Calculate final sentiment score
                if matched_tokens > 0:
                    sentiment_score = (positive_score - negative_score) / matched_tokens
                else:
                    sentiment_score = 0.0
                
                # Normalize to [-1, 1]
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
                
                # Determine polarity
                if sentiment_score > 0.1:
                    polarity = 'positive'
                elif sentiment_score < -0.1:
                    polarity = 'negative'
                else:
                    polarity = 'neutral'
                
                # Save or update FeatureVector
                FeatureVector.objects.update_or_create(
                    tweet=preprocessed.tweet,
                    defaults={
                        'sentiment_score': sentiment_score,
                        'positive_score': positive_score,
                        'negative_score': negative_score,
                        'polarity': polarity,
                        'tf_idf_vector': {},
                        'word_embedding': {},
                        'additional_features': {
                            'positive_ratio': positive_score / matched_tokens if matched_tokens > 0 else 0.0,
                            'negative_ratio': negative_score / matched_tokens if matched_tokens > 0 else 0.0,
                            'coverage': matched_tokens / total_tokens if total_tokens > 0 else 0.0,
                            'total_tokens': total_tokens,
                            'matched_tokens': matched_tokens,
                            'positive_words': positive_words[:10],
                            'negative_words': negative_words[:10],
                            'used_extended_dict': use_extended
                        }
                    }
                )
                
                success_count += 1
                
            except Exception as e:
                print(f"[ERROR] Processing tweet {preprocessed.tweet.id}: {str(e)}") # type: ignore
                error_count += 1
                continue
        
        dict_type = "base + extended" if use_extended else "base only"
        return (success_count, error_count, f"Successfully calculated sentiment scores for {success_count} tweets using {dict_type} dictionary")
        
    except Exception as e:
        print(f"[ERROR] calculate_sentiment_scores: {str(e)}")
        import traceback
        traceback.print_exc()
        return (0, 0, f"Error: {str(e)}")


def get_sentiment_statistics(dataset_id=None, split=None):
    """
    Get sentiment statistics for dataset
    
    Args:
        dataset_id: ID dataset
        split: 'train', 'test', or None (all)
    """
    try:
        if dataset_id:
            labels = Label.objects.filter(tweet__dataset_id=dataset_id)
        else:
            labels = Label.objects.all()
        
        # Filter by split if provided
        if split:
            labels = labels.filter(dataset_split=split)
        
        total = labels.count()
        
        if total == 0:
            return None
        
        positive = labels.filter(sentiment='positive').count()
        negative = labels.filter(sentiment='negative').count()
        neutral = labels.filter(sentiment='neutral').count()
        
        positive_pct = (positive / total * 100) if total > 0 else 0
        negative_pct = (negative / total * 100) if total > 0 else 0
        neutral_pct = (neutral / total * 100) if total > 0 else 0
        
        # Get feature vectors
        feature_vectors = FeatureVector.objects.filter(
            tweet_id__in=labels.values_list('tweet_id', flat=True)
        )
        
        avg_sentiment = 0.0
        sentiment_scores = feature_vectors.values_list('sentiment_score', flat=True)
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_pct': positive_pct,
            'negative_pct': negative_pct,
            'neutral_pct': neutral_pct,
            'avg_sentiment': avg_sentiment,
            'split': split if split else 'all'
        }
        
    except Exception as e:
        print(f'[ERROR] get_sentiment_statistics: {str(e)}')
        return None


def recalculate_with_extended_dict(dataset_id):
    """
    Recalculate sentiment scores menggunakan extended dictionary
    Dipakai setelah dictionary extension selesai
    """
    try:
        print(f"[INFO] Recalculating sentiment scores with extended dictionary...")
        
        # Calculate ulang dengan extended dict
        success, error, message = calculate_sentiment_scores(
            dataset_id=dataset_id,
            use_extended=True
        )
        
        if success > 0:
            print(f"[INFO] Successfully recalculated {success} tweets with extended dictionary")
            return (True, f"Recalculated {success} tweets with extended dictionary")
        else:
            return (False, message)
            
    except Exception as e:
        print(f"[ERROR] recalculate_with_extended_dict: {str(e)}")
        return (False, str(e))