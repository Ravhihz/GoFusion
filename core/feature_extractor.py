from collections import Counter
from gensim.models import FastText
import numpy as np
from .models import PreprocessedTweet, FeatureVector, Label, SentimentDictionary, ExtendedDictionary

def extract_word_embeddings(dataset_id=None, split='train', use_extended=False):
    """
    Extract WEIGHTED word embeddings (Formula 2 & 3 dari jurnal)
    
    Args:
        dataset_id: ID dataset
        split: 'train', 'test', or 'all'
        use_extended: True = use extended dict, False = base only
    
    Returns:
        tuple: (success_count, error_count, message)
    """
    try:
        if not dataset_id:
            return (0, 0, "Dataset ID required")
        
        # ===== STEP 1: GET LABELED TWEETS =====
        if split == 'all':
            labeled_tweet_ids = Label.objects.filter(
                tweet__dataset_id=dataset_id
            ).values_list('tweet_id', flat=True)
        else:
            labeled_tweet_ids = Label.objects.filter(
                tweet__dataset_id=dataset_id,
                dataset_split=split
            ).values_list('tweet_id', flat=True)
        
        if not labeled_tweet_ids:
            return (0, 0, f"No {split} tweets found")
        
        preprocessed_tweets = PreprocessedTweet.objects.filter(
            tweet_id__in=labeled_tweet_ids
        )
        
        if not preprocessed_tweets.exists():
            return (0, 0, f"No preprocessed {split} tweets")
        
        # ===== STEP 2: COLLECT SENTENCES FOR TRAINING =====
        all_sentences = []
        tweet_objects = []
        
        for tweet in preprocessed_tweets:
            if tweet.tokens and len(tweet.tokens) > 0:
                all_sentences.append(tweet.tokens)
                tweet_objects.append(tweet)
        
        if len(all_sentences) < 50:
            return (0, 0, f"Not enough data: {len(all_sentences)} tweets")
        
        print(f"[STEP 2] Training FastText on {len(all_sentences)} sentences...")
        
        # ===== STEP 3: TRAIN FASTTEXT MODEL =====
        model = FastText(
            sentences=all_sentences,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            sg=1,
            epochs=20,
            min_n=3,
            max_n=6,
            word_ngrams=1,
            bucket=2000000
        )
        
        print(f"[STEP 3] FastText trained! Vocab: {len(model.wv)} words")
        
        # ===== STEP 4: LOAD SENTIMENT DICTIONARY =====
        sentiment_dict = {}
        
        # Load BASE dictionary
        for word_entry in SentimentDictionary.objects.filter(source='base'):
            # Polarity value: positive = +weight, negative = -weight
            polarity_value = word_entry.weight if word_entry.polarity == 'positive' else -word_entry.weight
            sentiment_dict[word_entry.word] = polarity_value
        
        # Load EXTENDED dictionary (if requested)
        if use_extended:
            for word_entry in ExtendedDictionary.objects.all():
                # Use similarity_score as weight
                polarity_value = word_entry.sentiment_value if word_entry.polarity == 'positive' else -word_entry.sentiment_value
                sentiment_dict[word_entry.word] = polarity_value
        
        # ✅ PERBAIKAN 1: Find s_max (maximum absolute sentiment value)
        if sentiment_dict:
            s_max = max(abs(val) for val in sentiment_dict.values())
            print(f"[STEP 4] Sentiment dict loaded: {len(sentiment_dict)} words, s_max={s_max:.3f}")
        else:
            s_max = 1.0
            print("[WARNING] No sentiment dictionary found, using s_max=1.0")
        
        # ===== STEP 5: EXTRACT WEIGHTED EMBEDDINGS =====
        success_count = 0
        error_count = 0
        
        for tweet, tokens in zip(tweet_objects, all_sentences):
            try:
                # FORMULA 2: v̄ = Σ(wi × v(di)) / m
                weighted_sum = np.zeros(100)
                m = len(tokens)
                
                for token in tokens:
                    # Skip if token not in FastText vocabulary
                    if token not in model.wv:
                        continue
                    
                    # Get word vector v(di)
                    v_di = model.wv[token]
                    
                    # ✅ PERBAIKAN 2: Calculate weight wi using FORMULA 3
                    if token in sentiment_dict:
                        s_i = sentiment_dict[token]  # Can be positive or negative
                        # wi = 2 / (1 + e^(-5|s(i)/s_max|)) - 1
                        wi = (2.0 / (1.0 + np.exp(-5.0 * abs(s_i / s_max)))) - 1.0
                    else:
                        # ✅ PERBAIKAN 3: Kata tidak ada di dict → weight kecil tapi bukan 0
                        wi = 0.01  # Small weight instead of 0
                    
                    # Accumulate: wi × v(di)
                    weighted_sum += wi * v_di
                
                # Calculate weighted average: Σ(wi × v(di)) / m
                if m > 0:
                    weighted_avg = weighted_sum / m
                else:
                    weighted_avg = np.zeros(100)
                
                # Convert to list for JSON storage
                embedding_list = weighted_avg.tolist()
                
                # ===== STEP 6: SAVE TO DATABASE =====
                feature_vector = FeatureVector.objects.filter(tweet=tweet.tweet).first()
                if feature_vector:
                    feature_vector.word_embedding = {
                        'vector': embedding_list,
                        'dimension': 100,
                        'method': 'weighted_fasttext',
                        'formula': 'wi = 2/(1+e^(-5|s/smax|)) - 1',
                        'use_extended_dict': use_extended,
                        's_max': float(s_max)
                    }
                    feature_vector.save(update_fields=['word_embedding'])
                else:
                    FeatureVector.objects.create(
                        tweet=tweet.tweet,
                        word_embedding={
                            'vector': embedding_list,
                            'dimension': 100,
                            'method': 'weighted_fasttext',
                            'use_extended_dict': use_extended
                        },
                        sentiment_score=0.0,
                        positive_score=0.0,
                        negative_score=0.0,
                        polarity='neutral'
                    )
                
                success_count += 1
                
                if success_count % 50 == 0:
                    print(f"[PROGRESS] {success_count}/{len(all_sentences)} processed...")
                
            except Exception as e:
                print(f"[ERROR] Tweet {tweet.tweet.id}: {str(e)}")
                error_count += 1
                continue
        
        dict_type = "base + extended" if use_extended else "base only"
        print(f"[STEP 5] COMPLETE! {success_count} tweets, {error_count} errors")
        
        return (
            success_count, 
            error_count, 
            f"Successfully extracted weighted embeddings for {success_count} {split} tweets using {dict_type} dictionary"
        )
        
    except Exception as e:
        print(f"[ERROR] extract_word_embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return (0, 0, f"Error: {str(e)}")


def prepare_feature_vectors(dataset_id=None, split='train'):
    """
    Prepare final feature vectors for SVM training
    Combines: Sentiment Scores (3 features) + Word Embeddings (100 features) = 103 features
    
    Args:
        dataset_id: ID dataset
        split: 'train', 'test', or 'all'
    
    Returns:
        tuple: (X, y, feature_names)
    """
    try:
        if not dataset_id:
            return None, None, None
        
        # Get labeled tweets based on split
        if split == 'all':
            labeled_tweet_ids = Label.objects.filter(
                tweet__dataset_id=dataset_id
            ).values_list('tweet_id', flat=True)
        else:
            labeled_tweet_ids = Label.objects.filter(
                tweet__dataset_id=dataset_id,
                dataset_split=split
            ).values_list('tweet_id', flat=True)
        
        feature_vectors = FeatureVector.objects.filter(
            tweet_id__in=labeled_tweet_ids
        ).select_related('tweet')
        
        if not feature_vectors.exists():
            return None, None, None
        
        X = []
        y = []
        
        for fv in feature_vectors:
            try:
                # Get label
                label = Label.objects.filter(tweet=fv.tweet).first()
                if not label:
                    continue
                
                # Skip placeholder labels
                if label.labeled_by == 'system' and split == 'train':
                    continue
                
                # Check completeness
                if fv.sentiment_score is None:
                    print(f"[WARNING] Tweet {fv.tweet.id} missing sentiment scores") # type: ignore
                    continue
                
                if not fv.word_embedding or 'vector' not in fv.word_embedding:
                    print(f"[WARNING] Tweet {fv.tweet.id} missing word embeddings") # type: ignore
                    continue
                
                # ✅ Build feature vector: [sentiment_score, pos_score, neg_score, embedding_1...100]
                features = [
                    float(fv.sentiment_score),
                    float(fv.positive_score),
                    float(fv.negative_score),
                ]
                
                # Add word embedding (100 dimensions)
                embedding_vector = fv.word_embedding['vector']
                features.extend(embedding_vector)
                
                # Add to dataset
                X.append(features)
                
                # Convert label to numeric (multiclass: -1, 0, 1)
                if label.sentiment == 'positive':
                    y.append(1)
                elif label.sentiment == 'negative':
                    y.append(-1)
                else:
                    y.append(0)
                
            except Exception as e:
                print(f"[ERROR] Processing feature vector {fv.id}: {str(e)}") # type: ignore
                continue
        
        if not X:
            return None, None, None
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Feature names
        feature_names = ['sentiment_score', 'positive_score', 'negative_score']
        feature_names.extend([f'embedding_{i}' for i in range(100)])
        
        print(f"[INFO] Prepared {len(X)} {split} samples with {X.shape[1]} features")
        print(f"[INFO] Label distribution: Pos={np.sum(y==1)}, Neg={np.sum(y==-1)}, Neu={np.sum(y==0)}")
        
        return X, y, feature_names
        
    except Exception as e:
        print(f"[ERROR] prepare_feature_vectors: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def get_feature_statistics(dataset_id=None, split=None):
    """Get feature extraction statistics"""
    try:
        if dataset_id:
            if split:
                tweet_ids = Label.objects.filter(
                    tweet__dataset_id=dataset_id,
                    dataset_split=split
                ).values_list('tweet_id', flat=True)
                features = FeatureVector.objects.filter(tweet_id__in=tweet_ids)
            else:
                features = FeatureVector.objects.filter(tweet__dataset_id=dataset_id)
        else:
            features = FeatureVector.objects.all()
        
        total = features.count()
        
        if total == 0:
            return None
        
        # Count complete feature vectors
        sentiment_calculated = features.exclude(sentiment_score=None).count()
        embeddings_calculated = 0
        complete_features = 0
        
        for fv in features:
            if fv.word_embedding and 'vector' in fv.word_embedding:
                embeddings_calculated += 1
                
            if (fv.sentiment_score is not None and 
                fv.word_embedding and 'vector' in fv.word_embedding):
                complete_features += 1
        
        return {
            'total': total,
            'sentiment_calculated': sentiment_calculated,
            'embeddings_calculated': embeddings_calculated,
            'complete_features': complete_features,
            'split': split if split else 'all'
        }
        
    except Exception as e:
        print(f"[ERROR] get_feature_statistics: {str(e)}")
        return None