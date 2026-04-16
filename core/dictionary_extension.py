from gensim.models import FastText
import numpy as np
from collections import Counter
from .models import Dataset, Label, PreprocessedTweet, SentimentDictionary, ExtendedDictionary, Tweet
from core import models


def extend_dictionary_after_training(dataset_id, trained_model_id=None):
    """
    Extend dictionary AFTER SVM training (sesuai jurnal Figure 2)
    
    Workflow:
    1. Ambil base dictionary → sort by sentiment value
    2. Pilih Top-N positive + Bottom-N negative → Reference Words Set C
    3. Train FastText HANYA pada TRAINING SET tweets
    4. Untuk setiap kata baru (tidak ada di base dict):
       - Hitung cosine similarity dengan Reference Words
       - Jika similarity > threshold (0.7) → tambahkan ke extended dict
    
    Args:
        dataset_id: ID dataset yang sudah di-training
        trained_model_id: (Optional) ID model SVM yang sudah trained
    
    Returns:
        (success: bool, extended_count: int, skipped_count: int, message: str)
    """
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        print("[INFO] ========== DICTIONARY EXTENSION START ==========")
        print(f"[INFO] Dataset: {dataset.name}")
        
        # 1. Check if base dictionary exists
        base_dict_count = SentimentDictionary.objects.count()
        if base_dict_count == 0:
            return (False, 0, 0, 'Base dictionary not found. Please load dictionary first.')
        
        print(f"[INFO] Base dictionary loaded: {base_dict_count} words")
        
        # 2. Get ONLY TRAINING SET tweets (80%)
        train_labels = Label.objects.filter(
            tweet__dataset_id=dataset_id,
            dataset_split='train'
        )
        
        if not train_labels.exists():
            return (False, 0, 0, 'No training set found. Please split dataset first.')
        
        train_tweet_ids = train_labels.values_list('tweet_id', flat=True)
        print(f"[INFO] Training set: {len(train_tweet_ids)} tweets")
        
        # 3. Get preprocessed tweets ONLY for training set
        preprocessed_tweets = PreprocessedTweet.objects.filter(
            tweet_id__in=train_tweet_ids
        )
        
        if not preprocessed_tweets.exists():
            return (False, 0, 0, f'No preprocessed tweets found for training set')
        
        if preprocessed_tweets.count() < 50:
            return (False, 0, 0, f'Not enough training data. Found {preprocessed_tweets.count()} tweets, need at least 50.')
        
        print(f"[INFO] Preprocessed training tweets: {preprocessed_tweets.count()}")
        
        # 4. Collect tokens from training set
        all_sentences = []
        all_tokens = []
        
        for tweet in preprocessed_tweets:
            if tweet.tokens and len(tweet.tokens) > 0:
                all_sentences.append(tweet.tokens)
                all_tokens.extend(tweet.tokens)
        
        if not all_sentences:
            return (False, 0, 0, 'No tokens found in training set')
        
        token_freq = Counter(all_tokens)
        unique_tokens = set(all_tokens)
        
        print(f"[INFO] Collected {len(all_sentences)} sentences with {len(unique_tokens)} unique tokens")
        
        # 5. Bubble Sort Dictionary by Sentiment Value (sesuai jurnal)
        print("[INFO] Sorting base dictionary by sentiment value...")
        
        base_dict_entries = []
        for entry in SentimentDictionary.objects.all():
            # Polarity value: positive = +weight, negative = -weight
            polarity_value = entry.weight if entry.polarity == 'positive' else -entry.weight
            base_dict_entries.append({
                'word': entry.word,
                'polarity': entry.polarity,
                'value': polarity_value,
                'weight': entry.weight
            })
        
        # Bubble sort (descending by value)
        n = len(base_dict_entries)
        for i in range(n):
            for j in range(0, n - i - 1):
                if base_dict_entries[j]['value'] < base_dict_entries[j + 1]['value']:
                    base_dict_entries[j], base_dict_entries[j + 1] = base_dict_entries[j + 1], base_dict_entries[j]
        
        print(f"[INFO] Dictionary sorted. Range: {base_dict_entries[0]['value']:.3f} to {base_dict_entries[-1]['value']:.3f}")
        
        # 6. Select Top-N positive + Bottom-N negative (sesuai jurnal: N=50)
        N = 50
        top_N_positive = base_dict_entries[:N]  # Top 50 (most positive)
        bottom_N_negative = base_dict_entries[-N:]  # Bottom 50 (most negative)
        
        reference_set = top_N_positive + bottom_N_negative
        reference_words = [entry['word'] for entry in reference_set]
        
        print(f"[INFO] Reference Words Set C: Top {N} positive + Bottom {N} negative = {len(reference_words)} words")
        print(f"[INFO] Top positive examples: {[w['word'] for w in top_N_positive[:5]]}")
        print(f"[INFO] Top negative examples: {[w['word'] for w in bottom_N_negative[:5]]}")
        
        # 7. Get words NOT in base dictionary (new words)
        base_words = set([entry['word'] for entry in base_dict_entries])
        new_tokens = unique_tokens - base_words
        
        if not new_tokens:
            return (True, 0, 0, 'All tokens already in base dictionary')
        
        print(f"[INFO] Found {len(new_tokens)} new words not in base dictionary")
        
        # 8. Train FastText model on TRAINING SET
        print(f"[INFO] Training FastText on {len(all_sentences)} training sentences...")
        
        # model = FastText(
        #     sentences=all_sentences,
        #     vector_size=100,
        #     window=5,
        #     min_count=2,
        #     workers=4,
        #     sg=1,
        #     epochs=20,
        #     min_n=3,
        #     max_n=6,
        #     word_ngrams=1,
        #     bucket=2000000
        # )
        
        model = FastText(
            sentences=all_sentences,
            vector_size=100,
            window=5,
            min_count=3,      # ✅ Ubah dari 2 → 3
            workers=4,
            sg=1,
            epochs=10,        # ✅ Ubah dari 20 → 10
            min_n=3,
            max_n=6,
            word_ngrams=1,
            bucket=2000000
        )
        print("[INFO] FastText training completed")
        
        # 9. Calculate Similarity & Extend Dictionary
        # similarity_threshold = 0.7
        similarity_threshold = 0.95  # ✅ Ubah dari 0.7 → 0.95
        extended_count = 0
        skipped_count = 0

        deleted_count = ExtendedDictionary.objects.all().delete()[0]
        print(f"[INFO] Cleared {deleted_count} old extended dictionary entries")

        print(f"[INFO] Processing {len(new_tokens)} new tokens with threshold={similarity_threshold}...")

        # Create mapping reference_word -> (polarity, weight)
        reference_info_map = {}
        for entry in reference_set:
            # Polarity value: positive = +weight, negative = -weight
            polarity_value = entry['weight'] if entry['polarity'] == 'positive' else -entry['weight']
            
            reference_info_map[entry['word']] = {
                'polarity': entry['polarity'],
                'polarity_value': polarity_value  # ← Pakai ini (bisa negatif)
            }

        processed = 0
        for token in new_tokens:
            processed += 1
            
            if processed % 100 == 0:
                print(f"[INFO] Progress: {processed}/{len(new_tokens)} tokens processed, {extended_count} extended, {skipped_count} skipped")
            
            if token_freq[token] < 5: # ✅ Ubah dari 2 → 5
                skipped_count += 1
                continue
            
            if token not in model.wv:
                skipped_count += 1
                continue
            
            token_vector = model.wv[token]
            max_similarity = 0.0
            best_match = None
            best_info = None
            
            for ref_word in reference_words:
                if ref_word not in model.wv:
                    continue
                
                ref_vector = model.wv[ref_word]
                
                dot_product = np.dot(token_vector, ref_vector)
                norm_token = np.linalg.norm(token_vector)
                norm_ref = np.linalg.norm(ref_vector)
                
                if norm_token == 0 or norm_ref == 0:
                    continue
                
                similarity = dot_product / (norm_token * norm_ref)
                similarity = max(0.0, min(1.0, similarity))
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = ref_word
                    best_info = reference_info_map[ref_word]
            
            if max_similarity >= similarity_threshold and best_match and best_info:
                ExtendedDictionary.objects.create(
                    word=token,
                    reference_word=best_match,
                    similarity_score=round(max_similarity, 4),
                    sentiment_value=best_info['polarity_value'],
                    polarity=best_info['polarity']
                )
                extended_count += 1
            else:
                skipped_count += 1
        
        print("[INFO] ========== DICTIONARY EXTENSION COMPLETE ==========")
        print(f"[INFO] Extended: {extended_count} words")
        print(f"[INFO] Skipped: {skipped_count} words")
        print(f"[INFO] Threshold: {similarity_threshold}")
        
        # Show some examples
        examples = ExtendedDictionary.objects.all()[:10]
        if examples:
            print("[INFO] Extended dictionary examples:")
            for ex in examples:
                print(f"  - {ex.word} -> {ex.reference_word} ({ex.similarity_score:.3f}, {ex.polarity})")
        
        return (True, extended_count, skipped_count, f'Successfully extended dictionary with {extended_count} new words')
        
    except Exception as e:
        print(f"[ERROR] extend_dictionary_after_training: {str(e)}")
        import traceback
        traceback.print_exc()
        return (False, 0, 0, str(e))


def get_dictionary_statistics():
    """
    Get statistics about base and extended dictionaries
    """
    try:
        base_dict = SentimentDictionary.objects.all()
        extended_dict = ExtendedDictionary.objects.all()
        
        base_positive = base_dict.filter(polarity='positive').count()
        base_negative = base_dict.filter(polarity='negative').count()
        base_neutral = base_dict.filter(polarity='neutral').count()
        
        extended_positive = extended_dict.filter(polarity='positive').count()
        extended_negative = extended_dict.filter(polarity='negative').count()
        
        return {
            'base': {
                'total': base_dict.count(),
                'positive': base_positive,
                'negative': base_negative,
                'neutral': base_neutral
            },
            'extended': {
                'total': extended_dict.count(),
                'positive': extended_positive,
                'negative': extended_negative
            },
            'combined': {
                'total': base_dict.count() + extended_dict.count(),
                'positive': base_positive + extended_positive,
                'negative': base_negative + extended_negative
            }
        }
        
    except Exception as e:
        print(f"[ERROR] get_dictionary_statistics: {str(e)}")
        return None


def validate_extended_dictionary():
    """
    Validate extended dictionary quality
    Check for potential issues
    """
    try:
        extended = ExtendedDictionary.objects.all()
        
        if not extended.exists():
            return {
                'valid': False,
                'message': 'Extended dictionary is empty'
            }
        
        total = extended.count()
        high_similarity = extended.filter(similarity_score__gte=0.8).count()
        medium_similarity = extended.filter(similarity_score__gte=0.7, similarity_score__lt=0.8).count()
        
        avg_similarity = extended.aggregate(models.Avg('similarity_score'))['similarity_score__avg'] # type: ignore
        
        # Check for duplicates
        words = extended.values_list('word', flat=True)
        duplicates = len(words) - len(set(words))
        
        return {
            'valid': True,
            'total_words': total,
            'high_similarity': high_similarity,
            'medium_similarity': medium_similarity,
            'avg_similarity': avg_similarity,
            'duplicates': duplicates,
            'quality_score': (high_similarity / total * 100) if total > 0 else 0
        }
        
    except Exception as e:
        print(f"[ERROR] validate_extended_dictionary: {str(e)}")
        return {
            'valid': False,
            'message': str(e)
        }