import csv
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Avg, Count

from core.utils_dictionary_loader import load_base_dictionary_from_tsv
from .models import (
    FeatureVector, SVMModel, Dataset, TrainingStepLog, Tweet, Label, Prediction,
    SentimentDictionary, ExtendedDictionary, EvaluationMetrics
)
import json
from .predictor import predict_sentiment
from django.views.decorators.http import require_POST, require_http_methods


# ==================== MODELS VIEWS ====================

def models_list(request):
    """Display all trained models with comparison"""
    try:
        models = SVMModel.objects.all().order_by('-trained_at')
        
        total_models = models.count()
        active_models = models.filter(is_active=True).count()
        
        if total_models > 0:
            best_accuracy = models.order_by('-test_accuracy').first().test_accuracy # type: ignore
            avg_accuracy = models.aggregate(Avg('test_accuracy'))['test_accuracy__avg'] or 0
        else:
            best_accuracy = 0
            avg_accuracy = 0
        
        context = {
            'models': models,
            'total_models': total_models,
            'active_models': active_models,
            'best_accuracy': best_accuracy,
            'avg_accuracy': avg_accuracy,
        }
        
        return render(request, 'core/models_list.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading models: {str(e)}")
        return redirect('core:dashboard')


def model_detail(request, model_id):
    """Display detailed model information and evaluation metrics"""
    try:
        model = get_object_or_404(SVMModel, id=model_id)
        
        # Get evaluation metrics
        metrics = EvaluationMetrics.objects.filter(model=model).first()
        
        context = {
            'model': model,
            'metrics': metrics,
        }
        
        return render(request, 'core/model_detail.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading model: {str(e)}")
        return redirect('core:models_list')


def set_active_model(request, model_id):
    """Set a model as active"""
    if request.method == 'POST':
        try:
            SVMModel.objects.all().update(is_active=False)
            
            model = get_object_or_404(SVMModel, id=model_id)
            model.is_active = True
            model.save()
            
            messages.success(request, f'Model {model.name} berhasil diaktifkan!')
            
        except Exception as e:
            messages.error(request, f'Gagal mengaktifkan model: {str(e)}')
        
        return redirect('core:models_list')
    
    messages.error(request, 'Invalid request method')
    return redirect('core:models_list')


def delete_model(request, model_id):
    """Delete a model"""
    if request.method == 'POST':
        try:
            model = get_object_or_404(SVMModel, id=model_id)
            model_name = model.name
            
            # Delete associated metrics (cascade akan otomatis delete)
            # EvaluationMetrics.objects.filter(model=model).delete()  # Tidak perlu, cascade sudah handle
            
            # Delete model
            model.delete()
            
            messages.success(request, f"Model '{model_name}' deleted successfully")
            return redirect('core:models_list')  # ← UBAH INI
            
        except Exception as e:
            messages.error(request, f"Failed to delete model: {str(e)}")
            return redirect('core:models_list')  # ← UBAH INI
    
    messages.error(request, 'Invalid request method')
    return redirect('core:models_list')  # ← UBAH INI


# ==================== PREDICTION VIEWS ====================

def prediction_page(request):
    """Prediction page for single tweet"""
    try:
        active_model = SVMModel.objects.filter(is_active=True).first()
        
        context = {
            'active_model': active_model,
        }
        
        return render(request, 'core/prediction.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading prediction page: {str(e)}")
        return redirect('core:dashboard')


def predict_single_tweet(request):
    """Predict sentiment for a single tweet"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '').strip()
            
            if not text:
                return JsonResponse({
                    'success': False,
                    'message': 'Text is required'
                }, status=400)
            
            # Get active model
            model = SVMModel.objects.filter(is_active=True).first()
            if not model:
                return JsonResponse({
                    'success': False,
                    'message': 'No active model found. Please train a model first.'
                }, status=400)
            
            # Make prediction
            result = predict_sentiment(text, model)
            
            if result:
                return JsonResponse({
                    'success': True,
                    'prediction': result
                })
            else:
                return JsonResponse({
                    'success': False,
                    'message': 'Prediction failed'
                }, status=500)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'message': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'message': 'Invalid request method'
    }, status=405)


# ==================== ANALYTICS VIEWS ====================

def analytics_dashboard(request):
    """Analytics dashboard with charts and statistics"""
    try:
        # Overall stats
        total_tweets = Tweet.objects.count()
        labeled_tweets = Label.objects.exclude(sentiment='').count()
        total_models = SVMModel.objects.count()
        total_predictions = Prediction.objects.count()
        
        # Sentiment distribution from labeled tweets
        positive_count = Label.objects.filter(sentiment='positive').count()
        neutral_count = Label.objects.filter(sentiment='neutral').count()
        negative_count = Label.objects.filter(sentiment='negative').count()
        
        total_labeled = labeled_tweets if labeled_tweets > 0 else 1
        positive_percent = (positive_count / total_labeled) * 100
        neutral_percent = (neutral_count / total_labeled) * 100
        negative_percent = (negative_count / total_labeled) * 100
        
        # Model statistics
        models = SVMModel.objects.all()
        recent_models = models.order_by('-trained_at')[:5]
        
        active_models = models.filter(is_active=True).count()
        
        if models.exists():
            avg_train_accuracy = models.aggregate(Avg('train_accuracy'))['train_accuracy__avg'] or 0
            avg_test_accuracy = models.aggregate(Avg('test_accuracy'))['test_accuracy__avg'] or 0
            best_model = models.order_by('-test_accuracy').first()
            best_model_name = best_model.name if best_model else 'N/A'
        else:
            avg_train_accuracy = 0
            avg_test_accuracy = 0
            best_model_name = 'N/A'
        
        # Dataset overview
        datasets = Dataset.objects.all()
        for dataset in datasets:
            dataset.labeled_count = Label.objects.filter( # type: ignore
                tweet__dataset=dataset,
                sentiment__in=['positive', 'neutral', 'negative']
            ).count()
        
        # Dictionary stats
        base_dict_count = SentimentDictionary.objects.count()
        extended_dict_count = ExtendedDictionary.objects.count()
        total_dict_words = base_dict_count + extended_dict_count
        
        # Prediction timeline (last 7 days)
        from datetime import datetime, timedelta
        today = datetime.now().date()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
        
        prediction_counts = []
        for date_str in dates:
            count = Prediction.objects.filter(
                predicted_at__date=date_str
            ).count()
            prediction_counts.append(count)
        
        context = {
            'total_tweets': total_tweets,
            'labeled_tweets': labeled_tweets,
            'total_models': total_models,
            'total_predictions': total_predictions,
            
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'positive_percent': positive_percent,
            'neutral_percent': neutral_percent,
            'negative_percent': negative_percent,
            
            'recent_models': recent_models,
            'active_models': active_models,
            'avg_train_accuracy': avg_train_accuracy,
            'avg_test_accuracy': avg_test_accuracy,
            'best_model_name': best_model_name,
            
            'datasets': datasets,
            
            'base_dict_count': base_dict_count,
            'extended_dict_count': extended_dict_count,
            'total_dict_words': total_dict_words,
            
            'prediction_dates': json.dumps(dates),
            'prediction_counts': json.dumps(prediction_counts),
        }
        
        return render(request, 'core/analytics.html', context)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        messages.error(request, f"Error loading analytics: {str(e)}")
        return redirect('core:dashboard')


# ==================== DICTIONARY VIEWS ====================

def dictionary_view(request):
    try:
        word_type = request.GET.get('type', '')
        polarity = request.GET.get('polarity', '')
        search = request.GET.get('search', '')

        # ================= BASE =================
        base_words = SentimentDictionary.objects.filter(source='base')

        # ================= MANUAL =================
        manual_words = SentimentDictionary.objects.filter(source='manual')

        # ================= EXTENDED =================
        extended_words = ExtendedDictionary.objects.all()

        # ====== FILTER POLARITY ======
        if polarity:
            base_words = base_words.filter(polarity=polarity)
            manual_words = manual_words.filter(polarity=polarity)
            extended_words = extended_words.filter(polarity=polarity)

        # ====== FILTER SEARCH ======
        if search:
            base_words = base_words.filter(word__icontains=search)
            manual_words = manual_words.filter(word__icontains=search)
            extended_words = extended_words.filter(word__icontains=search)

        # ====== FILTER TYPE ======
        if word_type == 'base':
            manual_words = manual_words.none()
            extended_words = extended_words.none()

        elif word_type == 'manual':
            base_words = base_words.none()
            extended_words = extended_words.none()

        elif word_type == 'extended':
            base_words = base_words.none()
            manual_words = manual_words.none()

        # ====== COUNTS ======
        base_count = SentimentDictionary.objects.filter(source='base').count()
        manual_count = SentimentDictionary.objects.filter(source='manual').count()
        extended_count = ExtendedDictionary.objects.count()

        total_count = base_count + manual_count + extended_count

        # ====== POLARITY COUNTS ======
        base_positive = SentimentDictionary.objects.filter(source='base', polarity='positive').count()
        base_negative = SentimentDictionary.objects.filter(source='base', polarity='negative').count()

        manual_positive = SentimentDictionary.objects.filter(source='manual', polarity='positive').count()
        manual_negative = SentimentDictionary.objects.filter(source='manual', polarity='negative').count()

        extended_positive = ExtendedDictionary.objects.filter(polarity='positive').count()
        extended_negative = ExtendedDictionary.objects.filter(polarity='negative').count()

        # ====== PAGINATION BASE ======
        paginator = Paginator(base_words.order_by('word'), 50)
        page_number = request.GET.get('page', 1)
        base_words_page = paginator.get_page(page_number)

        context = {
            'base_words': base_words_page,
            'manual_words': manual_words.order_by('word'),
            'extended_words': extended_words.order_by('-similarity_score')[:100],

            'base_count': base_count,
            'manual_count': manual_count,
            'extended_count': extended_count,
            'total_count': total_count,

            'base_positive': base_positive,
            'base_negative': base_negative,
            'manual_positive': manual_positive,
            'manual_negative': manual_negative,
            'extended_positive': extended_positive,
            'extended_negative': extended_negative,
        }

        return render(request, 'core/dictionary_view.html', context)

    except Exception as e:
        import traceback
        traceback.print_exc()
        messages.error(request, f"Error loading dictionary: {str(e)}")
        return redirect('core:dashboard')
    
# ==================== Feature Extraction ====================

def feature_viewer(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)

    # =========================
    # Ambil filter dari UI
    # =========================
    split = request.GET.get("split", "all")        # train / test / all
    sentiment = request.GET.get("sentiment", "all")  # positive / negative / neutral / all

    # =========================
    # Query dasar FeatureVector
    # =========================
    qs = FeatureVector.objects.filter(
        tweet__dataset=dataset
    ).select_related("tweet")

    # =========================
    # Filter SPLIT (train / test)
    # =========================
    if split != "all":
        qs = qs.filter(tweet__label__dataset_split=split)

    # =========================
    # Filter SENTIMENT (LEWAT LABEL)
    # =========================
    if sentiment != "all":
        qs = qs.filter(tweet__label__sentiment=sentiment)

    # =========================
    # Statistik (SELALU DARI LABEL)
    # =========================
    base_label_qs = Label.objects.filter(tweet__dataset=dataset)

    if split != "all":
        base_label_qs = base_label_qs.filter(dataset_split=split)

    total_features = qs.count()

    positive_count = base_label_qs.filter(sentiment="positive").count()
    negative_count = base_label_qs.filter(sentiment="negative").count()
    neutral_count  = base_label_qs.filter(sentiment="neutral").count()

    # =========================
    # Status training (untuk banner UI)
    # =========================
    step4_done = TrainingStepLog.objects.filter(dataset=dataset, step=4).exists()
    step45_done = FeatureVector.objects.filter(
        tweet__dataset=dataset,
        additional_features__has_key="pca_features"
    ).exists()
    step5_done = TrainingStepLog.objects.filter(dataset=dataset, step=5).exists()

    # =========================
    # Pagination
    # =========================
    paginator = Paginator(qs.order_by("id"), 10)
    page_number = request.GET.get("page")
    features = paginator.get_page(page_number)

    # =========================
    # Enrich object untuk template
    # =========================
    for f in features:
        # Word embedding
        emb_raw = f.word_embedding or []

        # Normalisasi embedding ke list
        if isinstance(emb_raw, dict):
            emb = list(emb_raw.values())
        elif isinstance(emb_raw, (list, tuple)):
            emb = list(emb_raw)
        else:
            emb = []

        f.embedding_dims = len(emb)
        f.embedding_preview = emb[:8]


        # PCA
        pca_raw = []
        if f.additional_features:
            pca_raw = f.additional_features.get("pca_features", [])

        if isinstance(pca_raw, dict):
            pca_vec = list(pca_raw.values())
        elif isinstance(pca_raw, (list, tuple)):
            pca_vec = list(pca_raw)
        else:
            pca_vec = []

        f.pca_dims = len(pca_vec)
        f.pca_preview = pca_vec[:8]


    # =========================
    # Context ke template
    # =========================
    context = {
        "dataset": dataset,
        "features": features,

        # statistik
        "total_features": total_features,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,

        # status training
        "step4_done": step4_done,
        "step45_done": step45_done,
        "step5_done": step5_done,

        # filter state
        "current_split": split,
        "current_sentiment": sentiment,
    }

    return render(request, "core/feature_viewer.html", context)



@require_http_methods(["GET"])
def feature_detail_api(request, feature_id):
    """
    API endpoint untuk mendapatkan detail feature vector
    Dipanggil oleh tombol 'Show Full Vector'
    URL: /api/feature-detail/<feature_id>/
    """
    try:
        f = FeatureVector.objects.select_related('tweet__label').get(id=feature_id)
        
        # Build response data
        response_data = {
            'success': True,
            'feature_id': f.id, # type: ignore
            'tweet': {
                'id': f.tweet.id, # type: ignore
                'text': f.tweet.text,
                'created_at': f.tweet.created_at.isoformat() if f.tweet.created_at else None
            },
            'label': {
                'sentiment': None,
                'dataset_split': None
            },
            'sentiment_features': {
                'sentiment_score': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'polarity': 'neutral'
            },
            'word_embedding': None,
            'pca_features': None,
            'dimensions': {}
        }
        
        # Get label info
        if hasattr(f.tweet, 'label'):
            response_data['label']['sentiment'] = f.tweet.label.sentiment # type: ignore
            response_data['label']['dataset_split'] = f.tweet.label.dataset_split # type: ignore
        
        # Sentiment features
        response_data['sentiment_features'] = {
            'sentiment_score': float(f.sentiment_score) if f.sentiment_score else 0.0,
            'positive_score': float(f.positive_score) if f.positive_score else 0.0,
            'negative_score': float(f.negative_score) if f.negative_score else 0.0,
            'polarity': f.polarity if f.polarity else 'neutral'
        }
        
        # Word embedding
        if f.word_embedding and 'vector' in f.word_embedding:
            embedding_vector = f.word_embedding['vector']
            response_data['word_embedding'] = {
                'dimension': len(embedding_vector),
                'method': f.word_embedding.get('method', 'weighted_fasttext'),
                'vector_preview': embedding_vector[:10],  # First 10 values
                'vector_full': embedding_vector  # Full vector
            }
            response_data['dimensions']['embedding'] = len(embedding_vector)
        
        # PCA features
        if f.additional_features and 'pca_features' in f.additional_features:
            pca_vector = f.additional_features['pca_features']
            response_data['pca_features'] = {
                'dimension': len(pca_vector),
                'variance_explained': f.additional_features.get('pca_variance', 0.0),
                'vector_preview': pca_vector[:10],  # First 10 values
                'vector_full': pca_vector  # Full vector
            }
            response_data['dimensions']['pca'] = len(pca_vector)
        
        return JsonResponse(response_data)
        
    except FeatureVector.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': f'Feature vector with ID {feature_id} not found'
        }, status=404)
    
    except Exception as e:
        import traceback
        print(f"[ERROR] feature_detail_api: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }, status=500)



def export_features_csv(request, dataset_id):
    """
    Export features to CSV for analysis
    Includes: tweet_id, text, sentiment_scores, word_embeddings (first 10), pca_features (first 10)
    """
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Get all features
        features = FeatureVector.objects.filter(
            tweet__dataset_id=dataset_id
        ).select_related('tweet', 'tweet__label').order_by('tweet_id')
        
        # Create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="features_{dataset.name}.csv"'
        
        writer = csv.writer(response)
        
        # Header
        header = [
            'Tweet_ID', 'Text', 'Cleaned_Text', 'Sentiment', 'Split',
            'Positive_Score', 'Negative_Score', 'Neutral_Score',
            'Embedding_Dim', 'Embedding_1', 'Embedding_2', 'Embedding_3', 'Embedding_4', 'Embedding_5',
            'PCA_Dim', 'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5'
        ]
        writer.writerow(header)
        
        # Data rows
        for feature in features:
            # Get label
            try:
                label = feature.tweet.label # type: ignore
                sentiment = label.sentiment
                split = label.dataset_split
            except:
                sentiment = 'unknown'
                split = 'unknown'
            
            # Get cleaned text
            try:
                cleaned = feature.tweet.preprocessed.after_stemming # type: ignore
            except:
                cleaned = ''
            
            # Word embeddings (first 5)
            emb_dims = 0
            emb_values = [0.0] * 5
            if feature.word_embedding and 'vector' in feature.word_embedding:
                vector = feature.word_embedding['vector']
                emb_dims = len(vector)
                for i in range(min(5, len(vector))):
                    emb_values[i] = round(vector[i], 4)
            
            # PCA features (first 5)
            pca_dims = 0
            pca_values = [0.0] * 5
            if feature.additional_features and 'pca_features' in feature.additional_features:
                pca_vector = feature.additional_features['pca_features']
                pca_dims = len(pca_vector)
                for i in range(min(5, len(pca_vector))):
                    pca_values[i] = round(pca_vector[i], 4)
            
            row = [
                feature.tweet.tweet_id or feature.tweet.id, # type: ignore
                feature.tweet.text[:100],  # Limit text length
                cleaned[:100],
                sentiment,
                split,
                round(feature.positive_score or 0.0, 4),
                round(feature.negative_score or 0.0, 4),
                round(feature.neutral_score or 0.0, 4), # type: ignore
                emb_dims,
                *emb_values,
                pca_dims,
                *pca_values
            ]
            
            writer.writerow(row)
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        from django.contrib import messages
        messages.error(request, f"Export failed: {str(e)}")
        return redirect('core:feature_viewer', dataset_id=dataset_id)

def dictionary_evaluation_view(request):
    evaluations = (
        EvaluationMetrics.objects
        .select_related('model')
        .order_by('-created_at')
    )

    latest_results = {}

    for ev in evaluations:
        model_name = ev.model.name.lower()

        if 'extended' in model_name:
            key = 'Extended'
        elif 'manual' in model_name:
            key = 'Manual'
        else:
            key = 'Base'

        if key not in latest_results:
            latest_results[key] = ev

    ordered_types = ['Base', 'Manual', 'Extended']
    results = []

    for dtype in ordered_types:
        if dtype in latest_results:
            ev = latest_results[dtype]
            results.append({
                'dictionary': dtype,
                'accuracy': ev.accuracy,
                'precision': ev.precision,
                'recall': ev.recall,
                'f1_score': ev.f1_score
            })

    return render(request, 'core/dictionary_evaluation.html', {
        'results': results
    })

@require_POST
def reset_dictionary_all(request):
    base_deleted = SentimentDictionary.objects.all().delete()[0]
    extended_deleted = ExtendedDictionary.objects.all().delete()[0]

    return JsonResponse({
        'status': 'success',
        'message': f'Dictionary reset complete. Deleted {base_deleted} base/manual and {extended_deleted} extended entries.'
    })