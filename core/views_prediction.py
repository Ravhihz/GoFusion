import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from core.models import SVMModel, Dataset, Prediction, Tweet
import csv


def prediction_page(request):
    """
    Render prediction page UI
    URL: /prediction/
    """
    # Get active model
    active_model = SVMModel.objects.filter(is_active=True).order_by('-trained_at').first()
    
    # Get all datasets
    datasets = Dataset.objects.all().order_by('-uploaded_at')  # ✅ Fix: uploaded_at bukan created_at
    
    # Get recent predictions
    recent_predictions = Prediction.objects.select_related('tweet', 'model').order_by('-predicted_at')[:10]
    
    # Calculate prediction statistics
    total_predictions = Prediction.objects.count()
    positive_count = Prediction.objects.filter(predicted_sentiment='positive').count()
    neutral_count = Prediction.objects.filter(predicted_sentiment='neutral').count()
    negative_count = Prediction.objects.filter(predicted_sentiment='negative').count()
    
    context = {
        'active_model': active_model,
        'datasets': datasets,
        'recent_predictions': recent_predictions,
        'total_predictions': total_predictions,
        'positive_count': positive_count,
        'neutral_count': neutral_count,
        'negative_count': negative_count,
    }
    
    return render(request, 'core/prediction.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def predict_single_tweet(request):
    """
    Predict sentiment untuk single tweet
    URL: /api/prediction/single/
    """
    try:
        data = json.loads(request.body)
        tweet_text = data.get('text', '').strip()
        
        if not tweet_text:
            return JsonResponse({
                'success': False,
                'message': 'Tweet text is required'
            }, status=400)
        
        print(f"\n{'='*80}")
        print(f"[PREDICT SINGLE] Input: {tweet_text[:100]}...")
        print(f"{'='*80}")
        
        # Load active model
        model_record = SVMModel.objects.filter(is_active=True).order_by('-trained_at').first()
        
        if not model_record:
            return JsonResponse({
                'success': False,
                'message': 'No active model found. Please train a model first.'
            }, status=400)
        
        print(f"[INFO] Using model: {model_record.name}")
        print(f"[INFO] Trained at: {model_record.trained_at}")
        
        # Use predictor
        from core.predictor import predict_sentiment
        
        result = predict_sentiment(tweet_text, model_record)
        
        # Check for errors
        if 'error' in result:
            print(f"[ERROR] {result['error']}")
            return JsonResponse({
                'success': False,
                'message': result['error']
            }, status=500)
        
        # Return success
        print(f"\n{'='*80}")
        print(f"[RESULT] Sentiment: {result['predicted_sentiment']}")
        print(f"[RESULT] Confidence: {result['confidence']:.2f}%")
        print(f"{'='*80}\n")
        
        return JsonResponse({
            'success': True,
            'prediction': {
                'sentiment': result['predicted_sentiment'],
                'confidence': result['confidence']
            },
            'features': {
                'sentiment_score': result['sentiment_score'],
                'positive_score': result['positive_score'],
                'negative_score': result['negative_score'],
                'polarity': result['polarity']
            },
            'preprocessing': {
                'tokens': result['preprocessed_tokens'],
                'token_count': len(result['preprocessed_tokens'])
            }
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n[ERROR] Prediction failed: {str(e)}")
        print(error_trace)
        
        return JsonResponse({
            'success': False,
            'message': f'Prediction error: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_dataset(request):
    """
    Predict all tweets in a dataset
    URL: /api/prediction/dataset/
    """
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        
        if not dataset_id:
            return JsonResponse({
                'success': False,
                'message': 'Dataset ID is required'
            }, status=400)
        
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        print(f"\n[PREDICT DATASET] Dataset: {dataset.name}")
        
        # Load active model
        model_record = SVMModel.objects.filter(is_active=True).order_by('-trained_at').first()
        
        if not model_record:
            return JsonResponse({
                'success': False,
                'message': 'No active model found'
            }, status=400)
        
        # Predict all tweets
        from core.predictor import predict_dataset_tweets
        
        result = predict_dataset_tweets(dataset_id, model_record)
        
        if result['success']:
            return JsonResponse({
                'success': True,
                'message': f"Successfully predicted {result['total_predicted']} tweets",
                'statistics': {
                    'total': result['total_predicted'],
                    'positive': result['positive_count'],
                    'neutral': result['neutral_count'],
                    'negative': result['negative_count']
                }
            })
        else:
            return JsonResponse({
                'success': False,
                'message': result['error']
            }, status=500)
        
    except Dataset.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': f'Dataset not found'
        }, status=404)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@require_http_methods(["GET"])
def export_predictions(request):
    """
    Export predictions to CSV
    URL: /api/prediction/export/
    """
    try:
        # Get query parameters
        dataset_id = request.GET.get('dataset_id')
        sentiment = request.GET.get('sentiment')
        
        # Base query
        predictions = Prediction.objects.select_related('tweet', 'model').all()
        
        # Filter by dataset
        if dataset_id:
            predictions = predictions.filter(tweet__dataset_id=dataset_id)
        
        # Filter by sentiment
        if sentiment and sentiment != 'all':
            predictions = predictions.filter(predicted_sentiment=sentiment)
        
        # Order by prediction time
        predictions = predictions.order_by('-predicted_at')
        
        # Create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'Tweet ID',
            'Tweet Text',
            'Predicted Sentiment',
            'Confidence (%)',
            'Model',
            'Predicted At'
        ])
        
        for pred in predictions:
            writer.writerow([
                pred.tweet.id, # type: ignore
                pred.tweet.text,
                pred.predicted_sentiment,
                f"{pred.confidence_score:.2f}",
                pred.model.name,
                pred.predicted_at.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        print(f"[EXPORT] Exported {predictions.count()} predictions")
        
        return response
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)