from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from .models import Dataset, FeatureVector, Tweet, Label
import random
import json


def sampling_view(request, dataset_id):
    """Display sampling interface with current statistics"""
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Get relevant tweets only
        relevant_tweets = Tweet.objects.filter(
            dataset_id=dataset_id,
            is_relevant=True,
            relevance_checked=True
        )
        
        total_tweets = relevant_tweets.count()
        
        # Get sampling statistics
        sampled_tweets = Tweet.objects.filter(
            dataset_id=dataset_id,
            is_sampled=True
        )
        sampled_count = sampled_tweets.count()
        
        # Get label counts
        train_labels = Label.objects.filter(
            tweet__dataset_id=dataset_id,
            dataset_split='train'
        )
        test_labels = Label.objects.filter(
            tweet__dataset_id=dataset_id,
            dataset_split='test'
        )
        
        train_count = train_labels.count()
        test_count = test_labels.count()
        
        context = {
            'dataset': dataset,
            'total_tweets': total_tweets,
            'sampled_count': sampled_count,
            'train_count': train_count,
            'test_count': test_count,
        }
        
        return render(request, 'core/sampling.html', context)
        
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
        return redirect('core:dataset_detail', dataset_id=dataset_id)

def perform_sampling(request, dataset_id):
    """Execute sampling process"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            sample_size = int(data.get('sample_size', 500))
            train_ratio = float(data.get('train_ratio', 0.8))
            
            # Validate sample size
            if sample_size < 100:
                return JsonResponse({
                    'success': False,
                    'message': 'Sample size must be at least 100'
                }, status=400)
            
            if train_ratio < 0.5 or train_ratio > 0.9:
                return JsonResponse({
                    'success': False,
                    'message': 'Train ratio must be between 0.5 and 0.9'
                }, status=400)
            
            dataset = get_object_or_404(Dataset, id=dataset_id)
            
            # Get ONLY RELEVANT tweets
            relevant_tweets = Tweet.objects.filter(
                dataset_id=dataset_id,
                is_relevant=True,
                relevance_checked=True
            )
            total_available = relevant_tweets.count()
            
            print(f"[INFO] Starting sampling from {total_available} RELEVANT tweets")
            
            # Check if enough data
            if total_available < sample_size:
                return JsonResponse({
                    'success': False,
                    'message': f'Not enough tweets. Available: {total_available}, Requested: {sample_size}'
                }, status=400)
            
            # Get IDs of relevant tweets
            available_ids = list(relevant_tweets.filter(is_sampled=False).values_list('id', flat=True))
            if len(available_ids) < sample_size:
                available_ids = list(relevant_tweets.values_list('id', flat=True))
            
            # Random sampling
            random.seed(42)
            sampled_ids = random.sample(available_ids, sample_size)
            
            # Mark as sampled
            Tweet.objects.filter(id__in=sampled_ids).update(
                is_sampled=True,
                sampled_at=timezone.now()
            )
            
            # Split into train and test
            train_size = int(sample_size * train_ratio)
            test_size = sample_size - train_size
            
            random.shuffle(sampled_ids)
            train_ids = sampled_ids[:train_size]
            test_ids = sampled_ids[train_size:]
            
            print(f"[INFO] Split: {train_size} train, {test_size} test")
            
            # ===== CREATE LABELS FOR TRAIN SET =====
            train_labels_created = 0
            for tweet_id in train_ids:
                Label.objects.get_or_create(
                    tweet_id=tweet_id,
                    defaults={
                        'dataset_split': 'train',
                        'sentiment': '',  # Empty, untuk manual labeling
                        'labeled_by': '',
                        'confidence': 0.0
                    }
                )
                train_labels_created += 1
            
            # ===== CREATE LABELS FOR TEST SET (FIX!) =====
            test_labels_created = 0
            for tweet_id in test_ids:
                Label.objects.get_or_create(
                    tweet_id=tweet_id,
                    defaults={
                        'dataset_split': 'test',  # ← PENTING!
                        'sentiment': '',  # Empty, akan di-predict oleh model
                        'labeled_by': 'auto_test',  # Mark as test set
                        'confidence': 0.0
                    }
                )
                test_labels_created += 1
            
            print(f"[INFO] Created {train_labels_created} train labels")
            print(f"[INFO] Created {test_labels_created} test labels")
            
            return JsonResponse({
                'success': True,
                'message': f'Successfully sampled {sample_size} tweets (Train: {train_size}, Test: {test_size})',
                'sampled': sample_size,
                'train_count': train_size,
                'test_count': test_size,
                'train_labels_created': train_labels_created,
                'test_labels_created': test_labels_created
            })
            
        except ValueError as e:
            return JsonResponse({
                'success': False,
                'message': f'Invalid input: {str(e)}'
            }, status=400)
        except Exception as e:
            print(f"[ERROR] Sampling error: {str(e)}")
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


def reset_sampling(request, dataset_id):
    """
    Reset sampling - delete all labels and reset is_sampled flags
    """
    if request.method == 'POST':
        try:
            from django.db import transaction
            
            dataset = get_object_or_404(Dataset, id=dataset_id)
            
            print(f"[INFO] Resetting sampling for dataset: {dataset.name}")
            
            with transaction.atomic():
                # Count before deletion
                labels_count = Label.objects.filter(tweet__dataset_id=dataset_id).count()
                features_count = FeatureVector.objects.filter(tweet__dataset_id=dataset_id).count()
                
                # Delete all labels
                Label.objects.filter(tweet__dataset_id=dataset_id).delete()
                print(f"[INFO] Deleted {labels_count} labels")
                
                # Delete all feature vectors
                FeatureVector.objects.filter(tweet__dataset_id=dataset_id).delete()
                print(f"[INFO] Deleted {features_count} feature vectors")
                
                # Reset is_sampled flags
                updated = Tweet.objects.filter(dataset_id=dataset_id, is_sampled=True).update(
                    is_sampled=False,
                    sampled_at=None
                )
                print(f"[INFO] Reset {updated} tweets")
            
            # Update dataset statistics
            dataset.update_statistics()
            
            return JsonResponse({
                'success': True,
                'message': 'Sampling reset successfully',
                'deleted_labels': labels_count,
                'deleted_features': features_count,
                'reset_tweets': updated
            })
            
        except Exception as e:
            print(f"[ERROR] Reset sampling failed: {str(e)}")
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


def get_sampling_statistics(request, dataset_id):
    """
    Get detailed sampling statistics
    """
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Basic stats
        total = Tweet.objects.filter(dataset_id=dataset_id).count()
        relevant = Tweet.objects.filter(
            dataset_id=dataset_id, 
            is_relevant=True, 
            relevance_checked=True
        ).count()
        sampled = Tweet.objects.filter(dataset_id=dataset_id, is_sampled=True).count()
        
        # Train/test split
        train_labels = Label.objects.filter(tweet__dataset_id=dataset_id, dataset_split='train')
        test_labels = Label.objects.filter(tweet__dataset_id=dataset_id, dataset_split='test')
        
        train_count = train_labels.count()
        test_count = test_labels.count()
        
        # Manual labeling progress (train set only)
        labeled_train = train_labels.exclude(labeled_by='system').count()
        
        stats = {
            'total_tweets': total,
            'relevant_tweets': relevant,
            'sampled_tweets': sampled,
            'train_count': train_count,
            'test_count': test_count,
            'labeled_train': labeled_train,
            'unlabeled_train': train_count - labeled_train,
            'labeling_progress': round((labeled_train / train_count * 100), 2) if train_count > 0 else 0,
            'can_start_training': labeled_train >= 50,  # Minimum untuk training
        }
        
        return JsonResponse({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)