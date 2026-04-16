import csv
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.db.models import Count, Avg, Q
from .models import (
    Dataset, Tweet, PreprocessedTweet, Label, SentimentDictionary,
    ExtendedDictionary, SVMModel, EvaluationMetrics, FeatureVector
)
from django.core.paginator import Paginator
import pandas as pd
import json
import os
from django.conf import settings
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# ==================== HOME & DASHBOARD ====================

def home(request):
    """Dashboard homepage"""
    # Get statistics
    total_datasets = Dataset.objects.count()
    total_tweets = Tweet.objects.count()
    total_labeled = Label.objects.exclude(labeled_by='system').count()
    total_models = SVMModel.objects.count()
    
    # Recent datasets
    recent_datasets = Dataset.objects.order_by('-uploaded_at')[:5]
    
    context = {
        'total_datasets': total_datasets,
        'total_tweets': total_tweets,
        'total_labeled': total_labeled,
        'total_models': total_models,
        'recent_datasets': recent_datasets,
    }
    
    return render(request, 'core/dashboard.html', context)


# ==================== DATASET MANAGEMENT ====================

# ADD THIS TO views.py OR UPDATE EXISTING dataset_list() FUNCTION

def dataset_list(request):
    """Display list of all datasets with current status"""
    try:
        datasets = Dataset.objects.all().order_by('-uploaded_at')
        
        # Add preprocessing status for each dataset
        for dataset in datasets:
            total_tweets = Tweet.objects.filter(dataset=dataset).count()
            preprocessed_count = PreprocessedTweet.objects.filter(tweet__dataset=dataset).count()
            
            # Determine status
            if preprocessed_count == 0:
                dataset.status = 'not_preprocessed' # type: ignore
                dataset.status_label = 'Not Preprocessed' # type: ignore
                dataset.status_class = 'warning' # type: ignore
            elif preprocessed_count < total_tweets:
                dataset.status = 'preprocessing' # type: ignore
                dataset.status_label = f'Processing ({preprocessed_count}/{total_tweets})' # type: ignore
                dataset.status_class = 'info' # type: ignore
            else:
                dataset.status = 'preprocessed' # type: ignore
                dataset.status_label = 'Preprocessed' # type: ignore
                dataset.status_class = 'success' # type: ignore
            
            # Add counts for display
            dataset.total_tweets = total_tweets
            dataset.preprocessed_count = preprocessed_count # type: ignore # type: ignore
            dataset.preprocessing_percentage = (preprocessed_count / total_tweets * 100) if total_tweets > 0 else 0 # type: ignore
        
        context = {
            'datasets': datasets,
        }
        
        return render(request, 'core/dataset_list.html', context)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        messages.error(request, f"Error loading datasets: {str(e)}")
        return render(request, 'core/dataset_list.html', {'datasets': []})


def upload_dataset(request):
    """Upload and import new dataset - FIXED VERSION with duplicate handling"""
    if request.method == 'POST':
        try:
            # Get form data
            name = request.POST.get('name')
            description = request.POST.get('description', '')
            file = request.FILES.get('file')
            
            if not name or not file:
                messages.error(request, 'Please provide dataset name and file')
                return redirect('core:upload_dataset')
            
            # Read file
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(file, encoding='utf-8')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file)
            else:
                messages.error(request, 'Unsupported file format. Please upload CSV or Excel file.')
                return redirect('core:upload_dataset')
            
            # Support multiple text column names
            text_columns = ['text', 'tweet', 'full_text', 'content', 'message', 'tweet_text']
            text_column = None
            
            for col in text_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            if not text_column:
                messages.error(
                    request, 
                    f"File must contain one of these columns: {', '.join(text_columns)}. "
                    f"Found columns: {', '.join(df.columns)}"
                )
                return redirect('core:upload_dataset')
            
            print(f"[INFO] Detected text column: '{text_column}'")
            
            # Create dataset
            dataset = Dataset.objects.create(
                name=name,
                description=description,
                total_tweets=0  # Akan diupdate nanti
            )
            
            # Import tweets with duplicate tracking
            tweets_created = 0
            tweets_duplicate = 0
            tweets_error = 0
            tweets_to_create = []
            
            for idx, row in df.iterrows():
                try:
                    # Use detected text column
                    text = str(row[text_column]).strip()
                    
                    if not text or text.lower() in ['nan', 'none', '']:
                        continue
                    
                    # Get optional fields
                    tweet_id = row.get('id', row.get('tweet_id', row.get('id_str', f'tweet_{idx}')))
                    tweet_id = str(tweet_id) if tweet_id else f'tweet_{idx}'
                    username = row.get('username', row.get('user', row.get('screen_name', 'unknown')))
                    
                    # ========== CEK DUPLIKAT PER DATASET ==========
                    if Tweet.objects.filter(dataset=dataset, tweet_id=tweet_id).exists():
                        tweets_duplicate += 1
                        print(f"[SKIP] Duplicate tweet_id: {tweet_id}")
                        continue
                    # =============================================
                    
                    # Parse created_at
                    created_at = timezone.now()
                    if 'created_at' in row and pd.notna(row['created_at']):
                        try:
                            created_at = pd.to_datetime(row['created_at'])
                        except:
                            pass
                    
                    # Tambah ke batch
                    tweets_to_create.append(Tweet(
                        dataset=dataset,
                        text=text,
                        tweet_id=tweet_id,
                        username=username,
                        created_at=created_at
                    ))
                    tweets_created += 1
                    
                    # Bulk create setiap 1000 tweets (lebih cepat)
                    if len(tweets_to_create) >= 1000:
                        Tweet.objects.bulk_create(tweets_to_create, ignore_conflicts=True)
                        tweets_to_create = []
                    
                except Exception as e:
                    tweets_error += 1
                    print(f"[ERROR] Row {idx}: {e}")
                    continue
            
            # Insert sisa tweets
            if tweets_to_create:
                Tweet.objects.bulk_create(tweets_to_create, ignore_conflicts=True)
            
            # Update dataset total
            dataset.total_tweets = Tweet.objects.filter(dataset=dataset).count()
            dataset.save()
            
            # Message dengan detail
            message = f'Dataset "{name}" uploaded successfully! {tweets_created} tweets imported.'
            if tweets_duplicate > 0:
                message += f' {tweets_duplicate} duplicates skipped.'
            if tweets_error > 0:
                message += f' {tweets_error} errors.'
            
            messages.success(request, message)
            print(f"[SUMMARY] Created: {tweets_created}, Duplicates: {tweets_duplicate}, Errors: {tweets_error}")
            
            return redirect('core:dataset_detail', dataset_id=dataset.id) # type: ignore
            
        except Exception as e:
            messages.error(request, f'Error uploading dataset: {str(e)}')
            print(f"[ERROR] Upload failed: {e}")
            import traceback
            traceback.print_exc()
            return redirect('core:upload_dataset')
    
    return render(request, 'core/upload_dataset.html')

def dataset_detail(request, dataset_id):
    """Show dataset detail with workflow progress"""
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Check if preliminary check is done
        preliminary_done = Tweet.objects.filter(
            dataset=dataset,
            relevance_checked=True
        ).exists()
        
        # ✅ MAIN STATS - Adjust based on preliminary check
        if preliminary_done:
            # After preliminary: only show RELEVANT tweets
            total_tweets = Tweet.objects.filter(
                dataset=dataset,
                is_relevant=True  # ✅ Only relevant
            ).count()
            
            preprocessed_count = PreprocessedTweet.objects.filter(
                tweet__dataset=dataset,
                tweet__is_relevant=True  # ✅ Only relevant
            ).count()
        else:
            # Before preliminary: show ALL tweets
            total_tweets = Tweet.objects.filter(dataset=dataset).count()
            preprocessed_count = PreprocessedTweet.objects.filter(
                tweet__dataset=dataset
            ).count()
        
        # Preliminary check stats
        total_checked = Tweet.objects.filter(
            dataset=dataset, 
            relevance_checked=True
        ).count()
        
        relevant_count = Tweet.objects.filter(
            dataset=dataset,
            is_relevant=True,
            relevance_checked=True
        ).count()
        
        not_relevant_count = Tweet.objects.filter(
            dataset=dataset,
            is_relevant=False,
            relevance_checked=True
        ).count()
        
        # Calculate relevance percentage
        relevance_percentage = (relevant_count / total_checked * 100) if total_checked > 0 else 0
        
        # Duplicates (you can implement this later)
        duplicates_removed = 0
        
        # Sampling stats (only from relevant tweets)
        sampled_count = Label.objects.filter(
            tweet__dataset=dataset,
            tweet__is_relevant=True  # ✅ Only relevant
        ).count()
        
        train_count = Label.objects.filter(
            tweet__dataset=dataset,
            tweet__is_relevant=True,
            dataset_split='train'
        ).count()
        
        test_count = Label.objects.filter(
            tweet__dataset=dataset,
            tweet__is_relevant=True,
            dataset_split='test'
        ).count()
        
        # Labeling stats
        labeled_count = Label.objects.filter(
            tweet__dataset=dataset,
            tweet__is_relevant=True,
            dataset_split='train',
            sentiment__in=['positive', 'neutral', 'negative']
        ).count()
        
        # Feature vectors
        features_count = FeatureVector.objects.filter(
            tweet__dataset=dataset,
            tweet__is_relevant=True
        ).count()
        
        # Models
        models_count = SVMModel.objects.filter(
            dataset=dataset
        ).count() if hasattr(SVMModel, 'dataset') else 0
        
        # Workflow status
        can_preprocess = total_tweets > 0
        can_preliminary = preprocessed_count > 0
        can_sampling = relevant_count >= 100
        can_labeling = sampled_count > 0
        can_training = labeled_count >= train_count if train_count > 0 else False
        
        context = {
            'dataset': dataset,
            'total_tweets': total_tweets,  # ✅ Now shows only relevant after preliminary
            'preprocessed_count': preprocessed_count,
            
            # Preliminary check results
            'preliminary_done': preliminary_done,
            'total_checked': total_checked,
            'relevant_count': relevant_count,
            'not_relevant_count': not_relevant_count,
            'duplicates_removed': duplicates_removed,
            'relevance_percentage': round(relevance_percentage, 1),
            
            # Sampling stats
            'sampled_count': sampled_count,
            'train_count': train_count,
            'test_count': test_count,
            
            # Labeling stats
            'labeled_count': labeled_count,
            
            # Training stats
            'features_count': features_count,
            'models_count': models_count,
            
            # Workflow flags
            'can_preprocess': can_preprocess,
            'can_preliminary': can_preliminary,
            'can_sampling': can_sampling,
            'can_labeling': can_labeling,
            'can_training': can_training,
        }
        
        return render(request, 'core/dataset_view.html', context)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        messages.error(request, f"Error loading dataset: {str(e)}")
        return redirect('core:dataset_list')


def delete_dataset(request, dataset_id):
    """Delete dataset"""
    if request.method == 'POST':
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id)
            dataset_name = dataset.name
            dataset.delete()
            
            messages.success(request, f"Dataset '{dataset_name}' deleted successfully")
            return JsonResponse({'success': True})
            
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)
    
    return JsonResponse({'success': False, 'message': 'Invalid request'}, status=405)


# ==================== PREPROCESSING ====================

def preprocess_dataset(request, dataset_id):
    """Show preprocessing page (GET) and handle execution (POST)"""
    
    # ========== HANDLE GET - Show preprocessing page ==========
    if request.method == 'GET':
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id)
            
            total_tweets = Tweet.objects.filter(dataset=dataset).count()
            preprocessed_count = PreprocessedTweet.objects.filter(tweet__dataset=dataset).count()
            progress = (preprocessed_count / total_tweets * 100) if total_tweets > 0 else 0
            
            context = {
                'dataset': dataset,
                'total_tweets': total_tweets,
                'preprocessed_count': preprocessed_count,
                'progress': round(progress, 2)
            }
            
            return render(request, 'core/preprocessing.html', context)
            
        except Exception as e:
            messages.error(request, f"Error loading preprocessing page: {str(e)}")
            return redirect('core:dataset_list')
    
    # ========== HANDLE POST - Execute preprocessing ==========
    if request.method == 'POST':
        try:
            # ✅ TAMBAHKAN INI - Definisikan dataset di awal POST block
            dataset = get_object_or_404(Dataset, id=dataset_id)
            
            use_stemming = request.POST.get('use_stemming') == 'on'
            
            # Get tweets
            tweets = Tweet.objects.filter(dataset=dataset, text__isnull=False)
            total = tweets.count()
            
            if total == 0:
                messages.warning(request, "No tweets to preprocess")
                return redirect('core:preprocess_dataset', dataset_id=dataset_id)
            
            # Delete old preprocessed data
            PreprocessedTweet.objects.filter(tweet__dataset=dataset).delete()
            
            # Initialize tools
            stem_factory = StemmerFactory()
            stemmer = stem_factory.create_stemmer()
            
            stop_factory = StopWordRemoverFactory()
            stopword_remover = stop_factory.create_stop_word_remover()
            
            # NORMALIZATION DICTIONARY
            normalization_dict = {
                # Kata negasi
                'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak',
                'ngga': 'tidak', 'enggak': 'tidak', 'kagak': 'tidak', 'gk': 'tidak',
                'g': 'tidak', 'tdk': 'tidak',
                
                # Kata kerja umum
                'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
                'buat': 'untuk', 'utk': 'untuk', 'bikin': 'membuat',
                'kasih': 'beri', 'ksh': 'beri', 'kasi': 'beri',
                'liat': 'lihat', 'lt': 'lihat', 'lht': 'lihat',
                'pake': 'pakai', 'pk': 'pakai', 'pkai': 'pakai',
                
                # Kata sambung
                'dgn': 'dengan', 'krn': 'karena', 'karna': 'karena', 'krna': 'karena',
                'kalo': 'jika', 'klo': 'jika', 'tapi': 'tetapi', 'tp': 'tetapi',
                
                # Kata sifat/keterangan
                'emang': 'memang', 'emg': 'memang', 'bgt': 'sangat', 'banget': 'sangat',
                'bener': 'benar', 'bnr': 'benar', 'cuma': 'hanya', 'cm': 'hanya',
                'makin': 'semakin', 'mkn': 'semakin',
                
                # Kata tanya
                'gmn': 'bagaimana', 'gimana': 'bagaimana', 'bgmn': 'bagaimana',
                'bgmana': 'bagaimana', 'knp': 'kenapa', 'knapa': 'kenapa',
                'kenapa': 'mengapa', 'kpn': 'kapan', 'dimana': 'di mana', 'dmn': 'di mana',
                
                # Kata umum lainnya
                'aja': 'saja', 'aj': 'saja', 'doang': 'saja', 'juga': 'juga', 'jg': 'juga',
                'jadi': 'jadi', 'jd': 'jadi', 'sama': 'sama', 'sm': 'sama', 'ama': 'sama',
                'org': 'orang', 'orng': 'orang', 'jgn': 'jangan', 'hrs': 'harus',
                'msh': 'masih', 'lagi': 'sedang', 'lg': 'sedang', 'terus': 'terus',
                'trus': 'terus', 'trs': 'terus', 'banyak': 'banyak', 'bnyk': 'banyak',
                'byk': 'banyak', 'semua': 'semua', 'smua': 'semua', 'smw': 'semua',
                'soal': 'masalah', 'ttg': 'tentang', 'tentang': 'mengenai', 'tntg': 'mengenai',
                
                # Kata ganti orang
                'gue': 'saya', 'gw': 'saya', 'gua': 'saya', 'ane': 'saya',
                'ente': 'anda', 'elu': 'kamu', 'lu': 'kamu', 'lo': 'kamu',
                
                # Kata waktu
                'dulu': 'dahulu', 'dl': 'dahulu', 'duluan': 'dahulu',
                'nanti': 'nanti', 'ntar': 'nanti', 'nt': 'nanti',
                'sekarang': 'sekarang', 'skrng': 'sekarang', 'skr': 'sekarang', 'skg': 'sekarang',
                'kemarin': 'kemarin', 'kmrn': 'kemarin', 'besok': 'besok', 'bsk': 'besok',
                
                # Kata lainnya
                'mau': 'ingin', 'mo': 'ingin', 'mw': 'ingin', 'pengen': 'ingin', 'pgn': 'ingin',
                'bisa': 'dapat', 'bs': 'dapat', 'tau': 'tahu', 'tw': 'tahu',
                'kayak': 'seperti', 'kyk': 'seperti', 'kek': 'seperti',
                'gitu': 'begitu', 'gini': 'begini', 'coba': 'coba', 'cb': 'coba',
                'belum': 'belum', 'blm': 'belum', 'blom': 'belum',
                'ada': 'ada', 'ad': 'ada', 'pas': 'saat',
                'abis': 'habis', 'abs': 'habis', 'yuk': 'ayo', 'yok': 'ayo', 'ayok': 'ayo',
                'gausah': 'tidak usah', 'gasusah': 'tidak usah',
                
                # Singkatan
                'dll': 'dan lain lain', 'dsb': 'dan sebagainya', 'dst': 'dan seterusnya',
                'mksh': 'terima kasih', 'thanks': 'terima kasih', 'thx': 'terima kasih',
                'pls': 'tolong', 'plz': 'tolong', 'please': 'tolong',
                'btw': 'ngomong ngomong', 'fyi': 'informasi', 'asap': 'segera',
                'ok': 'oke', 'okay': 'oke',
                
                # Partikel (dihapus)
                'sih': '', 'dong': '', 'deh': '', 'kok': '', 'lho': '', 'loh': '',
                'nih': '', 'tuh': '', 'wkwk': '', 'wkwkwk': '', 'haha': '', 'hehe': '', 'hihi': '',
            }
            
            success_count = 0
            error_count = 0
            
            for tweet in tweets:
                try:
                    original_text = tweet.text
                    
                    # STEP 1: Remove URLs, mentions, hashtags
                    text = re.sub(r'http\S+|www\S+|https\S+', '', original_text, flags=re.MULTILINE)
                    text = re.sub(r'@\w+|#\w+', '', text)
                    
                    # STEP 2: Remove punctuation
                    after_punctuation = re.sub(r'[^\w\s]', '', text)
                    
                    # STEP 3: Remove numbers
                    after_remove_numbers = re.sub(r'\d+', '', after_punctuation)
                    
                    # STEP 4: Remove extra whitespace
                    after_cleaning = ' '.join(after_remove_numbers.split())
                    
                    # STEP 5: Case folding
                    after_case_folding = after_cleaning.lower()
                    
                    # STEP 6: Normalization
                    words = after_case_folding.split()
                    normalized_words = [normalization_dict.get(w, w) for w in words if normalization_dict.get(w, w)]
                    after_normalization = ' '.join(normalized_words)
                    
                    # STEP 7: Remove stopwords
                    after_stopword = stopword_remover.remove(after_normalization)
                    
                    # STEP 8: Stemming
                    after_stemming = stemmer.stem(after_stopword) if use_stemming else after_stopword
                    
                    # STEP 9: Tokenization
                    tokens = [t for t in after_stemming.split() if len(t) > 2]
                    
                    # Save
                    PreprocessedTweet.objects.create(
                        tweet=tweet,
                        original_text=original_text,
                        after_remove_punctuation=after_punctuation,
                        after_cleaning=after_cleaning,
                        after_case_folding=after_case_folding,
                        after_normalization=after_normalization,
                        after_stopword=after_stopword,
                        after_stemming=after_stemming if use_stemming else '',
                        tokens=tokens
                    )
                    
                    success_count += 1
                    if success_count % 100 == 0:
                        print(f"[INFO] Processed {success_count}/{total} tweets...")
                    
                except Exception as e:
                    print(f"[ERROR] Tweet {tweet.id}: {str(e)}") # type: ignore
                    error_count += 1
                    continue
            
            messages.success(request, f"Preprocessing completed! {success_count} tweets processed, {error_count} errors")
            return redirect('core:dataset_detail', dataset_id=dataset_id)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messages.error(request, f"Preprocessing failed: {str(e)}")
            return redirect('core:dataset_detail', dataset_id=dataset_id)


def preprocess_status(request, dataset_id):
    """Get preprocessing status via AJAX"""
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        total = Tweet.objects.filter(dataset=dataset).count()
        processed = PreprocessedTweet.objects.filter(tweet__dataset=dataset).count()
        
        progress = (processed / total * 100) if total > 0 else 0
        
        return JsonResponse({
            'success': True,
            'total': total,
            'processed': processed,
            'progress': round(progress, 2),
            'is_complete': processed >= total
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=400)

def preprocessing_visualizer(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Ambil 5 data preprocessed pertama
    preprocessed_samples = PreprocessedTweet.objects.filter(
        tweet__dataset=dataset
    ).select_related('tweet').order_by('tweet__id')[:5]
    
    # Statistik
    total_tweets = Tweet.objects.filter(dataset=dataset).count()
    preprocessed_count = PreprocessedTweet.objects.filter(
        tweet__dataset=dataset
    ).count()
    
    progress = int((preprocessed_count / total_tweets * 100)) if total_tweets > 0 else 0
    
    context = {
        'dataset': dataset,
        'samples': preprocessed_samples,
        'total_tweets': total_tweets,
        'preprocessed_count': preprocessed_count,
        'sample_count': preprocessed_samples.count(),
        'progress': progress,
    }
    
    return render(request, 'core/preprocessing_visualizer.html', context)

# ==================== LABELING ====================

def labeling_view(request, dataset_id):
    """Manual labeling interface"""
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Get ONLY training labels - WITH CORRECT RELATIONSHIP NAME
        training_labels = Label.objects.filter(
            tweet__dataset_id=dataset_id,
            dataset_split='train'
        ).select_related(
            'tweet',
            'tweet__preprocessed'  # ✅ CORRECT: 'preprocessed' not 'preprocessedtweet'
        )
        
        total_train = training_labels.count()
        
        # Unlabeled = empty sentiment
        unlabeled = training_labels.exclude(
            sentiment__in=['positive', 'neutral', 'negative']
        )
        
        unlabeled_count = unlabeled.count()
        labeled_count = total_train - unlabeled_count
        
        # Pagination
        paginator = Paginator(unlabeled, 50)
        page = request.GET.get('page', 1)
        current_batch = paginator.get_page(page)
        
        # Progress percentage
        progress = (labeled_count / total_train * 100) if total_train > 0 else 0
        
        context = {
            'dataset': dataset,
            'labels': current_batch,
            'total_labels': total_train,
            'labeled_count': labeled_count,
            'unlabeled_count': unlabeled_count,
            'progress_percentage': round(progress, 1),
        }
        
        return render(request, 'core/labeling.html', context)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        messages.error(request, f"Error: {str(e)}")
        return redirect('core:dataset_detail', dataset_id=dataset_id)


def save_label_batch(request, dataset_id):
    """
    Save multiple labels at once (batch save for 50 tweets)
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            labels_data = data.get('labels', [])
            
            if not labels_data:
                return JsonResponse({
                    'success': False,
                    'message': 'No labels provided'
                }, status=400)
            
            dataset = get_object_or_404(Dataset, id=dataset_id)
            
            saved_count = 0
            errors = []
            
            for label_item in labels_data:
                try:
                    tweet_id = label_item.get('tweet_id')
                    sentiment = label_item.get('sentiment')
                    
                    if not tweet_id or not sentiment:
                        errors.append(f"Invalid data for tweet {tweet_id}")
                        continue
                    
                    if sentiment not in ['positive', 'neutral', 'negative']:
                        errors.append(f"Invalid sentiment '{sentiment}' for tweet {tweet_id}")
                        continue
                    
                    # Get label for this tweet
                    label = Label.objects.filter(
                        tweet_id=tweet_id,
                        tweet__dataset=dataset,      # ✅ Access via tweet relationship
                        dataset_split='train'        # ✅ Use dataset_split field
                    ).first()
                    
                    if not label:
                        errors.append(f"Label not found for tweet {tweet_id}")
                        continue
                    
                    # Update label
                    label.sentiment = sentiment
                    label.labeled_by = request.user.username if request.user.is_authenticated else 'manual'
                    label.labeled_at = timezone.now()
                    label.confidence = 1.0
                    label.save()
                    
                    saved_count += 1
                    
                except Exception as e:
                    errors.append(f"Error saving tweet {tweet_id}: {str(e)}")
            
            return JsonResponse({
                'success': True,
                'message': f"Saved {saved_count} labels",
                'saved_count': saved_count,
                'errors': errors if errors else []
            })
            
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


def save_label(request, dataset_id):
    """
    Save/update single label
    Used for: editing existing labels, quick corrections
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            tweet_id = data.get('tweet_id')
            sentiment = data.get('sentiment')
            
            if not tweet_id or not sentiment:
                return JsonResponse({
                    'success': False,
                    'message': 'Tweet ID and sentiment are required'
                }, status=400)
            
            if sentiment not in ['positive', 'neutral', 'negative']:
                return JsonResponse({
                    'success': False,
                    'message': 'Invalid sentiment value'
                }, status=400)
            
            # Get label for this tweet
            label = Label.objects.filter(
                tweet_id=tweet_id,
                tweet__dataset_id=dataset_id,  # ✅ Access via tweet relationship
                dataset_split='train'          # ✅ Use dataset_split field
            ).first()
            
            if not label:
                return JsonResponse({
                    'success': False,
                    'message': 'Label not found for this tweet'
                }, status=404)
            
            # Update label
            label.sentiment = sentiment
            label.labeled_by = request.user.username if request.user.is_authenticated else 'manual'
            label.labeled_at = timezone.now()
            label.confidence = 1.0
            label.save()
            
            return JsonResponse({
                'success': True,
                'message': 'Label saved successfully',
                'tweet_id': tweet_id,
                'sentiment': sentiment
            })
            
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


def labeling_statistics(request, dataset_id):
    """Get labeling statistics"""
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        train_labels = Label.objects.filter(
            tweet__dataset=dataset,
            dataset_split='train'
        )
        
        total = train_labels.count()
        labeled = train_labels.exclude(labeled_by='system').count()
        
        positive = train_labels.filter(sentiment='positive').exclude(labeled_by='system').count()
        negative = train_labels.filter(sentiment='negative').exclude(labeled_by='system').count()
        neutral = train_labels.filter(sentiment='neutral').exclude(labeled_by='system').count()
        
        return JsonResponse({
            'success': True,
            'statistics': {
                'total': total,
                'labeled': labeled,
                'unlabeled': total - labeled,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'progress': round((labeled / total * 100), 2) if total > 0 else 0
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


def labeled_tweets_list(request, dataset_id):
    """Show list of labeled tweets"""
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Get all labeled tweets (train set only, exclude system)
        labeled_tweets = Label.objects.filter(
            tweet__dataset=dataset,
            dataset_split='train'
        ).exclude(
            labeled_by='system'
        ).select_related('tweet').order_by('-labeled_at')
        
        context = {
            'dataset': dataset,
            'labeled_tweets': labeled_tweets,
            'total': labeled_tweets.count()
        }
        
        return render(request, 'core/labeled_tweets_list.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading labeled tweets: {str(e)}")
        return redirect('core:dataset_detail', dataset_id=dataset_id)

@login_required
def export_labeled_tweets(request, dataset_id):
    """Export labeled tweets to CSV format"""
    dataset = get_object_or_404(Dataset, id=dataset_id)  # Hapus owner=request.user
    labels = Label.objects.filter(
        tweet__dataset=dataset
    ).select_related('tweet').order_by('-labeled_at')
    
    response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
    filename = f'labeled_tweets_{dataset.name}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv'
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    # Add BOM for Excel compatibility with UTF-8
    response.write('\ufeff')
    
    writer = csv.writer(response)
    
    # Header sesuai revisi dosen: username, created at, text, id tweet, label, tgl, jam
    writer.writerow([
        'Username',
        'Created At',
        'Tweet Text',
        'Tweet ID',
        'Label',
        'Tanggal Labeling',
        # 'Jam Labeling'
    ])
    
    for label in labels:
        writer.writerow([
            label.tweet.username or 'Unknown',
            label.tweet.created_at.strftime('%Y-%m-%d %H:%M:%S') if label.tweet.created_at else '',
            label.tweet.text,
            label.tweet.tweet_id or '',
            label.sentiment.upper(),
            label.labeled_at.strftime('%Y-%m-%d') if label.labeled_at else '',
            # label.labeled_at.strftime('%H:%M:%S') if label.labeled_at else ''
        ])
    
    return response

# ==================== DICTIONARY ====================
def upload_dictionary(request):
    """Upload MANUAL sentiment dictionary"""
    if request.method == 'POST':
        try:
            file = request.FILES.get('file')

            if not file:
                messages.error(request, "File is required")
                return redirect('core:dictionary')

            # Read file
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
                df = pd.read_excel(file)
            else:
                messages.error(request, "Invalid file format")
                return redirect('core:dictionary')

            # Validate columns
            required_cols = ['word', 'weight', 'polarity']
            if not all(col in df.columns for col in required_cols):
                messages.error(request, f"File must contain columns: {required_cols}")
                return redirect('core:dictionary')

            created = 0
            updated = 0

            for _, row in df.iterrows():
                obj, is_created = SentimentDictionary.objects.update_or_create(
                    word=str(row['word']).lower(),
                    defaults={
                        'weight': float(row['weight']),
                        'polarity': str(row['polarity']).lower(),
                        'source': 'manual'   # 🔥 INI KUNCI
                    }
                )

                if is_created:
                    created += 1
                else:
                    updated += 1

            messages.success(
                request,
                f"Manual dictionary uploaded: {created} added, {updated} updated"
            )

            return redirect('core:dictionary')

        except Exception as e:
            messages.error(request, f"Error uploading dictionary: {str(e)}")
            return redirect('core:dictionary')

    return redirect('core:dictionary')



def extended_dictionary_view(request):
    """View extended dictionary"""
    extended_dict = ExtendedDictionary.objects.all().order_by('-similarity_score')
    
    # Statistics
    total = extended_dict.count()
    positive = extended_dict.filter(polarity='positive').count()
    negative = extended_dict.filter(polarity='negative').count()
    
    context = {
        'dictionary': extended_dict[:100],
        'total': total,
        'positive': positive,
        'negative': negative
    }
    
    return render(request, 'core/extended_dictionary.html', context)


def dictionary_statistics(request):
    """Get dictionary statistics"""
    try:
        from .dictionary_extension import get_dictionary_statistics
        
        stats = get_dictionary_statistics()
        
        return JsonResponse({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)

# ==================== EVALUATION ====================

def evaluation_view(request, model_id):
    """Evaluation page for model"""
    model = get_object_or_404(SVMModel, id=model_id)
    
    try:
        metrics = EvaluationMetrics.objects.get(model=model)
    except EvaluationMetrics.DoesNotExist:
        metrics = None
    
    context = {
        'model': model,
        'metrics': metrics
    }
    
    return render(request, 'core/evaluation.html', context)


def evaluation_data(request, model_id):
    """Get evaluation data (JSON)"""
    try:
        model = get_object_or_404(SVMModel, id=model_id)
        metrics = EvaluationMetrics.objects.get(model=model)
        
        return JsonResponse({
            'success': True,
            'metrics': {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'confusion_matrix': metrics.confusion_matrix,
                'classification_report': metrics.classification_report
            }
        })
        
    except EvaluationMetrics.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': 'Evaluation metrics not found'
        }, status=404)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


# ==================== ANALYTICS ====================

def analytics_view(request):
    """Analytics dashboard with charts"""
    # Get statistics
    total_datasets = Dataset.objects.count()
    total_tweets = Tweet.objects.count()
    total_labeled = Label.objects.exclude(labeled_by='system').count()
    total_models = SVMModel.objects.count()
    
    # Get datasets with stats
    datasets = []
    for dataset in Dataset.objects.all():
        tweets = Tweet.objects.filter(dataset=dataset).count()
        labeled = Label.objects.filter(
            tweet__dataset=dataset
        ).exclude(labeled_by='system').count()
        
        positive = Label.objects.filter(
            tweet__dataset=dataset,
            sentiment='positive'
        ).exclude(labeled_by='system').count()
        
        negative = Label.objects.filter(
            tweet__dataset=dataset,
            sentiment='negative'
        ).exclude(labeled_by='system').count()
        
        neutral = Label.objects.filter(
            tweet__dataset=dataset,
            sentiment='neutral'
        ).exclude(labeled_by='system').count()
        
        datasets.append({
            'id': dataset.id, # type: ignore
            'name': dataset.name,
            'total_tweets': tweets,
            'labeled_count': labeled,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'is_preprocessed': dataset.is_preprocessed, # type: ignore
        })
    
    context = {
        'total_datasets': total_datasets,
        'total_tweets': total_tweets,
        'total_labeled': total_labeled,
        'total_models': total_models,
        'datasets': datasets,
    }
    
    return render(request, 'core/analytics.html', context)


def sentiment_distribution(request):
    """Get sentiment distribution for charts (AJAX)"""
    try:
        positive = Label.objects.filter(sentiment='positive').exclude(labeled_by='system').count()
        negative = Label.objects.filter(sentiment='negative').exclude(labeled_by='system').count()
        neutral = Label.objects.filter(sentiment='neutral').exclude(labeled_by='system').count()
        
        return JsonResponse({
            'success': True,
            'distribution': {
                'positive': positive,
                'negative': negative,
                'neutral': neutral
            }
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=400)


def model_comparison(request):
    """Get model comparison data for charts (AJAX)"""
    try:
        models = SVMModel.objects.all()
        
        model_data = []
        for model in models:
            model_data.append({
                'name': model.name,
                'train_accuracy': float(model.train_accuracy),
                'test_accuracy': float(model.test_accuracy)
            })
        
        return JsonResponse({
            'success': True,
            'models': model_data
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=400)
    
