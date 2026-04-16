"""
Manual Checking Views
Handles quality control review after preliminary check
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone
from django.core.paginator import Paginator
from django.db.models import Q, Count
from .models import Dataset, Tweet


# ============================================================================
# DASHBOARD VIEW
# ============================================================================

def manual_check_dashboard(request, dataset_id):
    """
    Dashboard untuk manual checking - overview statistics dan progress
    
    Shows:
    - Total relevant tweets from preliminary check
    - Pending review count
    - Accepted/rejected counts
    - Progress percentage
    """
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Calculate statistics
    # Total tweets yang is_relevant=True (lolos preliminary)
    total_relevant = Tweet.objects.filter(
        dataset=dataset,
        is_relevant=True
    ).count()
    
    # Pending: lolos preliminary tapi belum di-manual check
    pending_count = Tweet.objects.filter(
        dataset=dataset,
        is_relevant=True,
        relevance_checked=False
    ).count()
    
    # Accepted: lolos preliminary DAN lolos manual check
    accepted_count = Tweet.objects.filter(
        dataset=dataset,
        is_relevant=True,
        relevance_checked=True
    ).count()
    
    # Rejected: gagal manual check (di-uncheck)
    rejected_count = Tweet.objects.filter(
        dataset=dataset,
        is_relevant=False,
        relevance_checked=True
    ).count()
    
    # Calculate progress
    checked_total = accepted_count + rejected_count
    progress = (checked_total / total_relevant * 100) if total_relevant > 0 else 0
    
    context = {
        'dataset': dataset,
        'total_relevant': total_relevant,
        'pending_count': pending_count,
        'accepted_count': accepted_count,
        'rejected_count': rejected_count,
        'progress': round(progress, 1)
    }
    
    return render(request, 'core/manual_check_dashboard.html', context)


# ============================================================================
# LIST VIEW (CHECKLIST INTERFACE)
# ============================================================================

def manual_check_list(request, dataset_id):
    """
    List view dengan checklist interface untuk manual review
    
    Features:
    - Filter by status (pending/accepted/rejected/all)
    - Pagination (50 tweets per page)
    - Quick toggle buttons
    - Bulk actions
    """
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Get filter parameter
    status_filter = request.GET.get('status', 'pending')
    search_query = request.GET.get('search', '').strip()
    
    # Base query: only tweets from preliminary check (is_relevant=True initially)
    tweets = Tweet.objects.filter(dataset=dataset)
    
    # Apply status filter
    if status_filter == 'pending':
        # Tweets yang belum di-manual check
        tweets = tweets.filter(
            is_relevant=True,
            relevance_checked=False
        )
    elif status_filter == 'accepted':
        # Tweets yang lolos manual check
        tweets = tweets.filter(
            is_relevant=True,
            relevance_checked=True
        )
    elif status_filter == 'rejected':
        # Tweets yang ditolak manual check
        tweets = tweets.filter(
            is_relevant=False,
            relevance_checked=True
        )
    elif status_filter == 'all':
        # Semua tweets yang lolos preliminary
        tweets = tweets.filter(is_relevant=True)
    
    # Apply search filter
    if search_query:
        tweets = tweets.filter(text__icontains=search_query)
    
    # Order by ID
    tweets = tweets.order_by('id')
    
    # Get statistics for badges
    stats = {
        'pending': Tweet.objects.filter(
            dataset=dataset,
            is_relevant=True,
            relevance_checked=False
        ).count(),
        'accepted': Tweet.objects.filter(
            dataset=dataset,
            is_relevant=True,
            relevance_checked=True
        ).count(),
        'rejected': Tweet.objects.filter(
            dataset=dataset,
            is_relevant=False,
            relevance_checked=True
        ).count(),
    }
    
    # Pagination (50 tweets per page)
    paginator = Paginator(tweets, 50)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    context = {
        'dataset': dataset,
        'page_obj': page_obj,
        'status_filter': status_filter,
        'search_query': search_query,
        'pending_count': stats['pending'],
        'accepted_count': stats['accepted'],
        'rejected_count': stats['rejected'],
    }
    
    return render(request, 'core/manual_check_list.html', context)


# ============================================================================
# AJAX ACTIONS
# ============================================================================

def manual_check_toggle(request, tweet_id):
    """
    Toggle is_relevant status via AJAX (quick action button)
    
    Response JSON:
    {
        'success': true/false,
        'tweet_id': int,
        'is_relevant': true/false
    }
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'message': 'Invalid request method'
        }, status=405)
    
    tweet = get_object_or_404(Tweet, id=tweet_id)
    
    # Toggle is_relevant
    tweet.is_relevant = not tweet.is_relevant
    tweet.relevance_checked = True
    tweet.checked_by = request.user.username if request.user.is_authenticated else 'admin'
    tweet.checked_at = timezone.now()
    tweet.save()
    
    return JsonResponse({
        'success': True,
        'tweet_id': tweet.id, # type: ignore
        'is_relevant': tweet.is_relevant
    })


def accept_all_pending(request, dataset_id):
    """
    Accept all pending tweets (mark relevance_checked=True, keep is_relevant=True)
    Used when user is confident all remaining tweets are relevant
    
    Response JSON:
    {
        'success': true/false,
        'message': string,
        'count': int
    }
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'message': 'Invalid request method'
        }, status=405)
    
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Update all pending tweets (is_relevant=True, relevance_checked=False)
    updated_count = Tweet.objects.filter(
        dataset=dataset,
        is_relevant=True,
        relevance_checked=False
    ).update(
        relevance_checked=True,
        checked_by=request.user.username if request.user.is_authenticated else 'admin',
        checked_at=timezone.now()
    )
    
    return JsonResponse({
        'success': True,
        'message': f'{updated_count} tweets accepted and ready for sampling',
        'count': updated_count
    })


# ============================================================================
# BULK ACTIONS
# ============================================================================

def manual_check_bulk_update(request, dataset_id):
    """
    Bulk accept/reject selected tweets
    
    POST params:
    - tweet_ids: list of tweet IDs
    - action: 'accept' or 'reject'
    """
    if request.method != 'POST':
        messages.error(request, 'Invalid request method')
        return redirect('core:manual_check_list', dataset_id=dataset_id)
    
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Get selected tweet IDs
    tweet_ids = request.POST.getlist('tweet_ids')
    action = request.POST.get('action')
    
    # Validation
    if not tweet_ids:
        messages.error(request, 'No tweets selected')
        return redirect('core:manual_check_list', dataset_id=dataset_id)
    
    if action not in ['accept', 'reject']:
        messages.error(request, 'Invalid action')
        return redirect('core:manual_check_list', dataset_id=dataset_id)
    
    # Get tweets
    tweets = Tweet.objects.filter(
        id__in=tweet_ids,
        dataset=dataset
    )
    
    count = tweets.count()
    username = request.user.username if request.user.is_authenticated else 'admin'
    
    # Perform action
    if action == 'accept':
        tweets.update(
            is_relevant=True,
            relevance_checked=True,
            checked_by=username,
            checked_at=timezone.now()
        )
        messages.success(request, f'✓ {count} tweets accepted')
    
    elif action == 'reject':
        tweets.update(
            is_relevant=False,
            relevance_checked=True,
            checked_by=username,
            checked_at=timezone.now()
        )
        messages.warning(request, f'✗ {count} tweets rejected')
    
    # Redirect back to list
    status_filter = request.GET.get('status', 'pending')
    return redirect(f"{request.path.replace('/bulk/', '/list/')}?status={status_filter}")


# ============================================================================
# HELPER FUNCTIONS (Optional)
# ============================================================================

def get_manual_check_statistics(dataset):
    """
    Helper function to get manual check statistics
    Returns dict with counts and percentages
    """
    stats = Tweet.objects.filter(dataset=dataset).aggregate(
        total=Count('id'),
        total_relevant=Count('id', filter=Q(is_relevant=True)),
        pending=Count('id', filter=Q(
            is_relevant=True,
            relevance_checked=False
        )),
        accepted=Count('id', filter=Q(
            is_relevant=True,
            relevance_checked=True
        )),
        rejected=Count('id', filter=Q(
            is_relevant=False,
            relevance_checked=True
        ))
    )
    
    # Calculate percentages
    if stats['total_relevant'] > 0:
        stats['acceptance_rate'] = (stats['accepted'] / stats['total_relevant']) * 100
        stats['rejection_rate'] = (stats['rejected'] / stats['total_relevant']) * 100
        stats['progress'] = ((stats['accepted'] + stats['rejected']) / stats['total_relevant']) * 100
    else:
        stats['acceptance_rate'] = 0
        stats['rejection_rate'] = 0
        stats['progress'] = 0
    
    return stats