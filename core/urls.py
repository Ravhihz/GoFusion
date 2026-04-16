# File: core/urls.py
"""
URL Configuration for GoFusion - Fusion Sentiment Analysis
Organized by workflow: Dataset → Preprocessing → Preliminary → Sampling → Labeling → Training → Prediction
"""

from django.urls import path

from . import views
from .views_preliminary import (
    preliminary_check_view,
    mark_relevance_ajax,
    bulk_mark_relevance,
    preliminary_manual_check_view,
    preliminary_statistics,
    skip_preliminary_check,
    reset_preliminary_check,
    export_preliminary_results,
    preview_keyword_filter,
    apply_keyword_filter
)
from .views_sampling import (
    sampling_view,
    perform_sampling,
    reset_sampling,
    get_sampling_statistics
)
from .views_training import (
    execute_step_predict_test,
    training_dashboard,
    execute_step_sentiment_scores,
    execute_step_word_embeddings,
    execute_step_pca,
    execute_step_train_svm,
    execute_step_extend_dictionary,
    execute_step_retrain_with_extended,
    get_training_progress,
    reset_training_data,
    get_training_status
)
from .views_prediction import (
    prediction_page,
    predict_single_tweet,
    predict_dataset,
    export_predictions
)
from .views_additional import (
    dictionary_evaluation_view,
    models_list,
    model_detail,
    reset_dictionary_all,
    set_active_model,
    delete_model,
    analytics_dashboard,
    dictionary_view,
    feature_viewer,
    feature_detail_api,
    export_features_csv
)

app_name = 'core'

urlpatterns = [
    # HOME & DASHBOARD
    path('', views.home, name='home'),
    path('dashboard/', views.home, name='dashboard'),
    
    # DATASET MANAGEMENT
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/upload/', views.upload_dataset, name='upload_dataset'),
    path('datasets/<int:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('datasets/<int:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),
    
    # WORKFLOW STEP 1: PREPROCESSING
    path('datasets/<int:dataset_id>/preprocess/', views.preprocess_dataset, name='preprocess_dataset'),
    path('datasets/<int:dataset_id>/preprocess/status/', views.preprocess_status, name='preprocess_status'),
    path('datasets/<int:dataset_id>/preprocess/visualizer/', views.preprocessing_visualizer, name='preprocessing_visualizer'),
    
    # WORKFLOW STEP 2A: PRELIMINARY FILTER (Keyword-based Auto Filter)
    path('datasets/<int:dataset_id>/preliminary/filter/', preview_keyword_filter, name='preliminary_filter'),
    path('api/datasets/<int:dataset_id>/preliminary/preview-keyword/', preview_keyword_filter, name='preview_keyword_filter'),
    path('api/datasets/<int:dataset_id>/preliminary/apply-keyword/', apply_keyword_filter, name='apply_keyword_filter'),
    
    # WORKFLOW STEP 2B: PRELIMINARY MANUAL CHECK (Manual Verification)
    path('datasets/<int:dataset_id>/preliminary/', preliminary_check_view, name='preliminary_check'),
    path('datasets/<int:dataset_id>/preliminary/manual/', preliminary_manual_check_view, name='preliminary_manual_check'),
    path('datasets/<int:dataset_id>/preliminary/export/', export_preliminary_results, name='export_preliminary_results'),
    
    # Preliminary Manual Check - API Endpoints
    path('api/preliminary/mark-relevance/', mark_relevance_ajax, name='mark_relevance_ajax'),
    path('api/datasets/<int:dataset_id>/preliminary/bulk-mark/', bulk_mark_relevance, name='bulk_mark_relevance'),
    path('api/datasets/<int:dataset_id>/preliminary/stats/', preliminary_statistics, name='preliminary_statistics'),
    path('api/datasets/<int:dataset_id>/preliminary/skip/', skip_preliminary_check, name='skip_preliminary_check'),
    path('api/datasets/<int:dataset_id>/preliminary/reset/', reset_preliminary_check, name='reset_preliminary_check'),
    
    # WORKFLOW STEP 3: SAMPLING (Train/Test Split)
    path('datasets/<int:dataset_id>/sampling/', sampling_view, name='sampling'),
    
    # Sampling - API Endpoints
    path('api/datasets/<int:dataset_id>/sampling/perform/', perform_sampling, name='perform_sampling'),
    path('api/datasets/<int:dataset_id>/sampling/reset/', reset_sampling, name='reset_sampling'),
    path('api/datasets/<int:dataset_id>/sampling/stats/', get_sampling_statistics, name='sampling_statistics'),
    
    # WORKFLOW STEP 4: LABELING (Manual Annotation)
    path('datasets/<int:dataset_id>/labeling/', views.labeling_view, name='labeling'),
    path('datasets/<int:dataset_id>/labeled-tweets/', views.labeled_tweets_list, name='labeled_tweets_list'),
    path('datasets/<int:dataset_id>/labeled-tweets/export/', views.export_labeled_tweets, name='export_labeled_tweets'),

    # Labeling - API Endpoints
    path('api/datasets/<int:dataset_id>/labeling/save/', views.save_label, name='save_label'),
    path('api/datasets/<int:dataset_id>/labeling/save-batch/', views.save_label_batch, name='save_label_batch'),
    path('api/datasets/<int:dataset_id>/labeling/stats/', views.labeling_statistics, name='labeling_statistics'),
    
    # WORKFLOW STEP 5-7: TRAINING (Fusion Method + SVM)
    path('datasets/<int:dataset_id>/training/', training_dashboard, name='training_dashboard'),
    
    # Training - Step Execution API
    path('api/datasets/<int:dataset_id>/training/step3/', execute_step_sentiment_scores, name='execute_step_sentiment_scores'),
    path('api/datasets/<int:dataset_id>/training/step4/', execute_step_word_embeddings, name='execute_step_word_embeddings'),
    path('api/datasets/<int:dataset_id>/training/step4-5/', execute_step_pca, name='execute_step_pca'),
    path('api/datasets/<int:dataset_id>/training/step5/', execute_step_train_svm, name='execute_step_train_svm'),
    path('api/datasets/<int:dataset_id>/training/step6/', execute_step_extend_dictionary, name='execute_step_extend_dictionary'),
    path('api/datasets/<int:dataset_id>/training/step7/', execute_step_retrain_with_extended, name='execute_step_retrain_with_extended'),
    
    # Training - Utility API
    path('api/datasets/<int:dataset_id>/training/progress/', get_training_progress, name='training_progress'),
    path('api/datasets/<int:dataset_id>/training/status/', get_training_status, name='get_training_status'),
    path('api/datasets/<int:dataset_id>/training/reset/', reset_training_data, name='reset_training_data'),
    path('api/datasets/<int:dataset_id>/training/predict-test/', execute_step_predict_test, name="predict_test_data"),
    
    # FEATURE EXTRACTION VIEWER
    path('datasets/<int:dataset_id>/features/', feature_viewer, name='feature_viewer'),
    path('datasets/<int:dataset_id>/features/export/', export_features_csv, name='export_features_csv'),
    
    # Feature Viewer - API Endpoints
    path('api/features/<int:feature_id>/', feature_detail_api, name='feature_detail_api'),
    
    # MODELS MANAGEMENT
    path('models/', models_list, name='models_list'),
    path('models/<int:model_id>/', model_detail, name='model_detail'),
    path('models/<int:model_id>/evaluation/', views.evaluation_view, name='evaluation'),
    path('models/<int:model_id>/delete/', delete_model, name='delete_model'),
    
    # Models - API Endpoints
    path('api/models/<int:model_id>/set-active/', set_active_model, name='set_active_model'),
    path('api/models/<int:model_id>/evaluation/', views.evaluation_data, name='evaluation_data'),
    
    # PREDICTION
    path('prediction/', prediction_page, name='prediction_page'),
    
    # Prediction - API Endpoints
    path('api/prediction/single/', predict_single_tweet, name='predict_single_tweet'),
    path('api/prediction/dataset/', predict_dataset, name='predict_dataset'),
    path('api/prediction/export/', export_predictions, name='export_predictions'),

    # SENTIMENT DICTIONARY
    path('dictionary/', dictionary_view, name='dictionary'),
    path('dictionary/upload/', views.upload_dictionary, name='upload_dictionary'),
    path('dictionary/extended/', views.extended_dictionary_view, name='extended_dictionary'),
    path('dictionary/reset/', reset_dictionary_all, name='reset_dictionary_all'),

    # Dictionary - API Endpoints
    path('api/dictionary/stats/', views.dictionary_statistics, name='dictionary_statistics'),
    
    # ANALYTICS & REPORTS
    path('analytics/', analytics_dashboard, name='analytics'),
    path('evaluation/dictionary/', dictionary_evaluation_view, name='dictionary_evaluation'),
]