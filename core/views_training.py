import os
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.db.models import Q
import numpy as np

from core.utils_evaluation import run_full_dictionary_evaluation
from .models import (
    Dataset,
    EvaluationMetrics,
    Prediction,
    TrainingSession,
    TrainingStepLog,
    Tweet,
    Label,
    FeatureVector,
    SVMModel,
    SentimentDictionary,
    ExtendedDictionary,
)
import json
import pickle as pkl

# ========================================
# ✅ CORRECT IMPORTS (NO MORE CLASSES!)
# ========================================
from .sentiment_calculator import (
    calculate_sentiment_scores,
    recalculate_with_extended_dict,
)
from .feature_extractor import extract_word_embeddings
from .dimensionality_reduction import apply_pca, prepare_features_for_svm
from .svm_classifier import MultiClassSVM, generate_confusion_matrix_multiclass, train_test_split
from .evaluation import calculate_metrics_multiclass
from .dictionary_extension import extend_dictionary_after_training

# ======================================================
# UTIL
# ======================================================
def step_done(dataset, step):
    return TrainingStepLog.objects.filter(dataset=dataset, step=step).exists()


def log_step(dataset, step):
    TrainingStepLog.objects.get_or_create(dataset=dataset, step=step)

# ======================================================
# DASHBOARD
# ======================================================
def training_dashboard(request, dataset_id):
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)

        # =====================
        # BASIC COUNTS
        # =====================
        train_labels = Label.objects.filter(
            tweet__dataset=dataset, dataset_split="train"
        )
        test_labels = Label.objects.filter(
            tweet__dataset=dataset, dataset_split="test"
        )

        train_count = train_labels.count()
        test_count = test_labels.count()
        labeled_count = train_labels.exclude(sentiment="").count()

        # =====================
        # FEATURE COUNTS
        # =====================
        features_count = FeatureVector.objects.filter(
            tweet__dataset=dataset,
            sentiment_score__isnull=False
        ).count()

        embeddings_count = FeatureVector.objects.filter(
            tweet__dataset=dataset,
            word_embedding__isnull=False
        ).count()

        pca_count = FeatureVector.objects.filter(
            tweet__dataset=dataset,
            additional_features__has_key="pca_features"
        ).count()

        # =====================
        # DICTIONARY COUNTS
        # =====================
        base_dict_count = SentimentDictionary.objects.count()
        extended_dict_count = ExtendedDictionary.objects.count()

        # =====================
        # MODEL INFO
        # =====================
        active_models = SVMModel.objects.filter(is_active=True).order_by("-trained_at")

        initial_model = active_models.filter(
            ~Q(name__icontains="extended")
        ).first()

        final_model = active_models.filter(
            Q(name__icontains="extended") | Q(name__icontains="final")
        ).first()

        # =====================
        # STEP LOGIC
        # =====================
        labeling_done = train_count > 0 and labeled_count == train_count

        step3_done = step_done(dataset, 3)
        step4_done = step_done(dataset, 4)
        step45_done = step_done(dataset, 45)
        step5_done = step_done(dataset, 5)
        step6_done = step_done(dataset, 6)
        step7_done = step_done(dataset, 7)

        step3_status = "completed" if step3_done else ("ready" if labeling_done else "locked")
        step4_status = "completed" if step4_done else ("ready" if step3_done else "locked")
        step45_status = "completed" if step45_done else ("ready" if step4_done else "locked")
        step5_status = "completed" if step5_done else ("ready" if step4_done else "locked")
        step6_status = "completed" if step6_done else ("ready" if step5_done else "locked")
        step7_status = "completed" if step7_done else ("ready" if step6_done else "locked")

        completed_steps = sum([
            labeling_done,
            step3_done,
            step4_done,
            step5_done,
            step6_done,
            step7_done,
        ])

        total_steps = 6  # PCA (4.5) optional
        overall_progress = round((completed_steps / total_steps) * 100, 1)
        current_step = min(completed_steps + 1, total_steps)

        # =====================
        # STEPS FOR TEMPLATE
        # =====================
        steps = [
            {
                "id": 3,
                "label": "Calculate Sentiment Scores",
                "status": step3_status,
                "locked": step3_status == "locked",
            },
            {
                "id": 4,
                "label": "Generate Word Embeddings",
                "status": step4_status,
                "locked": step4_status == "locked",
            },
            {
                "id": "4.5",
                "label": "Apply PCA (Optional)",
                "status": step45_status,
                "locked": step45_status == "locked",
            },
            {
                "id": 5,
                "label": "Train SVM Classifier",
                "status": step5_status,
                "locked": step5_status == "locked",
            },
            {
                "id": 6,
                "label": "Extend Dictionary",
                "status": step6_status,
                "locked": step6_status == "locked",
            },
            {
                "id": 7,
                "label": "Retrain with Extended Dictionary",
                "status": step7_status,
                "locked": step7_status == "locked",
            },
        ]

        # =====================
        # CONTEXT
        # =====================
        context = {
            "dataset": dataset,
            "train_count": train_count,
            "test_count": test_count,
            "labeled_count": labeled_count,
            "features_count": features_count,
            "embeddings_count": embeddings_count,
            "pca_count": pca_count,
            "base_dict_count": base_dict_count,
            "extended_dict_count": extended_dict_count,

            "model_exists": initial_model is not None,
            "model_accuracy": initial_model.train_accuracy if initial_model else 0,
            "final_model_exists": final_model is not None,
            "final_model_accuracy": final_model.train_accuracy if final_model else 0,

            "overall_progress": overall_progress,
            "completed_steps": completed_steps,
            "current_step": current_step,

            "step5_locked": step5_status == "locked",
            "steps": steps,
        }

        return render(request, "core/training_dashboard.html", context)

    except Exception as e:
        messages.error(request, f"Error loading dashboard: {e}")
        return redirect("core:dataset_list")



# =============================================================================
# STEP 3: Calculate Sentiment Scores
# =============================================================================


def execute_step_sentiment_scores(request, dataset_id):
    """
    Step 3: Calculate sentiment scores for all labeled tweets
    URL: api/datasets/<id>/training/step3/
    """
    if request.method == "POST":
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id)

            print(f"[INFO] STEP 3: Calculate Sentiment Scores")
            print(f"[INFO] Dataset: {dataset.name}")

            # ✅ Call FUNCTION directly (no class!)
            success, error, message = calculate_sentiment_scores(
                dataset_id=dataset_id, use_extended=False
            )

            if success > 0:
                log_step(dataset, 3)
                return JsonResponse(
                    {
                        "success": True,
                        "message": f"Calculated scores for {success} tweets",
                        "details": {
                            "processed": success,
                            "errors": error,
                            "method": "Base dictionary only",
                        },
                    }
                )
            else:
                return JsonResponse({"success": False, "message": message}, status=500)

        except Exception as e:
            import traceback

            traceback.print_exc()
            return JsonResponse({"success": False, "message": str(e)}, status=500)

    return JsonResponse(
        {"success": False, "message": "Invalid request method"}, status=405
    )


# =============================================================================
# STEP 4: Generate Word Embeddings
# =============================================================================


def execute_step_word_embeddings(request, dataset_id):
    """
    Step 4: Generate word embeddings using FastText
    URL: api/datasets/<id>/training/step4/
    """
    if request.method == "POST":
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id)

            print(f"[INFO] STEP 4: Generate Word Embeddings")
            print(f"[INFO] Dataset: {dataset.name}")

            # ✅ Call FUNCTION directly
            success, error, message = extract_word_embeddings(
                dataset_id=dataset_id,
                split="all",  # Generate for all labeled tweets (train + test)
                use_extended=False,
            )

            if success > 0:
                log_step(dataset, 4)
                return JsonResponse(
                    {
                        "success": True,
                        "message": f"Generated embeddings for {success} tweets",
                        "details": {
                            "processed": success,
                            "errors": error,
                            "method": "Weighted average FastText + Sentiment",
                            "dimensions": 100,
                        },
                    }
                )
            else:
                return JsonResponse({"success": False, "message": message}, status=500)

        except Exception as e:
            import traceback

            traceback.print_exc()
            return JsonResponse({"success": False, "message": str(e)}, status=500)

    return JsonResponse(
        {"success": False, "message": "Invalid request method"}, status=405
    )


# =============================================================================
# STEP 4.5: Apply PCA/Dimensionality Reduction (OPTIONAL)
# =============================================================================


def execute_step_pca(request, dataset_id):
    """
    Step 4.5: Apply PCA/Dimensionality Reduction (OPTIONAL)
    URL: api/datasets/<id>/training/step4-5/
    """
    if request.method == "POST":
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id)

            print(f"[INFO] STEP 4.5: Apply Dimensionality Reduction")
            print(f"[INFO] Dataset: {dataset.name}")

            # ✅ Call FUNCTION directly
            success, n_components, variance, message = apply_pca(
                dataset_id=dataset_id, target_contribution_rate=0.98
            )

            if success:
                log_step(dataset, 45)
                return JsonResponse(
                    {
                        "success": True,
                        "message": message,
                        "details": {
                            "n_components": n_components,
                            "variance_retained": f"{variance:.2f}%",
                        },
                    }
                )
            else:
                return JsonResponse({"success": False, "message": message}, status=500)

        except Exception as e:
            import traceback

            traceback.print_exc()
            return JsonResponse({"success": False, "message": str(e)}, status=500)

    return JsonResponse(
        {"success": False, "message": "Invalid request method"}, status=405
    )


# =============================================================================
# STEP 5: Train SVM Classifier
# =============================================================================


def execute_step_train_svm(request, dataset_id):
    """
    Step 5: Train initial SVM classifier (FULL SCRATCH)
    """
    if request.method == "POST":
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id)

            print(f"[INFO] STEP 5: Train SVM Classifier")
            print(f"[INFO] Dataset: {dataset.name}")

            # ===============================
            # 1. Prepare features
            # ===============================
            X, y, feature_names = prepare_features_for_svm(
                dataset_id=dataset_id, split="train", use_pca=True
            )

            if X is None or len(X) == 0:
                return JsonResponse(
                    {"success": False, "message": "No training data available"},
                    status=500,
                )

            # ===============================
            # 2. Train–test split (SCRATCH)
            # ===============================
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ===============================
            # 3. Train model (SCRATCH)
            # ===============================
            model = MultiClassSVM(C=1.0, max_iter=1000, class_weight='balanced')
            model.fit(X_train, y_train)

            # ===============================
            # 4. Prediction
            # ===============================
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # ===============================
            # 5. Evaluation (SCRATCH, RAPIH)
            # ===============================
            train_metrics = calculate_metrics_multiclass(y_train, y_pred_train)
            test_metrics = calculate_metrics_multiclass(y_test, y_pred_test)

            # ===============================
            # 6. Save model metadata
            # ===============================
            # Save model to pickle file
            model_dir = settings.MODELS_DIR
            os.makedirs(model_dir, exist_ok=True)

            model_filename = f"model_{dataset.name}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = os.path.join(model_dir, model_filename)

            with open(model_path, 'wb') as f:
                pkl.dump(model, f)

            print(f"[INFO] Model saved to {model_path}")

            # Save to database with file path
            svm_model = SVMModel.objects.create(
                name=f"SVM_Initial_{dataset.name}",
                dataset=dataset,
                description="Initial SVM model (scratch)",
                hyperparameters={
                    "C": 1.0,
                    "kernel": "linear",
                    "use_pca": True,
                    "dictionary": "base",
                    "model_type": "MultiClassSVM",
                    "model_file": model_filename  # ← Save filename
                },
                train_accuracy = round(train_metrics["accuracy"], 4),
                test_accuracy  = round(test_metrics["accuracy"], 4),
                is_active=True,
            )


            # ===============================
            # 7. Save evaluation metrics
            # ===============================
            cm, class_names = generate_confusion_matrix_multiclass(y_test, y_pred_test)
            # Save evaluation metrics
            EvaluationMetrics.objects.create(
                model=svm_model,
                accuracy=test_metrics["accuracy"],
                precision=test_metrics["precision"],
                recall=test_metrics["recall"],
                f1_score=test_metrics["f1_score"],
                confusion_matrix=cm.tolist(),
                classification_report=test_metrics["per_class"],
            )

            # ===============================
            # 8. Log step
            # ===============================
            log_step(dataset, 5)

            return JsonResponse(
                {
                    "success": True,
                    "message": "SVM trained successfully (scratch)",
                    "details": {
                        "train_accuracy": train_metrics["accuracy"],
                        "test_accuracy": test_metrics["accuracy"],
                    },
                }
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return JsonResponse({"success": False, "message": str(e)}, status=500)

    return JsonResponse(
        {"success": False, "message": "Invalid request method"}, status=405
    )


# =============================================================================
# STEP 6: Extend Dictionary (PLACEHOLDER - file not provided)
# =============================================================================


def execute_step_extend_dictionary(request, dataset_id):
    """
    Step 6: Extend sentiment dictionary using trained model
    URL: api/datasets/<id>/training/step6/
    """
    if request.method == "POST":
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id)

            print(f"[INFO] STEP 6: Extend Dictionary")
            print(f"[INFO] Dataset: {dataset.name}")

            # 🔎 Jalankan dictionary extension (FUNCTION ASLI KAMU)
            success, extended_count, skipped_count, message = (
                extend_dictionary_after_training(dataset_id)
            )

            # ❌ GAGAL → JANGAN CATAT LOG
            if not success:
                return JsonResponse({"success": False, "message": message}, status=500)

            # ❌ TIDAK ADA KATA BARU → JANGAN CATAT LOG
            if extended_count == 0:
                return JsonResponse(
                    {
                        "success": False,
                        "message": "Dictionary extension finished but no new words were added",
                    },
                    status=500,
                )

            # ✅ BERHASIL → CATAT STEP 6
            log_step(dataset, 6)

            return JsonResponse(
                {
                    "success": True,
                    "message": message,
                    "details": {
                        "extended_words": extended_count,
                        "skipped_words": skipped_count,
                    },
                }
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return JsonResponse({"success": False, "message": str(e)}, status=500)

    return JsonResponse(
        {"success": False, "message": "Invalid request method"}, status=405
    )


# =============================================================================
# STEP 7: Retrain with Extended Dictionary
# =============================================================================


def execute_step_retrain_with_extended(request, dataset_id):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "Invalid request"}, status=405)

    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)

        print("[STEP 7] Retrain with Extended Dictionary")

        # =====================================================
        # 1. Recalculate sentiment scores (EXTENDED)
        # =====================================================
        success, error, msg = calculate_sentiment_scores(
            dataset_id=dataset_id,
            use_extended=True
        )
        if success == 0:
            return JsonResponse({"success": False, "message": msg}, status=500)

        # =====================================================
        # 2. Regenerate embeddings (EXTENDED)
        # =====================================================
        success, error, msg = extract_word_embeddings(
            dataset_id=dataset_id,
            split="all",
            use_extended=True
        )
        if success == 0:
            return JsonResponse({"success": False, "message": msg}, status=500)

        # 🔐 PENTING: log ulang step 4
        log_step(dataset, 4)

        # =====================================================
        # 3. Apply PCA ULANG
        # =====================================================
        pca_success, n_comp, var, msg = apply_pca(
            dataset_id=dataset_id,
            target_contribution_rate=0.98
        )
        if not pca_success:
            return JsonResponse({"success": False, "message": msg}, status=500)

        log_step(dataset, 45)

        # =====================================================
        # 4. Prepare features for SVM
        # =====================================================
        X, y, _ = prepare_features_for_svm(
            dataset_id=dataset_id,
            split="train",
            use_pca=True
        )

        if X is None or len(X) == 0:
            return JsonResponse({"success": False, "message": "No training data"}, status=500)

        # =====================================================
        # 5. Train FINAL SVM
        # =====================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = MultiClassSVM(C=1.0, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = calculate_metrics_multiclass(y_train, y_pred_train)
        test_metrics = calculate_metrics_multiclass(y_test, y_pred_test)

        # Save model to pickle file
        model_dir = settings.MODELS_DIR
        os.makedirs(model_dir, exist_ok=True)

        model_filename = f"model_{dataset.name}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = os.path.join(model_dir, model_filename)

        with open(model_path, 'wb') as f:
            pkl.dump(model, f)

        print(f"[INFO] Model saved to {model_path}")

        # Save to database with file path
        svm_model = SVMModel.objects.create(
            name=f"SVM_Final_Extended_{dataset.name}",
            dataset=dataset,
            description="Final SVM model with Extended Dictionary",
            hyperparameters={
                "C": 1.0,
                "kernel": "linear",
                "use_pca": True,
                "dictionary": "extended",
                "model_type": "MultiClassSVM",
                "model_file": model_filename  # ← Save filename
            },
            train_accuracy = round(train_metrics["accuracy"], 4),
            test_accuracy  = round(test_metrics["accuracy"], 4),
            is_active=True,
        )

        cm, class_names = generate_confusion_matrix_multiclass(y_test, y_pred_test)
        # Save evaluation metrics
        EvaluationMetrics.objects.create(
            model=svm_model,
            accuracy=test_metrics["accuracy"],
            precision=test_metrics["precision"],
            recall=test_metrics["recall"],
            f1_score=test_metrics["f1_score"],
            confusion_matrix=cm.tolist(),
            classification_report=test_metrics["per_class"],
        )


        # =====================================================
        # 6. LOG STEP 7 (TERAKHIR)
        # =====================================================
        log_step(dataset, 7)

        return JsonResponse({
            "success": True,
            "message": "Final model trained successfully with extended dictionary",
            "details": {
                "train_accuracy": train_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "pca_components": n_comp
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "message": str(e)}, status=500)

# =============================================================================
# UTILITY APIS
# =============================================================================


def get_training_progress(request, dataset_id):
    """
    Get training progress for AJAX polling
    URL: api/datasets/<id>/training/progress/
    """
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)

        train_count = Label.objects.filter(
            tweet__dataset_id=dataset_id, dataset_split="train"
        ).count()

        labeled_count = (
            Label.objects.filter(tweet__dataset_id=dataset_id, dataset_split="train")
            .exclude(sentiment="")
            .count()
        )

        features_count = FeatureVector.objects.filter(
            tweet__dataset_id=dataset_id, sentiment_score__isnull=False
        ).count()

        embeddings_count = FeatureVector.objects.filter(
            tweet__dataset_id=dataset_id, word_embedding__isnull=False
        ).count()

        # Calculate progress percentages
        labeling_progress = (
            (labeled_count / train_count * 100) if train_count > 0 else 0
        )
        features_progress = (
            (features_count / train_count * 100) if train_count > 0 else 0
        )
        embeddings_progress = (
            (embeddings_count / train_count * 100) if train_count > 0 else 0
        )

        progress = {
            "labeling": {
                "count": labeled_count,
                "total": train_count,
                "percentage": round(labeling_progress, 1),
            },
            "features": {
                "count": features_count,
                "total": train_count,
                "percentage": round(features_progress, 1),
            },
            "embeddings": {
                "count": embeddings_count,
                "total": train_count,
                "percentage": round(embeddings_progress, 1),
            },
        }

        return JsonResponse({"success": True, "progress": progress})

    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=500)


def get_training_status(request, dataset_id):
    """
    Get current training status for reset modal
    URL: api/datasets/<id>/training/status/
    """
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)

        # Count labels
        train_labels = Label.objects.filter(
            tweet__dataset_id=dataset_id, dataset_split="train"
        ).count()

        test_labels = Label.objects.filter(
            tweet__dataset_id=dataset_id, dataset_split="test"
        ).count()

        # Count features
        feature_vectors = FeatureVector.objects.filter(
            tweet__dataset_id=dataset_id
        ).count()

        # Count models
        svm_models = SVMModel.objects.filter(is_active=True).count()

        # Count extended dictionary
        extended_dict = ExtendedDictionary.objects.count()

        status = {
            "labels": {
                "total": train_labels + test_labels,
                "train": train_labels,
                "test": test_labels,
            },
            "feature_vectors": feature_vectors,
            "svm_models": svm_models,
            "extended_dictionary": extended_dict,
        }

        return JsonResponse({"success": True, "status": status})

    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=500)


def reset_training_data(request, dataset_id):
    """
    Reset training progress based on selected options
    URL: api/datasets/<id>/training/reset/
    """
    if request.method == "POST":
        try:
            from django.db import transaction
            import json

            dataset = get_object_or_404(Dataset, id=dataset_id)
            data = json.loads(request.body)

            keep_labels = data.get("keep_labels", True)
            reset_features = data.get("reset_features", True)
            reset_models = data.get("reset_models", True)
            reset_extended_dict = data.get("reset_extended_dict", True)

            summary = []

            with transaction.atomic():

                # ===============================
                # RESET MODELS
                # ===============================
                if reset_models:
                    model_count = SVMModel.objects.filter(is_active=True).count()
                    SVMModel.objects.filter(is_active=True).update(is_active=False)
                    summary.append(f"Deactivated {model_count} models")

                # ===============================
                # RESET FEATURE VECTORS
                # ===============================
                if reset_features:
                    fv_count = FeatureVector.objects.filter(
                        tweet__dataset_id=dataset_id
                    ).count()
                    FeatureVector.objects.filter(
                        tweet__dataset_id=dataset_id
                    ).delete()
                    summary.append(f"Deleted {fv_count} feature vectors")

                # ===============================
                # RESET EXTENDED DICTIONARY
                # ===============================
                if reset_extended_dict:
                    ext_count = ExtendedDictionary.objects.count()
                    ExtendedDictionary.objects.all().delete()
                    summary.append(f"Deleted {ext_count} extended dictionary entries")

                # ===============================
                # RESET LABELS (OPTIONAL)
                # ===============================
                if not keep_labels:
                    label_count = Label.objects.filter(
                        tweet__dataset_id=dataset_id
                    ).count()
                    Label.objects.filter(tweet__dataset_id=dataset_id).delete()
                    Tweet.objects.filter(dataset_id=dataset_id).update(
                        is_sampled=False,
                        sampled_at=None
                    )
                    summary.append(f"Deleted {label_count} labels")

                # ===============================
                # 🔴 RESET TRAINING STEP LOGS (KRITIS)
                # ===============================
                step_count = TrainingStepLog.objects.filter(dataset=dataset).count()
                TrainingStepLog.objects.filter(dataset=dataset).delete()
                summary.append(f"Reset {step_count} training steps")

            return JsonResponse({
                "success": True,
                "message": "Training reset successfully",
                "summary": summary
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"success": False, "message": str(e)}, status=500)

    return JsonResponse(
        {"success": False, "message": "Invalid request method"}, status=405
    )

@csrf_exempt
@require_http_methods(["POST"])
def execute_step_predict_test(request, dataset_id):
    """
    Step: Predict Test Data
    Prediksi sentimen untuk data testing menggunakan trained model
    """
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        print("\n" + "=" * 80)
        print(f"[INFO] ===== STEP: PREDICT TEST DATA =====")
        print(f"[INFO] Dataset: {dataset.name}")
        print("=" * 80)
        
        # ============================
        # 1. LOAD TRAINED MODEL
        # ============================
        print("\n[STEP 1] Loading trained SVM model...")
        
        # Ambil model terakhir yang sudah trained (prioritas extended)
        model_record = SVMModel.objects.filter(
            name__icontains='extended'
        ).order_by('-trained_at').first()
        
        if not model_record:
            # Fallback ke model base jika extended belum ada
            model_record = SVMModel.objects.order_by('-trained_at').first()
        
        if not model_record:
            return JsonResponse({
                'status': 'error',
                'message': 'No trained model found. Please train SVM first.'
            }, status=400)
        
        print(f"[INFO] Model loaded: {model_record.name}")
        print(f"[INFO] Trained at: {model_record.trained_at}")
        print(f"[INFO] Support vectors: {model_record.support_vectors}")
        
        # Load weights & bias dari database (FULL SCRATCH - NO PKL FILE)
        weights = np.array(model_record.weights) # type: ignore
        bias = model_record.bias # type: ignore
        
        print(f"[INFO] Weights shape: {weights.shape}")
        print(f"[INFO] Bias: {bias:.4f}")
        
        # ============================
        # 2. LOAD TEST DATA
        # ============================
        print("\n[STEP 2] Loading test data...")
        
        # Ambil semua data testing yang sudah dilabel
        test_labels = Label.objects.filter(
            tweet__dataset=dataset,
            dataset_split='test'  # Hanya data testing
        ).select_related('tweet')
        
        if test_labels.count() == 0:
            return JsonResponse({
                'status': 'error',
                'message': 'No test data found. Please check sampling process.'
            }, status=400)
        
        print(f"[INFO] Total test samples: {test_labels.count()}")
        
        # ============================
        # 3. PREPARE FEATURES
        # ============================
        print("\n[STEP 3] Preparing test features...")
        
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        reverse_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        X_test = []
        y_true = []
        tweet_data = []
        
        processed = 0
        errors = 0
        
        for label_obj in test_labels:
            try:
                # Ambil feature vector
                fv = FeatureVector.objects.get(tweet=label_obj.tweet)
                
                # Gunakan final_vector (setelah PCA) jika ada, atau embedding_vector
                if fv.final_vector: # type: ignore
                    features = np.array(fv.final_vector) # type: ignore
                elif fv.embedding_vector: # type: ignore
                    features = np.array(fv.embedding_vector) # type: ignore
                else:
                    print(f"[WARNING] No features for tweet {label_obj.tweet.id}")
                    errors += 1
                    continue
                
                X_test.append(features)
                y_true.append(label_map[label_obj.sentiment])
                tweet_data.append({
                    'tweet_obj': label_obj.tweet,  # Simpan object Tweet
                    'actual_sentiment': label_obj.sentiment
                })
                
                processed += 1
                
                # Progress update
                if processed % 20 == 0:
                    print(f"[PROGRESS] {processed}/{test_labels.count()} features prepared...")
                
            except FeatureVector.DoesNotExist:
                print(f"[WARNING] FeatureVector not found for tweet {label_obj.tweet.id}")
                errors += 1
                continue
            except Exception as e:
                print(f"[ERROR] Error processing tweet {label_obj.tweet.id}: {str(e)}")
                errors += 1
                continue
        
        if len(X_test) == 0:
            return JsonResponse({
                'status': 'error',
                'message': 'No valid feature vectors found for test data.'
            }, status=400)
        
        X_test = np.array(X_test)
        y_true = np.array(y_true)
        
        print(f"[INFO] Features prepared: {len(X_test)}")
        print(f"[INFO] Errors: {errors}")
        print(f"[INFO] Feature shape: {X_test.shape}")
        
        # ============================
        # 4. PREDICTION (Manual SVM)
        # ============================
        print("\n[STEP 4] Making predictions...")
        
        def predict_svm(X, weights, bias):
            """
            Manual SVM prediction (full scratch)
            Returns predictions and decision scores
            """
            # Decision function: f(x) = w·x + b
            scores = np.dot(X, weights.T) + bias
            
            # Untuk multi-class One-vs-Rest, pilih class dengan score tertinggi
            if len(scores.shape) > 1:
                predictions = np.argmax(scores, axis=1)
                # Decision value = max score
                decision_values = np.max(scores, axis=1)
            else:
                # Binary classification
                predictions = (np.sign(scores) + 1) / 2  # Convert -1,1 to 0,1
                predictions = predictions.astype(int)
                decision_values = np.abs(scores)
            
            return predictions, scores, decision_values
        
        y_pred, decision_scores, decision_values = predict_svm(X_test, weights, bias)
        
        print(f"[INFO] Predictions completed for {len(y_pred)} samples")
        
        # ============================
        # 5. CALCULATE METRICS
        # ============================
        print("\n[STEP 5] Calculating evaluation metrics...")
        
        def calculate_metrics(y_true, y_pred):
            """Calculate accuracy, precision, recall, F1-score manually"""
            # Confusion Matrix
            cm = np.zeros((3, 3), dtype=int)
            for i in range(len(y_true)):
                cm[y_true[i]][y_pred[i]] += 1
            
            # Accuracy
            accuracy = np.sum(y_true == y_pred) / len(y_true)
            
            # Per-class metrics
            metrics = {}
            for class_idx in range(3):
                tp = cm[class_idx][class_idx]
                fp = np.sum(cm[:, class_idx]) - tp
                fn = np.sum(cm[class_idx, :]) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                support = int(np.sum(cm[class_idx, :]))
                
                metrics[class_idx] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': support
                }
            
            # Weighted average
            total_support = sum([m['support'] for m in metrics.values()])
            weighted_precision = sum([m['precision'] * m['support'] for m in metrics.values()]) / total_support if total_support > 0 else 0
            weighted_recall = sum([m['recall'] * m['support'] for m in metrics.values()]) / total_support if total_support > 0 else 0
            weighted_f1 = sum([m['f1'] * m['support'] for m in metrics.values()]) / total_support if total_support > 0 else 0
            
            return accuracy, weighted_precision, weighted_recall, weighted_f1, cm, metrics
        
        accuracy, precision, recall, f1, cm, class_metrics = calculate_metrics(y_true, y_pred)
        
        # Print metrics
        print(f"\n{'='*80}")
        print(f"EVALUATION METRICS - TEST DATA")
        print(f"{'='*80}")
        print(f"\n{'Metric':<25} {'Score':<15} {'Percentage'}")
        print(f"{'-'*55}")
        print(f"{'Accuracy':<25} {accuracy:<15.4f} {accuracy*100:.2f}%")
        print(f"{'Precision (Weighted)':<25} {precision:<15.4f} {precision*100:.2f}%")
        print(f"{'Recall (Weighted)':<25} {recall:<15.4f} {recall*100:.2f}%")
        print(f"{'F1-Score (Weighted)':<25} {f1:<15.4f} {f1*100:.2f}%")
        
        print(f"\n{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        print(f"{'-'*58}")
        for ci, m in class_metrics.items():
            print(f"{reverse_map[ci]:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']}")
        
        print(f"\nConfusion Matrix:")
        print(f"           Negative  Neutral  Positive")
        for i, row in enumerate(cm):
            print(f"{reverse_map[i]:<10} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
        
        # ============================
        # 6. SAVE PREDICTIONS
        # ============================
        print(f"\n[STEP 6] Saving predictions to database...")
        
        saved_count = 0
        updated_count = 0
        
        for i in range(len(y_pred)):
            try:
                # Hitung confidence score dari decision scores
                if len(decision_scores.shape) > 1:
                    # Multi-class: Softmax-like confidence
                    exp_scores = np.exp(decision_scores[i] - np.max(decision_scores[i]))
                    softmax_probs = exp_scores / np.sum(exp_scores)
                    confidence = float(softmax_probs[y_pred[i]])
                else:
                    # Binary: Sigmoid-like
                    confidence = float(1 / (1 + np.exp(-decision_values[i])))
                
                # Save atau update prediction sesuai model Anda
                prediction, created = Prediction.objects.update_or_create(
                    tweet=tweet_data[i]['tweet_obj'],  # OneToOneField ke Tweet
                    defaults={
                        'model': model_record,  # ForeignKey ke SVMModel
                        'predicted_sentiment': reverse_map[y_pred[i]],
                        'confidence_score': confidence,
                        'decision_value': float(decision_values[i]),
                        'predicted_at': timezone.now()
                    }
                )
                
                if created:
                    saved_count += 1
                else:
                    updated_count += 1
                
                if (saved_count + updated_count) % 20 == 0:
                    print(f"[PROGRESS] {saved_count + updated_count}/{len(y_pred)} predictions saved...")
                
            except Exception as e:
                print(f"[ERROR] Error saving prediction for tweet {tweet_data[i]['tweet_obj'].id}: {str(e)}")
                continue
            
        run_full_dictionary_evaluation()

        print(f"[INFO] New predictions: {saved_count}")
        print(f"[INFO] Updated predictions: {updated_count}")
        
        # ============================
        # 7. UPDATE MODEL TEST ACCURACY
        # ============================
        model_record.test_accuracy = round(accuracy, 4)
        model_record.save()
        
        print("\n" + "=" * 80)
        print(f"[PREDICT TEST] COMPLETE!")
        print(f"[INFO] Total predictions: {saved_count + updated_count}")
        print(f"[INFO] Accuracy: {accuracy*100:.2f}%")
        print(f"[INFO] Precision: {precision*100:.2f}%")
        print(f"[INFO] Recall: {recall*100:.2f}%")
        print(f"[INFO] F1-Score: {f1*100:.2f}%")
        print(f"[INFO] Errors: {errors}")
        print("=" * 80)
        
        return JsonResponse({
            'status': 'success',
            'message': f'Test data predicted successfully!',
            'data': {
                'total_predictions': saved_count + updated_count,
                'new': saved_count,
                'updated': updated_count,
                'errors': errors,
                'metrics': {
                    'accuracy': round(accuracy * 100, 2),
                    'precision': round(precision * 100, 2),
                    'recall': round(recall * 100, 2),
                    'f1_score': round(f1 * 100, 2),
                    'confusion_matrix': cm.tolist(),
                    'per_class': {
                        reverse_map[k]: {
                            'precision': round(v['precision'] * 100, 2),
                            'recall': round(v['recall'] * 100, 2),
                            'f1_score': round(v['f1'] * 100, 2),
                            'support': v['support']
                        }
                        for k, v in class_metrics.items()
                    }
                }
            }
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n[ERROR] {str(e)}")
        print(error_trace)
        
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def evaluate_model(request, dataset_id):
    """
    Evaluasi model dengan training atau testing data
    """
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        data = json.loads(request.body)
        eval_type = data.get('type', 'train')  # 'train' or 'test'
        
        print(f"\n{'='*80}")
        print(f"[EVALUATION] Evaluating model on {eval_type} data")
        print(f"[EVALUATION] Dataset: {dataset.name}")
        print(f"{'='*80}")
        
        # Load model
        model_record = SVMModel.objects.filter(
            name__icontains='extended'
        ).order_by('-trained_at').first()
        
        if not model_record:
            model_record = SVMModel.objects.order_by('-trained_at').first()
        
        if not model_record:
            return JsonResponse({
                'status': 'error',
                'message': 'No trained model found'
            }, status=400)
        
        # Load data
        labels = Label.objects.filter(
            tweet__dataset=dataset,
            dataset_split=eval_type
        ).select_related('tweet')
        
        if labels.count() == 0:
            return JsonResponse({
                'status': 'error',
                'message': f'No {eval_type} data found'
            }, status=400)
        
        print(f"[INFO] Total samples: {labels.count()}")
        
        # Prepare features
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        reverse_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        X = []
        y_true = []
        
        for label_obj in labels:
            try:
                fv = FeatureVector.objects.get(tweet=label_obj.tweet)
                features = np.array(fv.final_vector if fv.final_vector else fv.embedding_vector) # type: ignore
                X.append(features)
                y_true.append(label_map[label_obj.sentiment])
            except:
                continue
        
        X = np.array(X)
        y_true = np.array(y_true)
        
        # Predict
        weights = np.array(model_record.weights) # type: ignore
        bias = model_record.bias # type: ignore
        
        scores = np.dot(X, weights.T) + bias
        y_pred = np.argmax(scores, axis=1) if len(scores.shape) > 1 else np.sign(scores).astype(int)
        
        # Calculate metrics
        def calc_metrics(yt, yp):
            cm = np.zeros((3, 3), dtype=int)
            for i in range(len(yt)):
                cm[yt[i]][yp[i]] += 1
            
            acc = np.sum(yt == yp) / len(yt)
            
            metrics = {}
            for ci in range(3):
                tp = cm[ci][ci]
                fp = np.sum(cm[:, ci]) - tp
                fn = np.sum(cm[ci, :]) - tp
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                
                metrics[ci] = {'precision': p, 'recall': r, 'f1': f, 'support': int(np.sum(cm[ci, :]))}
            
            ts = sum([m['support'] for m in metrics.values()])
            wp = sum([m['precision'] * m['support'] for m in metrics.values()]) / ts
            wr = sum([m['recall'] * m['support'] for m in metrics.values()]) / ts
            wf = sum([m['f1'] * m['support'] for m in metrics.values()]) / ts
            
            return acc, wp, wr, wf, cm, metrics
        
        acc, prec, rec, f1, cm, class_metrics = calc_metrics(y_true, y_pred)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"[RESULTS] {eval_type.upper()} DATA EVALUATION")
        print(f"{'='*80}")
        print(f"\n{'Metric':<20} {'Score':<15} {'Percentage'}")
        print(f"{'-'*50}")
        print(f"{'Accuracy':<20} {acc:<15.4f} {acc*100:.2f}%")
        print(f"{'Precision':<20} {prec:<15.4f} {prec*100:.2f}%")
        print(f"{'Recall':<20} {rec:<15.4f} {rec*100:.2f}%")
        print(f"{'F1-Score':<20} {f1:<15.4f} {f1*100:.2f}%")
        
        print(f"\n{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        print(f"{'-'*60}")
        for ci, m in class_metrics.items():
            print(f"{reverse_map[ci]:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']}")
        
        print(f"\n{'Confusion Matrix':^40}")
        print(f"           Negative  Neutral  Positive")
        for i, row in enumerate(cm):
            print(f"{reverse_map[i]:<10} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
        
        print(f"\n{'='*80}\n")
        
        return JsonResponse({
            'status': 'success',
            'type': eval_type,
            'metrics': {
                'accuracy': round(acc * 100, 2),
                'precision': round(prec * 100, 2),
                'recall': round(rec * 100, 2),
                'f1_score': round(f1 * 100, 2),
                'confusion_matrix': cm.tolist(),
                'per_class': {
                    reverse_map[k]: {
                        'precision': round(v['precision'] * 100, 2),
                        'recall': round(v['recall'] * 100, 2),
                        'f1_score': round(v['f1'] * 100, 2),
                        'support': v['support']
                    }
                    for k, v in class_metrics.items()
                }
            }
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
