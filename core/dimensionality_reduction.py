# File: core/dimensionality_reduction.py
import numpy as np
from .models import FeatureVector, Label

def apply_pca(dataset_id, target_contribution_rate=0.98):
    """
    Apply PCA Dimensionality Reduction (Section III.D jurnal)
    
    Formula 4-10:
    1. Calculate mean (Formula 5)
    2. Covariance matrix (Formula 6-7)
    3. Eigen decomposition (Formula 8)
    4. Sort eigenvalues (Formula 9)
    5. Select k components (Formula 21: CR ≥ 98%)
    6. Project to lower dimension (Formula 10)
    
    Args:
        dataset_id: ID dataset
        target_contribution_rate: Target variance (default: 0.98 = 98%)
    
    Returns:
        tuple: (success, n_components, variance_explained, message)
    """
    try:
        # ===== STEP 1: LOAD FEATURE VECTORS =====
        feature_vectors = FeatureVector.objects.filter(
            tweet__dataset_id=dataset_id,
            sentiment_score__isnull=False,
            word_embedding__isnull=False
        ).select_related('tweet')
        
        if feature_vectors.count() < 50:
            return (False, 0, 0.0, "Not enough samples for PCA (need ≥50)")
        
        print(f"[PCA STEP 1] Loading {feature_vectors.count()} feature vectors...")
        
        # ===== STEP 2: BUILD FEATURE MATRIX X =====
        X = []
        fv_ids = []
        
        for fv in feature_vectors:
            # Original features: 3 sentiment scores + 100 embedding = 103 dims
            features = [
                fv.sentiment_score if fv.sentiment_score else 0.0,
                fv.positive_score if fv.positive_score else 0.0,
                fv.negative_score if fv.negative_score else 0.0,
            ]
            
            # Add word embedding (100 dims)
            if fv.word_embedding and 'vector' in fv.word_embedding:
                embedding_vector = fv.word_embedding['vector']
                features.extend(embedding_vector)
            else:
                features.extend([0.0] * 100)
            
            X.append(features)
            fv_ids.append(fv.id) # type: ignore
        
        X = np.array(X, dtype=np.float64)  # Shape: (n_samples, 103)
        n_samples, m_features = X.shape
        
        print(f"[PCA STEP 2] Feature matrix: {X.shape}")
        print(f"[PCA STEP 2] Original dimensions: {m_features}")
        
        # ===== STEP 3: CALCULATE MEAN (FORMULA 5) =====
        X_mean = np.mean(X, axis=0)  # Shape: (103,)
        
        # ===== STEP 4: CENTER THE DATA =====
        X_centered = X - X_mean
        
        print(f"[PCA STEP 3-4] Data centered (mean={X_mean[:3]}...)")
        
        # ===== STEP 5: COVARIANCE MATRIX (FORMULA 6-7) =====
        C = np.cov(X_centered.T)  # Shape: (103, 103)
        
        print(f"[PCA STEP 5] Covariance matrix: {C.shape}")
        
        # ===== STEP 6: EIGEN DECOMPOSITION (FORMULA 8) =====
        eigenvalues, eigenvectors = np.linalg.eig(C)
        
        # Convert to real numbers
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        print(f"[PCA STEP 6] Eigenvalues computed: {len(eigenvalues)}")
        
        # ===== STEP 7: SORT EIGENVALUES (FORMULA 9) =====
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"[PCA STEP 7] Top 5 eigenvalues: {eigenvalues[:5]}")
        
        # ===== STEP 8: SELECT k COMPONENTS (FORMULA 21) =====
        cumsum_eigenvalues = np.cumsum(eigenvalues)
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = cumsum_eigenvalues / total_variance
        
        n_components = np.argmax(explained_variance_ratio >= target_contribution_rate) + 1
        n_components = max(n_components, 10)
        n_components = min(n_components, m_features - 1)
        
        variance_explained = explained_variance_ratio[n_components - 1]
        
        print(f"[PCA STEP 8] Contribution Rate: {variance_explained*100:.2f}%")
        print(f"[PCA STEP 8] Selected components: {n_components} (from {m_features})")
        
        # ===== STEP 9: PROJECT TO LOWER DIMENSION (FORMULA 10) =====
        U = eigenvectors[:, :n_components]  # Shape: (103, k)
        X_reduced = np.dot(X_centered, U)   # Shape: (n_samples, k)
        
        print(f"[PCA STEP 9] Reduced feature matrix: {X_reduced.shape}")
        
        # ===== STEP 10: SAVE TO DATABASE =====
        print(f"[PCA STEP 10] Updating {len(fv_ids)} feature vectors...")
        
        updated_count = 0
        for i, fv_id in enumerate(fv_ids):
            fv = FeatureVector.objects.get(id=fv_id)
            
            if not fv.additional_features:
                fv.additional_features = {}
            
            fv.additional_features['pca_features'] = X_reduced[i].tolist()
            fv.additional_features['pca_dims'] = n_components
            fv.additional_features['pca_variance'] = float(variance_explained)
            fv.additional_features['original_dims'] = m_features
            
            fv.save(update_fields=['additional_features'])
            updated_count += 1
            
            if updated_count % 100 == 0:
                print(f"[PROGRESS] {updated_count}/{len(fv_ids)} updated...")
        
        # ========================================
        # ✅ STEP 11: SAVE PCA MODEL FOR PREDICTION
        # ========================================
        print(f"\n[PCA STEP 11] Saving PCA transformation for prediction...")
        
        import pickle
        import os
        from django.conf import settings
        from .models import Dataset
        
        dataset = Dataset.objects.get(id=dataset_id)
        pca_dir = settings.MODELS_DIR
        os.makedirs(pca_dir, exist_ok=True)
        
        pca_file = os.path.join(pca_dir, f'pca_{dataset.name}.pkl')
        
        pca_data = {
            'mean': X_mean.tolist(),
            'components': U.tolist(),
            'n_components': n_components,
            'variance_explained': float(variance_explained),
            'original_dims': m_features,
            'eigenvalues': eigenvalues[:n_components].tolist()
        }
        
        with open(pca_file, 'wb') as f:
            pickle.dump(pca_data, f)
        
        print(f"[INFO] PCA model saved to: {pca_file}")
        
        print(f"\n[PCA COMPLETE] {m_features} → {n_components} dimensions")
        print(f"[PCA COMPLETE] Variance explained: {variance_explained*100:.2f}%")
        
        return (
            True, 
            n_components, 
            variance_explained,
            f"PCA completed: {m_features}→{n_components} dims ({variance_explained*100:.2f}% variance)"
        )
        
    except Exception as e:
        print(f"[ERROR] apply_pca: {str(e)}")
        import traceback
        traceback.print_exc()
        return (False, 0, 0.0, f"Error: {str(e)}")


def prepare_features_for_svm(dataset_id, split='train', use_pca=True):
    """
    Prepare feature vectors for SVM training
    
    Args:
        dataset_id: ID dataset
        split: 'train', 'test', or 'all'
        use_pca: True = use PCA features, False = use original 103-dim
    
    Returns:
        tuple: (X, y, feature_names)
    """
    try:
        # Get labeled tweets
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
        ).select_related('tweet__label')
        
        if not feature_vectors.exists():
            return None, None, None
        
        X = []
        y = []
        
        for fv in feature_vectors:
            try:
                label = Label.objects.filter(tweet=fv.tweet).first()
                if not label:
                    continue
                
                # Skip system labels for training
                if split == 'train' and label.labeled_by == 'system':
                    continue
                
                # Build feature vector
                if use_pca and fv.additional_features and 'pca_features' in fv.additional_features:
                    # Use PCA-reduced features
                    features = fv.additional_features['pca_features']
                else:
                    # Use original 103-dim features
                    features = [
                        fv.sentiment_score,
                        fv.positive_score,
                        fv.negative_score,
                    ]
                    
                    if fv.word_embedding and 'vector' in fv.word_embedding:
                        features.extend(fv.word_embedding['vector'])
                    else:
                        features.extend([0.0] * 100)
                
                X.append(features)
                y.append(label.sentiment)
                
            except Exception as e:
                print(f"[ERROR] Processing FV {fv.id}: {str(e)}") # type: ignore
                continue
        
        if not X:
            return None, None, None
        
        X = np.array(X)
        y = np.array(y)
        
        # ========================================
        # ✅ SHUFFLE DATA
        # ========================================
        print(f"\n[INFO] Shuffling {split} data to prevent bias...")
        print(f"[DEBUG] Before shuffle - first 20 labels: {y[:20]}")
        
        np.random.seed(42)
        shuffle_indices = np.random.permutation(len(X))
        X = X[shuffle_indices]
        y = y[shuffle_indices]
        
        print(f"[DEBUG] After shuffle - first 20 labels: {y[:20]}")
        
        from collections import Counter
        distribution = Counter(y)
        print(f"[INFO] Label distribution: {distribution}")
        
        # Feature names
        if use_pca and len(X[0]) < 103:
            n_dims = len(X[0])
            feature_names = [f'pca_{i}' for i in range(n_dims)]
        else:
            feature_names = ['sentiment_score', 'positive_score', 'negative_score']
            feature_names.extend([f'embedding_{i}' for i in range(100)])
        
        print(f"[FEATURES] Prepared {len(X)} {split} samples with {X.shape[1]} features")
        
        return X, y, feature_names
        
    except Exception as e:
        print(f"[ERROR] prepare_features_for_svm: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None
