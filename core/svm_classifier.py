import numpy as np
from datetime import datetime


class LinearSVM:
    def __init__(self, C=1.0, max_iter=1000, tol=1e-3, random_state=42):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.w = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        
        # Label encoding attributes
        self.label_mapping = {}  # Maps original labels to -1, 1
        self.inverse_label_mapping = {}  # Maps -1, 1 back to original labels
        
    def _encode_labels(self, y):
        """
        Convert string labels to numeric values (-1, 1)
        
        Args:
            y: Array of labels (can be strings or numbers)
            
        Returns:
            y_encoded: Numeric array with values -1 and 1
        """
        unique_labels = np.unique(y)
        
        # Create mapping if not exists (during training)
        if not self.label_mapping:
            if len(unique_labels) > 2:
                raise ValueError(f"LinearSVM only supports binary classification. Got {len(unique_labels)} classes: {unique_labels}")
            
            # Map first unique label to -1, second to 1
            self.label_mapping = {
                unique_labels[0]: -1,
                unique_labels[1]: 1 if len(unique_labels) > 1 else 1
            }
            self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
            
            print(f"[SVM] Label mapping created: {self.label_mapping}")
        
        # Encode labels
        y_encoded = np.array([self.label_mapping[label] for label in y])
        
        return y_encoded
    
    def _decode_labels(self, y_encoded):
        """
        Convert numeric labels back to original format
        
        Args:
            y_encoded: Numeric array with values -1 and 1
            
        Returns:
            y_original: Array in original label format
        """
        y_original = np.array([self.inverse_label_mapping[int(label)] for label in y_encoded])
        return y_original
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)
        
        print(f'[SVM] Training Linear SVM')
        print(f'[SVM] Samples: {n_samples}, Features: {n_features}')
        print(f'[SVM] C: {self.C}, Max iterations: {self.max_iter}')
        
        # Convert string labels to numeric (-1, 1)
        print(f'[SVM] Original label types: {type(y[0])}, unique values: {np.unique(y)}')
        y_encoded = self._encode_labels(y)
        print(f'[SVM] Encoded labels: {np.unique(y_encoded)}')
        
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        K = np.dot(X, X.T)
        
        print(f'[SVM] Starting optimization...')
        
        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alphas)
            
            for i in range(n_samples):
                j = self._get_random_j(i, n_samples)
                
                eta = K[i, i] + K[j, j] - 2 * K[i, j]
                if eta <= 0:
                    continue
                
                L, H = self._compute_L_H(self.alphas[i], self.alphas[j], y_encoded[i], y_encoded[j])
                
                if L == H:
                    continue
                
                E_i = self._decision_function_single(X, y_encoded, i, K) - y_encoded[i]
                E_j = self._decision_function_single(X, y_encoded, j, K) - y_encoded[j]
                
                alpha_j_old = self.alphas[j]
                self.alphas[j] = alpha_j_old + y_encoded[j] * (E_i - E_j) / eta
                self.alphas[j] = np.clip(self.alphas[j], L, H)
                
                if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                    continue
                
                self.alphas[i] = self.alphas[i] + y_encoded[i] * y_encoded[j] * (alpha_j_old - self.alphas[j])
                
                b1 = self.b - E_i - y_encoded[i] * (self.alphas[i] - alpha_prev[i]) * K[i, i] - y_encoded[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                b2 = self.b - E_j - y_encoded[i] * (self.alphas[i] - alpha_prev[i]) * K[i, j] - y_encoded[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                
                if 0 < self.alphas[i] < self.C:
                    self.b = b1
                elif 0 < self.alphas[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2
            
            diff = np.linalg.norm(self.alphas - alpha_prev)
            
            if (iteration + 1) % 100 == 0:
                print(f'[SVM] Iteration {iteration + 1}/{self.max_iter}, diff: {diff:.6f}')
            
            if diff < self.tol:
                print(f'[SVM] Converged at iteration {iteration + 1}')
                break
        
        sv_indices = self.alphas > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y_encoded[sv_indices]  # Store encoded labels
        self.alphas = self.alphas[sv_indices]
        
        self.w = np.dot((self.alphas * self.support_vector_labels).T, self.support_vectors)
        
        print(f'[SVM] Training completed')
        print(f'[SVM] Support vectors: {len(self.support_vectors)} / {n_samples}')
        
        return self
    
    def _get_random_j(self, i, n_samples):
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
    
    def _compute_L_H(self, alpha_i, alpha_j, y_i, y_j):
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        return L, H
    
    def _decision_function_single(self, X, y, i, K):
        return np.sum(self.alphas * y * K[:, i]) + self.b
    
    def decision_function(self, X):
        return np.dot(X, self.w) + self.b # type: ignore
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted labels in original format (strings or numbers)
        """
        scores = self.decision_function(X)
        y_pred_encoded = np.sign(scores)
        
        # Convert back to original label format
        y_pred_original = self._decode_labels(y_pred_encoded)
        
        return y_pred_original


def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics for any label type (string or numeric)
    """
    classes = np.unique(y_true)
    
    metrics = {
        'accuracy': np.mean(y_true == y_pred) * 100,
        'per_class': {}
    }
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Handle both string and numeric class labels
        if isinstance(cls, (int, np.integer)):
            if cls == 1:
                class_name = 'positive'
            elif cls == -1:
                class_name = 'negative'
            else:
                class_name = 'neutral'
        else:
            class_name = str(cls)
        
        metrics['per_class'][class_name] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'support': int(np.sum(y_true == cls))
        }
    
    return metrics


def generate_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix for any label type (string or numeric)
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_cls) & (y_pred == pred_cls))
    
    # Handle both string and numeric class labels
    class_names = []
    for cls in classes:
        if isinstance(cls, (int, np.integer)):
            if cls == 1:
                class_names.append('positive')
            elif cls == -1:
                class_names.append('negative')
            else:
                class_names.append('neutral')
        else:
            class_names.append(str(cls))
    
    return cm, class_names


# ============================================================================
# MULTI-CLASS SVM (One-vs-Rest Strategy)
# Pure scratch implementation - NO sklearn!
# ============================================================================

class MultiClassSVM:
    def __init__(self, C=1.0, max_iter=1000, tol=1e-3, random_state=42, class_weight=None):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.class_weight = class_weight  # ← TAMBAH INI
        self.classifiers = {}
        self.classes = None
        self.n_classes = 0
    
    def fit(self, X, y):
        """Train one binary classifier per class (One-vs-Rest)"""
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        print(f"[MultiClass SVM] Training {self.n_classes} binary classifiers")
        print(f"[MultiClass SVM] Classes: {self.classes}")
        
        for i, target_class in enumerate(self.classes):
            print(f"\n[Classifier {i+1}/{self.n_classes}] '{target_class}' vs others...")
            
            # ✅ PERBAIKAN: Binary labels sebagai NUMERIC (1 vs -1)
            y_binary = np.where(y == target_class, 1, -1)
            
            n_pos = np.sum(y_binary == 1)
            n_neg = np.sum(y_binary == -1)
            print(f"  Samples: {n_pos} positive, {n_neg} negative")
            
            # Train binary SVM
            clf = LinearSVM(C=self.C, max_iter=self.max_iter,
                        tol=self.tol, random_state=self.random_state)
            clf.fit(X, y_binary)
            
            self.classifiers[target_class] = clf
            print(f"  ✓ Trained successfully")
        
        print(f"\n[MultiClass SVM] All classifiers trained!")
        return self
    
    def decision_function(self, X):
        """Get decision scores from all classifiers"""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes))
        
        for i, cls in enumerate(self.classes): # type: ignore
            clf = self.classifiers[cls]
            scores[:, i] = clf.decision_function(X)
        
        return scores
    
    def predict(self, X):
        """Predict class with highest score"""
        scores = self.decision_function(X)
        max_indices = np.argmax(scores, axis=1)
        predictions = self.classes[max_indices] # type: ignore
        return predictions
    
    def predict_proba(self, X):
        """Get probability estimates (softmax of scores)"""
        scores = self.decision_function(X)
        
        # Softmax for numerical stability
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probabilities


def calculate_metrics_multiclass(y_true, y_pred):
    """
    Calculate metrics for multi-class classification
    Pure scratch implementation!
    """
    classes = np.unique(y_true)
    
    accuracy = np.mean(y_true == y_pred) * 100
    
    per_class_metrics = {}
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[str(cls)] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'support': int(np.sum(y_true == cls))
        }
    
    # Macro-averaged F1
    f1_scores = [m['f1_score'] for m in per_class_metrics.values()]
    macro_f1 = np.mean(f1_scores)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class': per_class_metrics
    }


def generate_confusion_matrix_multiclass(y_true, y_pred):
    """
    Generate confusion matrix for multi-class
    Pure scratch implementation!
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_cls) & (y_pred == pred_cls))
    
    class_names = [str(cls) for cls in classes]
    
    return cm, class_names