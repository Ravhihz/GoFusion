from django.utils import timezone
from core.models import Label, EvaluationMetrics, Prediction
from core.evaluation import calculate_metrics_multiclass


def evaluate_dictionary_scenario(dictionary_type):
    predictions = Prediction.objects.filter(
        model_name__icontains=dictionary_type
    ).select_related('tweet')

    if not predictions.exists():
        return None

    y_true = []
    y_pred = []

    for pred in predictions:
        try:
            label = Label.objects.get(
                tweet=pred.tweet,
                dataset_split='test'
            )
            y_true.append(label.sentiment)
            y_pred.append(pred.predicted_sentiment)
        except Label.DoesNotExist:
            continue

    if len(y_true) == 0:
        return None

    metrics = calculate_metrics_multiclass(y_true, y_pred)

    EvaluationMetrics.objects.create(
        dictionary_type=dictionary_type,
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1_score'],
        confusion_matrix=metrics['confusion_matrix'],
        evaluated_at=timezone.now()
    )

    return metrics

def run_full_dictionary_evaluation():
    results = {}

    for dict_type in ['base', 'manual', 'extended']:
        metrics = evaluate_dictionary_scenario(dict_type)
        if metrics:
            results[dict_type] = metrics

    return results
