import csv
import os

from django.conf import settings
from core.models import SentimentDictionary


def load_tsv_dictionary(tsv_path, polarity):
    created = 0
    skipped = 0

    with open(tsv_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue

            word = row[0].strip().lower()
            try:
                weight = float(row[1])
            except:
                continue

            obj, is_created = SentimentDictionary.objects.get_or_create(
                word=word,
                defaults={
                    'weight': weight,
                    'polarity': polarity,
                    'source': 'base'
                }
            )

            if is_created:
                created += 1
            else:
                skipped += 1

    return created, skipped

def load_base_dictionary_from_tsv():
    """
    Load BASE dictionary otomatis dari file TSV (positive & negative)
    Dipanggil oleh view, bukan shell
    """

    # Cegah load ulang
    if SentimentDictionary.objects.filter(source='base').exists():
        return

    base_dir = settings.BASE_DIR

    pos_path = os.path.join(base_dir, 'data/dictionaries/positive.tsv')
    neg_path = os.path.join(base_dir, 'data/dictionaries/negative.tsv')

    if os.path.exists(pos_path):
        load_tsv_dictionary(pos_path, polarity='positive')

    if os.path.exists(neg_path):
        load_tsv_dictionary(neg_path, polarity='negative')

    print('[INFO] Base dictionary loaded automatically from TSV')

