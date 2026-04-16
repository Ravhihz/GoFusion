from django.core.management.base import BaseCommand
from core.models import SentimentDictionary
from django.conf import settings
import pandas as pd


class Command(BaseCommand):
    help = 'Load initial sentiment dictionary from TSV files'

    def handle(self, *args, **kwargs):
        positive_file = settings.DICTIONARIES_DIR / 'positive.tsv'
        negative_file = settings.DICTIONARIES_DIR / 'negative.tsv'
        
        SentimentDictionary.objects.all().delete()
        self.stdout.write('Cleared existing dictionary')
        
        df_positive = pd.read_csv(positive_file, sep='\t')
        df_positive = df_positive.drop_duplicates(subset=['word'], keep='first')
        
        count_positive = 0
        for _, row in df_positive.iterrows():
            word = str(row['word']).strip()
            weight = float(row['weight'])
            
            if not word or word == 'nan':
                continue
            
            SentimentDictionary.objects.get_or_create(
                word=word,
                defaults={
                    'weight': weight,
                    'polarity': 'positive',
                    'source': 'base'
                }
            )
            count_positive += 1
        
        self.stdout.write(self.style.SUCCESS(f'Loaded {count_positive} positive words'))
        
        df_negative = pd.read_csv(negative_file, sep='\t')
        df_negative = df_negative.drop_duplicates(subset=['word'], keep='first')
        
        count_negative = 0
        for _, row in df_negative.iterrows():
            word = str(row['word']).strip()
            weight = float(row['weight'])
            
            if not word or word == 'nan':
                continue
            
            SentimentDictionary.objects.get_or_create(
                word=word,
                defaults={
                    'weight': weight,
                    'polarity': 'negative',
                    'source': 'base'
                }
            )
            count_negative += 1
        
        self.stdout.write(self.style.SUCCESS(f'Loaded {count_negative} negative words'))
        self.stdout.write(self.style.SUCCESS(f'Total: {count_positive + count_negative} words'))