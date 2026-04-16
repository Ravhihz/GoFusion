# tweets/migrations/0XXX_change_tweet_id_constraint.py

from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('core', '0007_svmmodel_dataset'),  # Sesuaikan dengan migration terakhir kamu
    ]

    operations = [
        # 1. Hapus unique constraint dari tweet_id
        migrations.AlterField(
            model_name='tweet',
            name='tweet_id',
            field=models.CharField(max_length=100),  # Tanpa unique=True
        ),
        
        # 2. Tambah unique_together untuk dataset + tweet_id
        migrations.AlterUniqueTogether(
            name='tweet',
            unique_together={('dataset', 'tweet_id')},
        ),
        
        # 3. Tambah index untuk performa
        migrations.AddIndex(
            model_name='tweet',
            index=models.Index(fields=['dataset', 'tweet_id'], name='tweets_dataset_tweet_idx'),
        ),
    ]