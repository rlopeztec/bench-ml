# Generated by Django 3.1.4 on 2021-01-26 06:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('benchmark', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='file_raw',
            name='data_type',
            field=models.CharField(default='Gene Expression', max_length=200),
            preserve_default=False,
        ),
    ]
