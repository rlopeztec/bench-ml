# Generated by Django 3.1.4 on 2021-04-26 05:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('benchmark', '0004_file_raw_regression'),
    ]

    operations = [
        migrations.AlterField(
            model_name='file_raw',
            name='regression',
            field=models.CharField(max_length=200),
        ),
    ]
