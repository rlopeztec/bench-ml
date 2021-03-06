# Generated by Django 3.1.4 on 2021-01-10 07:30

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='File_raw',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_name', models.CharField(max_length=200)),
                ('file_desc', models.CharField(max_length=200)),
                ('number_features', models.CharField(max_length=200)),
                ('methods', models.CharField(max_length=200)),
                ('percent_train', models.CharField(max_length=200)),
                ('dist_class', models.CharField(max_length=200)),
                ('target_class', models.CharField(max_length=200)),
                ('file_type', models.CharField(max_length=200)),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
            ],
        ),
        migrations.CreateModel(
            name='Model_Run',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_ran', models.CharField(max_length=200)),
                ('notes', models.CharField(max_length=200)),
                ('epochs', models.IntegerField(default=50)),
                ('weighted_accuracy', models.FloatField(default=0)),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
                ('file_raw', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='benchmark.file_raw')),
            ],
        ),
        migrations.CreateModel(
            name='Question',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_text', models.CharField(max_length=200)),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
            ],
        ),
        migrations.CreateModel(
            name='Web_User',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=200)),
                ('password', models.CharField(max_length=200)),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
            ],
        ),
        migrations.CreateModel(
            name='Model_Run_Steps',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('order', models.IntegerField(default=0)),
                ('step', models.CharField(max_length=200)),
                ('value', models.CharField(max_length=200)),
                ('model_run', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='benchmark.model_run')),
            ],
        ),
        migrations.CreateModel(
            name='Model_Run_Features',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('method', models.CharField(max_length=200)),
                ('num_features', models.CharField(max_length=200)),
                ('accuracy_score', models.FloatField(default=0)),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
                ('model_run', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='benchmark.model_run')),
            ],
        ),
        migrations.AddField(
            model_name='file_raw',
            name='web_user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='benchmark.web_user'),
        ),
        migrations.CreateModel(
            name='Confusion_Matrix',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('x', models.IntegerField()),
                ('y', models.IntegerField()),
                ('quantity', models.IntegerField()),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
                ('model_run_features', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='benchmark.model_run_features')),
            ],
        ),
        migrations.CreateModel(
            name='Classification_Report',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('class_name', models.CharField(max_length=200)),
                ('precision', models.FloatField(default=0)),
                ('recall', models.FloatField(default=0)),
                ('f1_score', models.FloatField(default=0)),
                ('support', models.FloatField(default=0)),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
                ('model_run_features', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='benchmark.model_run_features')),
            ],
        ),
        migrations.CreateModel(
            name='Class_Number',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(max_length=200)),
                ('to_number', models.IntegerField()),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
                ('file_raw', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='benchmark.file_raw')),
            ],
        ),
        migrations.CreateModel(
            name='Choice',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('choice_text', models.CharField(max_length=200)),
                ('votes', models.IntegerField(default=0)),
                ('question', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='benchmark.question')),
            ],
        ),
    ]
