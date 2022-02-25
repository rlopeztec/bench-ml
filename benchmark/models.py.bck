import datetime
from django.db import models
from django.utils import timezone

# users
class Web_User(models.Model):
    def __str__(self):
        return self.id
    username = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

# files loaded
class File_raw(models.Model):
    def __str__(self):
        return self.file_name
    def get_file_desc(self):
        return self.file_desc
    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now
    web_user = models.ForeignKey(Web_User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=200)
    file_desc = models.CharField(max_length=200)
    number_features = models.CharField(max_length=200)
    methods = models.CharField(max_length=200)
    percent_train = models.CharField(max_length=200)
    dist_class = models.CharField(max_length=200)
    target_class = models.CharField(max_length=200)
    file_type = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

# Create your models here.
class Model_Run(models.Model):
    def __str__(self):
        return self.id
    def get_model_ran(self):
        return self.model_ran
    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now
    file_raw = models.ForeignKey(File_raw, on_delete=models.CASCADE)
    model_ran = models.CharField(max_length=200)
    notes = models.CharField(max_length=200)
    epochs = models.IntegerField(default=50)
    weighted_accuracy = models.FloatField(default=0)
    pub_date = models.DateTimeField('date published')

# Create your models here.
class Model_Run_Steps(models.Model):
    def __str__(self):
        return self.id
    model_run = models.ForeignKey(Model_Run, on_delete=models.CASCADE)
    order = models.IntegerField(default=0)
    step = models.CharField(max_length=200)
    value = models.CharField(max_length=200)

# Create your models here.
class Model_Run_Features(models.Model):
    def __str__(self):
        return self.id
    model_run = models.ForeignKey(Model_Run, on_delete=models.CASCADE)
    method = models.CharField(max_length=200)
    num_features = models.CharField(max_length=200)
    accuracy_score = models.FloatField(default=0)
    pub_date = models.DateTimeField('date published')

# Create your models here.
class Classification_Report(models.Model):
    def __str__(self):
        return self.id
    model_run_features = models.ForeignKey(Model_Run_Features, on_delete=models.CASCADE)
    class_name = models.CharField(max_length=200)
    precision = models.FloatField(default=0)
    recall = models.FloatField(default=0)
    f1_score = models.FloatField(default=0)
    support = models.FloatField(default=0)
    pub_date = models.DateTimeField('date published')

# Create your models here.
class Confusion_Matrix(models.Model):
    def __str__(self):
        return self.id
    model_run_features = models.ForeignKey(Model_Run_Features, on_delete=models.CASCADE)
    x = models.IntegerField()
    y = models.IntegerField()
    quantity = models.IntegerField()
    pub_date = models.DateTimeField('date published')

# Create your models here.
class Class_Number(models.Model):
    def __str__(self):
        return self.id
    file_raw = models.ForeignKey(File_raw, on_delete=models.CASCADE)
    description = models.CharField(max_length=200)
    to_number = models.IntegerField()
    pub_date = models.DateTimeField('date published')

# Create your models here.
class Question(models.Model):
    def __str__(self):
        return self.question_text
    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

# not sure if this should go here
datetime.timedelta(days=1)

class Choice(models.Model):
    def __str__(self):
        return self.choice_text
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
