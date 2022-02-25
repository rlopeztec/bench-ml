from tensorflow.keras.backend import name_scope, clear_session
with name_scope('INIT'):
    from django.utils import timezone
    from .models import Classification_Report

class ClassificationReport:
    def saveClassificationReport(self, idModelRunFeatures, classificationReport):
        print('cr type', type(classificationReport))
        print('cr len', len(classificationReport))
        print(classificationReport)
        for cn in classificationReport:
            if cn not in ('accuracy', 'macro avg', 'weighted avg'):
                print('cr i', cn)
                cr = Classification_Report(model_run_features_id=idModelRunFeatures, class_name=cn, precision=classificationReport[cn]['precision'], recall=classificationReport[cn]['recall'], f1_score=classificationReport[cn]['f1-score'], support=classificationReport[cn]['support'], pub_date=timezone.now())
                cr.save()

