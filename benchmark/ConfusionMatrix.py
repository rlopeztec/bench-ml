from tensorflow.keras.backend import name_scope, clear_session
with name_scope('INIT'):
    from django.utils import timezone
    from .models import Confusion_Matrix

class ConfusionMatrix:
    def saveConfusionMatrix(self, idModelRunFeatures, confusionMatrix):
        print('cm type', type(confusionMatrix))
        len_cm = len(confusionMatrix)
        print('cm len', len_cm)
        print(confusionMatrix)
        for x in range(len_cm):
            for y in range(len_cm):
                cr = Confusion_Matrix(model_run_features_id=idModelRunFeatures, x=x, y=y, quantity=confusionMatrix[x][y], pub_date=timezone.now())
                cr.save()

