import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.backend import name_scope
from tensorflow.keras.utils import to_categorical

from .ClassificationReport import ClassificationReport
from .ConfusionMatrix import ConfusionMatrix
from .models import Classification_Report, Confusion_Matrix
from .Utils import Utils

class SupportVectorMachine():

    def runModel(self, idModelRunFeatures, trainInput, testInput, parametersList, targetClass='Outcome', saveDb=False, idFileRaw=None, benchmarkDir='benchmark'):
        print('SupportVectorMachine.runModel', trainInput, testInput, parametersList, targetClass, saveDb, idFileRaw)
        with name_scope('SETUP'):
            df = pd.read_csv(trainInput, delimiter='\t')
            dfTest = pd.read_csv(testInput, delimiter='\t')
            sc = StandardScaler()

            X_train = sc.fit_transform(df.drop(targetClass, axis=1))
            y_train_tmp = df[targetClass].values
            y_train_tmp, foundList = Utils.convertToIntegers(self, y_train_tmp, True, idFileRaw, {})
            y_train = to_categorical(y_train_tmp)

            X_test = sc.fit_transform(dfTest.drop(targetClass, axis=1))
            y_test_tmp = dfTest[targetClass].values
            y_test_tmp, foundList = Utils.convertToIntegers(self, y_test_tmp, True, idFileRaw, foundList)
            y_test = to_categorical(y_test_tmp)

            print('X_train.shape', X_train.shape)
            print('y_train.shape', y_train.shape)
            print('X_test.shape', X_test.shape)
            print('y_test.shape', y_test.shape)
            print(y_train_tmp)

            y_test_class = np.argmax(y_test, axis=1)

            c=0
            rows = 0
            outcome = [0 for i in range(0,y_train.shape[1])]
            y_train_rf = [0 for i in range(0,len(y_train))]
            for i in y_train[0:]:
                for j in range(0,y_train.shape[1]):
                    if i[j] > 0:
                        y_train_rf[rows] = j
                        outcome[j] += 1
                        c += 1
                rows+=1
   
            for mod in [SVC()]:
                history = mod.fit(X_train, y_train_rf)

                # save accuracy and loss curve images
                Utils.saveImageValidation(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_accuracy.png', 'SVC', X_train, y_train_rf, 'accuracy', 'gamma', -6, -1, 12)
                Utils.saveImageValidation(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_loss.png', 'SVC', X_train, y_train_rf, 'neg_mean_squared_log_error', 'gamma', -6, -1, 12)

                y_pred = mod.predict(X_test)
                print("="*80)
                print(mod)
                print("-"*80)
                accuracyScore = accuracy_score(y_test_class, y_pred)
                print("Accuracy score: {:0.3}".format(accuracyScore))
                print("Confusion Matrix:")
                print(confusion_matrix(y_test_class, y_pred))
                print()

                classificationReport = classification_report(y_test_class, y_pred, output_dict=True)
                confusionMatrix = confusion_matrix(y_test_class, y_pred)

                # save to database
                if saveDb:
                    ClassificationReport.saveClassificationReport(self,idModelRunFeatures, classificationReport)
                    ConfusionMatrix.saveConfusionMatrix(self,idModelRunFeatures, confusionMatrix)

        return accuracyScore

