# from here https://www.tensorflow.org/tutorials/images/cnn
import argparse
from tensorflow.keras.backend import name_scope
from tensorflow.keras import datasets, layers, models, backend
from .ClassificationReport import ClassificationReport
from .ConfusionMatrix import ConfusionMatrix
from .Utils import Utils

# deep learning benchmark
#RENE HOW TO DO THIS??? %load_ext tensorboard
import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from django.utils import timezone
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D
from tensorflow.keras.optimizers import Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,RMSprop,SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Flatten, MaxPool2D, MaxPooling2D, MaxPooling1D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#backend.clear_session()

from .models import Classification_Report, Confusion_Matrix

class ModelConv1D:

    def runModel(self, idModelRunFeatures, trainInput, testInput, parametersList, epochs=50, targetClass='Outcome', saveDb=False, idFileRaw=None):
        # deep learning benchmark
        print('ModelConv1D        .runModel', trainInput, testInput, parametersList, epochs, targetClass, saveDb, idFileRaw)
        #with name_scope('SETUP'):
        df = pd.read_csv(trainInput, delimiter='\t')
        dfTest = pd.read_csv(testInput, delimiter='\t')
        sc = StandardScaler()

        X_train = sc.fit_transform(df.drop(targetClass, axis=1))
        y_train_tmp = df[targetClass].values

        # convert list from string to integer
        y_train_tmp,foundList = Utils.convertToIntegers(self, y_train_tmp, True, idFileRaw, {})
        y_train = to_categorical(y_train_tmp)

        X_test = sc.fit_transform(dfTest.drop(targetClass, axis=1))
        y_test_tmp = dfTest[targetClass].values
        y_test_tmp,foundList = Utils.convertToIntegers(self, y_test_tmp, True, idFileRaw, foundList)
        y_test = to_categorical(y_test_tmp)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        print('X_train.shape', X_train.shape)
        print('y_train.shape', y_train.shape)
        print('X_test.shape', X_test.shape)
        print('y_test.shape', y_test.shape)

        #print(y_train_tmp)
        in_shape = X_train.shape[1:]
        n_classes = len(np.unique(y_train_tmp))
        print('np.unique', np.unique(y_train_tmp))
        print('in_shape', in_shape, n_classes)

        # input data: (batch_size, height, width, depth)
        #with name_scope('MODEL'):
        model = Sequential()

        # playing with CNN
        #model.add(Conv1D(64, 2, input_shape=in_shape, activation='relu'))
        model.add(Conv1D(int(parametersList['filters']), 2, input_shape=in_shape, activation=parametersList['activation']))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(16, activation=parametersList['activation']))
        model.add(Dense(n_classes, activation=parametersList['activation_output']))
        model.compile(globals()[parametersList['optimizer']](lr=float(parametersList['learning_rate'])),
                      loss=parametersList['loss'],
                      metrics=['accuracy'])
        print(model.summary())

        #with name_scope('CALLBACK'):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        #with name_scope('FIT'):
        #model.fit(X_train, y_train, epochs=epochs, verbose=2, validation_split=0.1)
        history = model.fit(X_train, y_train, epochs=epochs, verbose=2, validation_split=0.1, callbacks=[tensorboard_callback])
        Utils.saveImages(self, 'benchmark/static/benchmark/images/'+ str(idModelRunFeatures) + '_accuracy.png', history, 'accuracy', 'val_accuracy')
        Utils.saveImages(self, 'benchmark/static/benchmark/images/'+ str(idModelRunFeatures) + '_loss.png', history, 'loss', 'val_loss')

        #with name_scope('PREDICT'):
        y_pred = model.predict(X_test)

        #with name_scope('SERIES_NOTHING'):
        y_test_class = np.argmax(y_test, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        #print(pd.Series(y_test_class).value_counts() / len(y_test_class))

        #with name_scope('ACCURACY'):
        accuracyScore = accuracy_score(y_test_class, y_pred_class)
        print("Accuracy score: {:0.3}".format(accuracyScore))
        print(classification_report(y_test_class, y_pred_class))
        print(confusion_matrix(y_test_class, y_pred_class))

        classificationReport = classification_report(y_test_class, y_pred_class, output_dict=True)
        confusionMatrix = confusion_matrix(y_test_class, y_pred_class)

        # save to database
        if saveDb:
            ClassificationReport.saveClassificationReport(self,idModelRunFeatures, classificationReport)
            ConfusionMatrix.saveConfusionMatrix(self,idModelRunFeatures, confusionMatrix)
        return accuracyScore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='input trained file')
    parser.add_argument('-test', help='input test file')
    parser.add_argument('-e', help='epochs to run')
    parser.add_argument('-db', help='Set to False or True to save to database')
    args = parser.parse_args()
    if args.db and args.db != 0 and args.db != 'False' and args.db != 'false':
        saveDb = True
    else:
        saveDb = False
    c = ModelConv1D()
    #c.runModel('benchmark/files/gtex_6_1000_RF_train_80.tsv', 
    #           'benchmark/files/gtex_6_1000_RF_test_20.tsv', 
    #            10)
    c.runModel(args.train, args.test, {'filters':32, 'activation':'relu', 'activation_output':'softmax', 'optimizer':'Adam', 'loss':'categorical_crossentropy', 'learning_rate':0.05, 'model_option':1}, int(args.e), 'Outcome', saveDb)

