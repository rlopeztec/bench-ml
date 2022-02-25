# from here https://www.tensorflow.org/tutorials/images/cnn
import sys
import argparse
from tensorflow.keras.backend import name_scope
from tensorflow.keras import datasets, layers, models, backend
from .ClassificationReport import ClassificationReport
from .ConfusionMatrix import ConfusionMatrix
from .Utils import Utils

# deep learning benchmark
#RENE Shall we implment tensorboad??? %load_ext tensorboard
import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from django.utils import timezone
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Dropout
from tensorflow.keras.optimizers import Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,RMSprop,SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Flatten, MaxPool2D, MaxPooling2D, MaxPooling1D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#backend.clear_session()

from .models import Classification_Report, Confusion_Matrix

class ModelConv1D:

    def runModel(self, idModelRunFeatures, trainInput, testInput, parametersList, targetClass='Outcome', saveDb=False, idFileRaw=None, benchmarkDir='benchmark', regression='off'):
        # deep learning benchmark
        print('ModelConv1D.runModel', trainInput, testInput, parametersList, targetClass, saveDb, idFileRaw, regression, file=sys.stderr)
        #with name_scope('SETUP'):
        df = pd.read_csv(trainInput, delimiter='\t')
        dfTest = pd.read_csv(testInput, delimiter='\t')
        sc = StandardScaler()

        #print('X_train df:',df.shape, df,file=sys.stderr)
        #TODO RENE REMOVE X_train = sc.fit_transform(df.drop(targetClass, axis=1))
        X_train_RENE_REMOVE = sc.fit_transform(df.drop(targetClass, axis=1))
        X_train = df.drop(targetClass, axis=1).to_numpy()
        print('X_train sc type:', type(X_train_RENE_REMOVE), file=sys.stderr)
        print('X_train df type:', type(X_train), file=sys.stderr)
        y_train_tmp = df[targetClass].values
        print('X_train sc:',X_train.shape, X_train,file=sys.stderr)
        print('y_train_tmp shape 1:', y_train_tmp.shape, 'type(y_train_tmp):', type(y_train_tmp))
        print('y_train_tmp 1:', y_train_tmp)

        # convert list from string to integer
        y_train_tmp,foundList = Utils.convertToIntegers(self, y_train_tmp, True, idFileRaw, {})
        print('y_train_tmp shape 2:', 'type(y_train_tmp):', type(y_train_tmp))
        #TODO AVOID TEST ERROR RENE print('y_train_tmp shape 2:', y_train_tmp.shape)
        #print('y_train_tmp 2:', y_train_tmp)

        n_classes = len(np.unique(y_train_tmp))
        len_classes = len(y_train_tmp)

        y_train = to_categorical(y_train_tmp)
        print('y_train shape 3:', y_train.shape, 'type(y_train):', type(y_train))
        print('y_train 3:', y_train)

        #TODO RENE REMOVE X_test = sc.fit_transform(dfTest.drop(targetClass, axis=1))
        X_test = dfTest.drop(targetClass, axis=1).to_numpy()
        y_test_tmp = dfTest[targetClass].values
        y_test_tmp,foundList = Utils.convertToIntegers(self, y_test_tmp, True, idFileRaw, foundList)
        y_test = to_categorical(y_test_tmp)

        print('X_train.shape 1:', X_train.shape)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        print('X_train.shape 2:', X_train.shape)
        print('y_train.shape:', y_train.shape)
        print('len(y_train):', len(y_train), 'len(y_train[0]):', len(y_train[0]))
        print('y_train:', y_train)
        print('X_test.shape:', X_test.shape)
        print('y_test.shape:', y_test.shape)

        #print(y_train_tmp)
        in_shape = X_train.shape[1:]
        print('np.unique:', np.unique(y_train_tmp))
        print('in_shape:', in_shape, 'n_classes:', n_classes)

        # input data: (batch_size, height, width, depth)
        #with name_scope('MODEL'):
        model = Sequential()

        # CNN Model Option 1
        if parametersList['model_option'] == '1':
            #model.add(Conv1D(64, 2, input_shape=in_shape, activation='relu'))
            model.add(Conv1D(int(parametersList['filters']), kernel_size=int(parametersList['kernels']), input_shape=in_shape, activation=parametersList['activation']))
            model.add(MaxPooling1D(pool_size=int(parametersList['poolsize'])))
            model.add(Flatten())
            model.add(Dense(int(int(parametersList['filters'])/2), activation=parametersList['activation']))

        # CNN Model Option 2
        if parametersList['model_option'] == '2':
            #model.add(Conv1D(64, 2, input_shape=in_shape, activation='relu'))
            model.add(Conv1D(int(parametersList['filters']), kernel_size=int(parametersList['kernels']), input_shape=in_shape, activation=parametersList['activation']))
            model.add(MaxPooling1D(pool_size=int(parametersList['poolsize'])))
            #TODO 20211007
            model.add(Conv1D(int(int(parametersList['filters'])/2), kernel_size=int(parametersList['kernels']), activation=parametersList['activation']))
            #model.add(Dense(int(int(parametersList['filters'])/2), activation=parametersList['activation']))
            model.add(MaxPooling1D(pool_size=int(parametersList['poolsize'])))
            model.add(Flatten())
            model.add(Dense(int(int(parametersList['filters'])/4), activation=parametersList['activation']))

        # CNN Model Option 3
        if parametersList['model_option'] == '3':
            #model.add(Conv1D(64, 2, input_shape=in_shape, activation='relu'))
            model.add(Conv1D(int(parametersList['filters']), kernel_size=int(parametersList['kernels']), input_shape=in_shape, activation=parametersList['activation']))
            # TODO 20211007 remover dropout
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            #model.add(MaxPooling1D(pool_size=int(parametersList['poolsize'])))
            #model.add(Dense(int(int(parametersList['filters'])/2), activation=parametersList['activation']))
            model.add(Conv1D(int(int(parametersList['filters'])/2), kernel_size=3, activation=parametersList['activation']))
            #model.add(MaxPooling1D(pool_size=int(parametersList['poolsize'])))
            #model.add(Dense(int(int(parametersList['filters'])/4), activation=parametersList['activation']))
            model.add(Conv1D(int(int(parametersList['filters'])/4), kernel_size=2, activation=parametersList['activation']))
            model.add(MaxPooling1D(pool_size=int(parametersList['poolsize'])))
            model.add(Flatten())
            model.add(Dense(int(int(parametersList['filters'])/2), activation=parametersList['activation']))

        # CNN Model Option 4
        if parametersList['model_option'] == '4':
            #model.add(Conv1D(64, 2, input_shape=in_shape, activation='relu'))
            model.add(Conv1D(int(parametersList['filters']), kernel_size=int(parametersList['kernels']), input_shape=in_shape, activation=parametersList['activation']))
            model.add(MaxPooling1D(pool_size=int(parametersList['poolsize'])))
            model.add(Flatten())
            model.add(Dense(16, activation=parametersList['activation']))

        # if regression case like TCGA survival data
        if regression == 'on':
            #model.add(Dense(1, activation='softmax'))
            model.add(Dense(1))
        else:
            model.add(Dense(n_classes, activation=parametersList['activation_output']))

        optimizer=RMSprop(0.001)
        #model.compile(globals()[parametersList['optimizer']](lr=float(parametersList['learning_rate'])),
        #              loss=parametersList['loss'],
        #              metrics=['accuracy','mae'])
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['accuracy','mae'])
        print('model.summary:', model.summary(), file=sys.stderr)

        #with name_scope('CALLBACK'):
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        #with name_scope('FIT'):
        #history = model.fit(X_train, y_train, epochs=int(parametersList['epochs']), verbose=2, validation_split=0.1, callbacks=[tensorboard_callback])

        #change y_train (77,2406) to y_train_tmp (77,) for regression cases
        if regression == 'on':
            history = model.fit(X_train, y_train_tmp, epochs=int(parametersList['epochs']), verbose=2, validation_split=0.1)
            Utils.saveImagesRegression(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_mae.png', history, 'mae', 'mean_abs_error')
        else:
            history = model.fit(X_train, y_train, epochs=int(parametersList['epochs']), verbose=2, validation_split=0.1)
        Utils.saveImages(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_accuracy.png', history, 'accuracy', 'val_accuracy')
        Utils.saveImages(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_loss.png', history, 'loss', 'val_loss')

        #with name_scope('PREDICT'):
        y_pred_train = model.predict(X_train).flatten()
        y_pred_test = model.predict(X_test).flatten()
        y_pred = model.predict(X_test)

        #with name_scope('SERIES_NOTHING'):
        y_test_class = np.argmax(y_test, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        #print(pd.Series(y_test_class).value_counts() / len(y_test_class))

        #with name_scope('ACCURACY'):
        if regression == 'on':
            # for regression cases
            #errorRateTest = mean_absolute_error(y_test_tmp, y_pred_class)
            accuracyScore = 0
            [loss, accuracyScore, errorRateTrain] = model.evaluate(X_train, y_train_tmp)
            [loss, accuracyScore, errorRateTest]  = model.evaluate(X_test,  y_test_tmp)
            print("MAE Train: %.3f" % errorRateTrain, file=sys.stderr)
            print("MAE Test: %.3f" % errorRateTest, file=sys.stderr)
            print('y_train_tmp.type:', type(y_train_tmp), file=sys.stderr)
            print('y_train_tmp.shape:', y_train_tmp.shape, file=sys.stderr)
            print('y_train_tmp:', y_train_tmp, file=sys.stderr)
            print('y_pred_train.type:', type(y_pred_train), file=sys.stderr)
            print('y_pred_train.shape:', y_pred_train.shape, file=sys.stderr)
            print('y_pred_train:', y_pred_train, file=sys.stderr)
            plot1 = np.max(y_train_tmp)
            plot2 = np.max(y_pred_train)
            Utils.saveImagesPredictions(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_pred_train.png', y_train_tmp, y_pred_train, plot1, plot2, 'train', 'mean_abs_error')
            print('y_test_tmp.type:', type(y_test_tmp), file=sys.stderr)
            print('y_test_tmp.shape:', y_test_tmp.shape, file=sys.stderr)
            print('y_test_tmp.shape[0]:', y_test_tmp.shape[0], file=sys.stderr)
            print('y_test_tmp:', y_test_tmp, file=sys.stderr)
            print('y_pred_test.type:', type(y_pred_test), file=sys.stderr)
            print('y_pred_test.shape:', y_pred_test.shape, file=sys.stderr)
            print('y_pred_test:', y_pred_test, file=sys.stderr)
            print('y_pred:', y_pred, file=sys.stderr)
            print('X_test:', X_test, file=sys.stderr)
            Utils.saveImagesPredictions(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_pred_test.png', y_test_tmp, y_pred_test, plot1, plot2, 'test', 'mean_abs_error')
        else:
            # for not regression
            accuracyScore = accuracy_score(y_test_class, y_pred_class)
            print("Accuracy score 1: {:0.3}".format(accuracyScore), file=sys.stderr)
            errorRateTrain = 0
            errorRateTest  = 0

        print(classification_report(y_test_class, y_pred_class))
        print(confusion_matrix(y_test_class, y_pred_class))

        classificationReport = classification_report(y_test_class, y_pred_class, output_dict=True)
        confusionMatrix = confusion_matrix(y_test_class, y_pred_class)

        # save to database
        if saveDb:
            ClassificationReport.saveClassificationReport(self,idModelRunFeatures, classificationReport)
            ConfusionMatrix.saveConfusionMatrix(self,idModelRunFeatures, confusionMatrix)
        return accuracyScore,errorRateTrain, errorRateTest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='input trained file')
    parser.add_argument('-test', help='input test file')
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
    c.runModel(args.train, args.test, {'filters':32, 'activation':'relu', 'activation_output':'softmax', 'optimizer':'Adam', 'loss':'categorical_crossentropy', 'learning_rate':0.05, 'model_option':1}, 'Outcome', saveDb)

