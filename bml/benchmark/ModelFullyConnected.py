# from here https://www.tensorflow.org/tutorials/images/cnn
import argparse
from tensorflow.keras.backend import name_scope
from tensorflow.keras import datasets, layers, models, backend
from .ClassificationReport import ClassificationReport
from .ConfusionMatrix import ConfusionMatrix
from .Utils import Utils

#with name_scope('INIT'):
# deep learning benchmark
#RENE HOW TO DO THIS??? %load_ext tensorboard
import datetime
import pandas as pd
import numpy as np
from django.utils import timezone
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Dropout
from tensorflow.keras.optimizers import Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,RMSprop,SGD
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
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

class ModelFullyConnected:

    def runModel(self, idModelRunFeatures, trainInput, testInput, parametersList, targetClass='Outcome', saveDb=False, idFileRaw=None, benchmarkDir='benchmark', regression='off'):
        # deep learning benchmark
        print('ModelFullyConnected.runModel', trainInput, testInput, parametersList, targetClass, saveDb, idFileRaw)
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

        #i didn't have it but needed for non-regression cases
        y_test = to_categorical(y_test_tmp)

        print('X_train.shape', X_train.shape)
        print('y_train.shape', y_train.shape)
        print('X_test.shape', X_test.shape)
        print('y_test.shape', y_test.shape)
        print(y_train_tmp)

        in_shape = X_train.shape[1:]
        n_classes = len(np.unique(y_train_tmp))
        print('np.unique', np.unique(y_train_tmp))
        print('in_shape', in_shape, n_classes)

        # input data: (batch_size, height, width, depth)
        #with name_scope('MODEL'):
        model = Sequential()

        # playing with NN
        if parametersList['model_option'] == '1':
            print('OPTIONNNNNNNNNNNN 1')
            model.add(Dense(parametersList['filters'], input_shape=in_shape, activation=parametersList['activation'], name='CL1'))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/2), activation=parametersList['activation'], name='CL2'))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))

        if parametersList['model_option'] == '2':
            print('OPTIONNNNNNNNNNNN 2')
            model.add(Dense(parametersList['filters'], input_shape=in_shape, activation=parametersList['activation'], name='CL1'))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/2), activation=parametersList['activation'], name='CL2'))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/4), activation=parametersList['activation']))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/6), activation=parametersList['activation']))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))

        if parametersList['model_option'] == '3':
            print('OPTIONNNNNNNNNNNN 3')
            model.add(Dense(parametersList['filters'], input_shape=in_shape, activation=parametersList['activation'], name='CL1'))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/2), activation=parametersList['activation'], name='CL2'))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/4), activation=parametersList['activation']))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/6), activation=parametersList['activation']))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/8), activation=parametersList['activation']))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/10), activation=parametersList['activation']))
            if float(parametersList['dropout']) > 0:
                model.add(Dropout(float(parametersList['dropout'])))

        if parametersList['model_option'] == '4':
            print('OPTIONNNNNNNNNNNN 4')
            model.add(Dense(parametersList['filters'], input_shape=in_shape, activation=parametersList['activation'], name='CL1'))
            model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/2), activation=parametersList['activation'], name='CL2'))
            model.add(Dropout(float(parametersList['dropout'])))
            model.add(Dense(int(int(parametersList['filters'])/4), activation=parametersList['activation']))
            model.add(Dropout(float(parametersList['dropout'])))

        # if regression case like TCGA survival data
        if regression == 'on':
            model.add(Dense(1, activation='linear', name='CL3'))
        else:
            model.add(Dense(n_classes, activation=parametersList['activation_output'], name='CL3'))

        model.compile(globals()[parametersList['optimizer']](lr=float(parametersList['learning_rate'])),
                      loss=parametersList['loss'],
                      metrics=['accuracy'])

        print(model.summary())

        #with name_scope('CALLBACK'):
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        #with name_scope('FIT'):
        
        # if regression case like TCGA survival data
        if regression == 'on':
            w_y_train = y_train_tmp
        else:
            w_y_train = y_train

        #TODO RENE early stopping options are auto, min and max, maybe implement it later?
        earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=int(parametersList['earlystop']))

        if float(parametersList['earlystop']) > 0:
            #history = model.fit(X_train, y_train, epochs=int(parametersList['epochs']), verbose=2, validation_split=0.2, callbacks=[earlyStop, tensorboard_callback])
            history = model.fit(X_train, w_y_train, epochs=int(parametersList['epochs']), verbose=2, validation_split=0.2, callbacks=[earlyStop])
        else:
            #history = model.fit(X_train, y_train, epochs=int(parametersList['epochs']), verbose=2, validation_split=0.2, callbacks=[tensorboard_callback])
            history = model.fit(X_train, w_y_train, epochs=int(parametersList['epochs']), verbose=2, validation_split=0.2)

        Utils.saveImages(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_accuracy.png', history, 'accuracy', 'val_accuracy')
        Utils.saveImages(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_loss.png', history, 'loss', 'val_loss')
        # alternatively for plotting use:
        # modelLoss = pd.DataFrame(history.history)
        # modelLoss.plot()

        #with name_scope('PREDICT'):
        y_pred = model.predict(X_test)

        #with name_scope('SERIES_NOTHING'):
        y_test_class = np.argmax(y_test, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        print('test class:', len(y_test_class), y_test_class)
        print('pred class:', len(y_pred_class), y_pred_class)

        #with name_scope('ACCURACY'):
        accuracyScore = accuracy_score(y_test_class, y_pred_class)
        print("Accuracy score: {:0.3}".format(accuracyScore))

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
    parser.add_argument('-db', help='Set to False or True to save to database')
    args = parser.parse_args()
    if args.db and args.db != 0 and args.db != 'False' and args.db != 'false':
        saveDb = True
    else:
        saveDb = False
    c = ModelFullyConnected()
    #c.runModel('benchmark/files/gtex_6_1000_RF_train_80.tsv', 
    #           'benchmark/files/gtex_6_1000_RF_test_20.tsv', 
    #           {'filters':32, 'activation':'Relu', 'activation_output':'softmax, 'loss':'categorical_crossentropy', 'optimizer'='Adam', 'learning_rate'=0.05, 'model_option':1},
    #            10)
    c.runModel(args.train, args.test, {'filters':32, 'activation':'Relu', 'activation_output':'softmax', 'optimizer':'Adam', 'loss':'categorical_crossentropy', 'learning_rate':0.05, 'model_option':1}, 'Outcome', saveDb)

