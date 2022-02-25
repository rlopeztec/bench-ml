# from here https://www.tensorflow.org/tutorials/images/cnn
from keras.backend import name_scope
from keras import datasets, layers, models, backend
with name_scope('INIT'):
    # deep learning benchmark
    #RENE HOW TO DO THIS??? %load_ext tensorboard
    import datetime
    import pandas as pd
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Conv1D
    from keras.optimizers import Adam
    from keras.callbacks import TensorBoard
    from keras.layers import Flatten, MaxPool2D, MaxPooling2D, MaxPooling1D
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    #backend.clear_session()

class ModelFC:
    def runModel(self, trainInput, testInput, epochs):
        # deep learning benchmark
        print('ModelFC.runModel', trainInput, testInput, epochs)
        #df = pd.read_csv(trainInput, delimiter='\t')
        #dfTest = pd.read_csv(testInput, delimiter='\t')
        df = pd.read_csv('benchmark/files/gtex_6_1000_RF_train_80.tsv', delimiter='\t')
        dfTest = pd.read_csv('benchmark/files/gtex_6_1000_RF_test_20.tsv', delimiter='\t')
        sc = StandardScaler()

        X_train = sc.fit_transform(df.drop('Outcome', axis=1))
        y_train_tmp = df['Outcome'].values
        y_train = to_categorical(y_train_tmp)

        X_test = sc.fit_transform(dfTest.drop('Outcome', axis=1))
        y_test_tmp = dfTest['Outcome'].values
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
        model = Sequential()

        # playing with CNN
        model.add(Dense(64, 2, input_shape=(1001,1), activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(Adam(lr=0.05),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())

if __name__ == '__main__':
    c = ModelFC()
    c.runModel('benchmark/files/gtex_6_1000_RF_train_80.tsv', 
               'benchmark/files/gtex_6_1000_RF_test_20.tsv', 
                10)

