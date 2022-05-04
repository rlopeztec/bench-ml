# our own implementation of neural networks with backpropagation

import sys, argparse
from .ClassificationReport import ClassificationReport
from .ConfusionMatrix import ConfusionMatrix
from .Utils import Utils

import datetime
import pandas as pd
import numpy as np
from django.utils import timezone
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from .models import Classification_Report, Confusion_Matrix

# from our implementation
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

class ModelBNN:

    # Load CSV file
    def load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file, delimiter='\t')
            first = True
            for row in csv_reader:
                if first:
                    headers = row
                    first = False
                else:
                    if not row:
                        continue
                    dataset.append(row)
        return headers, dataset

    # Convert string column to float
    def str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())

    # Convert string column to integer
    def str_column_to_int(self, dataset, column):
        print(len(dataset), column)
        class_values = [row[column] for row in dataset]
        print('class values', len(class_values))
        unique = set(class_values)
        print('unique class values', len(unique))
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        print('lookup', len(lookup))
        for row in dataset:
            row[column] = lookup[row[column]]
        print('dataset', len(dataset))
        return lookup

    # Find the min and max values for each column
    def dataset_minmax(self, dataset):
        return [[min(column), max(column)] for column in zip(*dataset)]

    # Rescale dataset columns to the range 0-1
    def normalize_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    # Split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, debug, dataset, algorithm, n_folds, l_rate, n_epoch, n_hidden, hidden_layers):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        train_accuracy = list()
        test_accuracy = list()
        loss_train = []
        loss_test = []
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
            # predicted is from test data
            predicted, loss_train, loss_test, train_accuracy, test_accuracy = algorithm(debug, train_set, test_set, l_rate, n_epoch, n_hidden, hidden_layers)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores, train_accuracy, test_accuracy, loss_train, loss_test

    # Calculate neuron activation for an input
    def activate(self, debug, weights, inputs):
        activation = weights[-1]
        if debug:
            print('activation', activation)
        for i in range(len(weights)-1):
            if debug:
                print('i', i)
                print('weights', weights)
                print('weights[i]', weights[i])
                print('inputs', inputs)
                print('inputs[i]', inputs[i])
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation sigmoid function (TODO: pass as a parameter)
    # TODO: could be different for each layer
    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    # Forward propagate input to a network output
    # TODO: print weights, neurons and result of activation function
    # TODO: then design how it could work with several layers
    # TODO: and test in backward propagation
    def forward_propagate(self, debug, network, row):
        inputs = row
        if debug:
            print('row/inputs', inputs)
            print('network', network)
        for layer in network:
            if debug:
                print('network -> layer', layer)
            new_inputs = []
            for neuron in layer:
                if debug:
                    print('layer -> neuron', neuron)
                    print('layer -> neuron weights', neuron['weights'])
                activation = self.activate(debug, neuron['weights'], inputs)
                # activation function could be different for each layer
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(self, network, row, l_rate):
        #print('len network', len(network), 'layers')
        #print('len network 0', len(network[0]), 'nodes in hidden layer')
        #print('len network 1', len(network[1]), 'nodes in output layer')
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    #print('len neuron', len(neuron))
                    #print('len neuron weights', len(neuron['weights']))
                    #print('len inputs', len(inputs))
                    #print('j', j)
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']


    # evaluate loss function, expected vs predicted outputs
    def evaluate_loss_function(self, expected, predicted):
        cce = CategoricalCrossentropy()
        loss_result = cce(expected, predicted).numpy()
        return loss_result


    # Train a network for a fixed number of epochs
    def train_network(self, debug, network, train, test, l_rate, n_epoch, n_outputs):
        loss_train = []
        loss_test = []
        train_accuracy = []
        test_accuracy = []
        for _ in range(n_epoch):
            list_expected = []
            list_predicted = []
            cnt_predicted_train = 0
            for row in train:
                predicted = self.forward_propagate(debug, network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1

                list_predicted.append(predicted)
                list_expected.append(expected)
                self.backward_propagate_error(network, expected)
                self.update_weights(network, row, l_rate)
                if row[-1] == predicted.index(max(predicted)):
                    cnt_predicted_train += 1
            loss_train.append(self.evaluate_loss_function(list_expected, list_predicted))
            train_accuracy.append(cnt_predicted_train / len(train) * 100)

            # evaluate loss function for test data
            list_predicted = []
            list_expected = []
            cnt_predicted_test = 0
            for row in test:
                predicted = self.forward_propagate(debug, network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                list_predicted.append(predicted)
                list_expected.append(expected)
                # calculate accuracy
                if row[-1] == predicted.index(max(predicted)):
                    cnt_predicted_test += 1
            test_accuracy.append(cnt_predicted_test / len(test) * 100)
        return loss_train, loss_test, train_accuracy, test_accuracy

    # Initialize a network
    def initialize_network(self, n_inputs, n_hidden, n_outputs, hidden_layers):
        #print('n inputs', n_inputs)
        #print('n hidden', n_hidden)
        network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden[0])]
        network.append(hidden_layer)
        for i in range(hidden_layers-1):
            #print('i, i+1', i, i+1)
            #print('n_hidden(i+1)', n_hidden[i+1])
            hidden_layer = [{'weights':[random() for k in range(n_hidden[i] + 1)]} for j in range(n_hidden[i+1])]
            #print('pass 2')
            #print(hidden_layer)
            network.append(hidden_layer)
        output_layer = [{'weights':[random() for k in range(n_hidden[len(n_hidden)-1] + 1)]} for j in range(n_outputs)]
        network.append(output_layer)
        return network

    # Make a prediction with a network
    def predict(self, debug, network, row):
        outputs = self.forward_propagate(debug, network, row)
        return outputs.index(max(outputs))

    # Backpropagation Algorithm With Stochastic Gradient Descent
    def back_propagation(self, debug, train, test, l_rate, n_epoch, n_hidden, hidden_layers):
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        network = self.initialize_network(n_inputs, n_hidden, n_outputs, hidden_layers)
        if debug:
            print('len network', len(network), 'layers')
            for i in range(len(network)):
                print('len network', i, len(network[i]), 'nodes in layer')
        loss_train, loss_test, train_accuracy, test_accuracy = self.train_network(debug, network, train, test, l_rate, n_epoch, n_outputs)

        # evaluate test data and loss function for test
        predictions = list()
        for row in test:
            prediction = self.predict(debug, network, row)
            predictions.append(prediction)

        if debug:
            print('len train', len(train), 'rows in training set')
            print('len train 0', len(train[0]), '7 features + 1 label')
            print('len test', len(test), 'rows in test set')
            print('n inputs', n_inputs, 'features')
            print('n outputs', n_outputs, 'different values in output')
            print('len predictions', len(predictions), 'in test set')
            print('len network', len(network), 'layers')
            for i in range(len(network)):
                print('len network', i, len(network[i]), 'nodes in layer')
            #print('network', network)
            #print('train', train)
            print('')
        return(predictions, loss_train, loss_test, train_accuracy, test_accuracy)


    def runModel(self, idModelRunFeatures, trainInput, testInput, parametersList, targetClass='Outcome', saveDb=False, idFileRaw=None, benchmarkDir='benchmark', regression='off'):
        # out backpropagation method
        seed(1)
        # load and prepare data
        headers, dataset = self.load_csv(trainInput)
        print(type(dataset), len(dataset))

        for i in range(len(dataset[0])-1):
            self.str_column_to_float(dataset, i)

        # convert class column to integers
        self.str_column_to_int(dataset, len(dataset[0])-1)

        # normalize input variables
        minmax = self.dataset_minmax(dataset)

        # normalize input variables
        self.normalize_dataset(dataset, minmax)

        # evaluate algorithm
        n_folds = 5
        l_rate = 0.3
        n_epoch = 50
        n_hidden = [5] # number of nodes in the unique hidden layer
        hidden_layers = 1 # adding number of hidden layers
        scores, train_accuracy, test_accuracy, loss_train, loss_test = self.evaluate_algorithm(False, dataset, self.back_propagation, n_folds, l_rate, n_epoch, n_hidden, hidden_layers)
        print('Scores: %s' % scores)
        accuracyScore = sum(scores)/float(len(scores))
        print('Mean Accuracy train(of %i): %.3f%%' % (len(train_accuracy), sum(train_accuracy)/float(len(train_accuracy))))
        print('Mean Accuracy test (of %i): %.3f%%' % (len(test_accuracy), sum(test_accuracy)/float(len(test_accuracy))))

        print('loss train:', loss_train)
        print('loss test:', loss_test)
        Utils.saveImagesBNN(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_loss.png', 'loss', loss_train, loss_test, n_epoch)
        Utils.saveImagesBNN(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_accuracy.png', 'accuracy', train_accuracy, test_accuracy, n_epoch)

        '''
        Utils.saveImages(self, benchmarkDir+'/static/benchmark/images/'+ str(idModelRunFeatures) + '_accuracy.png', history, 'accuracy', 'val_accuracy')
        # alternatively for plotting use:
        # modelLoss = pd.DataFrame(history.history)
        # modelLoss.plot()

        y_pred = model.predict(X_test)

        y_test_class = np.argmax(y_test, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        #print(pd.Series(y_test_class).value_counts() / len(y_test_class))

        accuracyScore = accuracy_score(y_test_class, y_pred_class)
        print("Accuracy score: {:0.3}".format(accuracyScore))

        classificationReport = classification_report(y_test_class, y_pred_class, output_dict=True)
        confusionMatrix = confusion_matrix(y_test_class, y_pred_class)

        # save to database
        if saveDb:
            ClassificationReport.saveClassificationReport(self,idModelRunFeatures, classificationReport)
            ConfusionMatrix.saveConfusionMatrix(self,idModelRunFeatures, confusionMatrix)
        '''

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
    c = ModelBNN()
    #c.runModel('benchmark/files/gtex_6_1000_RF_train_80.tsv', 
    #           'benchmark/files/gtex_6_1000_RF_test_20.tsv', 
    #           {'filters':32, 'activation':'Relu', 'activation_output':'softmax, 'loss':'categorical_crossentropy', 'optimizer'='Adam', 'learning_rate'=0.05, 'model_option':1},
    #            10)
    c.runModel(args.train, args.test, {'filters':32, 'activation':'Relu', 'activation_output':'softmax', 'optimizer':'Adam', 'loss':'categorical_crossentropy', 'learning_rate':0.05, 'model_option':1}, 'Outcome', saveDb)

