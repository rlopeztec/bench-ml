# utils class
# i.e. convert descriptions in text to integers to be processed in deep learning
# learning curves: https://www.dataquest.io/blog/learning-curves-machine-learning/

import numpy as np
import matplotlib.pyplot as plt
from django.utils import timezone
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB

class Utils:

    def saveImageValidation(self, fileName, classifier='RandomForestClassifier', X=None, y=None, metric='accuracy', paramName='n_estimators', pv1=1, pv2=100, pv3=10):

            # import required libraries 
            from sklearn.datasets import load_digits 
            from sklearn.model_selection import validation_curve 
            # Loading dataset 
            dataset = load_digits()
            X, y = dataset.data, dataset.target 

            trainSizes = [1, int(len(X)*0.8/7), int(len(X)*0.8/7)*2, int(len(X)*0.8/7)*3, int(len(X)*0.8/7)*4, int(len(X)*0.8/7)*5, int(len(X)*0.8/7)*6, int(len(X)*0.8)]

            # Calculate accuracy on training and test with 5-fold cross validation
            train_sizes, train_score, test_score = learning_curve(globals()[classifier](), X, y, 
                                      train_sizes = trainSizes, 
                                      cv = 5,
                                      scoring = metric)

            # Calculating mean and standard deviation of training score 
            mean_train_score = np.mean(train_score, axis = 1) 
            std_train_score = np.std(train_score, axis = 1)

            # Calculating mean and standard deviation of testing score 
            mean_test_score = np.mean(test_score, axis = 1) 
            std_test_score = np.std(test_score, axis = 1) 

            # Plot mean accuracy scores for training and testing scores 
            plt.switch_backend('Agg')
            plt.plot(train_sizes, mean_train_score, label = "Training Score", color = 'b') 
            plt.plot(train_sizes, mean_test_score, label = "Cross Validation Score", color = 'g') 

            # Creating the plot 
            plt.title("Validation Curve with " + classifier) 
            plt.xlabel("training size") 
            plt.ylabel(metric) 
            plt.tight_layout() 
            plt.legend(loc = 'best') 
            plt.savefig(fileName)
            plt.close()

    def saveImagesPredictions(self, fileName, test_labels, test_predictions, plot1, plot2, metric, valMetric):
            plt.switch_backend('Agg')
            plt.scatter(test_labels, test_predictions)
            plt.axis('equal')
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            #_ = plt.plot([-100, 5000],[-100,5000])
            _ = plt.plot([0, plot1],[0, plot2])

            plt.title('model ' + metric)
            error = test_predictions - test_labels
            plt.hist(error, bins = 500)
            plt.xlabel("True Values")
            _ = plt.ylabel("Predictions")

            #plt.plot(history.epoch, np.array(history.history['mae']), label='Train')
            #plt.plot(history.epoch, np.array(history.history['val_mae']), label = 'Val')
            #plt.ylim([0,max(history.history['val_mae'])])
            plt.savefig(fileName)
            plt.close()

    def saveImagesRegression(self, fileName, history, metric, valMetric):
            plt.switch_backend('Agg')
            #plt.figure()
            plt.xlabel('epoch')
            plt.ylabel('Mean Abs Error')
            plt.plot(history.epoch, np.array(history.history['mae']), label='Train')
            plt.plot(history.epoch, np.array(history.history['val_mae']), label = 'Val')
            plt.title('model ' + metric)
            plt.legend()
            plt.ylim([0,max(history.history['val_mae'])])
            plt.savefig(fileName)
            plt.close()

    def saveImages(self, fileName, history, metric, valMetric):
            plt.switch_backend('Agg')
            plt.plot(history.history[metric])
            plt.plot(history.history[valMetric])
            plt.title('model ' + metric)
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(fileName)
            plt.close()

    def convertToIntegers(self, list, saveDb=False, file_raw_id=None, foundList={}):
        for i in list:
            try:
                aNumber = int(i)
                aNumber = True
            except:
                aNumber = False
                break
        if aNumber:
            return list, {}

        returnList = []
        if len(foundList) == 0:
            c = 0
        else:
            maxV = 0
            for k,v in foundList.items():
                if v > maxV:
                    maxV = v
            c = maxV+1
        for i in range(len(list)):
            if list[i] not in foundList:
                foundList[list[i]] = c
                c += 1
            returnList.append(foundList[list[i]])
        if saveDb:
            from .models import Class_Number
            for key,value in foundList.items():
                cn = Class_Number(file_raw_id=file_raw_id, description=key, to_number=value, pub_date=timezone.now())
                cn.save()
        #print(foundList)
        #print(returnList)
        return returnList, foundList

if __name__ == '__main__':
    print('Utils')
    u = Utils()
    u.convertToIntegers(['a','b'], False, 1)

