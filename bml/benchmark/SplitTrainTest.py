# split train and test data for gene importance scores file (i.e. gtex_6.tsv, gtex_6_1000_RF.tsv)
# split data by the percent indicated PER CLASS to have a balanced train/test data
# python benchmark_train_test_split.py -p 80 -c Outcome -i gtex_6_1000_RF.tsv -o gtex_6_1000_RF_train_80.tsv -t gtex_6_1000_RF_test_20.tsv
# Parameters: -p percent for training data, -i input file, -o output training file, -t output test file

import sys
import argparse
from csv import reader
from random import shuffle

class SplitTrainTest():

    ##########################################################################
    # read genes scoring file and return top n genes
    ##########################################################################
    def readGenesScoringFile(self, inputFile, fileType, targetClass):
        print('SplitTrainTest readGenesScoringFile', inputFile, fileType, targetClass, file=sys.stderr)
        retClasses = {}
        retLines = []
        targetCol = None
        if fileType == 'csv':
            delimiter = ','
        else:
            delimiter = '\t'

        # get input file as a list
        dataset = list()
        headers = list()
        with open(inputFile, 'r') as file:
            csv_reader = reader(file, delimiter=delimiter)
            first = True
            for row in csv_reader:
                if first:
                    headers = row
                    first = False
                else:
                    if not row:
                        continue
                    dataset.append(row)
        # get Label/Class position
        for i in range(0,len(headers)):
            if headers[i] == targetClass:
                targetCol = i
        if targetCol == None:
            raise Exception('COLUMN DOESNT EXIST', targetClass, 'in', inputFile)
        shuffle(dataset)
        for cols in dataset:
            #cols = line.rstrip().split(delimiter)
            retLines.append(cols)
            if cols[targetCol] in retClasses:
                retClasses[cols[targetCol]] = retClasses[cols[targetCol]] + 1
            else:
                retClasses[cols[targetCol]] = 1
        print('reading inputFile', inputFile, 'lines', len(dataset), 'classes', len(retClasses), file=sys.stderr)
        print(retClasses, file=sys.stderr)
        return targetCol, retLines, retClasses

    ##########################################################################
    # write output train and test files
    ##########################################################################
    def writeOutputFiles(self, percent, targetCol, retLines, retClasses, outputTraining, outputTest):
        linesTraining = 0
        linesTest = 0
        cClasses = {}
        trainingFile = open(outputTraining, 'w')
        testFile = open(outputTest, 'w')
        firstLine = True
        for cols in retLines:
            if cols[targetCol] in cClasses:
                cClasses[cols[targetCol]] = cClasses[cols[targetCol]] + 1
            else:
                cClasses[cols[targetCol]] = 1
            if firstLine or cClasses[cols[targetCol]] <= retClasses[cols[targetCol]] * (percent/100) or (0.5<len(retClasses)/len(retLines) and (percent/100)>linesTraining/len(retLines)):
                #if firstLine:
                    #print('trainingFile: firstLine', file=sys.stderr)
                #else:
                    #print('trainingFile:', cClasses[cols[targetCol]], '<=', retClasses[cols[targetCol]] * (percent/100), len(retClasses)/len(retLines), file=sys.stderr)
                trainingFile.write('\t'.join(cols))
                trainingFile.write('\n')
                linesTraining += 1
                if firstLine:
                    testFile.write('\t'.join(cols))
                    testFile.write('\n')
                    linesTest += 1
                    firstLine = False
            else:
                #print('testFile:', cClasses[cols[targetCol]], '>', retClasses[cols[targetCol]] * (percent/100), len(retClasses)/len(retLines), file=sys.stderr)
                testFile.write('\t'.join(cols))
                testFile.write('\n')
                linesTest += 1
        #print('training:', linesTraining, 'percent:', percent, 'test:', linesTest, file=sys.stderr)
        trainingFile.close()
        testFile.close()
        return linesTraining, linesTest

##########################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='porcent of training data')
    parser.add_argument('-c', help='target class (Outcome or Y)')
    parser.add_argument('-i', help='input scoring file')
    parser.add_argument('-o', help='output training file')
    parser.add_argument('-t', help='output test file')
    args = parser.parse_args()
    print('split training and test data')
    stt = SplitTrainTest()
    targetCol, retGenes, retClasses = stt.readGenesScoringFile(args.i, args.c)
    linesTraining, linesTest = stt.writeOutputFiles(int(args.p), targetCol, retGenes, retClasses, args.o, args.t)

