# split train and test data for gene importance scores file (i.e. gtex_6.tsv, gtex_6_1000_RF.tsv)
# split data by the percent indicated PER CLASS to have a balanced train/test data
# python benchmark_train_test_split.py -p 80 -c Outcome -i gtex_6_1000_RF.tsv -o gtex_6_1000_RF_train_80.tsv -t gtex_6_1000_RF_test_20.tsv
# Parameters: -p percent for training data, -i input file, -o output training file, -t output test file

import argparse

class SplitTrainTest():
    ##########################################################################
    # read genes scoring file and return top n genes
    ##########################################################################
    def readGenesScoringFile(self, inputFile, fileType, targetClass):
        print('SplitTrainTest readGenesScoringFile', inputFile, fileType, targetClass)
        retClasses = {}
        retGenes = []
        targetCol = None
        delimiter = '\t'
        if fileType == 'csv':
            delimiter = ','
        with open(inputFile, 'r') as input:
            c = 0
            for line in input:
                cols = line.rstrip().split(delimiter)
                retGenes.append(cols)
                if c == 0:
                    print('SplitTrainTest cols', cols)
                    for i in range(0,len(cols)):
                        if cols[i] == targetClass:
                            targetCol = i
                    if targetCol == None:
                        raise Exception('COLUMN DOESNT EXIST', targetClass, 'in', inputFile)
                else:
                    if cols[targetCol] in retClasses:
                        retClasses[cols[targetCol]] = retClasses[cols[targetCol]] + 1
                    else:
                        retClasses[cols[targetCol]] = 1
                c += 1
        print('reading', inputFile, 'lines', c, 'classes', len(retClasses))
        print(retClasses)
        return targetCol, retGenes, retClasses

    ##########################################################################
    # write output train and test files
    ##########################################################################
    def writeOutputFiles(self, percent, targetCol, retGenes, retClasses, outputTraining, outputTest):
        linesTraining = 0
        linesTest = 0
        cClasses = {}
        trainingFile = open(outputTraining, 'w')
        testFile = open(outputTest, 'w')
        firstLine = True
        for cols in retGenes:
            if cols[targetCol] in cClasses:
                cClasses[cols[targetCol]] = cClasses[cols[targetCol]] + 1
            else:
                cClasses[cols[targetCol]] = 1
            if firstLine or cClasses[cols[targetCol]] <= retClasses[cols[targetCol]] * (percent/100):
                trainingFile.write('\t'.join(cols))
                trainingFile.write('\n')
                linesTraining += 1
                if firstLine:
                    testFile.write('\t'.join(cols))
                    testFile.write('\n')
                    linesTest += 1
                    firstLine = False
            else:
                testFile.write('\t'.join(cols))
                testFile.write('\n')
                linesTest += 1
        print('training:', linesTraining, 'test', linesTest)
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

