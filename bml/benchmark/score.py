# score.py is a program spun from django's view views.py
# creates at least 5 files: #feats, #feats.scores, #feats.genes, #feats.train, #feats.test
# could create more files depending on the methods and #feats

import os, sys, subprocess, argparse
from datetime import datetime

import ScoreFeatures #otherwise it doesn't work with Popen
import SplitTrainTest #otherwise it doesn't work with Popen
import pca_filter_features #otherwise it doesn't work with Popen
import Pca3D #otherwise it doesn't work with Popen


# BUILD PCA IMAGE FILE FOR EACH METHOD
def buildPca(fileRawId, filename, targetClass, benchmarkDir):
    nameOnly = filename[filename.rfind('/')+1:len(filename)]
    Pca3D.buildPcaImage(filename, filename + '.genes', benchmarkDir+'/static/benchmark/images/' + nameOnly + '.pca.png', targetClass)
    print('RENE views, finish, after buildPca:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)


# SPLIT TRAIN/TEST DATA
def splitTrainTest(filename, featFile, fileType, dataType, targetClass, percentTrain):
    if dataType == 'DNA':
        targetCol, retLines, retClasses = SplitTrainTest.SplitTrainTest.readGenesScoringFile(None, featFile, 'tsv', targetClass)
    else:
        if dataType == 'Variants':
            targetCol,retLines,retClasses = SplitTrainTest.SplitTrainTest.readGenesScoringFile(None,featFile,'tsv',targetClass)
        else:
            targetCol,retLines,retClasses=SplitTrainTest.SplitTrainTest.readGenesScoringFile(None,filename,fileType,targetClass)

    print('After SPLIT:', targetCol, len(retLines), len(retClasses), datetime.now().strftime("%Y/%m/%d %H:%M:%S"),file=sys.stderr)
    linesTraining, linesTest = SplitTrainTest.SplitTrainTest.writeOutputFiles(None, percentTrain, targetCol, retLines, retClasses, featFile+'.train', featFile+'.test')
    print('After writeOutputFiles:', linesTraining, linesTest, datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)


# FOR EACH METHOD CREATE GENES SCORED FILE
def methods(filename, fileType, numFeatures, scoringMethods, targetClass, dataType, regression, percentTrain, benchmarkDir, rId, dimNames, xSeqs, ycols):

    # split data into train and test each features file
    methodsList = scoringMethods.split(',')
    featuresList = numFeatures.split(',')
    for method in methodsList:
        for numFeat in featuresList:

            genesScoredFile = filename+'.'+method+'.'+numFeat+'.scores'
            featFile = filename + '.' + method + '.' + numFeat
            print('Score method:', method, 'featFile1:', featFile, datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)

            # get top features from scored features (pca_filter_features.py)
            topGenes = pca_filter_features.readGenesScoringFile(genesScoredFile, numFeat)
            print('After readGSF:', len(topGenes), genesScoredFile, datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)

            # create genes files in single line in tsv format to be used in pca (pca_filter_features.py)
            headersGenes, dimNames, xSeqs, ycols = pca_filter_features.readHeadersFile(filename, topGenes, featFile+'.genes', fileType, dataType, targetClass, dimNames, xSeqs, ycols)
            print('after HeadersFile:', filename, len(headersGenes), len(dimNames), len(xSeqs), ycols.shape, datetime.now().strftime("%Y/%m/%d %H:%M:%S"),file=sys.stderr)

            # creates features file with target class and features selected  (pca_filter_features.py)
            #TODO RENE REVIEW MEMORY OMITTING 3 READS ON SAME FILE: SCORING, HEADERS, INPUTFILE
            pca_filter_features.readInputFile(filename,fileType, featFile, headersGenes, targetClass, dataType, dimNames, xSeqs, ycols)
            print('After readInputFile filename:', filename, datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)

            # split train/test data for each method
            splitTrainTest(filename, featFile, fileType, dataType, targetClass, percentTrain)

            # create pca image file for each method
            buildPca(rId, featFile, targetClass, benchmarkDir)


# CREATE SCORES FILE
def score(filename, fileType, numFeatures, scoringMethods, targetClass, dataType, regression, rId):

    print('score.py, before ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
    c = ScoreFeatures.ScoreFeatures()

    # genes scored file with multiple number of features and featFile are created here
    featureFiles,dimNames,xSeqs,yLabels,ycols = c.scoreTrain(filename, fileType, numFeatures, scoringMethods, targetClass, dataType, regression, rId)
    print('score.py, after ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)

    return featureFiles,dimNames,xSeqs,yLabels,ycols


# MANAGE THE PROCESS
def main(filename, fileType, numFeatures, scoringMethods, targetClass, dataType, regression, percentTrain, benchmarkDir, rId):

    # score features
    featureFiles,dimNames,xSeqs,yLabels,ycols = score(filename, fileType, numFeatures, scoringMethods, targetClass, dataType, regression, rId)

    # methods one by one
    metResults = methods(filename, fileType,numFeatures, scoringMethods, targetClass, dataType, regression, percentTrain, benchmarkDir, rId, dimNames, xSeqs, ycols)


# START HERE
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="input file w/cols as dimension and samples as rows")
    parser.add_argument('fileType', help="only csv or tsv")
    parser.add_argument('numFeatures', help="all or 10 or 50 or all,10,50,etc...")
    parser.add_argument('scoringMethods', help="only DT or RF or DT,RF")
    parser.add_argument('targetClass', help="i.e. Outcome or species")
    parser.add_argument('dataType', help="i.e. Gene Expression or DNA or Variants or Microbiomeor or other")
    parser.add_argument('regression', help="i.e. Gene Expression or DNA or Variants or Microbiomeor or other")
    parser.add_argument('percentTrain', help="i.e. percent of data used for training the rest is for test")
    parser.add_argument('benchmarkDir', help="i.e. directory where programs are located")
    parser.add_argument('rId', help="i.e. r.id for build pca")
    args = parser.parse_args()

    main(args.filename, args.fileType, args.numFeatures, args.scoringMethods, args.targetClass, args.dataType, args.regression, float(args.percentTrain), args.benchmarkDir, args.rId)
