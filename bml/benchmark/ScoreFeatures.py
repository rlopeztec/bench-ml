import numpy as np
import pandas as pd
import sys, argparse
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class ScoreFeatures():

    # for DNA convert bases to 0,1,2,3 and then hot encoding
    def oneHotEncoder(self, listSequences, labels, rId):
        retList = []
        integer_encoder = LabelEncoder()
        integer_encoder.fit(['A','C','G','T'])
        xinteger_encoded = integer_encoder.transform(['A','C','G','T'])
        integer_encoded = np.array(xinteger_encoded).reshape(-1, 1) 
        one_hot_encoder = OneHotEncoder(categories='auto')
        one_hot_encoder.fit(integer_encoded)
        ret_input_features = np.empty(0)
        cols = []
        cnt = 0

        scoreLog = open('/tmp/score.log.'+rId,'w')

        scoreLog.write('RENE one hot encoder.1: '+ str(type(listSequences)) + ' ' + str(len(listSequences)) + ' ' + str(datetime.now().strftime("%Y/%m/%d %H:%M:%S")) + '\n')
        print('RENE one hot encoder.1:', type(listSequences), len(listSequences), datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
        #print('RENE one hot encoder.1.1:', listSequences.shape, file=sys.stderr)
        for i in range(0,listSequences.shape[1]):
        #for i in range(0,1):
            #parSequences = listSequences.iloc[:, 0] # extract the 1st column only
            parSequences = listSequences.iloc[:, i]
            first = True

            #if first: print('RENE one hot encoder.1.5:', parSequences.shape, len(parSequences), file=sys.stderr)
            sequences = parSequences.values.tolist()
            #if first: print('RENE one hot encoder.2:', type(sequences), len(sequences), file=sys.stderr)
            #if first: print('RENE one hot encoder.2.5:', sequences[0], file=sys.stderr)
            sequences = list(filter(None, sequences))  # This removes empty sequences
            #if first: print('RENE one hot encoder.3:', type(sequences), len(sequences), file=sys.stderr)
            #if first: print('RENE one hot encoder.4:', sequences[0], file=sys.stderr) # ACGT
            #if first: print('RENE one hot encoder.5:', len(sequences), file=sys.stderr) #2000
            #if first: print('RENE one hot encoder.7:', len(sequences[0]), file=sys.stderr) #50

            if i%100 == 0:
                scoreLog.write('RENE one hot encoder.8: ' + str(i) + ' ' + str(type(sequences)) + ' ' + str(len(sequences)) + ' ' + str(len(sequences[0])) + ' ' + str(datetime.now().strftime("%Y/%m/%d %H:%M:%S")) + '\n')
                scoreLog.flush()

            input_features = []
            #first = True
            ind = 0
            for sequence in sequences:
                #if first: print('RENE IE 1, ind, sequence',ind, sequence, type(sequence), file=sys.stderr)
                ind += 1
                #sequence = 'C'
                #integer_encoded = integer_encoder.transform(list('ACGTACGT'))
                #if first: print('RENE IE 1, int enc', 'ACGTACGT', integer_encoded, file=sys.stderr)
                #integer_encoded = integer_encoder.transform(list('TGCATGCA'))
                #if first: print('RENE IE 1, int enc', 'TGCATGCA', integer_encoded, file=sys.stderr)

                #integer_encoded = integer_encoder.transform(list('A'))
                #if first: print('RENE IE 1, int enc', 'A', integer_encoded, file=sys.stderr)
                #integer_encoded = integer_encoder.transform(list('C'))
                #if first: print('RENE IE 2, int enc', 'C', integer_encoded, file=sys.stderr)
                #integer_encoded = integer_encoder.transform(list('G'))
                #if first: print('RENE IE 3, int enc', 'G', integer_encoded, file=sys.stderr)
                #integer_encoded = integer_encoder.transform(list('T'))
                #if first: print('RENE IE 4, int enc', 'T', integer_encoded, file=sys.stderr)

                #if first: print('RENE IE 1, ind, CC', ind, sequence, file=sys.stderr)
                #print('RENE IE 1, ind, CC', ind, sequence, file=sys.stderr)
                integer_encoded = integer_encoder.transform(list(sequence))
                #if first: print('RENE IE 2', integer_encoded, file=sys.stderr)
                integer_encoded = np.array(integer_encoded).reshape(-1, 1)
                #if first: print('RENE IE 3', integer_encoded, file=sys.stderr)
                #one_hot_encoded = one_hot_encoder.transform(integer_encoded)
                #if first: print('RENE IE 4', one_hot_encoded, file=sys.stderr)
                #if first: print('RENE IE 5', one_hot_encoded.toarray(), file=sys.stderr)
                newArray = []
                #cnt = 0
                #for i in one_hot_encoded.toarray():
                for i in one_hot_encoder.transform(integer_encoded).toarray():
                    for j in i:
                        newArray.append(j)
                        if first:
                            cols.append('s'+str(cnt))
                            cnt += 1
                    #if first: print('RENE IE 5.4:', cnt, len(cols), file=sys.stderr)
                #if first: print('RENE IE 5.7:', cnt, len(cols), file=sys.stderr)
                #if first: print('RENE IE 6:', newArray[0], file=sys.stderr)
                #if first: print('RENE IE 7:', newArray, file=sys.stderr)
                #if first: print('RENE IE 8:', len(newArray), file=sys.stderr)
                #input_features.append(one_hot_encoded.toarray())
                input_features.append(newArray)
                #if first: print('RENE input feats 1:', type(input_features),len(input_features),file=sys.stderr)
                if ind > 0:
                    first = False #TODO RENE REVERSE

            np.set_printoptions(threshold=40)
            #if first: print('RENE input feats 2:', len(input_features[0]), file=sys.stderr) #50
            #print('RENE input feats 3', len(input_features[0][0]), file=sys.stderr) #4

            # group features in a matrix
            if len(ret_input_features) == 0:
                ret_input_features = np.stack([el for el in input_features])
                #if first: print('RENE input feats 4:', ret_input_features.shape, file=sys.stderr) #50
            else:
                #if first: print('RENE input feats 5:', ret_input_features.shape, file=sys.stderr) #50
                inFeats = [el for el in input_features]
                ret_input_features = np.append(ret_input_features, inFeats, axis=1)
            #if first: print('RENE shape stack', ret_input_features.shape, file=sys.stderr)
            #if first: print("RENE Example sequence\n-----------------------", file=sys.stderr)
            #if first: print('RENE DNA Sequence #1:\n', sequences[0][:10],'...',sequences[0][-10:], file=sys.stderr)
            #if first: print('RENE One hot encoding of Sequence #1:\n', ret_input_features[0], file=sys.stderr)
            #if first: print('RENE One hot encoding of Sequence #1 T:\n', ret_input_features[0].T, file=sys.stderr)
            #if first: print('RENE len', len(ret_input_features), len(ret_input_features[0]), len(ret_input_features[0].T), file=sys.stderr)
            #if first: print('RENE shape after', ret_input_features.shape, file=sys.stderr)


        ##### PROCESSING LABELS NOW #####
        #print('Labels 1:', type(labels), len(labels), labels, file=sys.stderr)
        #labels = list(filter(None, labels)) # removes empty labels
        print('OneHotEncoder Labels 2:', len(labels), file=sys.stderr)
        print('ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
        one_hot_encoder = OneHotEncoder(categories='auto')
        print('OneHotEncoder Labels 3:', len(labels), file=sys.stderr)
        print('ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
        labels = np.array(labels).reshape(-1, 1)

        input_labels = one_hot_encoder.fit_transform(labels).toarray()

        print('OneHotEncoder Labels:', len(labels), file=sys.stderr)
        print('ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
        #print(len(labels), len(labels.T), len(labels.T[0]), 'Labels:\n',labels.T, file=sys.stderr)
        #print('One-hot labels:\n', len(input_labels), len(input_labels.T), input_labels.T, file=sys.stderr)
        #print('RENE cols:', len(cols), type(ret_input_features), type(input_labels), cols,file=sys.stderr)
        #print('RENE input features len:', len(ret_input_features), file=sys.stderr)
        #print('RENE input features:', ret_input_features, file=sys.stderr)
        #print('RENE input features shape', ret_input_features.shape, file=sys.stderr)
        ##print('RENE input labels shape', input_labels.shape, file=sys.stderr)

        scoreLog.close()

        return cols, pd.DataFrame(data=ret_input_features, columns=cols), pd.DataFrame(data=input_labels)

    def scoreTrain(self, fileName='gtex_6_100.tsv', fileType='tsv', featureList='all,10,50,100,1000', methods='RF,DT', targetClass="Outcome", dataType="Gene Expression", regression='off', rId=0):
        print('ScoreFeatures:', fileName, fileType, featureList, methods, targetClass, dataType, file=sys.stderr)
        print('ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
        cols = None
        X = None
        ytmp = None
        delimiter='\t'
        if fileType == 'csv':
            delimiter=','
        df = pd.read_csv(fileName, delimiter=delimiter)
        print('ScoreFeatures read_csv df.shape:', df.shape, file=sys.stderr)
        print('ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)

        #Features: Row#, Outcome, gene1, gene2, ...., gene50200
        X = df.drop(columns=[targetClass])
        y = df[targetClass]
        if dataType == 'DNA':
            cols,X,ytmp = self.oneHotEncoder(X, y, rId)
        if dataType == 'Variants':
            cols,X,ytmp = self.oneHotEncoder(X, y, rId)
        print('ScoreFeatures created X.shape:', X.shape, type(X), 'y shape,type:', y.shape, type(y),file=sys.stderr)
        print('ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
        #print('RENE X shape,type', X.shape, type(X), 'y shape,type', y.shape, type(y),file=sys.stderr)
        #print('RENE X df', X, file=sys.stderr)
        ##print('RENE TARGET CLASS df.Outcome', df.Outcome)
        #print('RENE y df', y, file=sys.stderr)
        #print('RENE ytmp df', ytmp, file=sys.stderr)

        # create a file for each combination of method and number of features
        featureFiles = []
        for method in methods.split(','):
            #print('RENE create a file for each combination', method, file=sys.stderr)
            if 'DT' == method:
                clf=DecisionTreeClassifier()
            else:
                clf=RandomForestClassifier(n_estimators=100)
            #print('RENE before fit', len(X), len(y), file=sys.stderr)
            #print('RENE before shape X', X.shape, file=sys.stderr)
            #print('RENE before shape y', y.shape, file=sys.stderr)
            clf.fit(X,y)

            # calculate feature importance
            print('ScoreFeatures after fit before feature importance:', method, file=sys.stderr)
            print('ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
            feature_imp = pd.Series( clf.feature_importances_,
                                     index=X.columns
                                   ).sort_values(ascending=False)
            print('ScoreFeatures after feature importance:', type(feature_imp), len(feature_imp), file=sys.stderr)
            print('ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)

            for noFeatures in featureList.split(','):
                #print('RENE create a file for each combination', method, noFeatures, file=sys.stderr)
                if noFeatures == 'all':
                    nFeat = -1
                else:
                    nFeat = int(noFeatures)

                # write output file, get the top features from here
                fname = fileName+'.'+method+'.'+noFeatures+'.scores'
                featureFiles.append(fname)
                c=0; v=0
                output = open(fname,'w')
                for key,value in feature_imp.items():
                    if key != 'Name':
                        #if c<24: print('feature importance:', c+1, fname, key,value, file=sys.stderr)
                        # breaks when there is only one feature because value is 0.0 add if c > 0
                        if value == 0 and c > 0 and regression != 'on':
                            #print('break:', c+1, fname, key,value, file=sys.stderr)
                            break
                        else:
                            v+=1
                        output.write(key+'\t'+str(value)+'\n')
                        #if c<24: print('feature importance write:', c+1, fname, key,value, file=sys.stderr)
                        c+=1
                        if nFeat != -1 and c >= nFeat:
                            break
                output.close()
                print('ScoreFeatures after for feature_imp, Scored', fname,'features', c, 'features w/values', v, file=sys.stderr)
                print('ScoreFeatures:', datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
        #raise Exception('stopping here')
        return featureFiles, cols, X, ytmp, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fileInput', help="input file w/cols as dimension and samples as rows")
    parser.add_argument('fileType', help="only csv or tsv")
    parser.add_argument('numFeatures', help="all or 10 or 50 or all,10,50,etc...")
    parser.add_argument('scoringMethods', help="only DT or RF or DT,RF")
    parser.add_argument('targetClass', help="i.e. Outcome or species")
    parser.add_argument('dataType', help="i.e. Gene Expression or DNA or Variants or Microbiomeor or other")
    parser.add_argument('regression', help="i.e. Gene Expression or DNA or Variants or Microbiomeor or other")
    args = parser.parse_args()
    print('Starting ScoreFeatures.py', args, file=sys.stderr)
    c = ScoreFeatures()
    #featureFiles = c.scoreTrain('gtex_6_100.tsv', 'tsv', 'all,10', 'RF,DT', 'Outcome', 'Gene Expression')
    featureFiles = c.scoreTrain(args.fileInput, args.fileType, args.numFeatures, args.scoringMethods, args.targetClass, args.dataType, args.regression)
    print('ScoreFeatures.py finished', file=sys.stderr)
