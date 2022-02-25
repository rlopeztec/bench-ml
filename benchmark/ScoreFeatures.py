import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ScoreFeatures():
    def scoreTrain(self, fileName='gtex_6_100.tsv', fileType='tsv', featureList='all,10,50,100,1000', methods='RF,DT', targetClass="Outcome"):
        print('ScoreFeatures:', fileName, fileType, featureList, methods, targetClass)
        delimiter='\t'
        if fileType == 'csv':
            delimiter=','
        df = pd.read_csv(fileName, delimiter=delimiter)
        print('RENE TARGET CLASS df', df)

        #Features: Row#, Outcome, gene1, gene2, ...., gene50200
        X = df.drop(columns=[targetClass])
        print('RENE X df', X)
        #print('RENE TARGET CLASS df.Outcome', df.Outcome)
        #y = df.Outcome # Labels
        y = df[targetClass] # Labels
        print('RENE y df', y)

        # create a file for each combination of method and number of features
        featureFiles = []
        for method in methods.split(','):
            if 'DT' == method:
                clf=DecisionTreeClassifier()
            else:
                clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X,y)

            # calculate feature importance
            feature_imp = pd.Series( clf.feature_importances_,
                                     index=X.columns
                                   ).sort_values(ascending=False)

            for noFeatures in featureList.split(','):
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
                        if c<24: print('feature importance:', c+1, fname, key,value)
                        if value == 0:
                            break
                        else:
                            v+=1
                        output.write(key+'\t'+str(value)+'\n')
                        if c<24: print('feature importance write:', c+1, fname, key,value)
                        c+=1
                        if nFeat != -1 and c >= nFeat:
                            break
                output.close()
                print('Scored', fname, 'features', c, 'features w/values', v)
        #raise Exception('stopping here')
        return featureFiles

if __name__ == '__main__':
    c = ScoreFeatures()
    featureFiles = c.scoreTrain('gtex_6_100.tsv', 'tsv', 'all,10', 'RF,DT', 'Outcome')
