# PCA about GTeX samples, not sure if all tissues or just the 6 tissues we selected for testing
# pip install matplotlib
# python Pca3D.py -i gtex_6.tsv -g genes_all.tsv -o rene.png

from mpl_toolkits import mplot3d #RENE what for?

import matplotlib
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
matplotlib.use('agg')
from sklearn.preprocessing import StandardScaler

import argparse
import numpy as np
import pandas as pd

# return colors for each target class (y)
def getColors(y):

    targets = np.unique(np.array(y))
    # getting descriptions and match with ids
    #colors = ['r', 'g', 'b', 'y', 'c', 'm']
    #colors = ['black', 'darkorange', 'green', 'royalblue', 'gold', 'slategrey']
    colors = []
    colorsList = list(mcolors.CSS4_COLORS.keys())
    colorsList.remove('snow')
    colorsList.remove('white')
    colorsList.remove('whitesmoke')
    colorsList.remove('aliceblue')
    colorsList.remove('seashell')
    colorsList.remove('ivory')
    colorsList.remove('honeydew')
    colorsList.remove('mintcream')
    colorsList.remove('azure')
    colorsList.remove('ghostwhite')
    colorsList.remove('antiquewhite')
    divideBy = int(len(colorsList) / len(targets))
    i = 0
    for t in targets:
        colors.append(colorsList[i])
        i += divideBy
    return targets,colors,divideBy

def buildPcaImage(inputFile, genesFile, outputFile, targetClass):
    print('buildPcaImage', inputFile, genesFile, outputFile, targetClass)
    # test files
    #inputFile = "head2.tsv"
    #genesFile = "gtex_tissues_genes_3.tsv"
    # all rows
    #inputFile = "gtex_6.tsv"
    #genesFile = "genes_all.tsv"

    # read genes file and transform to single list, because DF transforms to a two dimensional list
    df_genes = pd.read_csv(genesFile, header=None, sep='\t')

    # these are genes/features
    l_genes = []
    l_features = []
    l_features.append(targetClass)
    for i in df_genes.iloc[0]:
        l_genes.append(i)
        l_features.append(i)
    #print('-0:',l_genes, len(l_genes))
    #print('-1:',l_features, len(l_features))

    # load dataset into Pandas DataFrame
    df_tissues = pd.read_csv(inputFile, names=l_features, header=0, sep='\t', low_memory=False)
    #print('-2:',df_tissues, len(df_tissues))

    # Separating out the features
    x = df_tissues.loc[:, l_genes].values
    print('tissues target class:', len(df_tissues[targetClass]))
    print('tissues target descr:', df_tissues.describe())
    print('tissues target tc de:', df_tissues[targetClass].describe())
    # comment this line because replacing strings with integers happens later on the program
    #df_tissues[targetClass] = df_tissues[targetClass].replace(dictTargets)
    c=0
    #'''
    for i in x:
      if c<5:
          #print('values in sample:',i)
          cj=0
          for j in i:
              #print('\tgene value:',j)
              cj+=1
      else:
          break
      c+=1
    #'''
    # Separating out the target
    y = df_tissues.loc[:,[targetClass]].values
    print('XXXXXXXXXX', len(df_tissues.loc[:,[targetClass]]), len(y), targetClass)
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # getting colors for each target class (y)
    targets, colors,divideBy = getColors(y)

    # PCA project to 2D or 3D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    print('RENE ERROR', x)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    finalDf = pd.concat([principalDf, df_tissues[[targetClass]]], axis = 1)

    # Visualize 2D projection
    fig = plt.figure(figsize = (14,14))
    ax = fig.add_subplot(1,1,1)
    ax = plt.axes(projection="3d") 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('Tissues', fontsize = 20)
    print('YYYYYYYYYYYY', len(y), len(colors), len(targets))
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[targetClass] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                  ,finalDf.loc[indicesToKeep, 'principal component 2']
                  ,finalDf.loc[indicesToKeep, 'principal component 3']
                  ,c = color
                  ,s = 50)
    ax.legend(targets)
    ax.grid()
    fig.savefig(outputFile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',help='input file')
    parser.add_argument('-g',help='features list file in one row tsv')
    parser.add_argument('-o',help='output file')
    parser.add_argument('-tc',help='target class')
    args = parser.parse_args()
    print('building pca', args.o)
    buildPcaImage(args.i, args.g, args.o, args.tc)

