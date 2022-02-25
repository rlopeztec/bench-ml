# Creates PCA using target variable and features
# python pca_gtex_tissues.py

from mpl_toolkits import mplot3d
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')
import pandas as pd

class Pca3DPlotly:

    def createPca(self, gtex_tissues_file, genes_file):

        # read genes file and transform to single list, because DF transforms to a two dimensional list
        df_genes = pd.read_csv(genes_file, header=None, sep='\t')
        # these are genes/features
        l_genes = []
        l_features = []
        l_features.append('Outcome')
        for i in df_genes.iloc[0]:
            l_genes.append(i)
            l_features.append(i)
        #print('-0:',l_genes, len(l_genes))
        #print('-1:',l_features, len(l_features))
    
        # load dataset into Pandas DataFrame
        df_tissues = pd.read_csv(gtex_tissues_file, names=l_features, sep='\t', low_memory=False)
        #print('-2:',df_tissues, len(df_tissues))

        # Standardize the data
        from sklearn.preprocessing import StandardScaler
        # Separating out the features
        x = df_tissues.loc[:, l_genes].values
        df_tissues.Outcome = df_tissues.Outcome.replace({0:'Bladder', 1:'Vagina', 2:'Stomach'})
        c=0
        #'''
        for i in x:
          if c<5:
              print('values in sample:',i)
              cj=0
              for j in i:
                  #print('\tgene value:',j)
                  cj+=1
              print('\tgenes:',cj)
          else:
              break
          c+=1
        #'''
        # Separating out the target
        y = df_tissues.loc[:,['Outcome']].values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)

        # PCA project to 2D or 3D
        from sklearn.decomposition import PCA
        #pca = PCA(n_components=2) RENE TODO
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
                     #RENE, columns = ['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, df_tissues[['Outcome']]], axis = 1)

        # Visualize 2D projection
        fig = plt.figure(figsize = (14,14))
        ax = fig.add_subplot(1,1,1)
        ax = plt.axes(projection="3d") 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_zlabel('Principal Component 3', fontsize = 15) #RENE TODO
        ax.set_title('Tissues', fontsize = 20)
        targets = ['Bladder', 'Vagina', 'Stomach']
        colors = ['r', 'g', 'b']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['Outcome'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , finalDf.loc[indicesToKeep, 'principal component 3'] #RENE
                       , c = color
                       , s = 50)
        ax.legend(targets)
        ax.grid()
        fig.savefig('tissues1.png')

if __name__ == '__main__':
    pca = Pca3DPlotly()
    pca.createPca("gtex_6.tsv", "genes_all.tsv")

