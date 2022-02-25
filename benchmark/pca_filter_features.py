# filter x number of features from gtex file using importance scoring from genes (features_6_tissues)
# python pca_filter_features.py 1000 ../../data/gtex/features_6_tissues.tsv.RF.20201013.0006 headers_genes.tsv gtex_6.tsv tsv gtex_6_1000.tsv genes_all_1000.tsv tsv 1
# python pca_filter_features.py 1000 ../../data/gtex/features_6_tissues.tsv.DT.20201013.0010 headers_genes.tsv gtex_6.tsv tsv gtex_6_1000.tsv genes_all_1000.tsv tsv 1
# feed gtex_6_1000.tsv genes_all_1000.tsv to jupyter plotly_pca_3d_interactive

import argparse

##########################################################################
# read genes scoring file and return top n genes
##########################################################################
def readGenesScoringFile(genesFile, nGenes):
    print('readGenesScoringFile', genesFile, nGenes)
    retGenes = []
    with open(genesFile, 'r') as inputFile:
        c = 0
        for line in inputFile:
            if nGenes == 'all' or c < int(nGenes):
                retGenes.append(line.split('\t')[0])
                print('retGenes', line.split('\t')[0], line.split('\t'))
            else:
                break
            c += 1
    print('reading', genesFile, c, 'select', nGenes, 'genes', 'and got', len(retGenes), retGenes)
    #print(retGenes)
    return retGenes

##########################################################################
# read headers file and note position of gene if one of the top n genes
##########################################################################
def readHeadersFile(headersFile, topGenes, genesalloutFile, fileType):
    delimiter='\t'
    if fileType == 'csv':
        delimiter=','
    retHeaders = []
    retHeadersGenes = []
    genesOutput = open(genesalloutFile, 'w')
    with open(headersFile, 'r') as inputFile:
        line = inputFile.readline()
        c = 0
        cols = line.split(delimiter)
        first = True
        for gene in cols:
            print('gene in cols:', gene, len(cols))
            if gene in topGenes:
                print('gene in top:', gene, len(topGenes))
                if gene not in retHeadersGenes:
                    retHeaders.append(c)
                    retHeadersGenes.append(gene)
                    print('gene not in:', gene, len(retHeadersGenes))
                    if first:
                        first = False
                    else:
                        genesOutput.write('\t')
                    genesOutput.write(gene)
                else:
                    print('DUPLICATE', c, gene)
            c += 1
    print('headersFile:', headersFile, 'headers:', len(retHeaders), retHeaders)
    genesOutput.close()
    return retHeaders

##########################################################################
# read genes values file and keep genes if one of the top n genes
##########################################################################
def readInputFile(genesin, fileType, outputFile, headersGenes, targetClass):
    print('pca_filter_features.py readInputFile',  genesin, fileType, outputFile, headersGenes, targetClass)
    c = 0
    targetClassPos = -1
    delimiter='\t'
    if fileType == 'csv':
        delimiter=','
    outputFile = open(outputFile, 'w')
    with open(genesin, 'r') as inputFile:
        for line in inputFile:
            line = line.replace('\n','')
            listGenes = line.split(delimiter)

            # find the target class position
            if c == 0:
                for i in range(0,len(listGenes)):
                    if listGenes[i] == targetClass:
                        targetClassPos = i
                        break
            c += 1

            # add target class (y) to file as the 1st position
            outputFile.write(listGenes[targetClassPos] + '\t')

            cols = 0
            for gene in headersGenes:
                outputFile.write(listGenes[gene])
                if cols < len(headersGenes) - 1:
                    outputFile.write('\t')
                #if c < 3: print('gene value:', gene, listGenes[gene])
                cols += 1
            outputFile.write('\n')
    outputFile.close()
    #print('input lines', c)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n',      help='number of genes to keep')
    parser.add_argument('genes',  help='genes scoring file')
    parser.add_argument('headers', help='genes headers file')
    parser.add_argument('genesin', help='genes input values file')
    parser.add_argument('filetype', help='input file type')
    parser.add_argument('genesout', help='genes output values file')
    parser.add_argument('genesallout', help='ngenes output file')
    parser.add_argument('t',      help='file type tsv or csv')
    parser.add_argument('p',      help='target class position')
    args = parser.parse_args()
    #parser.add_argument('', help='')
    print('filter features')
    print(args)
    topGenes = readGenesScoringFile(args.genes, args.n)
    headersGenes = readHeadersFile(args.headers, topGenes, args.genesallout, args.t)
    readInputFile(args.genesin, args.filetype, args.genesout, headersGenes, int(args.p))

