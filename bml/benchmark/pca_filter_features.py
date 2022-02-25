# filter x number of features from gtex file using importance scoring from genes (features_6_tissues)
# python pca_filter_features.py 1000 ../../data/gtex/features_6_tissues.tsv.RF.20201013.0006 headers_genes.tsv gtex_6.tsv tsv gtex_6_1000.tsv genes_all_1000.tsv tsv 1
# python pca_filter_features.py 1000 ../../data/gtex/features_6_tissues.tsv.DT.20201013.0010 headers_genes.tsv gtex_6.tsv tsv gtex_6_1000.tsv genes_all_1000.tsv tsv 1
# feed gtex_6_1000.tsv genes_all_1000.tsv to jupyter plotly_pca_3d_interactive

import pandas as pd
import sys, argparse, subprocess, datetime
import ScoreFeatures
#from .ScoreFeatures import ScoreFeatures #otherwise it won't work with Popen from score.py

##########################################################################
# read genes scoring file and return top n genes
##########################################################################
def readGenesScoringFile(genesFile, nGenes):
    #print('readGenesScoringFile ' + genesFile + ' ' + nGenes, file=sys.stderr)
    topGenes = []
    #print(subprocess.Popen(['/home/rlopeztec/mysite/bml/benchmark/rene.sh']))
    with open(genesFile, 'r') as inputFile:
        c = 0
        for line in inputFile:
            if nGenes == 'all' or c < int(nGenes):
                topGenes.append(line.split('\t')[0])
                #print('topGenes', line.split('\t')[0], line.split('\t'))
            else:
                break
            c += 1
    #print('reading', genesFile, c, 'select', nGenes, 'genes', 'and got', len(topGenes), topGenes)
    #print(topGenes)
    return topGenes

##########################################################################
# read headers file and note position of gene if one of the top n genes
##########################################################################
def readHeadersFile(headersFile, topGenes, genesalloutFile, fileType, dataType, targetClass, dimNames, xSeqs, ycols):
    delimiter='\t'
    if fileType == 'csv':
        delimiter=','
    retHeaders = []
    retHeadersGenes = []
    genesOutput = open(genesalloutFile, 'w')

    # get headers from file or build headers from DNA sequencing data (ACGTs)
    if dataType not in ('DNA', 'Variants'):
        with open(headersFile, 'r') as inputFile:
            line = inputFile.readline()
            dimNames = line.split(delimiter)
            xSeqs, ycols = None, None
    c = 0
    first = True
    #print('RENE cols read headers file onehot:', len(cols), file=sys.stderr)
    for gene in dimNames:
        #print('RENE gene in cols:', gene, len(cols), len(topGenes), topGenes, file=sys.stderr)
        if gene in topGenes:
            #print('gene in top:', gene, len(topGenes))
            if gene not in retHeadersGenes:
                retHeaders.append(c)
                retHeadersGenes.append(gene)
                #print('gene not in:', gene, len(retHeadersGenes))
                if first:
                    first = False
                else:
                    genesOutput.write('\t')
                genesOutput.write(gene)
            else:
                print('DUPLICATE', c, gene)
        c += 1
    #print('headersFile:', headersFile, 'headers:', len(retHeaders), retHeaders)
    genesOutput.close()
    return retHeaders, dimNames, xSeqs, ycols

################################################################################
# read values file and keep genes if one of the top n genes in the output file #
################################################################################
def readInputFile(genesin, fileType, outputFile, headersGenes, targetClass, dataType, dimNames=None, xSeqs=None, ycols=None):

    #print('pca_filter_features readInputFile', genesin, fileType, outputFile, headersGenes, targetClass, datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)
    c = 0
    targetClassPos = -1
    delimiter='\t'
    if fileType == 'csv':
        delimiter=','
    outputFile = open(outputFile, 'w')

    # read original file and get only the columns scored as important
    if dataType == 'DNA' or dataType == 'Variants':
        # dimNames, xSeqs, ycols
        for i in range(0,len(xSeqs)):
            # write target/y class and feature headers
            if i == 0:
                #print('RENE y labels1:', type(ycols), file=sys.stderr)
                #print('RENE y labels2:', type(ycols.iloc[:0]), file=sys.stderr)
                #print('RENE y labels3:', i, type(ycols.iloc[i:0]), file=sys.stderr)
                #print('RENE y labels4:', str(ycols.iloc[i]), file=sys.stderr)
                #print('RENE y labels5:', type(ycols.iat[0]), file=sys.stderr)
                #print('RENE y labels6:', str(ycols.iat[i]), file=sys.stderr)
                #print('RENE y labels7:', str(ycols.iloc[i:i+1]), file=sys.stderr)
                #print('RENE y labels8:', str(ycols.iloc[:10]), file=sys.stderr)
                outputFile.write(targetClass)
                for header in headersGenes:
                    outputFile.write('\t' + dimNames[header])
                outputFile.write('\n')
            outputFile.write(str(ycols.iat[i]) + '\t')
            cols = 0
            for gene in headersGenes:
                #if i == 0:
                    #print('RENE xSeqs0:', headersGenes, file=sys.stderr)
                    #print('RENE xSeqs1:', len(xSeqs), i, gene, type(xSeqs), file=sys.stderr)
                    #print('RENE xSeqs2:', type(xSeqs.iat[i,gene]), file=sys.stderr)
                    #print('RENE xSeqs3:', str(int(xSeqs.iat[i,gene])), file=sys.stderr)
                outputFile.write(str(int(xSeqs.iat[i,gene])))
                if cols < len(headersGenes) - 1:
                    outputFile.write('\t')
                cols += 1
            outputFile.write('\n')
    else:
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
    #print('pca_filter_features, input lines', c, datetime.now().strftime("%Y/%m/%d %H:%M:%S"), file=sys.stderr)

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

    # last 3 parameters are dim names, x sequences, y labels for DNA data type
    readInputFile(args.genesin, args.filetype, args.genesout, headersGenes, int(args.p), None, None, None)

