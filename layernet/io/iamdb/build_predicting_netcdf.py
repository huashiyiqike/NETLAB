#Four params need to change
#testset_t, ncFileName, mean_std_flie_name, points_name


import os 
from get_lineStroke import getLineStroke
from get_xmlFileName import getXmlNames
import numpy as np
import netcdf_helpers

labels = ['unstroke', 'stroke']
inputs = []
targetPatterns = []
targetClasses = []
seqDims = []
seqLengths = []
seqTags = []

testset_t = "./opitions/trainset_big.txt"
xmlPrefix = "./lineStrokes/"
asciiPrefix = "./ascii/"
ncFileName = "./iamondb_p_big.nc"

trainFileList = file(testset_t).readlines()
for l in trainFileList:
    l = l.strip()
    print "loading file ", l
    #print l[0:7]
    curXmlDir = os.path.join(xmlPrefix, l.split('-')[0], l[0:7])

    xmlNames = getXmlNames(curXmlDir, l)
    for xmlName in xmlNames:
        seqTags.append(xmlName)
        xmlFilePath = os.path.join(curXmlDir, xmlName)
        curLineStroke = getLineStroke(xmlFilePath)
        #print len(curLine)
        inputs.append([0.0]*3)
        inputs.extend(curLineStroke)
        inputs = inputs[:-1]
        seqLengths.append(len(curLineStroke))
        seqDims.append([len(curLineStroke)])
        for coord in curLineStroke:
            targetPatterns.append(coord[:-1]);
            targetClasses.append([coord[-1]]);

inputsArr = np.array(inputs)
inputMeans = np.mean(inputsArr, 0)
inputStds = np.std(inputsArr, 0)
inputsArr[:, :-1] = (inputsArr[:, :-1] - inputMeans[:-1])/inputStds[:-1]
inputs = inputsArr.tolist()
targetPattArr = np.array(targetPatterns)
targetPatterns = ((targetPattArr - inputMeans[:-1])/inputStds[:-1]).tolist()

out = open("mean_std_big.txt", "w")
print >> out, "means %.12f,%.12f,%.12f" % (inputMeans.tolist()[0], inputMeans.tolist()[1], inputMeans.tolist()[2])

print >> out, "stds %.12f,%.12f,%.12f" % (inputStds.tolist()[0], inputStds.tolist()[1], inputStds.tolist()[2])
#print inputMeans
#print inputStds
#print inputs[:3]
#print targetPatterns[:3]
#print targetClasses[:3]
#print len(targetPatterns)
outData = open("points.txt", "w")
for p in inputs:
    for x in p:
        if x == p[-1]:
            print >> outData, x
        else:
            print >> outData, x,

ncFile = netcdf_helpers.NetCDFFile(ncFileName, 'w')

netcdf_helpers.createNcDim(ncFile, 'numSeqs', len(seqLengths))
netcdf_helpers.createNcDim(ncFile, 'numTimesteps', len(inputs))
netcdf_helpers.createNcDim(ncFile, 'inputPattSize', len(inputs[0]))
netcdf_helpers.createNcDim(ncFile, 'targetPattSize', len(targetPatterns[0]))
netcdf_helpers.createNcDim(ncFile, 'targetClassSize', len(targetClasses[0]))
netcdf_helpers.createNcDim(ncFile, 'numDims', 1)
netcdf_helpers.createNcDim(ncFile, 'numLabels', len(labels))
netcdf_helpers.createNcDim(ncFile, 'classOutputSize', len(labels))

netcdf_helpers.createNcStrings(ncFile, 'seqTags', seqTags, ('numSeqs', 'maxSeqTagLength'), 'sequence tags') 
netcdf_helpers.createNcStrings(ncFile, 'labels', labels, ('numLabels', 'maxLabelLength'), 'labels')
netcdf_helpers.createNcVar(ncFile, 'seqLengths', seqLengths, 'i', ('numSeqs', ), 'seq lengths')
netcdf_helpers.createNcVar(ncFile, 'seqDims', seqDims, 'i', ('numSeqs', 'numDims'), 'sequence dimensions')
netcdf_helpers.createNcVar(ncFile, 'inputs', inputs, 'f', ('numTimesteps', 'inputPattSize'), 'input patterns')
netcdf_helpers.createNcVar(ncFile, 'targetPatterns', targetPatterns, 'f', ('numTimesteps', 'targetPattSize'), 'real_t target patterns')
netcdf_helpers.createNcVar(ncFile, 'targetClasses', targetClasses, 'i', ('numTimesteps', 'targetClassSize'), 'class target patterns')

print "closing file", ncFileName
ncFile.close()
