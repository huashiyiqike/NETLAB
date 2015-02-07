import os 
from sets import Set

from get_lineStroke import getLineStroke
from get_targetStrings import getTargetString
from get_xmlFileName import getXmlNames
import netcdf_helpers
import numpy as np


targetStrings = []
wordTargetStrings = []
charSet = Set()
inputs = []
labels = []
seqDims = []
seqLengths = []
seqTags = []

# testset_t = "./iamdb/task1/trainset.txt"
testset_t = "./iamdb/task1/my.txt"
xmlPrefix = "./iamdb/lineStrokes/"
asciiPrefix = "./iamdb/ascii/"
ncFileName = "./iamdb/iamondb.nc"

trainFileList = file(testset_t).readlines()
for l in trainFileList:
    l = l.strip()
  #  print "loading file ", l
    # print l[0:7]
    curXmlDir = os.path.join(xmlPrefix, l.split('-')[0], l[0:7])
    curAsciiDir = os.path.join(asciiPrefix, l.split('-')[0], l[0:7])

    curAsciiFilePath = os.path.join(curAsciiDir, l + ".txt")
    [curTargetString, curWordTargetString, curCharSet] = getTargetString(curAsciiFilePath)
    targetStrings.extend(curTargetString)
    wordTargetStrings.extend(curWordTargetString)
    # print len(curTargetString)
    # print curCharSet
    charSet = charSet.union(curCharSet)

#     for i in range(len(curTargetString)):
#         print curWordTargetString[i]
#         print curTargetString[i]
    
    xmlNames = getXmlNames(curXmlDir, l)
    assert len(curTargetString) == len(xmlNames)
    for xmlName in xmlNames:
        seqTags.append(xmlName)
        xmlFilePath = os.path.join(curXmlDir, xmlName)
        curLineStroke = getLineStroke(xmlFilePath)
        # print len(curLine)
        inputs.extend(curLineStroke)
        seqLengths.append(len(curLineStroke))
        seqDims.append([len(curLineStroke)])

inputsArr = np.array(inputs)
inputMeans = np.mean(inputsArr, 0)
inputStds = np.std(inputsArr, 0)
inputsArr[:, :-1] = (inputsArr[:, :-1] - inputMeans[:-1]) / inputStds[:-1]
inputs = inputsArr.tolist()
# print inputMeans
# print inputStds
# print inputs

labels = list(charSet)
print len(labels), labels
# i = 0
# for s in targetStrings:
#    i += 1
#    print i
#    print s
# for point in inputs:
#    print point
# for s in wordTargetStrings:
#    print s
ncFile = netcdf_helpers.NetCDFFile(ncFileName, 'w')

netcdf_helpers.createNcDim(ncFile, 'numSeqs', len(seqLengths))
netcdf_helpers.createNcDim(ncFile, 'numTimesteps', len(inputs))
netcdf_helpers.createNcDim(ncFile, 'inputPattSize', len(inputs[0]))
netcdf_helpers.createNcDim(ncFile, 'numDims', 1)
netcdf_helpers.createNcDim(ncFile, 'numLabels', len(labels))

netcdf_helpers.createNcStrings(ncFile, 'seqTags', seqTags, ('numSeqs', 'maxSeqTagLength'), 'sequence tags') 
netcdf_helpers.createNcStrings(ncFile, 'labels', labels, ('numLabels', 'maxLabelLength'), 'labels')
netcdf_helpers.createNcStrings(ncFile, 'targetStrings', targetStrings, ('numSeqs', 'maxTargetStringLength'), 'target strings')
netcdf_helpers.createNcStrings(ncFile, 'wordTargetStrings', wordTargetStrings, ('numSeqs', 'maxWordTargetStringLength'), 'word target strings')
netcdf_helpers.createNcVar(ncFile, 'seqLengths', seqLengths, 'i', ('numSeqs',), 'seq lengths')
netcdf_helpers.createNcVar(ncFile, 'seqDims', seqDims, 'i', ('numSeqs', 'numDims'), 'sequence dimensions')
netcdf_helpers.createNcVar(ncFile, 'inputs', inputs, 'f', ('numTimesteps', 'inputPattSize'), 'input patterns')

print "closing file", ncFileName
ncFile.close()
