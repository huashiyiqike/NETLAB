#Four params need to change
#testset_t, ncFileName, mean_std_flie_name, points_name


import os 
from get_lineStroke_offset import getLineStrokeOffset
from get_xmlFileName import getXmlNames
from get_targetStrings_compress import getTargetStringCompress
from sets import Set
import numpy as np
import netcdf_helpers

labels = ['unstroke', 'stroke']
inputs = []
targetPatterns = []
targetClasses = []
seqDims = []
seqLengths = []
seqTags = []
chars = []
targetStrings = []
wordTargetStrings = []
charSet = Set()

testset_t = "./opitions/trainset_big.txt"
xmlPrefix = "./lineStrokes/"
asciiPrefix = "./ascii/"
ncFileName = "./iamondb_s_offset.nc"
meanStdName = "./mean_std_s_offset.txt"
charsetName = "./char_set.txt"

trainFileList = file(testset_t).readlines()
for l in trainFileList:
    l = l.strip()
    print "loading file ", l
    #print l[0:7]
    curXmlDir = os.path.join(xmlPrefix, l.split('-')[0], l[0:7])
    curAsciiDir = os.path.join(asciiPrefix, l.split('-')[0], l[0:7])

    curAsciiFilePath = os.path.join(curAsciiDir, l + ".txt")
    [curTargetString, curWordTargetString, curCharSet] = getTargetStringCompress(curAsciiFilePath)
    targetStrings.extend(curTargetString)
    wordTargetStrings.extend(curWordTargetString)
    #print len(curTargetString)
    #print curCharSet
    charSet = charSet.union(curCharSet)

    xmlNames = getXmlNames(curXmlDir, l)
    for xmlName in xmlNames:
        seqTags.append(xmlName)
        xmlFilePath = os.path.join(curXmlDir, xmlName)
        curLineStroke = getLineStrokeOffset(xmlFilePath)
        #print len(curLine)
        inputs.append([0.0] * 3)
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

chars = list(charSet)
print "all chars:", len(chars), chars
out = open(charsetName, 'w')
for c in chars:
    print >>out, c,
out.close()

out = open(meanStdName, "w")
print >> out, "means %.12f,%.12f,%.12f" % (inputMeans.tolist()[0], inputMeans.tolist()[1], inputMeans.tolist()[2])
print >> out, "stds %.12f,%.12f,%.12f" % (inputStds.tolist()[0], inputStds.tolist()[1], inputStds.tolist()[2])
out.close()
#print inputMeans
#print inputStds
#print inputs[:3]
#print targetPatterns[:3]
#print targetClasses[:3]
#print len(targetPatterns)
#outData = open("points_test.txt", "w")
#for p in inputs:
#    for x in p:
#        if x == p[-1]:
#            print >> outData, x
#        else:
#            print >> outData, x,

ncFile = netcdf_helpers.NetCDFFile(ncFileName, 'w')

netcdf_helpers.createNcDim(ncFile, 'numSeqs', len(seqLengths))
netcdf_helpers.createNcDim(ncFile, 'numTimesteps', len(inputs))
netcdf_helpers.createNcDim(ncFile, 'inputPattSize', len(inputs[0]))
netcdf_helpers.createNcDim(ncFile, 'targetPattSize', len(targetPatterns[0]))
netcdf_helpers.createNcDim(ncFile, 'targetClassSize', len(targetClasses[0]))
netcdf_helpers.createNcDim(ncFile, 'numDims', 1)
netcdf_helpers.createNcDim(ncFile, 'numLabels', len(labels))
netcdf_helpers.createNcDim(ncFile, 'numChars', len(chars))
netcdf_helpers.createNcDim(ncFile, 'classOutputSize', len(labels))

netcdf_helpers.createNcStrings(ncFile, 'seqTags', seqTags, ('numSeqs', 'maxSeqTagLength'), 'sequence tags') 
netcdf_helpers.createNcStrings(ncFile, 'labels', labels, ('numLabels', 'maxLabelLength'), 'labels')
netcdf_helpers.createNcStrings(ncFile, 'chars', chars, ('numChars', 'maxCharLength'), 'chars')
netcdf_helpers.createNcStrings(ncFile, 'targetStrings', targetStrings, ('numSeqs', 'maxTargetStringLength'), 'target strings')
netcdf_helpers.createNcStrings(ncFile, 'wordTargetStrings', wordTargetStrings, ('numSeqs', 'maxWordTargetStringLength'), 'word target strings')
netcdf_helpers.createNcVar(ncFile, 'seqLengths', seqLengths, 'i', ('numSeqs', ), 'seq lengths')
netcdf_helpers.createNcVar(ncFile, 'seqDims', seqDims, 'i', ('numSeqs', 'numDims'), 'sequence dimensions')
netcdf_helpers.createNcVar(ncFile, 'inputs', inputs, 'f', ('numTimesteps', 'inputPattSize'), 'input patterns')
netcdf_helpers.createNcVar(ncFile, 'targetPatterns', targetPatterns, 'f', ('numTimesteps', 'targetPattSize'), 'real_t target patterns')
netcdf_helpers.createNcVar(ncFile, 'targetClasses', targetClasses, 'i', ('numTimesteps', 'targetClassSize'), 'class target patterns')

print "closing file", ncFileName
ncFile.close()
