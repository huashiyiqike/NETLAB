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
    print "loading file ", l
    # print l[0:7]
    curXmlDir = os.path.join(xmlPrefix, l.split('-')[0], l[0:7])
    curAsciiDir = os.path.join(asciiPrefix, l.split('-')[0], l[0:7])

    curAsciiFilePath = os.path.join(curAsciiDir, l + ".txt")
    [curTargetString, curWordTargetString, curCharSet] = getTargetString(curAsciiFilePath)
    targetStrings.extend(curTargetString)
   # wordTargetStrings.extend(curWordTargetString)
    # print len(curTargetString)
    # print curCharSet
 #   charSet = charSet.union(curCharSet)

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
   #     seqDims.append([len(curLineStroke)])

inputsArr = np.array(inputs)
inputMeans = np.mean(inputsArr, 0)
inputStds = np.std(inputsArr, 0)

inputsArr[:, :-1] = (inputsArr[:, :-1] - inputMeans[:-1]) / inputStds[:-1]
inputs = inputsArr.tolist()
index = 0
file = open('test.txt', 'w');

file.write(str(max(seqLengths)) + '  ' + str(len(seqLengths)) + '  3\n')

for i in seqLengths:
  print i
  strs = str(i) + '\n'
  for j in range(0, i):
      strs += str(inputs[index][0]) + '  ' + str(inputs[index][1]) + '  ' + str(inputs[index][2]) + '\n'
      index += 1
  file.write(strs) 
  
file.write(str(inputMeans[0]) + ' ' + str(inputMeans[1]) + ' ' + str(inputMeans[2]) + '\n' + str(inputStds[0]) + ' ' + str(inputStds[1]) + ' ' + str(inputStds[2]))
