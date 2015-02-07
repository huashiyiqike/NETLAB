import re

inName = "./char_order.txt"
inLines = file(inName, 'r').readlines()

charDict = dict()
puncDict = dict()
for line in inLines:
    seq = line.split()
    if re.match('[a-zA-Z\.\,\'\"\-]', seq[1]):
        count = int(seq[2][1:])
        if charDict.has_key(seq[1]):
            charDict[seq[1]] += count
        else:
            charDict[seq[1]] = count
    elif re.match('\d', seq[1]):
        count = int(seq[2][1:])
        if puncDict.has_key('num'):
            puncDict['num'] += count
        else:
            puncDict['num'] = count
    else:
        count = int(seq[2][1:])
        if puncDict.has_key(seq[1]):
            puncDict[seq[1]] += count
        else:
            puncDict[seq[1]] = count

charList = sorted(charDict.iteritems(), key = lambda d : d[1])
puncDict = sorted(puncDict.iteritems(), key = lambda d : d[1])
print len(charList)
for t in charList:
    print t
print len(puncDict)
for t in puncDict:
    print t
s = "["
for t in puncDict:
    s += "\\" + t[0]
s += "]"
print s

newDict = dict()
for line in inLines:
    seq = line.split()
    if re.match("[\&\*\]\[\/\+\;\#\)\(\:\!\?\d]", seq[1]):
        count = int(seq[2][1:])
        if newDict.has_key("None"):
            newDict["None"] += count
        else:
            newDict["None"] = count
    else:
        count = int(seq[2][1:])
        if newDict.has_key(seq[1]):
            newDict[seq[1]] += count
        else:
            newDict[seq[1]] = count
newList = sorted(newDict.items(), key = lambda d : d[1])
print
print len(newList)
for t in newList:
    print t


