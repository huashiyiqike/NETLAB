import re
from sets import Set

def getTargetStringCompress(asciiFileName):
    print "reading label from ", asciiFileName.split('/')[-1]
    targetStrings = []
    wordTargetStrings = []
    charSet = Set()
    asciiFile = open(asciiFileName, 'r')
    asciiLines = asciiFile.readlines()
    flag = False
    for line in asciiLines:
        line = line.strip()
        if(flag and line != ""):
            wordTargetStrings.append(line)
            ts = ""
            words = line.split()
            for word in words:
                for c in word:
                    if re.match("[\&\*\]\[\/\+\;\#\)\(\:\!\?\d]", c):
                        c = 'Non'
                    ts += c + ' '
                    charSet.add(c)

            ts = ts.strip()
            targetStrings.append(ts)
            print ts

        if(re.match(r'^CSR', line)):
            #print "find CSR"
            flag = True
    #print charSet
    #l = list(charSet)
    #print l
    return [targetStrings, wordTargetStrings, charSet]


if __name__ == "__main__":
    fileName = "/home/lau/homework/IAM_OnDB/ascii/a01/a01-000/a01-000u.txt"
    getTargetStringCompress(fileName)
