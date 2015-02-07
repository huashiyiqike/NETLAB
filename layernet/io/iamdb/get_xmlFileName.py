import os

def getXmlNames(prefix, destname):
    xmlNames = []
    allNames = os.listdir(prefix)
    #print allNames
    dest = destname.split('-')[1]
    for name in allNames:
        mid = name.split('-')[1]
        if(mid == dest):
            xmlNames.append(name)
    return xmlNames


if __name__ == '__main__':
    prefix = "/home/lau/homework/IAM_OnDB/lineStrokes/a01/a01-001"
    name = "a01-001w"
    print getXmlNames(prefix, name)
