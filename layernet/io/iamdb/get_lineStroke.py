from xml.dom.minidom import parse
import matplotlib.pyplot as plt


def getCorner(xmlParse, cornerName):
    corner = xmlParse.getElementsByTagName(cornerName)[0]
    cornerCoord = []
    cornerCoord.append(float(corner.getAttribute('x')))
    cornerCoord.append(float(corner.getAttribute('y')))
    return cornerCoord

def minusCoords(coord1, coord2):
    res = []
    res.append(coord1[0] - coord2[0])
    res.append(coord1[1] - coord2[1])
    return res

def getLineStroke(xmlFileName):
 #   print "reading line stroke from xml file", xmlFileName.split('/')[-1]
    xmlParse = parse(xmlFileName);

    #descrip = xmlParse.getElementsByTagName('WhiteboardDescription')[0]
    #print descrip.childNodes

    loc = xmlParse.getElementsByTagName('SensorLocation')[0].getAttribute('corner')
    #print loc
    assert loc == 'top_left'
    
    bot_right = getCorner(xmlParse, 'DiagonallyOppositeCoords')
    bot_left = getCorner(xmlParse, 'VerticallyOppositeCoords')
    top_right = getCorner(xmlParse, 'HorizontallyOppositeCoords')
    #print bot_right, bot_left, top_right;
    top_left = [];
    top_left.append(bot_left[0]) 
    top_left.append(top_right[1])
    #print top_left
    inputSeq = []
    #plotSeq = []
    for stroke in xmlParse.getElementsByTagName('Stroke'):
        for point in stroke.getElementsByTagName('Point'):
            curCoord = []
            curCoord.append(float(point.getAttribute('x')))
            curCoord.append(float(point.getAttribute('y')))
            #print curCoord
            curCoord[0] = curCoord[0] - bot_left[0]   # start from 0
            curCoord[1] = bot_right[1] - curCoord[1]  # up down inverse
            #print curCoord
            #plotSeq.append(curCoord)
            curCoord.append(0)
            inputSeq.append(curCoord)
        inputSeq[-1][-1] = 1
    #return [inputSeq, plotSeq]
    return inputSeq

if __name__ == '__main__':
    testXml = "/home/lau/homework/IAM_OnDB/lineStrokes/a01/a01-000/a01-000u-05.xml"
    #[inputs, plotSeq] = getLineStroke(testXml)
    inputs = getLineStroke(testXml)
    ofile = open("testOut.txt", 'w')
    for point in inputs:
        print >> ofile, point

    x = []
    y = []
    
    plt.figure(1)
    plt.xlim(0, 7000)
    plt.ylim(0, 7000)
    for line in inputs:
        x.append(line[0])
        y.append(line[1])
        if line[2] == 1:
            plt.plot(x, y, 'k')
            x = []
            y = []

    plt.show()
