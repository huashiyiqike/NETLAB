from xml.dom.minidom import parse
import matplotlib.pyplot as plt
import numpy as np


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

def getLineStrokeOffset(xmlFileName):
    print "reading line stroke from xml file", xmlFileName.split('/')[-1]
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
    offsetSeq = []
    #plotSeq = []
    preCoord = []
    for stroke in xmlParse.getElementsByTagName('Stroke'):
        for point in stroke.getElementsByTagName('Point'):
            curCoord = []
            curCoord.append(float(point.getAttribute('x')))
            curCoord.append(float(point.getAttribute('y')))
            #print curCoord
            #curCoord[0] = curCoord[0] - bot_left[0]   # start from 0
            curCoord[1] = bot_right[1] - curCoord[1]  # up down inverse
            #print curCoord
            #plotSeq.append(curCoord)
            curCoord.append(0)
            inputSeq.append(curCoord)

            if(len(offsetSeq)):
                offsetSeq.append(map(lambda x: x[0] - x[1], zip(curCoord, preCoord)))
            else:
                offsetSeq.append([0.0, 0.0, 0.0])
            preCoord = curCoord
            offsetSeq[-1][-1] = 0

        inputSeq[-1][-1] = 1
        offsetSeq[-1][-1] = 1
    #return [inputSeq, plotSeq]
    return offsetSeq

if __name__ == '__main__':
    #testXml = "./lineStrokes/a01/a01-020/a01-020x-01.xml"
    testXml = "./lineStrokes/g04/g04-107/g04-107z-01.xml"

    #[inputs, plotSeq] = getLineStroke(testXml)
    offset = getLineStrokeOffset(testXml)
    ofile = open("testOut.txt", 'w')
    for point in offset:
        print >> ofile, point

    inputs = []
    for i in range(len(offset)):
        inputs.append(reduce(lambda offset1, offset2: map(lambda x: x[0] + x[1], zip(offset1, offset2)), offset[: i+1]))
        inputs[i][-1] = offset[i][-1]

    inArr = np.array(inputs)
    print inArr.shape
    print inArr
    xlim = max(max(inArr[:, 0]), abs(min(inArr[:, 0])))
    ylim = max(max(inArr[:, 1]), abs(min(inArr[:, 1])))
    lim = max(xlim, ylim)
       
    x = []
    y = []
    
    plt.figure(1)
    plt.xlim(0, lim)
    plt.ylim(- lim, lim)
    for line in inputs:
        x.append(line[0])
        y.append(line[1])
        if line[2] == 1:
            plt.plot(x, y, 'k')
            x = []
            y = []

    plt.show()
