import sys
import math

textfile, function, data = sys.argv[1], sys.argv[2], \
                           [float(k) for k in sys.argv[3:]]
weightFile = open(textfile, 'r')
weightList = []
for line in weightFile:
    weightList.append([float(k) for k in line.strip().split(' ')])


def transfer(n):
    global function
    if function == 'T1':
        return n
    elif function == 'T2':
        if n > 0: return n
        else: return 0
    elif function == 'T3':
        return 1/(1 + math.exp(-n))
    else:
        return -1 + 2/(1 + math.exp(-n))


def dotProduct(v1, v2): #v1 and v2 are same sized lists
    productList = []
    for k in range(len(v1)):
        productList.append(v1[k]*v2[k])
    return sum(productList)


def determineCells(inputs, weightLine):
    numInputs = len(inputs)
    numCells = int(len(weightLine)/numInputs)
    indexList = []
    for k in range(numCells):
        indexList.append(weightLine[k*numInputs:k*numInputs + numInputs])
    cellList = []
    for k in indexList:
        cellList.append(transfer(dotProduct(inputs, k)))
    return cellList


def feedForward(data, weightList):
    inputs = data
    nodeStruct = []
    for layer in range(len(weightList)-1):
        nodeStruct.append(inputs)
        inputs = determineCells(inputs, weightList[layer])
    finalWeights = weightList[len(weightList)-1]
    outputs = [inputs[k]*finalWeights[k] for k in range(len(inputs))]
    nodeStruct.append(outputs)
    print(nodeStruct)
    return outputs


print(' '.join(str(k) for k in feedForward(data, weightList)))