import sys
import math
import random

caseFile = open(sys.argv[1], 'r')
caseList = [line.strip().split(' ') for line in caseFile]
for i in caseList: i.remove('=>')
cases = {tuple([float(k) for k in i[:len(i)-1]]): float(i[len(i)-1]) for i in caseList}
caseIDs = [[float(k) for k in i[:len(i)-1]] for i in caseList]
targetList = [cases[key] for key in cases]
errList = [10, 10, 10, 10]
#print('CASES: {}\n CASEIDS: {}\n TARGETLIST: {}\n'.format(cases, caseIDs, targetList))

nodeStruct = [[1, 1, 1], # inputs
              [0, 0], # layer 1 cells
              [0]] # layer 2

# w11, w21, w31, w12, w22, w32
weightStruct = [[1, 0, 1, 0, 1, 0],
                [0, 1],
                [1]]


def transfer(n): # transfer function for neural network
    return 1/(1 + math.exp(-n))


def transDeriv(n): # derivative of that function
    return n*(1-n)


def dot(v1, v2): #v1 and v2 are same sized lists
    return sum(hadamard(v1, v2))


def hadamard(v1, v2):
    productList = []
    for k in range(len(v1)):
        productList.append(v1[k] * v2[k])
    return productList


def determineCells(inputs, weightLayer):
    numInputs = len(inputs)
    numCells = int(len(weightLayer)/numInputs)
    indexList = []
    for k in range(numCells):
        indexList.append(weightLayer[k*numInputs:k*numInputs + numInputs])
    cellList = []
    for k in indexList:
        cellList.append(transfer(dot(inputs, k)))
    return cellList


def feedForward(inputs, weightList):
    #print('INPUTS', inputs)
    currentCells = inputs
    nodeStruct = []
    for layer in range(len(weightList) - 1):
        nodeStruct.append(currentCells)
        currentCells = determineCells(currentCells, weightList[layer])
    nodeStruct.append(currentCells)
    finalWeights = weightList[len(weightList)-1]
    outputs = [currentCells[k]*finalWeights[k] for k in
               range(len(currentCells))]
    nodeStruct.append(outputs)
    #print('NODES FF', nodeStruct)
    return nodeStruct, outputs


def calcError(target, result):
    return .5*(target - result)**2


def calcErrorList(errList):
    return sum(errList)/len(errList)


def backProp(nodeStruct, weights, target, alpha):
    #print('NODES', nodeStruct)
    newWeights = [[*w] for w in weights]
    errStruct = [[*nodes] for nodes in nodeStruct]
    errStruct[3][0] = target - errStruct[3][0]
    #print(errStruct)
    #print('previousErrNode {} weight {} node {}'.format(errStruct[3][0], weights[2][0], transDeriv(errStruct[2][0])))
    errStruct[2][0] = errStruct[3][0]*weights[2][0]*transDeriv(errStruct[2][0])
    #print(errStruct)
    errStruct[1][0] = errStruct[2][0]*weights[1][0]*transDeriv(errStruct[1][0])
    #print(errStruct)
    errStruct[1][1] = errStruct[2][0]*weights[1][1]*transDeriv(errStruct[1][1])
    #print('ERRSTRUCT', errStruct)

    #print('ERR {} NODE {}'.format(errStruct[2][0], nodeStruct[1][0]))
    '''
    gradient = [[nodeStruct[0][1]*errStruct[1][0], nodeStruct[0][2]*errStruct[1][0], nodeStruct[0][0]*errStruct[1][0],
                 nodeStruct[0][0] * errStruct[1][1], nodeStruct[0][1]*errStruct[1][1], nodeStruct[0][2]*errStruct[1][1]],
                [nodeStruct[1][0] * errStruct[2][0], nodeStruct[1][1]*errStruct[2][0]], [nodeStruct[2][0]*errStruct[3][0]]]
    print('GRADIENT', gradient)
    '''
    newWeights[2][0] = nodeStruct[2][0]*errStruct[3][0]*alpha+weights[2][0]
    newWeights[1][0] = nodeStruct[1][0]*errStruct[2][0]*alpha+weights[1][0]
    newWeights[1][1] = nodeStruct[1][1]*errStruct[2][0]*alpha+weights[1][1]
    newWeights[0][0] = nodeStruct[0][0]*errStruct[1][0]*alpha+weights[0][0]
    newWeights[0][1] = nodeStruct[0][1]*errStruct[1][0]*alpha+weights[0][1]
    newWeights[0][2] = nodeStruct[0][2]*errStruct[1][0]*alpha+weights[0][2]
    newWeights[0][3] = nodeStruct[0][0]*errStruct[1][1]*alpha+weights[0][3]
    newWeights[0][4] = nodeStruct[0][1]*errStruct[1][1]*alpha+weights[0][4]
    newWeights[0][5] = nodeStruct[0][2]*errStruct[1][1]*alpha+weights[0][5]

    #print('NODE STRUCT', nodeStruct)
    #print('ERR STRUCT', errStruct)
    #print('NEW WEIGHTS', newWeights)

    return newWeights


def randomGenerateWeights(weights):
    for layer in weights:
        for weight in range(len(layer)):
            layer[weight] = random.randint(-2, 2)
    return weights

#weightStruct = randomGenerateWeights(weightStruct)
minError = 1
minWeights = []
minTestNum = 0
minErrList = []
minFF = []
alpha = 1
for k in range(200000):
    inputs = caseIDs[k%len(caseIDs)].copy() # k%len(caseIDs) is number case on
    inputs.append(1)
    #print('INP', inputs)
    initialNodes, result = feedForward(inputs, weightStruct)
    result = result[0]
    err = calcError(targetList[k%len(caseIDs)], result)
    errList[k%len(caseIDs)] = err
    totalErr = calcErrorList(errList)
    newWeights = backProp(initialNodes, weightStruct, targetList[k%len(caseIDs)], alpha)
    weightStruct = newWeights
    tempNodes, checkResult = feedForward(inputs, newWeights)
    checkResult = checkResult[0]
    err = calcError(targetList[k%len(caseIDs)], checkResult)
    tempErrList = [err for err in errList]
    tempErrList[k%len(caseIDs)] = err
    newErr = calcErrorList(tempErrList)
    alpha = 0.1 - .01 * newErr
    #print('\n NODES: {} \nWEIGHTS {} \nERRORS {}'.format(tempNodes, newWeights, errList))
    #if k - reset > 20000:
    #    weightStruct = randomGenerateWeights(weightStruct)
    #print('TESTNUM: {} newErr: {}'.format(k, newErr))
    if k - minTestNum > 20000:
        weightStruct = randomGenerateWeights(weightStruct)
    if newErr < minError:
        minError = newErr
        minWeights = newWeights
        minTestNum = reset = k
        minErrList = errList
        minFF = initialNodes
        #print('TEST NUM:', k)
        if newErr < .01:
            #print('minFF: ', minFF)
            #print('Errors: ', minErrList)
            print('layer cts: [3, 2, 1, 1]')
            for layer in newWeights:
                print(layer)
            quit()
        #print('newERR: {} ERRLIST: {}'.format(newErr, errList))
        #for layer in newWeights:
        #    print(layer)


print('Error:', minError)
print('layer cts: [3, 2, 1, 1]')
for layer in minWeights:
    print(layer)