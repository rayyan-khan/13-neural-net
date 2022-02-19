import sys
import math
import random

caseFile = open(sys.argv[1], 'r')
caseList = [line.strip().split(' ') for line in caseFile]
for i in caseList: i.remove('=>')
caseIDs = [[float(k) for k in i[:len(i)-1]] for i in caseList]
targetList = [float(i[len(i)-1]) for i in caseList]
errList = [10 for k in range(len(caseIDs))]
#print('CASES: {}\n CASEIDS: {}\n TARGETLIST: {}\n'.format(cases, caseIDs, targetList))

nodeStruct = [[1 for k in range(len(caseIDs[0]) + 1)], # inputs
              [0, 0], # layer 1 cells
              [0]] # layer 2

# w11, w21, w31, w12, w22, w32
weightStruct = [[1 for k in range(((len(caseIDs[0]) + 1) * 2))],
                [1, 1],
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
    #print('WEIGHTLIST', weightList)
    for layer in range(len(weightList) - 1):
        nodeStruct.append(currentCells)
        currentCells = determineCells(currentCells, weightList[layer])
    nodeStruct.append(currentCells)
    finalWeights = weightList[len(weightList)- 1]
    #print('CURRENT CELLS', currentCells, 'finalWEIGHTS', finalWeights)
    outputs = [currentCells[k]*finalWeights[k] for k in
               range(len(currentCells))]
    nodeStruct.append(outputs)
    return nodeStruct, outputs


def calcError(target, result):
    return .5*(target - result)**2


def calcErrorList(errList):
    return sum(errList)


def backProp(nodeStruct, weights, target, alpha):
    #print('NODES', nodeStruct)
    newWeights = [[*w] for w in weights]
    gradient = [[*w] for w in weights]
    errStruct = [[*nodes] for nodes in nodeStruct]
    errStruct[len(errStruct) - 1][0] = target - \
                                       nodeStruct[len(errStruct) - 1][0]
    errStruct[len(errStruct) - 2][0] = \
        errStruct[len(errStruct) - 1][0] * weights[len(errStruct) - 2][0] \
        * transDeriv(nodeStruct[len(errStruct) - 2][0])

    for index in range(len(errStruct) - 3, 0, -1):
        for i in range(len(errStruct[index])):
            errStruct[index][i] = transDeriv(nodeStruct[index][i]) * \
                                  dot(errStruct[index + 1], [weights[index][w]
                                                              for w in range(len(weights[index]))
                                                              if w % len(errStruct[index]) == i])

    #print('ERRSTRUCT', errStruct)

    for ind in range(len(nodeStruct) - 1):
        for nL in range(len(nodeStruct[ind])):
            for nR in range(len(nodeStruct[ind + 1])):
                gradient[ind][nL + nR*len(nodeStruct[ind])] = \
                    errStruct[ind + 1][nR]*nodeStruct[ind][nL]

    #print('GRADIENT:', gradient)

    for ind in range(len(newWeights)):
        for weight in range(len(newWeights[ind])):
            newWeights[ind][weight] = weightStruct[ind][weight] + gradient[ind][weight]*alpha

    return newWeights


def randomGenerateWeights(weights):
    for layer in weights:
        for weight in range(len(layer)):
            layer[weight] = random.randint(-2, 2)
    return weights


weightStruct = randomGenerateWeights(weightStruct)
minError = 1
minWeights = []
minTestNum = 0
minErrList = []
minFF = []
reset = 0
resetErr = 0
alpha = 0.1
for k in range(300000):
    inputs = caseIDs[k%len(caseIDs)].copy() # k%len(caseIDs) is number case on
    inputs.append(1)
    #print('INPUTS: {} EXPECTED OUTPUT: {}'.format(inputs, targetList[0]))
    initialNodes, result = feedForward(inputs, weightStruct)
    #print('FEED FORWARD:', initialNodes)
    #print('CURRENT WEIGHTS:', weightStruct)
    result = result[0]
    newWeights = backProp(initialNodes, weightStruct, targetList[k%len(caseIDs)], alpha)
    #print('NEW WEIGHTS:', newWeights)
    weightStruct = newWeights
    tempNodes, checkResult = feedForward(inputs, newWeights)
    #print('NEW NODES:', tempNodes)
    checkResult = checkResult[0]
    err = calcError(targetList[k%len(caseIDs)], checkResult)
    errList[k%len(caseIDs)] = err
    newErr = calcErrorList(errList)
    #alpha = 2*(0.1 - .01*newErr)
    #print('ALPHA', alpha)
    #print('\n NODES: {} \nWEIGHTS {} \nERRORS {}'.format(tempNodes, newWeights, errList))
    #print('INPUTS {} TARGET {}'.format(inputs[0:2], targetList[k % len(caseIDs)]))
    if k - reset > 20000 and newErr - resetErr < .0001:
        reset = k
        resetErr = newErr
        #print('RAND NEW WEIGHTS')
        weightStruct = randomGenerateWeights(weightStruct)
    #print('TESTNUM: {} newErr: {}'.format(k, newErr))
    counter = 0
    if newErr < minError and k - reset > 4:
        minError = newErr
        minWeights = newWeights
        minTestNum = k
        minErrList = errList
        minFF = initialNodes
        #print('TEST NUM:', k, 'newErr:', minError)
        if counter % 1 == 0:
            counter += 1
            print(minError)
            #print('Errors: ', minErrList)
            #print('layer cts: [{}, 2, 1, 1]'.format(len(nodeStruct[0])))
            #for layer in newWeights:
            #    print(layer)
        counter1 = 0
        if newErr < .01 and counter1 % 50 == 0:
            counter1 += 1
            print('Errors: ', minErrList)
            print('layer cts: [{}, 2, 1, 1]'.format(len(nodeStruct[0])))
            for layer in newWeights:
                print(layer)
        if newErr < .009:
            #print('minFF: ', minFF)
            print('Errors: ', minErrList)
            print('layer cts: [{}, 2, 1, 1]'.format(len(nodeStruct[0])))
            for layer in newWeights:
                print(layer)
            quit()
        #print('newERR: {} ERRLIST: {}'.format(newErr, errList))
        #for layer in newWeights:
        #    print(layer)


print('Error:', minError)
print('layer cts: [{}, 2, 1, 1]'.format(len(nodeStruct[0])))
for layer in minWeights:
    print(layer)