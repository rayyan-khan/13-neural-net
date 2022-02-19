import sys
import math
import random

equation = sys.argv[1].strip()
print(equation)
if equation.find('<') != -1:
    sign = 'L' # less than
else: sign = 'G' # greater than
if '=' in equation: radius = float(equation[9:])
else: radius = float(equation[8:])
print(radius)

numCases = 100
errList = [5 for k in range(numCases)]

nodeStruct = [[1, 1, 1], # inputs
              [0, 0, 0, 0], # layer 1 cells
              [0, 0],
              [0]] # layer 2

# w11, w21, w31, w12, w22, w32
weightStruct = [[1 for k in range(12)],
                [1 for k in range(8)],
                [1, 1],
                [1]]


def transfer(n): # transfer function for neural network
    return 1/(1 + math.exp(-n))


def transDeriv(n): # derivative of that function
    return n*(1-n)


def dot(v1, v2): #v1 and v2 are same sized lists
    return sum(hadamard(v1, v2))


def hadamard(v1, v2): # returns a list with the respective
    productList = []  # indices of v1 and v2 multiplied
    for k in range(len(v1)):
        productList.append(v1[k] * v2[k])
    return productList


def determineCells(input, weightLayer):
    numInputs = len(input)
    numCells = int(len(weightLayer)/numInputs)
    indexList = []
    for k in range(numCells):
        indexList.append(weightLayer[k*numInputs:k*numInputs + numInputs])
    cellList = []
    for k in indexList:
        cellList.append(transfer(dot(input, k)))
    return cellList


def feedForward(inputs, weightList):
    currentCells = inputs
    nodeStruct = []
    for layer in range(len(weightList) - 1):
        nodeStruct.append(currentCells)
        currentCells = determineCells(currentCells, weightList[layer])
    nodeStruct.append(currentCells)
    finalWeights = weightList[len(weightList)- 1]
    outputs = [currentCells[k]*finalWeights[k] for k in
               range(len(currentCells))]
    nodeStruct.append(outputs)
    return nodeStruct, outputs


def calcError(target, result):
    return .5*(target - result)**2


def calcErrorList(errList):
    return sum(errList)/numCases


def reverse(n):
    if n == 1: return 0
    else: return 1


def backProp(nodeStruct, weights, target, alpha):
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

    for ind in range(len(nodeStruct) - 1):
        for nL in range(len(nodeStruct[ind])):
            for nR in range(len(nodeStruct[ind + 1])):
                gradient[ind][nL + nR*len(nodeStruct[ind])] = \
                    errStruct[ind + 1][nR]*nodeStruct[ind][nL]

    for ind in range(len(newWeights)):
        for weight in range(len(newWeights[ind])):
            newWeights[ind][weight] = weightStruct[ind][weight] + gradient[ind][weight]*alpha

    return newWeights


def randomGenerateWeights(weights):
    for layer in weights:
        for weight in range(len(layer)):
            layer[weight] = random.randint(-2, 2)
    return weights


def generateRandCase(n): # return [x, y] and whether or not
    global radius    # it is within circle (1 or 0)
    global sign
    # x, y always between -1.5, 1.5
    theta = random.random()*2*math.pi
    if n % 2: # return a pair within the circle
        randr = radius * random.uniform(0, 1)
        x = randr*math.cos(theta)
        y = randr*math.sin(theta)
    else: # return pair outside circle
        randr = radius * random.uniform(1, 1 + 1.5/radius)
        x = randr * math.cos(theta)
        y = randr * math.sin(theta)
    if sign == 'L':
        target = n % 2
    else:
        target = reverse(n % 2)
    return [x, y], target


def generateNearCase(n):
    # return [x, y] and whether or not
    global radius  # it is within circle (1 or 0)
    global sign
    # x, y always between -1.5, 1.5
    theta = random.random() * 2 * math.pi
    if n % 2:  # return a pair within the circle
        randr = radius - 0.00000001*n
        x = randr * math.cos(theta)
        y = randr * math.sin(theta)
    else:  # return pair outside circle
        randr = radius + 0.00000001*n
        x = randr * math.cos(theta)
        y = randr * math.sin(theta)
    if sign == 'L':
        target = n % 2
    else:
        target = reverse(n % 2)
    return [x, y], target


weightStruct = randomGenerateWeights(weightStruct)
minError = 1
minWeights = []
minTestNum = 0
minErrList = []
minFF = []
reset = 0
resetErr = 0
alpha = .1
for k in range(300000):
    if k < 300000:
        inputs, target = generateRandCase(k)
    else:
        inputs, target = generateNearCase(k)
    if not k % 5000: alpha = alpha * .975
    inputs.append(1)
    initialNodes, result = feedForward(inputs, weightStruct)
    result = result[0]
    err = calcError(target, result)
    errList[k%numCases] = err
    totalErr = calcErrorList(errList)
    newWeights = backProp(initialNodes, weightStruct, target, alpha)
    weightStruct = newWeights
    tempNodes, checkResult = feedForward(inputs, newWeights)
    checkResult = checkResult[0]
    err = calcError(target, checkResult)
    tempErrList = [err for err in errList]
    tempErrList[k%numCases] = err
    newErr = calcErrorList(tempErrList)
    if k - reset > 20000 and newErr - resetErr < .0001:
        reset = k
        resetErr = newErr
        weightStruct = randomGenerateWeights(weightStruct)
    if newErr < minError and k - reset > numCases:
        minError = newErr
        minWeights = newWeights
        minTestNum = k
        minErrList = errList
        minFF = initialNodes
        #print('TEST NUM:', k, 'newErr:', minError)
        if newErr < .001:
            #print('Errors: ', minErrList)
            print('layer cts: [{}, {}, {}, 1, 1]'
                  .format(len(nodeStruct[0]), len(nodeStruct[1]), len(nodeStruct[2])))
            for layer in newWeights:
                print(layer)
            quit()


print('Error:', minError)
print('layer cts: [{}, {}, {}, 1, 1]'
      .format(len(nodeStruct[0]), len(nodeStruct[1]), len(nodeStruct[2])))
for layer in minWeights:
    print(layer)