'''
Created on 27 mars. 2018

@author: John Henry Dahlberg
'''

from builtins import print
from numpy import *
import matplotlib.pyplot as plt
import copy
import itertools


class HopfieldNetwork(object):

    def __init__(self, X0, nEpochs, approach='synchronous', rho=0, theta=0):
        random.seed(0)
        self.P = X0.shape[0]
        self.N = X0.shape[1]
        self.nEpochs = nEpochs
        self.X0 = X0
        self.W = zeros((self.N, self.N))
        self.approach = approach
        self.rho = rho
        self.mean = (X0.max()+ X0.min())/2
        self.variance = abs(X0.max() - X0.min())/2
        self.theta = theta


    def hebbianLearning(self):
        print('Original input:')
        print(self.X0)
        deltaW = dot((self.X0.T - self.rho), (self.X0 - self.rho))/self.N
        self.W += deltaW
        if self.approach == 'asynchronous':
            fill_diagonal(self.W, 0.)

    def hopFieldNetworkAlgorithm(self, X):
        if self.approach == 'synchronous':
            previousX = None
            for iteration in range(self.nEpochs):
                X = self.recall(X, iteration)
                if (X == previousX).all():
                    print('Found minimum energy point')
                    return X, iteration + 1, True
                    break
                previousX = copy.copy(X)

            return X, iteration+1, False
        elif self.approach == 'asynchronous':
            previousX = None
            for iteration in range(self.nEpochs):
                X, totalNeurons = self.recall(X, iteration)
                if (X == previousX).all():
                    print('Found minimum energy point after ' + str(totalNeurons) + ' iterations')
                    return X, iteration + 1, True
                    break
                previousX = copy.copy(X)

            return X, iteration + 1, False

    def recall(self, X, iteration):
        if self.approach == 'synchronous':
            Y = sign(dot(self.W, X.T))
            Y[Y==0] = 1
            return Y.T
        elif self.approach == 'asynchronous':
            d = int(sqrt(X.shape[1]))
            neurons = arange(0, self.N)
            random.shuffle(neurons)
            for neuronIteration in range(self.N):
                firingNeuron = neurons[neuronIteration]
                #firingNeuron = randint(0, self.N-1)
                Wneuron = array([self.W[:, firingNeuron]]).T
                a = sign(dot(X, Wneuron) - self.theta)
                a[a == 0] = 1
                y = self.mean + self.variance*a
                X[0, firingNeuron] = y

                totalNeurons = iteration*self.N + neuronIteration
                if totalNeurons%100 == 0:
                    self.energy = self.energyFunction(X)
                    #print('Energy = ' + str(self.energy))
                    Y = X.reshape(d, d)
                    plt.imshow(Y)
                    plt.title('Recalling memory, ' + str(totalNeurons) + ' neurons iterated')
                    plt.pause(0.001)

            return X, totalNeurons

    def energyFunction(self, x):
        Wx = dot(self.W, x.T)
        energy = -dot(x, Wx)[0][0]

        return energy


def generateData():
    x1 = array([[-1, -1, 1, -1, 1, -1, -1, 1]])
    x2 = array([[-1, -1, -1, -1, -1, 1, -1, -1]])
    x3 = array([[-1, 1, 1, -1, -1, 1, -1, 1]])
    X = concatenate([x1, x2, x3], axis=0)

    x1 = array([[1, -1, 1, -1, 1, -1, -1, 1]])
    x2 = array([[1, -1, -1, -1, -1, 1, -1, -1]])
    x3 = array([[1, 1, 1, -1, 1, 1, -1, 1]])
    Xd = concatenate([x1, x2, x3], axis=0)

    return X, Xd

def convergenceTests():
    X0, Xd = generateData()

    # Recall original inputs
    nEpochs = 1
    hn = HopfieldNetwork(X0, nEpochs)
    hn.hebbianLearning()
    print('\nDifference original vector X0 with recalled(X0):')
    X, iterations, converged = hn.hopFieldNetworkAlgorithm(X0)
    print(X - X0)
    print('\n')

    # Recall distorted inputs, 1 iteration
    X, iterations, converged = hn.hopFieldNetworkAlgorithm(Xd)
    print('Difference distorted vector with recalled after ' + str(iterations) + ' itaration:')
    print(X - X0)
    print('\n')

    # Recall distorted inputs, 2 iteration
    hn.nEpochs = 3
    X, iterations, converged = hn.hopFieldNetworkAlgorithm(Xd)
    print('Difference distorted vector with recalled after ' + str(iterations) + ' itarations:')
    print(X - X0)
    print('\n')

    # Recall distorted inputs, 1 iteration
    X0inv = -X0
    X, iterations, converged = hn.hopFieldNetworkAlgorithm(X0inv)
    print('Difference negoted vector with recalled after ' + str(iterations) + ' itaration:')
    print(X - X0inv)
    print('\n')

    Xspurious = zeros((2 ** 4 + 2 * X.shape[0], X.shape[1]))
    Xspurious[:X.shape[0], :] = X0
    Xspurious[X.shape[0]:2 * X.shape[0], :] = -X0
    row = 2 * X.shape[0]
    X1, X2, X3 = X[0, :], X[1, :], X[2, :]
    s = -1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    Xspurious[row, :] = s ** l * sign(s ** i * X1 + s ** j * X2 + s ** k * X3)
                    row += 1

    Xspurious = unique(Xspurious, axis=0)
    nSpurious = Xspurious.shape[0]
    print('Number of spurious formula: ' + str(nSpurious))

    Xevery = array(list(itertools.product([-1, 1], repeat=X.shape[1])))
    Xspurious, iterations, converged = hn.hopFieldNetworkAlgorithm(Xevery)
    Xspurious = unique(Xspurious, axis=0)
    nSpurious = Xspurious.shape[0]
    print('Number of spurious automating every combination: ' + str(nSpurious))


def generatePictures(nTrainingPatterns=3, d=32, type=None, density=0):
    N = d ** 2
    #nTrainingPatterns = 100
    maxValue = 1000000
    densityThreashold = density*maxValue
    if type == 'randomSparse':
        pict = []
        X0 = zeros((nTrainingPatterns, N))
        for i in range(nTrainingPatterns):
            sparsePic = array(random.randint(0, maxValue, (d, d)))
            sparsePic[sparsePic <= densityThreashold] = 1
            sparsePic[sparsePic > densityThreashold] = 0
            pict.append(sparsePic)
            X0[i, :] = array(pict[i].reshape(1, -1))
    else:
        pict = genfromtxt('pict.dat', dtype=None, delimiter=',')
        pict = pict.reshape(11, d, d)
        X0 = zeros((nTrainingPatterns, N))
        for i in range(nTrainingPatterns):
            X0[i, :] = array(pict[i].reshape(1, -1))

    return pict, X0

def recallImages():
    nEpochs = 5
    nTrainingPatterns = 3
    pict, X0 = generatePictures(nTrainingPatterns)
    N = X0.shape[1]

    #X0[X0 == -1] = 0
    hn = HopfieldNetwork(X0, nEpochs, 'asynchronous')
    hn.hebbianLearning()

    i = 10
    hn.nEpochs = 20
    plt.figure()
    plt.imshow(pict[i])
    if i == 10:
        plt.title('Combination of 2 learned images')
    plt.figure()
    triggerVector = copy.copy(pict[i].reshape(1, -1))
    Y, iterations, converged = hn.hopFieldNetworkAlgorithm(triggerVector)
    print('Energy for p' + str(i) + ' = ' + str(hn.energy))

    # Assign random weights
    #hn.W = random.randn(N, N)
    #hn.W = 0.5*(hn.W+hn.W.T)
    #fill_diagonal(hn.W, 0.)
    #Y, iterations, converged = hn.hopFieldNetworkAlgorithm(triggerVector)

def sparsePatterns():

    nEpochs = 10

    M = 10
    N = 10000
    pict, X0s = generatePictures(d = int(sqrt(N)), nTrainingPatterns=M, type='randomSparse', density=0.1)

    P =X0s.shape[0]

    rho = 1/(N*P)*sum(X0s)
    theta = 0

    hn = HopfieldNetwork(X0s, nEpochs, 'asynchronous', rho, theta)
    hn.hebbianLearning()

    i = 0# Picture to recall
    plt.figure()
    plt.imshow(pict[i])
    plt.title('Original random, sparse, image')
    plt.figure()
    triggerVector = pict[i].reshape(1, -1)
    Y, iterations, converged = hn.hopFieldNetworkAlgorithm(triggerVector)
    print(converged)



def main():

    #convergenceTests()

    recallImages()

    # sparsePatterns()


if __name__ == '__main__':
    main()
    plt.show()
