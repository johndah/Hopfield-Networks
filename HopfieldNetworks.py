from numpy import *


class HopfieldNetwork(object):

    def __init__(self, X):
        self.P = X.shape[0]
        self.N = X.shape[1]
        self.X = X
        self.W = zeros((self.N, self.N))

    def hopFieldNetworkAlgorithm(self):
        self.hebbianLearning(self.X)


    def hebbianLearning(self, X):
        #for
        deltaW = dot(X.T, X)
        self.W += deltaW

        # for i in range(X.shape[1]):
        #    x = array([X[:, i]])
        #    print(x)
        #    self.W += dot(x.T, x)

    def recall(self, X):
        X = sign(dot(self.W, X.T))

        return X.T
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

def main():

    X, Xd = generateData()
    hn = HopfieldNetwork(X)
    hn.hopFieldNetworkAlgorithm()
    Xrecalled = hn.recall(X)
    print(Xrecalled - X)
    XDrecalled = hn.recall(Xd)
    print(XDrecalled - X)

if __name__ == '__main__':
    main()