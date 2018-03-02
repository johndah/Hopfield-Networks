from numpy import *


class HopfieldNetwork(object):

    def __init__(self, X):
        self.P = X.shape[0]
        self.N = X.shape[1]
        self.X = X
        self.W = zeros((self.N, self.N))

    def hopFieldNetworkAlgorithm(self):
        self.hebbianLearning(self.X)
        X = self.recall(self.X)
        print(X-self.X)

    def hebbianLearning(self, X):
        # for i in range(X.shape[1]):
        #    x = array([X[:, i]])
        #    print(x)
        #    self.W += dot(x.T, x)

        self.W += dot(X.T, X)
        #print(self.W)

    def recall(self, X):
        X = sign(dot(self.W, X.T))

        return X.T


def main():

    x1 = array([[-1, -1, 1, -1, 1, -1, -1, 1]])
    x2 = array([[-1, -1, -1, -1, -1, 1, -1, -1]])
    x3 = array([[-1, 1, 1, -1, -1, 1, -1, 1]])
    X = concatenate([x1, x2, x3], axis=0)
    hn = HopfieldNetwork(X)
    hn.hopFieldNetworkAlgorithm()

if __name__ == '__main__':
    main()