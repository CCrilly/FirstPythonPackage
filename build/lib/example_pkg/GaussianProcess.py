import numpy as np
import matplotlib.pyplot as plt


# Class for a simple Gaussian process with custom covariance kernels
class GP:

    def __init__(self, xvalues):
        self.xvalues = xvalues
        self.covarianceFunctionType = "Ornstein Uhlenbeck"
        self.p = 2
        self.l = 1
        self.nu = 3 / 2
        self.pepower = 2

    def setparameters(self, p, l, pepower=2, nu=3 / 2, kernel_type="Ornstein Uhlenbeck"):
        self.p = p
        self.l = l
        self.nu = nu
        self.pepower = pepower
        self.covarianceFunctionType = kernel_type
        print(f"Parameters have been set to p={self.p} and l={self.l} and nu={self.nu} and kernel={self.covarianceFunctionType}")

    def getparameters(self):
        print(
            f"X values: {self.xvalues} \nCovariance Function Of Type: {self.covarianceFunctionType} with p={self.p} "
            f"and l={self.l} and nu={self.nu}")

    # For matern only use the functions with nice properties
    def covarianceFunction(self, xi):
        if self.covarianceFunctionType == "Ornstein Uhlenbeck":
            return np.exp(-((np.abs(xi) ** self.p) / self.l))
        elif self.covarianceFunctionType == "Matern":
            if self.nu == 3 / 2:
                return (1 + ((np.sqrt(3) * np.abs(xi)) / self.l)) * np.exp((-np.sqrt(3) * np.abs(xi)) / self.l)
            elif self.nu == 5 / 2:
                return (1 + np.sqrt(5) * np.abs(xi) / self.l + 5*(np.abs(xi)**2)/(3*(self.l**2)))*np.exp((-np.sqrt(5) * np.abs(xi)) / self.l)
            else:
                raise TypeError("set nu to 3/2 or 5/2")
        elif self.covarianceFunctionType == "Periodic":
            return (2.1**2)*np.exp(-(np.abs(xi)**2)/(2*40**2))*np.exp((-np.sin(np.pi*np.abs(xi)/28)**2)/(2*0.37**2))
        elif self.covarianceFunctionType == "PE":
            return np.exp(-np.abs(xi)**self.pepower)

    def getKernelMatrix(self):
        if None in self.xvalues:
            raise TypeError("please specify as a NumPy array with full entries")
        else:
            self.kernelMatrix = self.covarianceFunction(np.abs(np.subtract.outer(self.xvalues, self.xvalues)))
        return self.kernelMatrix

    def plotCovarianceFunction(self):
        plt.plot(self.xvalues, self.covarianceFunction(self.xvalues))
        plt.show()

    def generateSamplePath(self, n):
        # For reproducibility purposes
        np.random.seed(123)
        self.currentSamplePath = np.random.multivariate_normal(mean=np.zeros(self.xvalues.shape), cov=self.kernelMatrix,
                                                               size=n)
        self.currentSamplePath = np.reshape(self.currentSamplePath, newshape=(n, np.size(self.xvalues)))

    def plotSamplePath(self):
        for i in range(np.shape(self.currentSamplePath)[0]):
            plt.plot(self.xvalues, self.currentSamplePath[i])
        plt.show()

    def posteriorMean(self, x):
        return np.matmul(self.covarianceFunction(np.subtract.outer(x, self.xvalues)), np.linalg.inv(self.covarianceFunction(np.subtract.outer(self.xvalues, self.xvalues))))
