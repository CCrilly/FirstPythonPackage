from GaussianProcess import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Setting up test function from R code
    def f(t):
        j = np.arange(-5, 6, 1)
        total = 0
        for _ in j:
            total += np.sin(2 * np.pi * t / (27 + (_ / 2)))
        out = 2 * np.sin(2 * np.pi * t / 27) * np.cos(2 * np.pi * t / 27) + np.exp(-np.abs(t) / 100) * (
                    2 / 5) * total + t / 50
        return out

    # Trying some Matern stuff - nu = 5/2
    mygptwo = GP(xvalues=np.arange(-5, 5, 0.01))
    mygptwo.setparameters(p=0.5, l=1, nu=5 / 2, kernel_type="Matern")
    mygptwo.getKernelMatrix()
    fig = plt.gcf()
    mygptwo.plotCovarianceFunction()
    fig.savefig(fname="MT52CovFunction.png", dpi=300)
    mygptwo.generateSamplePath(n=5)
    fig = plt.gcf()
    mygptwo.plotSamplePath()
    fig.savefig(fname="MT52CovPaths.png", dpi=300)

    # Trying some Matern stuff - nu = 3/2
    mygptwo = GP(xvalues=np.arange(-5, 5, 0.01))
    mygptwo.setparameters(p=0.5, l=1, nu=3 / 2, kernel_type="Matern")
    mygptwo.getKernelMatrix()
    fig = plt.gcf()
    mygptwo.plotCovarianceFunction()
    fig.savefig(fname="MT32CovFunction.png", dpi=300)
    mygptwo.generateSamplePath(n=5)
    fig = plt.gcf()
    mygptwo.plotSamplePath()
    fig.savefig(fname="MT32CovPaths.png", dpi=300)

    # Power Exponential - Santner Figure 2.6
    mygpthree = GP(xvalues=np.arange(-5, 5, 0.01))
    mygpthree.setparameters(p=0.5, l=1, nu=3 / 2, pepower = 2, kernel_type="PE")
    mygpthree.getKernelMatrix()
    fig = plt.gcf()
    mygpthree.plotCovarianceFunction()
    fig.savefig(fname="PE2CovFunction.png", dpi=300)
    fig = plt.gcf()
    mygpthree.generateSamplePath(n=5)
    mygpthree.plotSamplePath()
    fig.savefig(fname="PE2CovPaths.png", dpi=300)


    # Power Exponential - Santner Figure 2.6
    mygpthree = GP(xvalues=np.arange(-5, 5, 0.01))
    mygpthree.setparameters(p=0.5, l=1, nu=3 / 2, pepower = 0.75, kernel_type="PE")
    mygpthree.getKernelMatrix()
    fig = plt.gcf()
    mygpthree.plotCovarianceFunction()
    fig.savefig(fname="PE75CovFunction.png", dpi=300)
    fig = plt.gcf()
    mygpthree.generateSamplePath(n=5)
    mygpthree.plotSamplePath()
    fig.savefig(fname="PE75CovPaths.png", dpi=300)

    # Power Exponential - Santner Figure 2.6
    mygpthree = GP(xvalues=np.arange(-5, 5, 0.01))
    mygpthree.setparameters(p=0.5, l=1, nu=3 / 2, pepower = 0.2, kernel_type="PE")
    mygpthree.getKernelMatrix()
    fig = plt.gcf()
    mygpthree.plotCovarianceFunction()
    fig.savefig(fname="PE02CovFunction.png", dpi=300)
    fig = plt.gcf()
    mygpthree.generateSamplePath(n=5)
    mygpthree.plotSamplePath()
    fig.savefig(fname="PE02CovPaths.png", dpi=300)

    # Plotting example function
    ts = np.arange(-150, 150, 0.1)
    #plt.plot(ts, f(ts))
    #plt.show()

    # Creating some noiseless observations
    samplexs = np.random.normal(0, 50, 10)
    sampleys = f(samplexs)

    # Making a gp with the given x values
    mygpthree = GP(xvalues=samplexs)
    # Can change this to other covariance functions and will not perform as well as periodic
    mygpthree.setparameters(p=2, l=1, kernel_type="Periodic")

    plt.scatter(samplexs, sampleys)
    plt.show()

    k = np.matmul(mygpthree.posteriorMean(ts), np.transpose(sampleys))

    plt.plot(ts, k, 'g-')
    plt.plot(ts, f(ts), 'b')
    plt.scatter(samplexs, sampleys, c='r', marker='+')
    plt.ylim(-5, 5)
    plt.show()

    print(k)
    print(f(samplexs))

    mygp = GP(xvalues=np.arange(-1, 1, 0.01))

    mygp.setparameters(p=2, l=1)

    mygp.getparameters()

    mygp.getKernelMatrix()

    # fig = plt.gcf()
    mygp.plotCovarianceFunction()
    # fig.savefig(fname="p2CovFunction.png", dpi=300)

    mygp.generateSamplePath(n=10)

    # fig = plt.gcf()
    mygp.plotSamplePath()
    # fig.savefig(fname="p2Paths.png", dpi=300)

    # Non differentiable kernel
    mygp.setparameters(p=0.5, l=1)

    mygp.getparameters()

    mygp.getKernelMatrix()

    # fig = plt.gcf()
    mygp.plotCovarianceFunction()
    # fig.savefig(fname="p05CovFunction.png", dpi=300)

    mygp.generateSamplePath(n=10)

    # fig = plt.gcf()
    mygp.plotSamplePath()
    # fig.savefig(fname="p05Paths.png", dpi=300)


