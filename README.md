# DESCRIPTION -- First Python Package
This is a simple package containing one class - A gaussian process class with some different kernels from which you can draw samples and then plot them.


# Installation
# On linux run the following commands in the terminal to use the package

# create a virtual environment using
python3 -m venv <DIR>
source <DIR>/bin/activate

# still in the virtual environment run
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps First-Package-Conor-Crilly

# run python in the console using
python3

# import the package using
import example_pkg

# You could use the package as follows (making sure numpy and matplotlib are installed in the venv
import numpy as np
import matplotlib.pyplot as plt
from example_pkg import GaussianProcess as GP

mygptwo = GP(xvalues=np.arange(-5, 5, 0.01))
mygptwo.setparameters(p=0.5, l=1, nu=5 / 2, kernel_type="Matern")
mygptwo.getKernelMatrix()
fig = plt.gcf()
mygptwo.plotCovarianceFunction()

