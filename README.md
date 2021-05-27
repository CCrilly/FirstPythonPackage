# DESCRIPTION -- First Python Package
This is a simple package containing one class - A gaussian process class with some different kernels from which you can draw samples and then plot them. the method used to create the package was based on the tutorial at https://packaging.python.org/tutorials/packaging-projects/


# Installation
# On linux run the following commands in the terminal to use the package

# Create a virtual environment using
python3 -m venv <DIR>
source <DIR>/bin/activate

# Still in the virtual environment run
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps First-Package-Conor-Crilly

# Run python in the console using
python3

# Import the package using
import example_pkg

# You could use the package as follows (making sure numpy and matplotlib are installed in the venv) see main.py for more information
import numpy as np  
import matplotlib.pyplot as plt  
from example_pkg import GaussianProcess as GP  

mygptwo = GP(xvalues=np.arange(-5, 5, 0.01))  
mygptwo.setparameters(p=0.5, l=1, nu=5 / 2, kernel_type="Matern")  
mygptwo.getKernelMatrix()  
fig = plt.gcf()  
mygptwo.plotCovarianceFunction()
