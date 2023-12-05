# Functions

This folder contains the code needed to run the simulations. To simulate the propagation of the daisy quorum drive in space and time, we use a Crank-Nicolson finite difference method. 

`terms.py` contains the functions needed to determine each coefficients of the 256 equations in the system.

`control.py` contains some control functions to ensure that the system is correct.

`main_cst.py` simulates the system in an homogeneous 1D spatial environment (constant spatial steps).

`main_cst_2D.py` simulates the system in an homogeneous 2D spatial environment (constant spatial steps).

`main_var.py`simulates the system in an heterogeneous 1D spatial environment (the size of the spatial step is variable in space).


