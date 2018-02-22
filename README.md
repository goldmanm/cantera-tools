This module contains methods which can help you modify cantera mechanisms, run cantera models, and analyze the output. 

Checkout cookbook.ipynb to see what this package can do! 

The files and what they do:

file | function
--------|----------
`simulation.py` | functions for reducing cantera mechanisms and running simulations
`analysis.py` | functions for analyzing output from cantera (and functions in `simulation.py`
`rmg_and_cantera.py` | methods for converting Reaction Mechanism Generator output for easier reading in cantera
`kinetics_tools.py` | methods for calculating kinetics, without using RMG or Cantera
`plot_tools.py` | methods for improving matplotlib visuals

The dependencies for this package:

dependency | files applicable
------------|-------------
numpy | all modules
cantera | simulation, analysis, rmg_and_cantera
matplotlib | plot
rmg | rmg_and_cantera


The other files: `soln2cti.py` and `test_mechanism_from_solution.py` are used by `rmg_and_cantera.py` and are available 

To use this package, you need to have the files somewhere on your path. This varies by operating system. 


If you have any problem using this package, I'd love if you posted an issue. I'll get back as soon as I can!
