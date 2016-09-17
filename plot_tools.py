# -*- coding: utf-8 -*-
"""
This file contains functions that adjust matplotlib properties 
of figures to give adequate output to both presentations (larger fonts) and 
publications.(more information). Features include:

* automatically allowing latex math mode
* improving quality of image
* limiting number of tick labels in presentations

To get this function to work in Jupyter notebook, run `%matplotlib inline` 
before `plot_tools.xxxxxx()`. Here's an example:

```
import plot_tools
%matplotlib inline
plot_tools.presentation()
import matplotlib.pyplot as plt
```
"""

import matplotlib as mpl
#import matplotlib.pyplot as plt

def general_changes():
    mpl.rcParams['figure.dpi'] = 200
    mpl.rcParams['savefig.dpi'] = 200
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['text.latex.preamble'] = [r"\usepackage[version=4]{mhchem}"]

def set_linear_tick_locator_for_mpl(self,axis):
    """Hack to limit number of labels automatically
    Written by Schulfa Schwein on stack overflow: 
    http://stackoverflow.com/questions/10437689/matplotlib-globally-set-number-of-ticks-x-axis-y-axis-colorbar"
    """
    if isinstance(axis, mpl.axis.XAxis):
        axis.set_major_locator(mpl.ticker.MaxNLocator())
    elif isinstance(axis, mpl.axis.YAxis):
        axis.set_major_locator(mpl.ticker.MaxNLocator())

    axis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axis.set_minor_locator(mpl.ticker.NullLocator())
    axis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    
def set_log_tick_locator_for_mpl(self,axis):
    """Hack to limit number of labels automatically
    Written by Schulfa Schwein on stack overflow: 
    http://stackoverflow.com/questions/10437689/matplotlib-globally-set-number-of-ticks-x-axis-y-axis-colorbar"
    """

    axis.set_major_locator(mpl.ticker.LogLocator(self.base,numticks=4))
    axis.set_major_formatter(mpl.ticker.LogFormatterMathtext(self.base))
    axis.set_minor_locator(mpl.ticker.LogLocator(self.base, self.subs))
    axis.set_minor_formatter(mpl.ticker.NullFormatter())
    
def reduce_labels(nlabels = 3):
    # make MaxNLocator default
    mpl.scale.LinearScale.set_default_locators_and_formatters = set_linear_tick_locator_for_mpl
    mpl.scale.LogScale.set_default_locators_and_formatters = set_log_tick_locator_for_mpl
    mpl.ticker.MaxNLocator.default_params['nbins'] = nlabels-1 #this might 
    
def presentation():
    reduce_labels()
    mpl.rcParams['figure.figsize'] = (12.0,8.0) # default = (6.0, 4.0)
    mpl.rcParams['font.size'] = 33 # default = 10
    mpl.rcParams['legend.fontsize'] = 'small' # default = large
    mpl.rcParams['axes.linewidth'] = 2.0 # default = 1.0
    mpl.rcParams['lines.linewidth'] = 2.0 # default = 1.0
    mpl.rcParams['patch.linewidth'] = 1.0 # default = 1.0
    mpl.rcParams['grid.linewidth'] = 1.0 # default = 0.5
    mpl.rcParams['xtick.major.width'] = 2.0 # default = 0.5
    mpl.rcParams['xtick.major.size'] = 8.0 # default = 0.5
    mpl.rcParams['ytick.major.size'] = 8.0 # default = 0.5
    mpl.rcParams['ytick.major.width'] = 2.0 # default = 0.5
    mpl.rcParams['xtick.major.pad'] = 16 #default = 4.0 (prevents overlap labels)
    general_changes()
    
    # for later need to set default bold font for presentation mode, but not sure how.
    
    
    #latex_preamble = ['\usepackage{fontspec}','\setmathfont{Cambria Math}']
    #mpl.rcParams['axes.formatter.use_mathtext']=True
    #mpl.rcParams['mathtext.fontset'] = 'custom'
    #mpl.rcParams['mathtext.default'] = 'sf'
    #mpl.rcParams['xtick.minor.width'] = 1.0 # default = 0.5
    #mpl.rcParams['ytick.minor.width'] = 1.0 # default = 0.5


def publication():
    mpl.rcParams['figure.figsize'] = (12.0,8.0) # default = (6.0, 4.0)
    mpl.rcParams['font.size'] = 18 # default = 10
    mpl.rcParams['axes.linewidth'] = 2.0 # default = 1.0
    mpl.rcParams['lines.linewidth'] = 2.0 # default = 1.0
    mpl.rcParams['patch.linewidth'] = 1.0 # default = 1.0
    mpl.rcParams['grid.linewidth'] = 1.0 # default = 0.5
    mpl.rcParams['xtick.major.width'] = 1.0 # default = 0.5
    mpl.rcParams['xtick.minor.width'] = 1.0 # default = 0.5
    mpl.rcParams['ytick.major.width'] = 1.0 # default = 0.5
    mpl.rcParams['ytick.minor.width'] = 1.0 # default = 0.5