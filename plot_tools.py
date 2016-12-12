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

a future improvement might be to set tick labels on the upper and lower margins
instead of the middle to avoid conflicting letters

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import re
"""
This file contains tools that can be called at the begining of a 
script to modify all the matplotlib graphs. The two most useful methods
might be `presentation` and `publication`, which changes the detail on
how much information is in figures

I might want to look into compatibility with the seaborn package.

"""
def general_changes():
    mpl.rcParams['figure.figsize'] = (12.0,8.0) # default = (6.0, 4.0)
    improve_dpi()
    use_tex()

def use_tex():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['text.latex.preamble'] = [r"\usepackage[version=4]{mhchem}"]

def improve_dpi():
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300

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

    axis.set_major_locator(mpl.ticker.LogLocator(self.base,numticks=3))
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
 
    # make latex font visible
    #latex_preamble = [r'\usepackage[mathbf]{euler}'] # ideal depth but odd shape
    latex_preamble = [r'\usepackage{arev}'] # pretty visible 
    #latex_preamble = [r'\usepackage{cmbright}'] # too light color
    mpl.rcParams['text.latex.preamble'] = mpl.rcParams['text.latex.preamble'] + latex_preamble



def publication():
    mpl.rcParams['font.size'] = 18 # default = 10
    mpl.rcParams['axes.linewidth'] = 2.0 # default = 1.0
    mpl.rcParams['lines.linewidth'] = 2.0 # default = 1.0
    mpl.rcParams['patch.linewidth'] = 1.0 # default = 1.0
    mpl.rcParams['grid.linewidth'] = 1.0 # default = 0.5
    mpl.rcParams['xtick.major.width'] = 1.0 # default = 0.5
    mpl.rcParams['xtick.minor.width'] = 1.0 # default = 0.5
    mpl.rcParams['ytick.major.width'] = 1.0 # default = 0.5
    mpl.rcParams['ytick.minor.width'] = 1.0 # default = 0.5
    general_changes()



######################################################
# methods for usage within plotting functions
######################################################

def _get_color_list_from_colormap(name,number_colors=10):
    """
    returns list of length `number_colors` based on the default 
    colormap of `name`. number must be less than 256
    
    this can be implemented in matplotlib by
    
    ```
    colors = plot_tools._get_color_list_from_colormap('viridis',number_reactions)
    from cycler import cycler
    axis.set_prop_cycle(cycler('color',colors))
    ```
    
    This method does not work with LinearColorMaps
    
    seaborn.mpl_palette(name,n_colors) seems identical to this
    """

    colormap = plt.get_cmap(name)
    color_array = colormap.colors
    color_map_colors = len(color_array)
    if number_colors == 1:
        return [color_array[0]]
    if number_colors > color_map_colors:
        raise ValueError('number colors cannot be more than %i. currently is %i' % (color_map_colors,number_colors))
    # find distance between each node, and multiply by almost one to allow proper 'floor'ing of double
    distance_between_array_elements = float(color_map_colors) / (number_colors-1) * 0.999999999999
    colors = []
    for index in range(number_colors):
        color = color_array[int(index*distance_between_array_elements)]
        colors.append(color)
    return colors
    
def set_colors(number_colors=10,name = 'viridis'):
    """
    sets the color cycle to the `number_colors` using the colormap `name`.
    
    this allows for easy comparison of N set of parameters on one graph.
    For example, a full and reduced model with the same number of variables
    can be plotted with different icons ('-', '.', etc) without specifying
    colors each time.
    
    deprecated, use instead:
    
    ```
    import seaborn as sns
    color_palette = sns.mpl_palette('jet',10)
    sns.set_palette(color_palette)
    ```
    
    checkout color map options [here](http://matplotlib.org/users/colormaps.html)
    """
    
    colors = _get_color_list_from_colormap( name, number_colors)
    mpl.rcParams['axes.prop_cycle'] = cycler('color',colors)

def place_legend_outside_plot(axis,legend_entry= None):
    """
    this method takes an axis object and the text of the legend, and
    it places the legend to the right of the figure, making the figure
    less wide.
    
    usage in normal contexts:
    
    ```
    f = plt.figure()
    [run plotting functions]
    ax = f.get_axes()[0]
    ptt.place_legend_outside_plot(axis=ax,legend_entry= ['a','b','c'])
    ```
    """
    if legend_entry != None:
        axis.legend(legend_entry, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        axis.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
def latexify_legend(legend_string,mode = 'math'):
    """makes the legend string interpreted able to be interpreted
    by adding a 
    """
    prefix = r''
    sufix = r''
    if mode =='math':
        prefix = r'$'
        sufix = r"$"
    elif mode == 'chemistry':
        prefix = r'\ce{'
        sufix = r'}'
        #replace '=>' with '->'
        irreversible_reaction = re.compile(r'\s=>')
        for index, string in enumerate(legend_string):
            if irreversible_reaction.search(string) != None:
                legend_string[index] = re.sub(irreversible_reaction,' ->',string)
    elif mode =='no underscores':
        for index, string in enumerate(legend_string):
            legend_string[index] = string.replace(r'_',r' ')
    new_legend = []
    for index,string in enumerate(legend_string):
        string = string.replace('#',r'\#')
        new_legend.append(prefix + string + sufix)
    return new_legend
    
######################################################
# methods that plot functions and return the figure object
# this mostly involves methods used in data analysis of various projects
######################################################

def plot_multiple_comparisons(data,dictionary_of_plots,x_data = None,
                              log_plots = [],
                              title='',
                              xlabel='time',
                              yunit=''):
    """
    This method takes in a data frame and plots multiple plots each named
    by the key in `dictionary_of_plots` with the column(s) given by the list
    in the values of `dictionary_of_plots`. 
    
    x_data is a pd.Series object for the data to be plotted. If the x_axis is
    meant to be the index of the dataframe `data`, then it doesn't need to be
    specified.
    
    title = string above all plots
    xlabel = plt.xlabel
    ynit = unit appended to the ylabel
    
    returns the figure object,whose axes can be accessed with output.
    
    used in the isotope project
    """
    number_ratios = len(dictionary_of_plots.keys())
    f, axes = plt.subplots(number_ratios,1, sharex=True,figsize=(12,6*number_ratios))
    
    axes[0].set_title(title)
    for plot_name, axis in zip(dictionary_of_plots.keys(),axes):
        isotopomers = dictionary_of_plots[plot_name]
        sum_of_isotopomers = data.filter(isotopomers).sum('columns')
        for isotopomer in isotopomers:
            axis.plot(x_data,data[isotopomer]/sum_of_isotopomers)
        #axis.set_xscale('log')
        if plot_name in log_plots:
            axis.set_yscale('log')
        #else:
            #axis.set_ylim((-.1,1.1))
        #axis.legend(dictionary_of_plots[plot_name], loc='best')
        place_legend_outside_plot(axis,dictionary_of_plots[plot_name])
        axis.set_ylabel(plot_name + ' (' + yunit+ ')')
    axes[-1].set_xlabel(xlabel)
    return f
