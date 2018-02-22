import pandas as pd
"""
This file will be deprecated.
"""

from analysis import *
from rmg_and_cantera import *
from simulation import *

def remove_ignition(df, percent_cutoff=0.98):
    """
    returns a dataframe with time points after `percent_cutoff` removed.

    deprecated
    """
    end_time = df['time (s)'].iloc[-1]
    data_before_ignition=df[df['time (s)'] < end_time*percent_cutoff]
    return data_before_ignition

def integrate_data(df, integration_column):
    """
    Takes a data frame and performs an integration 
    of each column over the `integration_column`, 
    which is a pandas.Series object. This
    uses the method of right Reman sum.

    deprecated
    """
    time_intervals = integration_column.diff()
    time_intervals.iloc[0] = 0
    return df.mul(time_intervals, axis='index')
    
def diff_data(df, differentiation_column):
    """
    Takes a data frame and performs an derivative
    of each column with respect to the `differentiation_column`, 
    which is a pandas.Series object.

    deprecated
    """
    if isinstance(df,pd.DataFrame):
        numerator_difference = df.diff(axis='index')
    else:
        numerator_difference = df.diff()
    denominator_difference = differentiation_column.diff()
    derivative = numerator_difference.div( denominator_difference, axis='index')
    derivative.iloc[0] = 0
    return derivative

def _prepare_string_for_re_processing(string):
    """ used for allowing parenthesis in species when searching reactions

    deprecated
    """
    return string.replace('(','\(').replace(')','\)')