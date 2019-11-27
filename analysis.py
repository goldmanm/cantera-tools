# -*- coding: utf-8 -*-
import numpy as np
import cantera as ct
import pandas as pd
import re
import warnings
import copy
###################################
# 3b. output data analysis
###################################

def branching_ratios(df, solution, compound, production = False):
    """
    This method looks at the consumption pathways of `compound` over
    all time points in the data set.
    It outputs a pandas.DataFrame which contains columns of pertinant reactions
    and values of the branching ratio of each reaction which is defined as 
    
    $BR_{i} = \frac{ROC_i}{\Sigma_{j=0}^{j=N} ROC_j   }$
    
    where $i$ is the reaction in question, $ROC$ is the rate of consumption of
    the desired species, and $N$ is the number of reactions, and $BR$ is the branching ratio.
    
    df = dataframe of run data
    solution = cantera solution object
    compound = species string which you want to identify
    production = if True, shows the reactions forming species X
    
    This method only works on forward reactions
    """
    reaction_dataframe = weight_reaction_dataframe_by_stoich_coefficients(df,solution,compound)
    
    if not production:
        #only keep consumption
        consumption_terms = reaction_dataframe[reaction_dataframe < 0]
        df = consumption_terms.dropna('columns','all')
    else:
        production_terms = reaction_dataframe[reaction_dataframe > 0]
        df = production_terms.dropna('columns','all')
    
    total = df.sum('columns')
    branching_ratios = df.div(total,'index')
    branching_ratios = branching_ratios.fillna(0)
    
    #sort from most important
    importance_index = branching_ratios.sum('index').sort_values(ascending=False)
    branching_ratios = branching_ratios.reindex(importance_index.index,axis='columns')

    return branching_ratios

def consumption_pathways(solution,df,species, time = 'all'):
    """
    returns the total rate of production for a particular species at the specified
    time(s). Postive values indicate production, negative values indicate consumption
    
    If multiple times are given or the keyword 'all' is used, the output is a DataFrame
    with indexes the various times. If only one time is supplied, the output is a
    Series. 

    solution = cantera solution object
    df = pandas dataframe of reactions
    species = string of species
    time = number describing the time points to determine consumption (or list of numbers)
    """

    if time=='all':
        time = list(df.index)
    if isinstance(time,list):
        # recursively run consumption_pathways
        consumption_values = []
        for t in time:
            consumption_values.append(consumption_pathways(solution=solution,
                                            df=df,
                                            species=species,
                                            time= t))
        consumption_values = pd.DataFrame(consumption_values, index=time)
        # sort by total sum of flux
        sorted_index = consumption_values.sum('index').sort_values().keys()
        return consumption_values[sorted_index]

    # the time is not a list, return a pd.Series
    try:
        reactions_weighted = find_reactions(solution, df,species).loc[time,:]
    except KeyError:
        reactions_weighted = find_reactions(solution, df,species).loc[return_nearest_time_index(time,df.index, index=False),:]

    # weight by stoichiometric_coefficients
    stoich_coeffs = [obtain_stoichiometry_of_species(solution, species, reaction) for reaction in reactions_weighted.index]
    stoich_coeff_dict = pd.Series(dict(zip(reactions_weighted.index,stoich_coeffs)))
    # pandas was having some bug, so manually rewrote the line below
    #reactions_weighted *= stoich_coeff_dict
    for index in stoich_coeff_dict.index:
        reactions_weighted[index] *= stoich_coeff_dict[index]
    return reactions_weighted.sort_values()

def quasi_steady_state(df, species):
    """
    This method outputs the key parameter, $\frac{|ROP-ROC|}{ROP}$, in quasi steady state
    approximation. 
    
    df = pd.DataFrame containing get_rop_and_roc_series
    species = string of species to use
    
    returns a pd.Series of the qss apprixmation: $\frac{|ROP-ROC|}{ROP}$
    """
    return (df['production',species] - df['consumption',species]).abs() / df['production',species]
    

def compare_species_profile_at_one_time(desired_time, df1,df2,
                                        minimum_return_value=1e-13,
                                       time_string = 'time (s)'):
    """
    compares the species profile between two models closest to the desired time
    returns a pandas.Series object with the relative species concentrations
    given by `compare_2_data_sets`
    """
    
    time_index_1 = return_nearest_time_index(desired_time,df1[time_string])
    time_index_2 = return_nearest_time_index(desired_time,df2[time_string])
    
    time_slice_1 = find_species(df1).loc[time_index_1]
    time_slice_2 = find_species(df2).loc[time_index_2]
    return _compare_2_data_sets(time_slice_1,time_slice_2,minimum_return_value)

def _compare_2_data_sets(model1, model2, minimum_return_value = 1000,diff_returned=0.0):
    """given two pd.Series of data, returns a pd.Series with the relative
    differences between the two sets. This requires one of the values to be
    above the `minimum_return_cutoff` and the difference to be above `diff_returned`
    
    The difference is returned as $\frac{model1 - model2}{\min(model1,model2)}$.
    Where the minimum merges the two datasets using the minimum value at each index. 
    
    """
    #ensure all values are the same
    model1 = copy.deepcopy(model1)[model2.index].dropna()
    model2 = copy.deepcopy(model2)[model1.index].dropna()
    minimum_value = pd.DataFrame({'model1':model1,'model2':model2}).min(1)
    compared_values = ((model1-model2)/minimum_value).dropna()
    for label in compared_values.index:
        not_enough_value = (model1[label] < minimum_return_value and model2[label] < minimum_return_value)
        not_enough_difference = abs(compared_values[label]) < diff_returned
        if  not_enough_value or not_enough_difference:
            compared_values[label] = np.nan
    compared_values = compared_values.dropna()
    return compared_values.sort_values()
    
    
    
    
    
def return_nearest_time_index(desired_time,time_series,index=True):
    """
    input the desired time, double, and time_series, pd.Series,
    returns the index of the time_series. 
    If you want the actual time value, change index=False
    """
    # commented out due to error in mp.argmin
    #nearest_value = lambda value, array: np.argmin(abs(value-array))
    #if index:
    #    return nearest_value(desired_time,time_series)
    #return time_series[nearest_value(desired_time,time_series)]
    deviation_list = abs(desired_time-time_series)
    min_deviation = min(deviation_list)
    index_value = list(deviation_list).index(min_deviation)
    if index:
        return index_value
    return time_series[index_value]

def obtain_stoichiometry_of_species(solution, species, reaction):
    """
    this method finds a reaction string in the cantera solution file, and 
    returns its stoichiometric coefficient of the specified species.
    Returns a negative value if the species is a reactant.
    
    solution = cantera solution object
    species = string of species name
    reaction = reaction string or list of reaction strings.
    
    Stoichiometry is calculated by: product_stoich_coeff - reactant_stoich_coeff
    """
    # recursively deal with lists of reactions
    if not isinstance(reaction,str):
        coefficients = np.empty(len(reaction))
        for index, reaction_string in enumerate(reaction):
            coefficients[index] = obtain_stoichiometry_of_species(solution,species,reaction_string)
        return coefficients
    # deal with individual reactions
    assert isinstance(reaction,str)
    reaction_index = solution.reaction_equations().index(reaction)
    reactant_stoich_coeff = solution.reactant_stoich_coeff(species, reaction_index)
    product_stoich_coeff = solution.product_stoich_coeff(species, reaction_index)
    if product_stoich_coeff > 0 or reactant_stoich_coeff > 0:
        return product_stoich_coeff - reactant_stoich_coeff
    raise Exception('Species {} is not in reaction {}'.format(species,reaction))

def weight_reaction_dataframe_by_stoich_coefficients(df, solution, species):
    """
    returns a dataframe of reactions over time weighted by the stoichiometric
    coefficient of the species string `species`. 
    """
    reactions = find_reactions( solution, df, species)
    reaction_strings = list(reactions.columns)
    stoichiometries = obtain_stoichiometry_of_species(solution,
                                                      species,
                                                      reaction_strings)
    return reactions * stoichiometries


def find_reactions(solution, df,species):
    """
    finds the reaction columns in the net_reaction dataframe which contain 
    the species specified and returns them.
    """
    included_columns = []
    rxn_string_to_rxn_index = dict(zip(solution.reaction_equations(),range(solution.n_reactions)))
    for rxn_name in df.columns:
        sln_index = rxn_string_to_rxn_index[rxn_name]
        try:
            if solution.product_stoich_coeff(species,sln_index) !=0 or \
                        solution.reactant_stoich_coeff(species,sln_index) !=0:
                included_columns.append(rxn_name)
        except KeyError:
            print("Error obtained in find_reactions,\ncheck to ensure the columns in `df`\ncorrespond to the reactions in `solution`")
            raise
    df_my_reactions = df[included_columns]
    
    if df_my_reactions.empty:
        raise Exception('No reactions found for species {}'.format(species))
    return df_my_reactions
