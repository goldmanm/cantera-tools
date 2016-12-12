import math
import numpy as np
from scipy import integrate
import cantera as ct
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import warnings
import plot_tools as ptt
import copy
"""
This module contains script methods for analysis of cantera mechanisms.
The dependencies for this analysis are imported above. The section is 
divided into:

1. methods for getting cantera running & saving the data
    a. setup cantera solution objects
    b. reduce mechanisms 
    c. run cantera
    d. save data
2. methods for analyzing the cantera 'Solution' object
3. methods for analyzing output from cantera runs using the run data (from part 1)
4. methods for analyzing output using output data and cantera 'Solution' object



"""


###################################
# 1a. system setup
###################################

def get_initial_mole_fractions(stoich_ratio, 
                               fuel_mole_ratios, 
                               oxygen_per_fuel_at_stoich_list, 
                               fuels = None):
    """
    this method obtains your initial mole fractions for fuel in air.
    the product is returned as a dictionary  with nitrogen, oxygen, and then
    the fuels are listed.
    
    stoich_ratio = oxygen to fuel stoichiometric ratio in the system
    fuels = list of cantera fuel names for output dictionary. If ommitted, a list is returned
    fuel_mole_ratios = list of molar ratios of various fuels (must sum to 1)
    oxygen_per_fuel_at_stoic_list = a list containing the number of oxygen 
            molecules necessary for full combustion of each molecule. For 
            example, Methane would be 2.
    """
    #errror check
    np.testing.assert_allclose(sum(fuel_mole_ratios),1.)#,"mole ratios of fuels needs to add to one")
    assert len(fuel_mole_ratios) ==len(oxygen_per_fuel_at_stoich_list)
    
    combined_oxygen_per_fuel = np.sum(np.multiply(fuel_mole_ratios,oxygen_per_fuel_at_stoich_list))
    
    total_fuel = sum(fuel_mole_ratios)
    total_oxygen = total_fuel * combined_oxygen_per_fuel / stoich_ratio
    total_nitrogen = total_oxygen * .79/.21
    total_species = total_fuel + total_oxygen + total_nitrogen
    mole_fractions = np.concatenate(([total_nitrogen, total_oxygen],fuel_mole_ratios),0)
    normalized_mole_fractions = mole_fractions/total_species
    if fuels:
        fuel_zip =  [(fuels[index],normalized_mole_fractions[index+2]) for index in range(len(fuels))]
        air_zip = [('N2',normalized_mole_fractions[0]),('O2',normalized_mole_fractions[1])]
        mole_fraction_dictionary = dict(air_zip+fuel_zip)
        for species, fraction in list(mole_fraction_dictionary.items()):
            if fraction < 1e-10:
                del mole_fraction_dictionary[species]
        return mole_fraction_dictionary
    else:
        return normalized_mole_fractions







def create_mechanism(full_model_file_path,kept_reaction_equations='all',non_reactive_species = ['AR','N2','HE']):
    """
    input the full model and a list of reaction equations that you'd like to keep.
    returns a Cantera.Solution object of the mechanism with only the cooresponding
    reactions and their species.
    """
    desired_file = full_model_file_path
    spec = ct.Species.listFromFile(desired_file)
    rxns = ct.Reaction.listFromFile(desired_file)

    if kept_reaction_equations=='all':
        return ct.Solution(full_model_file_path)
    else:
        reduced_reactions = reduce_reactions_in_mechanism(rxns,kept_reaction_equations)
        reduced_species = eliminate_species_from_mechanism(spec,reduced_reactions,non_reactive_species)
        return ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                          species=reduced_species, reactions=reduced_reactions)
    
###################################
# 1b. mechanism reduction
###################################

def reduce_reactions_in_mechanism(reaction_list, kept_reaction_equations):
    """
    finds reactions that match the form of the reaction equations in 
    kept_reaction_equations. It returns just the reactions that are meant
    to be in the mechanism.
    
    reaction_list = list of cantera Reaction objects
    kept_reaction_equations = list of strings of reaction equations to keep.
    
    This does not check for equations not in kept_reaction_equations. must be fixed
    """
    reduced_reaction_list = []
    found_reaction = np.full(len(kept_reaction_equations), False, dtype=bool)
    for reaction in reaction_list:
        if reaction.equation in kept_reaction_equations:
            reduced_reaction_list.append(reaction)
            found_reaction[kept_reaction_equations.index(reaction.equation)] = True
    if not all(found_reaction):
        reactions_missed = np.array(kept_reaction_equations)[~ found_reaction]
        raise Exception('Reactions not found in solution or appear twice in the kept_reaction_list: ' + \
                                        str(reactions_missed) + \
                                        str())
    return reduced_reaction_list

def eliminate_species_from_mechanism(species_list, kept_reactions,inert_species):
    """
    finds all the species in kept_reactions, and returns a list of species
    objects of those species. inert_species are automatically kept.
    """
    
    reacting_species = []
    for reaction in kept_reactions:
        reacting_species += list(reaction.reactants.keys()) + list(reaction.products.keys())
    # remove duplicates and add inert
    reduced_species_name_list = list(set(reacting_species)) + inert_species
    
    reduced_species_list = []
    for species in species_list:
        if species.name in reduced_species_name_list:
            reduced_species_list.append(species)
            
    return reduced_species_list

###################################
# 1c. run mechanism
###################################

def find_ignition_delay(solution, conditions, 
                      condition_type = 'adiabatic-constant-volume',
                      output_profile = False,
                      temp_final = 965,
                      output_reactions = True,
                      output_rop_roc = False):
    """
    This method finds the ignition delay of a cantera solution object with
    an option to return all species and reactions as a pandas.DataFrame object
    which can be stored. The simulation uses a hard-temperature cutoff to determine
    ignition. 
    
    This method returns a tuple with ignition delay and the DataFrame (or `None`)
    if not specified. 
    
    `solution` = Cantera.Solution object
    `conditions` = tuple of temperature, pressure, and mole fraction initial species
    `condition_type` = string describing the run type, currently only 'adiabatic-constant-volume' supported
    `output_profile` = should the program save simulation results and output them
    `temp_final` = the temperature which the ignition is reported
    `output_reactions` = should the data contain reactions as well. If
            output_profile is False, this has no effect
    """
    solution.TPX = conditions    
    if condition_type == 'adiabatic-constant-volume':
        reactor = ct.IdealGasReactor(solution)
        simulator = ct.ReactorNet([reactor])
        solution = reactor.kinetics
    else:
        raise NotImplementedError('only adiabatic constant volume is supported')
        
    # setup data storage
    if output_profile:
        df = pd.DataFrame()
        df = append_data_to_df(simulator, solution, df, add_rxns=output_reactions)
    else:
        df = None
        
    if output_rop_roc:
        rop_roc = pd.DataFrame()
        rop_roc = append_rop_and_roc_to_dataframe(solution, rop_roc)
    else:
        rop_roc = None
    
        
    # run simulation
    time_final = 500 #seconds
    while simulator.time < time_final and reactor.T < temp_final:
        simulator.step(time_final)
        if output_profile:
            df = append_data_to_df(simulator,solution,df, add_rxns=output_reactions)
        if output_rop_roc:
            rop_roc = append_rop_and_roc_to_dataframe(solution, rop_roc)
    
    return simulator.time, df, rop_roc


###################################
# 1d. saving data
###################################

def append_data_to_df(simulator,solution, df, basics= ['time','temperature','pressure','density'],
                      add_species = True, species_names='all',
                      add_rxns = False, reaction_names='all'):
    """
    appends the current conditions of a Solution object contianing ReactorNet
    object (simulator) to the pandas.dataframe object (df). 
    
    The optional parameters define what is saved.
    basics = a list of state variables to save: options are time, temperature
        pressure, density, volume, cp and cv.
    add_species = save the concentrations of species in kmol/m3.
    species_names = list of species names to be saved (default is all)
    add_reactions = save the concentration of reactions in kmol/m3s
    reaction_names = list of reaction strings to be saved (default is all)
    
    
    The following are enabled for the basics conditions:
    * time
    * temperature
    * pressure
    * density
    * volume
    * cp (constant pressure heat capacity)
    * cv (constant volume heat capacity)
    """
    
    # this term is to suppress warning messages during initialization
    if df.empty:
        initialization = True
    else:
        initialization = False
    
    conditions = {}
    # add regular conditions
    if 'time' in basics:
        conditions['time (s)'] = simulator.time
    if 'temperature' in basics:
        conditions['temperature (K)'] = solution.T
    if 'pressure' in basics:
        conditions['pressure (Pa)'] = solution.P 
    if 'density' in basics:
        conditions['density (kmol/m3)'] = solution.density_mole
    if 'volume' in basics:
        conditions['volume (m3)'] = solution.volume_mole
    if 'cp' in basics:
        conditions['heat capacity, cp (J/kmol/K)'] = solution.cp_mole
    if 'cv' in basics:
        conditions['heat capacity, cv (J/kmol/K)'] = solution.cv_mole
#    if '' in basics:
#        conditions[''] = solution.
    # end adding regular conditions


    if add_species:
        if species_names=='all':
            species_recorded = solution.species_names
        else:
            species_recorded = species_names
        mole_fractions = solution.mole_fraction_dict()
        for name in species_recorded:
            try:
                conditions[name] = mole_fractions[name] * solution.density_mole
            except KeyError:
                conditions[name] = 0
                # sends warning if user typed species incorrectly
                if name in species_names and not initialization:
                    warnings.warn('{} is not listed in the mole fraction dictionary and may be mispelled.'.format(name))
    
    if add_rxns:
        if reaction_names=='all':
            reaction_names = solution.reaction_equations()
            
        reaction_rates = __get_rxn_rate_dict(solution.reaction_equations(),solution.net_rates_of_progress)
        for name in reaction_names:
            try:
                conditions[name] = reaction_rates[name]
            except KeyError:
                conditions[name] = 0
                warnings.warn('{} is not listed in the reaction names.'.format(name))
    
    return df.append(conditions, ignore_index=True)

def append_forward_and_reverse_reactions_to_dataframe(solution, df):
    """
    This method appends the forward and reverse reactions to the dataframe
    
    I've never used this method, could be deprecitated in the future
    """
    reaction_equations = solution.reaction_equations()
    forward_reactions = pd.Series(__get_rxn_rate_dict(reaction_equations,solution.forward_rates_of_progress))
    reverse_reactions = pd.Series(__get_rxn_rate_dict(reaction_equations,solution.reverse_rates_of_progress))
    
    forward_reactions.index = pd.MultiIndex.from_product(['forward',forward_reactions.index])
    reverse_reactions.index = pd.MultiIndex.from_product(['reverse',reverse_reactions.index])
    
    return df.append(pd.concat([forward_reactions,reverse_reactions]), ignore_index=True)
    

def append_rop_and_roc_to_dataframe(solution, df):
    """
    appends rate of production and rate of consumption to dataframe (kmol/m3s)
    This is inherently separated from the other method because this stores
    extra data that may only be useful for quasi-steady state analysis
    """
    species = solution.species_names
    production = pd.Series(__get_rxn_rate_dict(species,solution.creation_rates))
    consumption = pd.Series(__get_rxn_rate_dict(species,solution.destruction_rates))
    
    assert isinstance(production.index,pd.indexes.base.Index)
    assert isinstance(consumption.index,pd.indexes.base.Index)
    
    production.index = pd.MultiIndex.from_product([['production'],production.index])
    consumption.index = pd.MultiIndex.from_product([['consumption'],consumption.index])
     
    return df.append(pd.concat([production,consumption]), ignore_index=True)
        

def __get_rxn_rate_dict(reaction_equations, net_rates):
    """
    makes a dictionary out of the two inputs. If identical reactions are encountered,
    called duplicates in Cantera, the method will merge them and sum the rate together
    """
    rxn_dict = {}
    for equation, rate in zip(reaction_equations, net_rates):
        try:
            rxn_dict[equation] += rate
        except KeyError:
            rxn_dict[equation] = rate
    return rxn_dict



###################################
# 2. cantera mechanism analysis
###################################

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
    # change this to a try/except statement to deal with arrays properly.;..though this won't work since 
    # strings are iterable.
    
    
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

def branching_ratios(df, solution, compound):
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
    """
    reaction_dataframe = weight_reaction_dataframe_by_stoich_coefficients(df,solution,compound)
    
    #only keep consumption
    consumption_terms = reaction_dataframe[reaction_dataframe < 0]
    consumption_terms = consumption_terms.dropna('columns','all')
    #consumption_terms = consumption_terms.fillna(0)
    
    total = consumption_terms.sum('columns')
    #print(total)
    branching_ratios = consumption_terms.div(total,'index')
    branching_ratios = branching_ratios.fillna(0)
    
    #sort from most important
    importance_index = branching_ratios.sum('index').sort_values(ascending=False)
    branching_ratios = branching_ratios.reindex_axis(importance_index.index,axis='columns')

    return branching_ratios
    
def weight_reaction_dataframe_by_stoich_coefficients(df, solution, species):
    """
    returns a dataframe of reactions over time weighted by the stoichiometric
    coefficient of the species string `species`. 
    """
    
    reactions = find_reactions(df, species)
    reaction_strings = list(reactions.columns)
    stoichiometries = obtain_stoichiometry_of_species(solution,
                                                      species,
                                                      reaction_strings)
    return reactions * stoichiometries
    


###################################
# 3a. output data processing methods
# these methods are less likely to be useful by themselves.
# many methods in 3b call these
###################################

def remove_ignition(df, percent_cutoff=0.98):
    "returns a dataframe with time points after `percent_cutoff` removed."
    end_time = df['time (s)'].iloc[-1]
    data_before_ignition=df[df['time (s)'] < end_time*percent_cutoff]
    return data_before_ignition

def integrate_data(df, integration_column):
    """
    Takes a data frame and performs an integration 
    of each column over the `integration_column`, 
    which is a pandas.Series object. This
    uses the method of right Reman sum.
    """
    time_intervals = integration_column.diff()
    time_intervals.iloc[0] = 0
    return df.mul(time_intervals, axis='index')
    

def find_reactions(df,species='any'):
    """
    finds the reaction columns in the dataframe and returns them
    if a string, species, is specified, it will only return reactions
    with the matching species.
    """
    
    # find reaction columns
    df_reactions = df.loc[:,['=' in column for column in df.columns]]
    if species =='any':
        return df_reactions
    string = _prepare_string_for_re_processing(species)
    expression = r'(\A|\s)%s(\Z|\s)' %(string)
    df_my_reactions = df_reactions.loc[:,[re.compile(expression).search(column) != None for column in df_reactions.columns]]
    if df_my_reactions.empty:
        raise Exception('No reactions found for species {}'.format(species))
    return df_my_reactions

def _prepare_string_for_re_processing(string):
    """ used for allowing parenthesis in species when searching reactions"""
    return string.replace('(','\(').replace(')','\)')


def find_species(df):
    """
    finds the species columns in the dataframe and returns them
    if a string, species, is specified, it will only return reactions
    with the matching species.
    """
    # find reaction columns
    df_not_reactions = df.loc[:,['=' not in column for column in df.columns]]
    # hard coded properties. not the ideal way
    properties = ['time (s)','temperature (K)','density (kmol/m3)','cp (J/kmol/K)','cv (J/kmol/K)','pressure (Pa)','volume (m3)']
    not_states_map = [all([state not in column for state in properties]) for column in df_not_reactions.columns]
    if all(not_states_map):
        raise Exception('did not find states')
    df_not_reactions_states = df_not_reactions.loc[:,not_states_map]
    return df_not_reactions_states

def return_nearest_time_index(desired_time,time_series,index=True):
    """
    input the desired time, double, and time_series, pd.Series,
    returns the index of the time_series. 
    If you want the actual time value, change index=False
    """
    nearest_value = lambda value, array: np.argmin(abs(value-array))
    if index:
        return nearest_value(desired_time,time_series)
    return time_series[nearest_value(desired_time,time_series)]

###################################
# 3b. output data analysis
###################################

def consumption_pathways(solution,df,species='any',ignore_ignition=True):
    """
    returns the total rate of production 
    for a particular species
    over the entire simulation using
    the forward difference approximation.
    
    Postive values indicate production, negative values indicate consumption
    """

    df_reactions_weighted = integrate_data(find_reactions(df,species), df['time (s)'])
    if ignore_ignition:
        last_index = remove_ignition(df).shape[0] 
    else:
        last_index = df.shape[0]-1
    reactions_weighted = df_reactions_weighted[df.index<last_index].sum()
    
    if species != 'any': # weight to stoich coefficients
        stoich_coeffs = [obtain_stoichiometry_of_species(solution, species, reaction) for reaction in reactions_weighted.index]
        stoich_coeff_dict = pd.Series(dict(zip(reactions_weighted.index,stoich_coeffs)))
        reactions_weighted *= stoich_coeff_dict
    return reactions_weighted.sort_values()

def quasi_steady_state(df, species):
    """
    This method outputs the key parameter in quasi steady state
    approximation. 
    
    df = pd.DataFrame with format similar to `append_rop_and_roc_to_dataframe`
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
    


###################################
# 4. plotting
# this includes plotting functions specific to cantera 
###################################
