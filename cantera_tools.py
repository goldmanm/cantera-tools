import math
import numpy as np
from scipy import integrate
import cantera as ct
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import warnings
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

def get_initial_mole_fractions(stoich_ratio, fuel_mole_ratios, oxygen_per_fuel_at_stoic_list, fuels = None):
    """
    this method obtains your initial mole fractions for fuel in air.
    the product is returned as a list with nitrogen, oxygen, and then
    the fuels are listed.
    
    stoich_ratio = oxygen to fuel stoichiometric ratio in the system
    fuels = list of strings for output dictionary. If ommitted, a list is returned
    fuel_mole_ratios = list of molar ratios of various fuels (must sum to 1)
    oxygen_per_fuel_at_stoic_list = a list containing the number of oxygen 
            molecules necessary for full combustion of each molecule. For 
            example, Methane would be 2.
    """
    #errror check
    np.testing.assert_allclose(sum(fuel_mole_ratios),1.)#,"mole ratios of fuels needs to add to one")
    assert len(fuel_mole_ratios) ==len(oxygen_per_fuel_at_stoic_list)
    
    combined_oxygen_per_fuel = np.sum(np.multiply(fuel_mole_ratios,oxygen_per_fuel_at_stoic_list))
    
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
        return ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                      species=spec, reactions=rxns)
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
    
    This does not check for equations not in kept_reaction_equations. must be fixed
    """
    reduced_reaction_list = []
    for reaction in reaction_list:
        if reaction.equation in kept_reaction_equations:
            reduced_reaction_list.append(reaction)        
    return reduced_reaction_list

def eliminate_species_from_mechanism(species_list, kept_reactions,inert_species):
    """
    finds all the species in kept_reactions, and returns a list of species
    objects of those species. inert_species are automatically kept.
    """
    
    reacting_species = []
    for reaction in kept_reactions:
        reacting_species += reaction.reactants.keys() + reaction.products.keys()
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
    
    The following are enabled for the basics conditions:
    * time
    * temperature
    * pressure
    * density
    * volume
    * cp (constant pressure heat capacity)
    * cv (constant volume heat capacity)
    """
    
    
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
        conditions['heat capacity, cp (J/kmol/K)'] = solution.cv_mole
#    if '' in basics:
#        conditions[''] = solution.
    # end adding regular conditions


    if add_species:
        if species_names=='all':
            species_names = solution.species_names
        mole_fractions = solution.mole_fraction_dict()
        for name in species_names:
            try:
                conditions[name] = mole_fractions[name] * solution.density_mole
            except KeyError:
                conditions[name] = 0
                warnings.warn('%s is not listed in the mole fraction dictionary. If this occurs after the first iteration, you may have typed the species incorrectly')
    
    if add_rxns:
        if reaction_names=='all':
            reaction_names = solution.reaction_equations()
            
        reaction_rates = __get_rxn_rate_dict(solution.reaction_equations(),solution.net_rates_of_progress)
        for name in reaction_names:
            try:
                conditions[name] = reaction_rates[name]
            except KeyError:
                conditions[name] = 0
                warnings.warn('%s is not listed in the reaction names. If this occurs after the first iteration, you may have typed the species incorrectly')
    
    return df.append(conditions, ignore_index=True)

    
def __get_rxn_rate_dict(reaction_equations, net_rates):
    """
    makes a dictionary out of the two inputs, adding together duplicate
    reactions
    """
    rxn_dict = {}
    for equation, rate in zip(reaction_equations, net_rates):
        try:
            rxn_dict[equation] += rate
        except KeyError:
            rxn_dict[equation] = rate
    return rxn_dict
