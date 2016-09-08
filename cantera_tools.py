import math
import numpy as np
from scipy import integrate
from IPython.display import display, Image
import cantera as ct
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re

"""



Setup of System



"""


def get_initial_mole_fractions(stoich_ratio, fuel_mole_ratios, oxygen_per_fuel_at_stoic_list):
    """
    this method obtains your initial mole fractions for fuel in air.
    the product is returned as a list with nitrogen, oxygen, and then
    the fuels are listed.
    
    stoich_ratio = oxygen to fuel stoichiometric ratio in the system
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
    return mole_fractions/total_species


"""



Running System



"""



"""



Mechanism Reduction



"""

def reduce_reactions_in_mechanism(reaction_list, kept_reaction_equations):
    """
    finds reactions that match the form of the reaction equations in 
    kept_reaction_equations. It returns just the reactions that are meant
    to be in the mechanism
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
    # remove duplicates
    reduced_species_list = list(set(reacting_species))

    for species in species_list:
        if species.name in inert_species:
            reduced_species_list.append(species)
            
    return reduced_species_list

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


    
