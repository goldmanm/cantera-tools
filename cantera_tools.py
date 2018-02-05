import numpy as np
import cantera as ct
import pandas as pd
import re
import warnings
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
    a. smaller, core methods (derivatives, etc.)
    b. more applied method (view consumption pathways, QSSA)
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

def run_simulation(solution,  times, conditions=None,
                      condition_type = 'adiabatic-constant-volume',
                      output_species = True,
                      output_reactions = True,
                      output_directional_reactions = False,
                      output_rop_roc = False,
                      atol = 1e-15,
                      rtol = 1e-9):
    """
    This method iterates through the cantera solution object and outputs information
    about the simulation as a pandas.DataFrame object.
    
    This method returns a dictionary with the reaction conditions data, species data,
    net reaction data, forward/reverse reaction data, and the rate of production 
    and consumption (or `None` if a variable not specified). 
    
    `solution` = Cantera.Solution object
    `conditions` = tuple of temperature, pressure, and mole fraction initial 
                species
    `times` = an iterable of times which you would like to store information in
    `condition_type` = string describing the run type, currently supports 
                'adiabatic-constant-volume' and 'constant-temperature-and-pressure'
    `output_species` = output a DataFrame of species' concentrations
    `output_reactions` = output a DataFrame of net reaction rates
    `output_directional_reactions` = output a DataFrame of directional reaction rates
    `output_rop_roc` = output a DataFrame of species rates of consumption & production
    """
    if conditions is not None:
        solution.TPX = conditions
    if condition_type == 'adiabatic-constant-volume':
        reactor = ct.IdealGasReactor(solution)
    elif condition_type == 'constant-temperature-and-pressure':
        reactor = ct.IdealGasConstPressureReactor(solution, energy='off')
    else:
        raise NotImplementedError('only "adiabatic-constant-volume" or "constant-temperature-and-pressure" is supported. {} input'.format(condition_type))
    simulator = ct.ReactorNet([reactor])
    solution = reactor.kinetics
    simulator.atol = atol
    simulator.rtol = rtol
    # setup data storage
    outputs = {}
    outputs['conditions'] = pd.DataFrame()
    if output_species:
        outputs['species'] = pd.DataFrame()
    if output_reactions:
        outputs['net_reactions'] = pd.DataFrame()
    if output_directional_reactions:
        outputs['directional_reactions'] = pd.DataFrame()
    if output_rop_roc:
        outputs['rop'] = pd.DataFrame()

    for time in times:
        simulator.advance(time)
        # save data
        outputs['conditions'] = outputs['conditions'].append(
                                get_conditions_series(simulator,solution),
                                ignore_index = True)
        if output_species:
            outputs['species'] = outputs['species'].append(
                                get_species_series(solution),
                                ignore_index = True)
        if output_reactions:
            outputs['net_reactions'] = outputs['net_reactions'].append(
                                get_reaction_series(solution),
                                ignore_index = True)
        if output_directional_reactions:
            outputs['directional_reactions'] = outputs['directional_reactions'].append(
                                get_forward_and_reverse_reactions_series(solution),
                                ignore_index = True)
        if output_rop_roc:
            outputs['rop'] = outputs['rop'].append(
                                get_rop_and_roc_series(solution),
                                ignore_index = True)

    # set indexes as time
    time_vector = outputs['conditions']['time (s)']
    for output in outputs.values():
        output.set_index(time_vector,inplace=True)

    return outputs

def run_simulation_till_conversion(solution, species, conversion,conditions=None,
                      condition_type = 'adiabatic-constant-volume',
                      output_species = True,
                      output_reactions = True,
                      output_directional_reactions = False,
                      output_rop_roc = False,
                      atol = 1e-15,
                      rtol = 1e-9,):
    """
    This method iterates through the cantera solution object and outputs information
    about the simulation as a pandas.DataFrame object.

    This method returns a dictionary with the reaction conditions data, species data,
    net reaction data, forward/reverse reaction data, and the rate of production 
    and consumption (or `None` if a variable not specified) at the specified conversion value.

    `solution` = Cantera.Solution object
    `conditions` = tuple of temperature, pressure, and mole fraction initial 
                species
    `species` = a string of the species label (or list of strings) to be used in conversion calculations
    `conversion` = a float of the fraction conversion to stop the simulation at
    `condition_type` = string describing the run type, currently supports 
                'adiabatic-constant-volume' and 'constant-temperature-and-pressure'
    `output_species` = output a Series of species' concentrations
    `output_reactions` = output a Series of net reaction rates
    `output_directional_reactions` = output a Series of directional reaction rates
    `output_rop_roc` = output a DataFrame of species rates of consumption & production
    """
    if conditions is not None:
        solution.TPX = conditions
    if condition_type == 'adiabatic-constant-volume':
        reactor = ct.IdealGasReactor(solution)
    if condition_type == 'constant-temperature-and-pressure':
        reactor = ct.IdealGasConstPressureReactor(solution, energy='off')
    else:
        raise NotImplementedError('only adiabatic constant volume is supported')
    simulator = ct.ReactorNet([reactor])
    solution = reactor.kinetics
    simulator.atol = atol
    simulator.rtol = rtol

    if isinstance(species,str):
        target_species_indexes = [solution.species_index(species)]
    else: # must be a list or tuple
        target_species_indexes = [solution.species_index(s) for s in species]
    starting_concentration = sum([solution.concentrations[target_species_index] for target_species_index in target_species_indexes])
    proper_conversion = False
    new_conversion = 0
    while not proper_conversion:
        error_count = 0
        while error_count >= 0:
            try:
                simulator.step()
                error_count = -1
            except:
                error_count += 1
                if error_count > 10:
                    print('Might not be possible to achieve conversion at T={0}, P={1}, with concentrations of {2} obtaining a conversion of {3} at time {4} s.'.format(solution.T,solution.P,zip(solution.species_names,solution.X), new_conversion,simulator.time))
                    raise
        new_conversion = 1-sum([solution.concentrations[target_species_index] for target_species_index in target_species_indexes])/starting_concentration
        if new_conversion > conversion:
            proper_conversion = True
    #print 'terminated at {0} with conversion {1}.'.format(simulator.time, new_conversion)
    # setup data storage
    outputs = {}
    outputs['conditions'] = get_conditions_series(simulator,solution)
    if output_species:
        outputs['species'] = get_species_series(solution)
    if output_reactions:
        outputs['net_reactions'] = get_reaction_series(solution)
    if output_directional_reactions:
        outputs['directional_reactions'] = get_forward_and_reverse_reactions_series(solution)
    if output_rop_roc:
        outputs['rop'] = get_rop_and_roc_series(solution)

    return outputs


def find_ignition_delay(solution, conditions, 
                      condition_type = 'adiabatic-constant-volume',
                      output_profile = False,
                      temp_final = 965,
                      time_final = 1000,
                      skip_data = 150,
                      output_reactions = True,
                      output_rop_roc = False):
    """
    This method finds the ignition delay of a cantera solution object with
    an option to return all species and reactions as a pandas.DataFrame object
    which can be stored. 
    
    The method calculates ignition delay by going until the temperature is near
    `temp_final`, and then it locates the maximum change in temperature with
    time, $\frac{\delta T}{\delta t}$. The time value corresponding with the max
    is the ignition delay
    
    This method returns a tuple with ignition delay and the species and 
    reaction data, and the rate of production and consumption (or `None`
    if not specified). 
    
    `solution` = Cantera.Solution object
    `conditions` = tuple of temperature, pressure, and mole fraction initial species
    `condition_type` = string describing the run type, currently only 'adiabatic-constant-volume' supported
    `output_profile` = should the program save simulation results and output them (True),
                        or should it just give the ignition delay (False)
    `temp_final` = the temperature which the ignition is reported
    `time_final` = the time to cut off the simulation if the temperature never
                    reaches `temp_final`
    `skip_data` = an integer which reduces storing each point of data.
                    storage space scales as 1/`skip_data`
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
        df = df.append(get_data_series(simulator, solution, 
                                       add_rxns=output_reactions), ignore_index = True)
    else:
        df = None
        
    if output_rop_roc:
        rop_roc = pd.DataFrame()
        rop_roc = rop_roc.append(get_rop_and_roc_series(solution))
    else:
        rop_roc = None
    
        
    # run simulation
    old_time = -1
    old_temp = reactor.T
    max_dTdt = 0
    max_dTdt_time = 0
    data_storage = 1e8
    while simulator.time < time_final:
        simulator.step(time_final)
        if data_storage > skip_data:
            data_storage = 1
            if time_final == 500 and reactor.T > temp_final:
                time_final = simulator.time * 1.03 # go just beyond the final temperature
            if output_profile:
                df = df.append(get_data_series(simulator,solution,
                                               add_rxns=output_reactions),ignore_index = True)
            if output_rop_roc:
                rop_roc = rop_roc.append(get_rop_and_roc_series(solution), ignore_index = True)
            
            # find ignition delay
            dTdt = (reactor.T - old_temp) / (simulator.time - old_time)
            if dTdt > max_dTdt:
                max_dTdt = dTdt
                max_dTdt_time = simulator.time
            old_temp = reactor.T
            old_time = simulator.time
        data_storage += 1
    
    return max_dTdt_time, df, rop_roc

###################################
# 1d. saving data
###################################

def get_data_series(simulator,solution, basics= ['time','temperature','pressure','density'],
                      add_species = True, species_names='all',
                      add_rxns = False, reaction_names='all'):
    """
    a wrapper function of `get_conditions_series`, `get_species_series`, and 
    `get_reaction_series`, which may be depreciated in the future
    """

    conditions = get_conditions_series(simulator,solution, basics)

    if add_species:
        species_series = get_species_series(solution, species_names)
        conditions = pd.concat([conditions,species_series])

    if add_rxns:
        rxn_series = get_reaction_series(solution, reaction_names)
        conditions = pd.concat([conditions,rxn_series])

    return conditions

def get_conditions_series(simulator, solution, basics= ['time','temperature','pressure','density']):
    """
    returns the current conditions of a Solution object contianing ReactorNet
    object (simulator) as a pd.Series.
    
    simulator = the ReactorNet object of the simulation
    solution = solution object to pull values from
    basics =a list of state variables to save 
    
    The following are enabled for the conditions:
    * time
    * temperature
    * pressure
    * density
    * volume
    * cp (constant pressure heat capacity)
    * cv (constant volume heat capacity)
    """
    conditions = pd.Series()
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
    return conditions

def get_species_series(solution, species_names = 'all'):
    """
    returns a pandas.Series of the desired species' concentrations
    
    solution = the cantera.Solution object for the simulation
    species_names = list of species names to be saved (default is all)
    """
    series = pd.Series()
    if species_names=='all':
        species_recorded = solution.species_names
    else:
        species_recorded = species_names
    mole_fractions = solution.mole_fraction_dict()
    for name in species_recorded:
        try:
            series[name] = mole_fractions[name] * solution.density_mole
        except KeyError:
            series[name] = 0
            # sends warning if user typed species incorrectly
            if name not in solution.species_names:
                warnings.warn('{} is not listed in the mole fraction dictionary and may be mispelled.'.format(name))
    return series

def get_reaction_series(solution, reaction_names = 'all'):
    """
    returns a pandas.Series of the desired reactions' net rates
    
    solution = the cantera.Solution object for the simulation
    species_names = list of reaction names to be saved (default is all)
    """
    series = pd.Series()
    if reaction_names=='all':
        reaction_names = solution.reaction_equations()

    reaction_rates = __get_rxn_rate_dict(solution.reaction_equations(),solution.net_rates_of_progress)
    for name in reaction_names:
        try:
            series[name] = reaction_rates[name]
        except KeyError:
            series[name] = 0
            warnings.warn('{} is not listed in the reaction names.'.format(name))
    return series

def get_forward_and_reverse_reactions_series(solution):
    """
    This method returns a series of the forward and reverse reactions
    """
    reaction_equations = solution.reaction_equations()
    forward_reactions = pd.Series(__get_rxn_rate_dict(reaction_equations,solution.forward_rates_of_progress))
    reverse_reactions = pd.Series(__get_rxn_rate_dict(reaction_equations,solution.reverse_rates_of_progress))
    
    forward_reactions.index = pd.MultiIndex.from_product([['forward'],forward_reactions.index], names = ['direction','reaction'])
    reverse_reactions.index = pd.MultiIndex.from_product([['reverse'],reverse_reactions.index], names = ['direction','reaction'])
    
    return pd.concat([forward_reactions,reverse_reactions])

def get_rop_and_roc_series(solution):
    """
    returns rate of production and rate of consumption to dataframe (kmol/m3s)
    This data is primarily useful for quasi-steady state analysis
    """
    species = solution.species_names
    production = pd.Series(__get_rxn_rate_dict(species,solution.creation_rates))
    consumption = pd.Series(__get_rxn_rate_dict(species,solution.destruction_rates))
    
    assert isinstance(production.index,pd.indexes.base.Index)
    assert isinstance(consumption.index,pd.indexes.base.Index)
    
    production.index = pd.MultiIndex.from_product([['production'],production.index])
    consumption.index = pd.MultiIndex.from_product([['consumption'],consumption.index])
     
    return pd.concat([production,consumption])

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

def save_flux_diagrams(solution, conditions, times,
                      condition_type = 'adiabatic-constant-volume',
                      path = '.', element = 'C', filename= 'flux_diagram',
                              filetype = 'png'):
    """
    This method is similar to run_simulation but it saves reaction path
    diagrams instead of returning objects.
    """
    solution.TPX = conditions
    if condition_type == 'adiabatic-constant-volume':
        reactor = ct.IdealGasReactor(solution)
    if condition_type == 'constant-temperature-and-pressure':
        reactor = ct.IdealGasConstPressureReactor(solution, energy='off')
    else:
        raise NotImplementedError('only adiabatic constant volume is supported')
    simulator = ct.ReactorNet([reactor])
    solution = reactor.kinetics

    for time in times:
        simulator.advance(time)
        
        save_flux_diagram(solution,filename=filename+'_{:.2e}s'.format(simulator.time),
                              path = path, element=element, filetype = filetype)

def save_flux_diagram(kinetics, path = '.', element = 'C', filename= 'flux_diagram',
                              filetype = 'png'):
    """
    makes a flux diagram and saves it to the designated file.
    
    kinetics is a solution object
    path is the path to the designated storage
    element is the element to be traced. Isotopes can also count
    filename is the filename to store it as
    filetype is the type to store the file as (any availalbe from http://graphviz.org/doc/info/output.html)
    """
    import os

    diagram = ct.ReactionPathDiagram(kinetics, element)
    diagram.label_threshold = 0.000001

    dot_file = 'temp.dot'
    img_file = filename + '.' + filetype
    img_path = os.path.join(path, img_file)

    diagram.write_dot(dot_file)


    error = os.system('dot {0} -T{2} -o{1} -Gdpi=200'.format(dot_file, img_path,filetype))

    if error:
        raise OSError('dot was not able to create the desired image')
    else:
        print("Wrote graphviz output file to '{0}'.".format(img_path))
    os.remove(dot_file)

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
    
    This method only works on forward reactions
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
    
    reactions = find_reactions( solution, df, species)
    reaction_strings = list(reactions.columns)
    stoichiometries = obtain_stoichiometry_of_species(solution,
                                                      species,
                                                      reaction_strings)
    return reactions * stoichiometries
    


###################################
# 3a. output data processing methods
# these methods are less likely to be useful by themselves.
# many methods in 3b call methods in 3a
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
    
def diff_data(df, differentiation_column):
    """
    Takes a data frame and performs an derivative
    of each column with respect to the `differentiation_column`, 
    which is a pandas.Series object.
    """
    if isinstance(df,pd.DataFrame):
        numerator_difference = df.diff(axis='index')
    else:
        numerator_difference = df.diff()
    denominator_difference = differentiation_column.diff()
    derivative = numerator_difference.div( denominator_difference, axis='index')
    derivative.iloc[0] = 0
    return derivative

def find_reactions(solution, df,species='any'):
    """
    finds the reaction columns in the dataframe and returns them
    if a string, species, is specified, it will only return reactions
    with the matching species.
    """
    # find reaction columns
    df_reactions = df.loc[:,['=' in column for column in df.columns]]
    if species =='any':
        return df_reactions
    # not needed since using solution object
    #string = _prepare_string_for_re_processing(species)
    #expression = r'(\A|\s)%s(\Z|\s)' %(string)
    #df_my_reactions = df_reactions.loc[:,[re.compile(expression).search(column) != None for column in df_reactions.columns]]
    included_columns = []
    reaction_strings = solution.reaction_equations()
    #print df_reactions.shape
    #print len(reaction_strings)
    for index, rxn_name in enumerate(reaction_strings):
        if solution.product_stoich_coeff(species,index) !=0 or solution.reactant_stoich_coeff(species,index) !=0:
            included_columns.append(rxn_name)
    #print included_columns
    df_my_reactions = df_reactions[included_columns]
    
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
###################################
# 3b. output data analysis
###################################

def consumption_pathways(solution,df,species='any',ignore_ignition=True, time = 'all'):
    """
    returns the total rate of production 
    for a particular species
    over the entire simulation using
    the forward difference approximation.
    
    Postive values indicate production, negative values indicate consumption
    
    time = 'all' uses average of values. giving a double returns the conpumption
    pathways at one time
    """
    
    if time=='all':
        df_reactions_weighted = integrate_data(find_reactions(solution, df,species), df['time (s)'])
        if ignore_ignition:
            last_index = remove_ignition(df).shape[0] 
        else:
            last_index = df.shape[0]-1
        reactions_weighted = df_reactions_weighted[df.index<last_index].sum()
    else:
        try:
            reactions_weighted = find_reactions(solution, df,species).loc[time,:]
        except KeyError:
            reactions_weighted = find_reactions(solution, df,species).loc[return_nearest_time_index(time,df['time (s)'], index=False),:]
    if species != 'any': # weight to stoich coefficients
        stoich_coeffs = [obtain_stoichiometry_of_species(solution, species, reaction) for reaction in reactions_weighted.index]
        stoich_coeff_dict = pd.Series(dict(zip(reactions_weighted.index,stoich_coeffs)))
        # pandas was having some bug, so manually rewrote the line below
        #reactions_weighted *= stoich_coeff_dict
        for index in stoich_coeff_dict.index:
            reactions_weighted[index] *= stoich_coeff_dict[index]
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
# 4. rename species in chemkin and save a cantera
###################################

def make_string_labels_independent(species):
    """
    This method accepts a list of species objects, converts their label to SMILES,
    and makes sure none of their labels conflict
    If a conflict occurs, the second occurance will have '-2' added, third '-3'...
    """
    from rmgpy.molecule import Molecule
    
    labels = set()
    for spec in species:
        duplicate_index = 1
        if spec.molecule: # use smiles string
            # see if the label is already valid smiles
            try:
                Molecule().fromSMILES(spec.label)
                potential_label = spec.label
            except:
                potential_label = spec.molecule[0].toSMILES()
        else:
            potential_label = spec.label
        unnumbered_label = potential_label
        while potential_label in labels:
            duplicate_index += 1
            potential_label = unnumbered_label + '-{}'.format(duplicate_index)
        spec.label = potential_label
        labels.add(potential_label)

def obtain_cti_file_nicely_named(chemkinfilepath, readComments = True, 
                                 original_ck_file = 'chem_annotated.inp'):
    """
    Given a chemkin file path, this method reads in the chemkin file and species
    dictionary into RMG objects, renames the species more intuitively, and then
    saves the output as 'input_nicely_named.cti' to the same file path.
    """
    from rmgpy.chemkin import loadChemkinFile
    import os
    import soln2cti

    chemkinPath = os.path.join(chemkinfilepath, original_ck_file)
    speciesDictPath = os.path.join(chemkinfilepath,'species_dictionary.txt')
    species, reactions = loadChemkinFile(chemkinPath, speciesDictPath, readComments = readComments, useChemkinNames=False)
    make_string_labels_independent(species)
    for spec in species:
        if len(spec.molecule) == 0:
            print spec
    # convert species
    ct_species = [spec.toCantera(useChemkinIdentifier = False) for spec in species]
    # convert reactions
    # since this can return a rxn or list of reactions, this allows to make a list based on the returned type
    ct_reactions = []
    for rxn in reactions:
        ct_rxn = rxn.toCantera(useChemkinIdentifier = False)
        if isinstance(ct_rxn, list):
            ct_reactions.extend(ct_rxn)
        else:
            ct_reactions.append(ct_rxn)

    # save new file
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                  species=ct_species, reactions=ct_reactions)
    new_file = soln2cti.write(gas)
    # move the file to new location
    os.rename(new_file, os.path.join(chemkinfilepath,'input_nicely_named.cti'))
    # save species dictionary
    dictionary = ''
    for spec in species:
        dictionary += spec.toAdjacencyList() + '\n\n'
    f = open(os.path.join(chemkinfilepath,'species_dictionary_nicely_named.txt'),'w')
    f.write(dictionary)
    f.close()