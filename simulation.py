# -*- coding: utf-8 -*-
import numpy as np
import cantera as ct
import pandas as pd
import re
import warnings
import copy

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


def create_mechanism(full_model_file_path,kept_reaction_equations='all', remove_reaction_equations=None,
                     non_reactive_species = ['AR','N2','HE']):
    """
    This is a convenience method for reducing mechanisms when reading cantera
    files.

    input the full model and a list of reaction equations that you'd like to keep
    or a list of reaction equations to remove.

    This method should retain or remove all instances of the reactions

    returns a Cantera.Solution object of the mechanism with only the cooresponding
    reactions and their species.
    """
    desired_file = full_model_file_path
    spec = ct.Species.listFromFile(desired_file)
    rxns = ct.Reaction.listFromFile(desired_file)

    if remove_reaction_equations is not None:
        if isinstance(remove_reaction_equations,list):
            rxn_index = 0 
            while rxn_index < len(rxns):

                rxn_eq = rxns[rxn_index].equation

                if rxn_eq in remove_reaction_equations:
                    del rxns[rxn_index]
                else:
                    rxn_index += 1
            reduced_species = eliminate_species_from_mechanism(spec,rxns,non_reactive_species)
            return ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                      species=reduced_species, reactions=rxns)
        else:
            raise TypeError('remove reaction equations must be a list if specified. It is currently {}'.format(remove_reaction_equations))


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
                      rtol = 1e-9,
                      temperature_values=None):
    """
    This method iterates through the cantera solution object and outputs information
    about the simulation as a pandas.DataFrame object.
    
    This method returns a dictionary with the reaction conditions data, species data,
    net reaction data, forward/reverse reaction data, and the rate of production 
    and consumption (or `None` if a variable not specified). 
    
    `solution` = Cantera.Solution object
    `conditions` = tuple of temperature, pressure, and mole fraction initial 
                species (will be deprecated. Set parameters before running)
    `times` = an iterable of times which you would like to store information in
    `condition_type` = string describing the run type
    `output_species` = output a DataFrame of species' concentrations
    `output_reactions` = output a DataFrame of net reaction rates
    `output_directional_reactions` = output a DataFrame of directional reaction rates
    `output_rop_roc` = output a DataFrame of species rates of consumption & production

    condition_types supported
    #########################
    'adiabatic-constant-volume' - assumes no heat transfer and no volume change
    'constant-temperature-and-pressure' - no solving energy equation or changing
                            rate constants
    'constant-temperature-and-volume' - no solving energy equation but allows
                            for pressure to change with reactions
    'specified-temperature-constant-volume' - the temperature profile specified
                            `temperature_values`, which corresponds to the
                            input `times`, alters the temperature right before
                            the next time step is taken. Constant volume is assumed.
    """
    if conditions is not None:
        solution.TPX = conditions
    if condition_type == 'adiabatic-constant-volume':
        reactor = ct.IdealGasReactor(solution)
    elif condition_type == 'constant-temperature-and-pressure':
        reactor = ct.IdealGasConstPressureReactor(solution, energy='off')
    elif condition_type == 'constant-temperature-and-volume':
        reactor = ct.IdealGasReactor(solution, energy='off')
    elif condition_type == 'specified-temperature-constant-volume':
        reactor = ct.IdealGasReactor(solution, energy='off')
        if temperature_values is None:
            raise AttributeError('Must specify temperature with `temperature_values` parameter')
        elif len(times) != len(temperature_values):
            raise AttributeError('`times` (len {0}) and `temperature_values` (len {1}) must have the same length.'.format(len(times),len(temperature_values)))
    else:
        supported_types = ['adiabatic-constant-volume','constant-temperature-and-pressure',
                           'constant-temperature-and-volume','specified-temperature-constant-volume']
        raise NotImplementedError('only {0} are supported. {1} input'.format(supported_types, condition_type))
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

    for time_index, time in enumerate(times):
        if condition_type == 'specified-temperature-constant-volume':
            solution.TD = temperature_values[time_index], solution.density
            reactor = ct.IdealGasReactor(solution, energy='off')
            simulator = ct.ReactorNet([reactor])
            solution = reactor.kinetics
            simulator.atol = atol
            simulator.rtol = rtol
            if time_index > 0:
                simulator.set_initial_time(times[time_index-1])
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
                      skip_data = 150,
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
    `skip_data` = an integer which reduces storing each point of data.
                    storage space scales as 1/`skip_data`
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

    if isinstance(species,str):
        target_species_indexes = [solution.species_index(species)]
    else: # must be a list or tuple
        target_species_indexes = [solution.species_index(s) for s in species]
    starting_concentration = sum([solution.concentrations[target_species_index] for target_species_index in target_species_indexes])

    proper_conversion = False
    new_conversion = 0
    skip_count = 1e8
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
        
        # save data
        if skip_count > skip_data or proper_conversion:
            skip_count = 0
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
        skip_count += 1

    # set indexes as time
    time_vector = outputs['conditions']['time (s)']
    for output in outputs.values():
        output.set_index(time_vector,inplace=True)

    return outputs


def find_ignition_delay(solution, conditions=None, 
                      condition_type = 'adiabatic-constant-volume',
                      output_profile = False,
                      output_species = True,
                      output_reactions = True,
                      output_directional_reactions = False,
                      output_rop_roc = False,
                      temp_final = 965,
                      time_final = 1000,
                      skip_data = 150,):
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
    `output_species` = output a Series of species' concentrations
    `output_reactions` = output a Series of net reaction rates
    `output_directional_reactions` = output a Series of directional reaction rates
    `output_rop_roc` = output a DataFrame of species rates of consumption & production
    `temp_final` = the temperature which the ignition is reported
    `time_final` = the time to cut off the simulation if the temperature never
                    reaches `temp_final`
    `skip_data` = an integer which reduces storing each point of data.
                    storage space scales as 1/`skip_data`
    """
    if conditions is not None:
        solution.TPX = conditions
    if condition_type == 'adiabatic-constant-volume':
        reactor = ct.IdealGasReactor(solution)
        simulator = ct.ReactorNet([reactor])
        solution = reactor.kinetics
    else:
        raise NotImplementedError('only adiabatic constant volume is supported')
        
    # setup data storage
    outputs = {}
    if output_profile:
        outputs['conditions'] = pd.DataFrame()
        if output_species:
            outputs['species'] = pd.DataFrame()
        if output_reactions:
            outputs['net_reactions'] = pd.DataFrame()
        if output_directional_reactions:
            outputs['directional_reactions'] = pd.DataFrame()
        if output_rop_roc:
            outputs['rop'] = pd.DataFrame()

    # run simulation
    max_time = time_final
    old_time = -1
    old_temp = reactor.T
    max_dTdt = 0
    max_dTdt_time = 0
    data_storage = 1e8 # large number to ensure first data point taken
    while simulator.time < time_final:
        simulator.step()
        if data_storage > skip_data:
            data_storage = 1
            if time_final == max_time and reactor.T > temp_final:
                time_final = simulator.time * 1.01 # go just beyond the final temperature
            if output_profile:
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
            
            # find ignition delay
            dTdt = (reactor.T - old_temp) / (simulator.time - old_time)
            if dTdt > max_dTdt:
                max_dTdt = dTdt
                max_dTdt_time = simulator.time
            old_temp = reactor.T
            old_time = simulator.time
        data_storage += 1
    # set indexes as time
    if output_profile:
        time_vector = outputs['conditions']['time (s)']
        for output in outputs.values():
            output.set_index(time_vector,inplace=True)
    # save ignition_delay
    outputs['ignition_delay'] = max_dTdt_time
    return outputs

def save_flux_diagrams(solution, times, conditions=None, 
                      condition_type = 'adiabatic-constant-volume',
                      path = '.', element = 'C', filename= 'flux_diagram',
                              filetype = 'png'):
    """
    This method is similar to run_simulation but it saves reaction path
    diagrams instead of returning objects.
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

    for time in times:
        simulator.advance(time)

        save_flux_diagram(solution,filename=filename+'_{:.2e}s'.format(simulator.time),
                              path = path, element=element, filetype = filetype)

###################################
# 1d. saving data helper methods
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
    
    assert isinstance(production.index,pd.Index)
    assert isinstance(consumption.index,pd.Index)
    
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