# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:09:09 2018

@author: mark
"""

import unittest
import simulation
import numpy as np

class TestRunSimulation(unittest.TestCase):

    def setUp(self):
        """A function that is run before all unit tests in this class."""
        self.solution = simulation.create_mechanism('gri30.cti')
        self.temperature = 1000
        self.pressure = 2e5
        self.mole_fractions = {'C3H8': 1, 'N2': 1, 'O2':5}
        self.times = np.linspace(0,0.2,11)
        self.solution.TPX = self.temperature, self.pressure, self.mole_fractions

    def test_constant_temperature_and_pressure(self):
        outputs = simulation.run_simulation(self.solution,self.times,
                                            condition_type = 'constant-temperature-and-pressure',
                                            output_species = False,
                                            output_reactions = False,
                                            output_directional_reactions = False,
                                            output_rop_roc = False)
        conditions = outputs['conditions']
        temperatures = conditions['temperature (K)'].values
        pressures = conditions['pressure (Pa)'].values
        volumes = conditions['volume (m3)'].values
        int_energies = conditions['internal energy (J/kg)']
        print(conditions)
        self.assertAlmostEqual(min(pressures)/max(pressures),1)
        self.assertAlmostEqual(pressures[0], self.pressure)
        self.assertAlmostEqual(min(temperatures)/max(temperatures),1)
        self.assertAlmostEqual(temperatures[0], self.temperature)
        self.assertNotAlmostEqual(min(volumes)/max(volumes),1)
        self.assertNotAlmostEqual(min(int_energies)/max(int_energies),1)

    def test_constant_temperature_and_volume(self):
        outputs = simulation.run_simulation(self.solution,self.times,
                                            condition_type = 'constant-temperature-and-volume',
                                            output_species = False,
                                            output_reactions = False,
                                            output_directional_reactions = False,
                                            output_rop_roc = False)
        conditions = outputs['conditions']
        temperatures = conditions['temperature (K)'].values
        pressures = conditions['pressure (Pa)'].values
        volumes = conditions['volume (m3)'].values
        int_energies = conditions['internal energy (J/kg)']
        print(conditions)
        self.assertNotAlmostEqual(min(pressures)/max(pressures),1)
        self.assertNotAlmostEqual(min(int_energies)/max(int_energies),1)
        self.assertAlmostEqual(min(volumes)/max(volumes),1)
        self.assertAlmostEqual(min(temperatures)/max(temperatures),1)
        self.assertAlmostEqual(temperatures[0], self.temperature)

    def test_adiabatic_constant_volume(self):
        outputs = simulation.run_simulation(self.solution,self.times,
                                            condition_type = 'adiabatic-constant-volume',
                                            output_species = False,
                                            output_reactions = False,
                                            output_directional_reactions = False,
                                            output_rop_roc = False)
        conditions = outputs['conditions']
        temperatures = conditions['temperature (K)'].values
        pressures = conditions['pressure (Pa)'].values
        volumes = conditions['volume (m3)'].values
        int_energies = conditions['internal energy (J/kg)']
        print(conditions)
        self.assertNotAlmostEqual(min(pressures)/max(pressures),1)
        self.assertNotAlmostEqual(min(temperatures)/max(temperatures),1)
        self.assertAlmostEqual(min(int_energies)/max(int_energies),1,places=5)
        self.assertAlmostEqual(min(volumes)/max(volumes),1)

    def test_specified_temperature_constant_volume_changing_T(self):
        temperatures = np.linspace(900, 1000,len(self.times))
        print temperatures
        outputs = simulation.run_simulation(self.solution,self.times,
                                            condition_type = 'specified-temperature-constant-volume',
                                            output_species = False,
                                            output_reactions = False,
                                            output_directional_reactions = False,
                                            output_rop_roc = False,
                                            temperature_values = temperatures)
        conditions = outputs['conditions']
        temperatures = conditions['temperature (K)'].values
        pressures = conditions['pressure (Pa)'].values
        volumes = conditions['volume (m3)'].values
        int_energies = conditions['internal energy (J/kg)']
        print(conditions)
        self.assertNotAlmostEqual(min(int_energies)/max(int_energies),1)
        self.assertNotAlmostEqual(min(temperatures)/max(temperatures),1)
        self.assertAlmostEqual(min(temperatures),900)
        self.assertAlmostEqual(max(temperatures),1000)
        self.assertAlmostEqual(min(volumes)/max(volumes),1)

    def test_that_specified_temp_performs_same_as_constant_temperature(self):
        temperatures = np.ones(len(self.times)) * self.temperature
        outputs_specs = simulation.run_simulation(self.solution,self.times,
                                            condition_type = 'specified-temperature-constant-volume',
                                            output_species = False,
                                            output_reactions = False,
                                            output_directional_reactions = False,
                                            output_rop_roc = False,
                                            temperature_values = temperatures)
        self.solution.TPX = self.temperature, self.pressure, self.mole_fractions
        outputs_const = simulation.run_simulation(self.solution,self.times,
                                            condition_type = 'constant-temperature-and-volume',
                                            output_species = False,
                                            output_reactions = False,
                                            output_directional_reactions = False,
                                            output_rop_roc = False)
        print(outputs_specs)
        print(outputs_const)
        # make sure all are within 1000
        self.compare_data_frames(outputs_const['conditions'],outputs_specs['conditions'])

    def test_specified_temp_over_different_time_steps(self):
        """compare the first iteration of one simulation with the last iteration
        of one taking smaller steps"""
        temperatures = np.ones(len(self.times)) * self.temperature
        outputs1 = simulation.run_simulation(self.solution,self.times,
                                            condition_type = 'specified-temperature-constant-volume',
                                            output_species = False,
                                            output_reactions = False,
                                            output_directional_reactions = False,
                                            output_rop_roc = False,
                                            temperature_values = temperatures)
        times2 = np.linspace(0,self.times[1])
        temperatures2 = np.ones(len(times2))*temperatures[0]
        self.solution.TPX = self.temperature, self.pressure, self.mole_fractions
        outputs2 = simulation.run_simulation(self.solution,times2,
                                            condition_type = 'specified-temperature-constant-volume',
                                            output_species = False,
                                            output_reactions = False,
                                            output_directional_reactions = False,
                                            output_rop_roc = False,
                                            temperature_values = temperatures2)
        self.compare_data_frames(outputs1['conditions'].iloc[1,:],outputs2['conditions'].iloc[-1,:])

    def compare_data_frames(self,df1,df2,tol=1e-3):
        relative_difference = (df1-df2)/df2
        relative_difference.fillna(0,inplace=True)
        print(relative_difference)
        self.assertTrue((relative_difference > -1*tol).values.all() & (relative_difference <tol).values.all())
