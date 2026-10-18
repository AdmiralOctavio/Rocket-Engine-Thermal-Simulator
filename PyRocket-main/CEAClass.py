#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#
#####################################################################

import numpy as np
import cea

g0              = 9.80665										# standard gravity [m/s^2]
T_reactant      = 298.15										# temperature at which the propellants enter the chamber [K]
stations        = ['chamber', 'throat', 'exit']
new_propellants = {}											# propellants that are not in the CEA thermo database, see add_new_propellant


def add_new_propellant(name, formula, h_f):
	"""Adds a fuel or oxidiser that is not present in the thermo database of NASA CEA

	Args:
		name (str): name of the propellant, to be used in the blends of PropLibrary.py
		formula (dict): number of atoms of each element in the molecule, e.g. {'C': 2, 'H': 6, 'O': 1}
		h_f (float): enthalpy of formation at 298.15 K in J/mol
	"""
	new_propellants[name] = (formula, h_f)


def reactant_mixture(propellants):
	"""Converts propellants into the reactant names, reactant weights and mixture enthalpy h/R that CEA needs

	Args:
		propellants (list): [propellant, mass share] pairs. A propellant is either the name of a single species (str)
			or a blend from PropLibrary.py (dict of species name and weight percentage)
	"""
	weights = {}
	h = 0
	for propellant, share in propellants:
		blend = {propellant: 100} if isinstance(propellant, str) else propellant
		for name, percent in blend.items():
			w = share * percent / sum(blend.values())
			if name in new_propellants:
				# CEA only needs the elements and the enthalpy of a reactant, so a new propellant is passed on as its atoms
				formula, h_f = new_propellants[name]
				atoms    = list(formula)
				atom_w   = cea.Mixture(atoms).moles_to_weights(np.array([float(formula[atom]) for atom in atoms]))
				h       += w * h_f / (atom_w.sum() * 1e-3)
				species  = zip(atoms, w * atom_w / atom_w.sum())
			else:
				h       += w * cea.Mixture([name]).calc_property(cea.ENTHALPY, np.array([1.0]), T_reactant)
				species  = [(name, w)]
			for species_name, species_w in species:
				weights[species_name] = weights.get(species_name, 0) + species_w

	return list(weights), np.array(list(weights.values())), h / cea.R


class CEA():
	def __init__(self, chamber_pressure, ambient_pressure=1.0):
		self.chamber_pressure = chamber_pressure
		self.ambient_pressure = ambient_pressure

	def solve(self, propellants, location, expansion_ratio):
		if location not in stations:
			raise ValueError('Invalid location, use "chamber," "throat" or "exit"')
		idx = stations.index(location)

		names, weights, hc = reactant_mixture(propellants)
		reactants = cea.Mixture(names)
		products  = cea.Mixture(names, products_from_reactants=True)
		solver    = cea.RocketSolver(products, reactants=reactants, transport=True)

		# equilibrium expansion for the gas properties, expansion with frozen composition for the performance
		eq     = cea.RocketSolution(solver)
		frozen = cea.RocketSolution(solver)
		solver.solve(eq,     weights, self.chamber_pressure / 1e5, supar=[expansion_ratio], iac=True, hc=hc)
		solver.solve(frozen, weights, self.chamber_pressure / 1e5, supar=[expansion_ratio], iac=True, hc=hc, n_frz=1)
		if not (eq.converged and frozen.converged):
			raise ValueError('CEA did not converge, check propellants, mixture ratio and chamber pressure')

		self.MW             = eq.MW[idx]
		self.gamma          = eq.gamma_s[idx]
		self.Pr             = eq.Pr_fr[idx]
		self.isp            = frozen.Isp_vacuum[-1] / g0																						# vacuum Isp [s]
		self.cstar          = frozen.c_star[-1]																								# m/s
		self.ispAmb         = (eq.Isp_vacuum[-1] - eq.c_star[-1] * expansion_ratio * self.ambient_pressure / self.chamber_pressure) / g0	# s
		self.mole_fractions = eq.mole_fractions																								# species name: [chamber, throat, exit]
		self.Cp             = eq.cp_fr[idx] * 1e3																								# coversion to J/kg/K
		self.R              = 8314.46 / self.MW																								# J/kg/K
		self.mu             = eq.viscosity[idx] * 1e-4																						# coversion to Pa*s
		self.k              = eq.conductivity_fr[idx] * 0.1																					# coversion to W/m/K
		self.Tc             = eq.T[0]																											# K
		self.Pc             = self.chamber_pressure
		self.rho 			= self.Pc / (self.R * self.Tc)
		self.h              = eq.enthalpy[0] * 1e3																							# J/kg


class BipropCEA(CEA):
	def __init__(self, fuel, oxidiser, chamber_pressure, ambient_pressure=1.0):
		"""Calcualtes hot gas properties using NASA CEA and converts to SI units.

		Args:
			fuel (str or dict): fuel from the NASA CEA thermo database or from PropLibrary.py
			oxidiser (str or dict): oxidiser from the NASA CEA thermo database or from PropLibrary.py
			chamber_pressure (float): combustion chamber total pressure in Pa
			ambient_pressure (float, optional): ambient pressure in Pa. Defaults to 1.0.
		"""
		super().__init__(chamber_pressure, ambient_pressure)
		self.fuel     = fuel
		self.oxidiser = oxidiser

	def metric_cea_output(self, location, mixture_ratio, expansion_ratio):
		self.solve([[self.fuel, 1 / (1 + mixture_ratio)], [self.oxidiser, mixture_ratio / (1 + mixture_ratio)]], location, expansion_ratio)


class MonopropCEA(CEA):
	def __init__(self, propellant, chamber_pressure, ambient_pressure=1.0):
		"""Calcualtes hot gas properties using NASA CEA and converts to SI units.

		Args:
			propellant (str or dict): propellant from the NASA CEA thermo database or from PropLibrary.py
			chamber_pressure (float): combustion chamber total pressure in Pa
			ambient_pressure (float, optional): ambient pressure in Pa. Defaults to 1.0.
		"""
		super().__init__(chamber_pressure, ambient_pressure)
		self.propellant = propellant

	def metric_cea_output(self, location, expansion_ratio):
		self.solve([[self.propellant, 1.0]], location, expansion_ratio)
