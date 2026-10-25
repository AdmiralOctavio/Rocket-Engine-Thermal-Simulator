#####################################################################
#                           PyRocket								#
# 2D Regenertive Cooling Simulation for Bipropellant Rocket Engines #
#                                                                   #
# Creator:  Joanthan Neeser                                         #
# Date:     15.12.2022                                              #
# Version:  2.3  													#
# License:	GNU GENERAL PUBLIC LICENSE V3							#                                          
#####################################################################

import os
import numpy as np

from IsentropicRelations import Isentropic
from CEAClass import BipropCEA
from CoolantClass import Coolant
from GeomClass import ChamberGeometry, CoolingGeometry
from Output import Output1D, Settings2D
from FilmCooling import FilmCooling

import MaterialLib as matlib
import PropLibrary as proplib

# IMPORTANT OBSERVATIONS OF THIS TOOL
# 
# This tool has been used to design DLX-1000A, a regeneratively cooled Nitrous/Ethanol engine.
# This tool originally predicted that CHF would be exceeded by a factor of around 2.5-3x around the throat and nozzle.
# In practice, the engine survive without issue for a 2s and 3s test burn. Exceeding CHF would yield
# failure within 0.1s, ergo either:
# 1) CHF is being grossly underestimated
# 2) CHF is not a critically defining factor of failure
#
# Please take this into consideration when simulating your own engines. 
# - Flavio Cicero


# CEA input
fuel            = proplib.ethanol80               # choose existing fuel or oxidiser from rocketcea or create new fuel or oxidiser blend in PropLibrary
ox              = 'N2O'
hot_gas_method  = 'cinjarev'                         # use either 'cinjarev', 'standard-bartz' or 'modified-bartz'

# Operating point of combustion chamber 
eta_c_star      = 0.90                                # c* efficiency

MR              = 4.4
m_dot           = 0.518                                # total mass flow [kg/s]
m_dot_f_ideal   = m_dot / (MR + 1)
m_dot_f         = 0.113
m_dot_ox        = m_dot - m_dot_f_ideal
Pc              = 34e5                               # Chamber pressure [Pa]

# Coolant input
cooling_fluid   = ['Ethanol', 'Water']                         # needs to be list of str, use CoolProp fluid names
fluid_mass_frac = [0.8, 0.2]    			                 # mass fractions of the cooling fluid (needs to add up to 1)
m_dot_coolant   = m_dot_f                 		 	 # mass flow through the cooling channels
inlet_temp      = 288                     		 # inlet temperature [K]
inlet_pressure  = 45e5                               # Cooling channel inlet pressure [Pa]

cooling_method  = 'gnielinski-rough'                       # use either 'gnielinski', 'gnielinski-rough' (no laminar flow, for rough channels), 'dittus-boelter' or 'dittus-boelter-simple', pay attention to suitable Re number range
boiling         = True                               # nucleate boiling at cooling channel walls above the saturation temperature of the coolant, False for forced convection only
chf_method      = 'hall-mudawar'                            # critical heat flux of the coolant, use either 'meyer' (measured for ethanol), 'gambill' (any coolant, lowest values) or 'hall-mudawar' (mixture properties and channel diameter, corrected with the ethanol measurements)
post_chf        = 'meyer'                            # beyond the critical heat flux, use either 'meyer' (ethanol above 17 bar keeps boiling up to 1.55x the CHF, measured by Meyer et al.) or 'fallback' (section solved again with forced convection only, the most conservative)
ambient_temp    = 288.15							 # ambient temperature for radiation boundary condition [K] (must be type float)!!

# Film cooling settings for coolant originating at the injector face
m_dot_film = 0.35 * m_dot_coolant                      # set mass flow to 0.0 to deactivate film cooing
injection_velocity = 10								  # [m/s]
injector_diameter  = 0.4e-3							  # [m]

# Additional rings of film cooling ports in the chamber wall, which take their coolant from the cooling channels
# one entry per ring: [x coordinate of the chamber contour as in the plots [m], mass flow [kg/s], injection velocity [m/s], port diameter [m]]
# EXAMPLE of a ring 20 mm upstream of the throat: film_ports = [[-20e-3, 0.1 * m_dot_coolant, 10, 0.4e-3]]
# film_ports = [[-50e-3, 0.25*m_dot_coolant, 10, 0.4e-3], [-10e-3, 0.1*m_dot_coolant, 10, 0.4e-3]]
film_ports = []

# Chamber geometry
D_c       = 50e-3                                   # chamber diamter [m]
D_t       = 12.5e-3                                    # throat diameter [m]
D_e       = 27.56e-3                                    # exit diamter [m]
L_cyl     = 50e-3                                   # cylindrical chamber length [m]
r_1       = 50e-3                                    # converging section radius [m]
r_2       = 0.4*D_t                                  # throat converging section radius [m]
r_n       = 0.2*D_t                                  # throat diverging radius [m]
phi_conv  = 30                                       # convergence angle [deg]
phi_div   = 24                                       # divergence angle [deg]
phi_e     = 15                                       # exit angle [deg]
step_size = 0.0025                                     # step size along the chamber contour [m]
material  = matlib.GRCop42                             # use entry from MaterialLibrary. Make sure temperature dependent properties are specified

# cooling channel geometry; h_c, psi, t_w_i can be functions of x 
n         = 15                                       # number of cooling channels [int]
h_c       = 1e-3                                     # radial height of cooling channels [m] CAN BE FLOAT OR FUNCTION
# EXAMPLE of function input for psi: psi = lambda x: 1/3 * (1 - 0.0001 * x) 
psi       = 0.45                                      # fill factor of the cooling channels; fraction of the circumferecne covered by the cooling channels (0 - 1) CAN BE FLOAT OR FUNCTION
t_w_i     = 0.9e-3                                   # inner chamber wall thickness [m] CAN BE FLOAT OR FUNCTION
t_w_o     = 1.2e-3                                     # outer chamber wall thickness [m]
start_idx = -1										 # starting index, use 0 for injector side and -1 for nozzle

# 2D Section Simulation settings
cell_size    = 0.15 * t_w_i                          # cell size in 2D section solver, will heavily impact performance
time_step    = (cell_size)**2 / (material.alpha)     # time step in 2D section solver (cell_size)**2 / (material.alpha)
adaptive_up  = 1.4                                   # UP factor for adaptive time step. Increase for faster convergence but less stability
tolerance    = 5e-2                                  # maximum temperature difference between time steps
max_iter     = 300									 # maximum number of iterations before termination
save_fig     = True                                  # save figures to folder in Output Class
print_result = True                                  # print intermittant maximum temperatures, useful for DEBUGGING
run_time     = 'steady_state'                        # use 'steady_state' as default. use time in [s] if transient solution is desired


# add thermocouple locations for model validation (effectively temperature logging points in the 2D solution)
log_TC = False                                       # temperature at thermocouple locations is to be logged
TC_x   = [10e-3, 20e-3, 30e-3]                       # x coordinate of thermocouples 
TC_r   = [36e-3, 36e-3, 36e-3]                       # radial position of thermocouples               


# save folder path for output class
save_path = "SectionImages"


######################################
# GAS, COOLANT and GEOMETRY SETUP
######################################

# create folder to store sim files
try:
    os.mkdir(save_path)
except:
    pass
    
# generate chamber geometry
chamber = ChamberGeometry(D_c, D_t, D_e, L_cyl, r_1, r_2, r_n, phi_conv, phi_div, phi_e, step_size)
chamber.contour()
chamber.plot_contour(save_path)
geometry = chamber.geometry

# generate cooling channel geometry
cooling_geom = CoolingGeometry(geometry, h_c, psi, t_w_i, t_w_o, n)
cooling_geom.channel_geometry()
cooling_geom.set_thermocouples(TC_x, TC_r)


# generate settings class for 2D thermal sim
settings = Settings2D(cell_size, time_step, tolerance, max_iter, save_fig, print_result, run_time, start_idx, adaptive_up, log_TC, cooling_geom.thermocouples)

# generate output class 
output = Output1D(geometry, save_path)

# calculate hot gas properties from NASA CEA
cea = BipropCEA(fuel, ox, Pc) 
cea.metric_cea_output('throat', MR, chamber.expansion_ratio)

# isentropic properties for the hot gas along the chamber contour
gas = Isentropic(Pc, cea.Tc, cea.gamma, cea.Pr, geometry[:,0], geometry[:,1])
gas.calculate()

# coolant Properties from CoolProp library
coolant = Coolant(cooling_fluid, ws=fluid_mass_frac, P=inlet_pressure, T=inlet_temp)     

# film cooling model, the film from the injector face is the first ring of ports
film = FilmCooling(cea, gas, coolant, geometry, m_dot, [[geometry[0,0], m_dot_film, injection_velocity, injector_diameter]] + film_ports)
film.calculate()
