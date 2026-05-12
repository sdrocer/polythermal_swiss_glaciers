import numpy as np
from scipy.integrate import quad

"""
This script calculates the drilling speed of a hot-water drill based on the equations in the images.
The equations are extracted from the paper "Cold ice in an alpine glacier and ice dynamics at the margin of the Greenland Ice Sheet" by Claudia Ryser's PhD thesis.
"""

# water temperature as a function of depth
def Tw(z, T_in, lambda_, T_wall):
    return T_in * np.exp(-z / lambda_) + T_wall

# drilling speed
def v_drill(z, T_in, lambda_, T_wall):
    return C(A0, d) * c_w * rho_w * Q_w * Tw(z, T_in, lambda_, T_wall)

# specific drilling rate
def C(A0, d):
    return A0 / (d**2 * (L_i * rho_i + c_i * rho_i * T_ice))**(1/3)

# drill time simple calculation
def t_simple(z, T_in, lambda_, T_wall):
    return z / v_drill(z, T_in, lambda_, T_wall)

# Constants
c_i = 2009 # J/kg/K
c_w = 4186 # J/kg/K
L_i = 334000 # J/kg
rho_i = 917 # kg/m^3
rho_w = 1000 # kg/m^3
T_ice = -10 # degC (temperature of the ice)
Q = 60 # l/min (flow rate of the water)
Q_w = Q / 60 / 1000 # m^3/s (flow rate of the water)
A0 = 7.1510e-5 # m^2 kWh^(-2/3) (specific drilling rate constant)
d = 0.0035 # m (diameter of the drill, converted from 3.5 mm)
T_wall = 0 # degC (temperature of the ice)
T_in = 80 # degC (temperature of the water)
Z = 0.1 # eff. total thermal resistance (m^2·K/W)
lambda_ = rho_w * c_w * Q_w * Z # W/(m·K) (thermal conductivity of the ice)
R = 0.4 # borehole radius (m)

# Variables
z = 100 # m

# Calculate the results
T = Tw(z, T_in, lambda_, T_wall)          # Temperature at the depth z
C_ = C(A0, d)                             # Specific drilling rate
v = v_drill(z, T_in, lambda_, T_wall)     # Drilling speed
drill_time = t(z, T_in, lambda_, T_wall) / 60  # Drill time needed to reach the depth z in minutes

# Print all results
print(f"Temperature at the depth z={z} m: {T} degC")
print(f"Specific drilling rate: {C_} m/s")
print(f"Drilling speed: {v} m/s")
print(f"Drill time simple calculation: {drill_time} minutes")