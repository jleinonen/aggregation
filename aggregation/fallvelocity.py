"""
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import mcs


g = 9.81 # gravitational acceleration [m/s^2]
rho_i = 917.6 # density of ice [kg/m^3]


def fall_velocity(agg, T=273.15, P=1000e2, method="HW"):
    """The fall velocity of an aggregate.

    Args:
        agg: An Aggregate object.
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].
        method: The method used to compute the fall velocity.
            If "HW", use the method from Heymsfield and Westbrook (2010),
            https://doi.org/10.1175/2010JAS3379.1
            If "KC", use the method from Khvorostyanov and Curry (2005),
            https://doi.org/10.1175/JAS3622.1

    Returns:
        The fall velocity [m/s]
    """
    area = agg.vertical_projected_area()
    D_max = mcs.minimum_covering_sphere(agg.X)[1]*2
    mass = rho_i * agg.X.shape[0] * agg.grid_res**3
    if method=="HW":
        return fall_velocity_HW2(area, mass, D_max, T, P)
    elif method=="KC":
        return fall_velocity_KC(area, mass, D_max, T, P)
    else:
        raise ValueError("Unknown fall velocity method.")


def fall_velocity_HW(area, mass, D_max, T, P):
    """Alternative method for the Heymsfield-Westbrook fall velocity.

    See fall_velocity_HW2 for an explanation of the arguments.
    """

    k = 0.5 # defined in the paper
    delta_0 = 8.0
    C_0 = 0.35

    rho_air = air_density(T, P)
    area_proj = area/((np.pi/4.0) * D_max**2)
    eta = air_dynamic_viscosity(T)

    Xstar = 8.0 * rho_air * mass * g / (np.pi * area_proj**(1.0-k)
        * eta**2) # eq 9
    Re = delta_0**2/4.0 * ( (1.0+((4.0*np.sqrt(Xstar)) /
        (delta_0**2.0*np.sqrt(C_0))))**0.5 - 1 )**2 # eq10

    return eta * Re / (rho_air*D_max)


def fall_velocity_HW2(area, mass, D_max, T, P):
    """The Heymsfield-Westbrook fall velocity.

    Args:
        area: Projected area [m^2].
        mass: Particle mass [kg].
        D_max: Particle maximum dimension [m].
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The fall velocity [m/s].
    """
    do_i = 8.0
    co_i = 0.35

    rho_air = air_density(T, P)
    eta = air_dynamic_viscosity(T)

    # modified Best number eq. on p. 2478
    Ar = area / (np.pi/4)
    Xbest = rho_air * 8.0 * mass * g * D_max / (eta**2 * np.pi * 
        np.sqrt(Ar))

    # Re-X eq. on p. 2478
    c1 = 4.0 / ( do_i**2 * np.sqrt(co_i) )
    c2 = 0.25 * do_i**2
    bracket = np.sqrt(1.0 + c1*np.sqrt(Xbest)) - 1.0
    Re = c2*bracket**2

    return eta * Re / (rho_air * D_max)


def fall_velocity_KC(area, mass, D_max, T, P):
    """The Khvorostyanov-Curry fall velocity.

    Args:
        area: Projected area [m^2].
        mass: Particle mass [kg].
        D_max: Particle maximum dimension [m].
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The fall velocity [m/s].
    """
    C0 = 0.6 # p. 4348
    delta0 = 5.83 # p. 4348
    C1 = 4.0/(delta0**2 * np.sqrt(C0)) # appendix
    Ct = 1.6 # p. 4345
    k = 2 # p. 4345
    X0 = 2.8e6 # p. 4345

    eta = air_dynamic_viscosity(T)
    rho_air = air_density(T, P)

    X = 2.0 * rho_air * mass * g * D_max**2 / (area * eta**2) # 2.4b
    X_sqrt = np.sqrt(X)

    h = np.sqrt(1 + C1*X_sqrt)
    b_Re = C1*X_sqrt / (2 * (h-1) * h) # 2.8
    a_Re = (delta0**2/4.0) * (h-1)**2 / X**b_Re # 2.7
    Re = a_Re * X**b_Re # 2.6

    # 2.2
    Cd = C0*(1+delta0/np.sqrt(Re))**2
    
    # 3.2
    X_rel = (X/X0)**k
    psi = (1.0 + X_rel)/(1.0 + Ct*X_rel)
    # 3.1
    Cd /= psi

    # 2.1
    return np.sqrt(2*mass*g / (rho_air*area*Cd)) 
    

def air_kinematic_viscosity(T, P):
    """The kinematic viscosity of air.

    Args:
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The kinematic viscosity [m^2/s].
    """
    rho = air_density(T, P)
    mu = air_dynamic_viscosity(T)
    return mu/rho


def air_dynamic_viscosity(T):
    """The kinematic viscosity of air.

    Args:
        T: Ambient temperature [K].

    Returns:
        The kinematic viscosity [Pa/s].
    """
    mu0 = 1.716e-5
    T0 = 273.15
    C = 111.0
    return mu0 * ((T0+C)/(T+C)) * (T/T0)**1.5


def air_density(T, P):
    """The density of air.

    Args:
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The kinematic viscosity [Pa/s].
    """
    R = 28704e-2 # gas constant for air
    return P / (T*R)
