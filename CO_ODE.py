import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd


def mass_balance_CO_ODE(vars):
    Membrane, Component_properties, Fibre_Dimensions = vars
    
    Fibre_Dimensions["Number_Fibre"] = Membrane["Area"] / (Fibre_Dimensions["Length"] * math.pi * Fibre_Dimensions["D_out"])  # number of fibres in the module

    J = len(Membrane["Feed_Composition"])
        
    #Number of elements N
    J = len(Membrane["Feed_Composition"])
    min_elements = [3]  # minimum of 3 elements
    for i in range(J):  # (Coker and Freeman, 1998)
        N_i = (Membrane["Area"] * (1 - Membrane["Feed_Composition"][i] + 0.005) * Membrane["Permeance"][i] * Membrane["Pressure_Feed"] * Membrane["Feed_Composition"][i]) / (Membrane["Feed_Flow"] * 0.005)
        min_elements.append(N_i)
    n_elements = min(round(max(min_elements)), 1000)
     
    Membrane["Feed_Composition"] = np.array(Membrane["Feed_Composition"])
    Membrane["Sweep_Composition"] = np.array(Membrane["Sweep_Composition"])

    def membrane_odes(z, var, params):
        Membrane, Component_properties, Fibre_Dimensions = params
        J = len(Membrane["Feed_Composition"])


        # Unpack variables
        '''
        U_r_x = var[:J]  # Retentate composition flows
        U_p_y = var[J:2*J]  # Permeate composition flows
        
        U_r = sum(U_r_x)
        U_p = sum(U_p_y)

        x = U_r_x / U_r
        y = U_p_y / U_p if U_p != 0 else np.zeros_like(U_p_y)
        '''
        epsilon = 1e-8
        # Define the ODEs for each component
        dx_dz = np.zeros(J)
        dy_dz = np.zeros(J)
 
        for i in range(J):
            dx_dz[i] = 1/Membrane["Total_Flow"] * ( - Membrane["Permeance"][i] * (Fibre_Dimensions["D_out"] * math.pi * Fibre_Dimensions["Number_Fibre"]) * (Membrane["Pressure_Feed"] * var[i]/(sum(var[:J])+epsilon) - Membrane["Pressure_Permeate"] * var[J+i]/(sum(var[J:2*J])+epsilon)) ) # change in component flow in retentate is negative of permeation
            dy_dz[i] = - dx_dz[i]
        

        return np.concatenate((dx_dz, dy_dz))

    # Initial conditions
    U_x_N = Membrane["Feed_Composition"] * Membrane["Feed_Flow"] / Membrane["Total_Flow"]
    U_y_N = Membrane["Sweep_Composition"] * Membrane["Sweep_Flow"] / Membrane["Total_Flow"]

    boundary = np.concatenate((U_x_N, U_y_N))

    params = (Membrane, Component_properties, Fibre_Dimensions)

    t_span = [0, Fibre_Dimensions['Length']]
    t_eval = np.linspace(t_span[0], t_span[1], max(250,n_elements))

    #solution = solve_ivp(membrane_odes, t_span, y0 = boundary, args=(params,), method='BDF', t_eval=t_eval)
    solution = solve_ivp(lambda z, var: membrane_odes(z, var, params), t_span, y0 = boundary, method='BDF', t_eval=t_eval)

    z_points = solution.t
    z_points_norm = z_points / np.max(z_points)    
    U_x_profile = solution.y[:J, :]
    U_y_profile = solution.y[J:2*J, :]

    # Calculate the compositions and flows
    x_profiles = U_x_profile / np.sum(U_x_profile, axis=0)
    y_profiles = U_y_profile / (np.sum(U_y_profile, axis=0) + 1e-12)
    Qr_profile = np.sum(U_x_profile, axis=0)
    Qp_profile = np.sum(U_y_profile, axis=0)

    data = {
        "norm_z": z_points_norm,
        **{f"x{i+1}": x_profiles[i, :] for i in range(J)},
        **{f"y{i+1}": y_profiles[i, :] for i in range(J)},
        "cut_r/Qr": Qr_profile,
        "cut_p/Qp": Qp_profile,
    }
 

    # Create DataFrame
    profile = pd.DataFrame(data)

    x_ret = profile.iloc[-1, 1:J+1].values
    y_perm = profile.iloc[-1, J+1:2*J+1].values
    cut_r = profile.iloc[-1, 2*J+1]
    cut_p = profile.iloc[-1, 2*J+2]
    Qr = cut_r * Membrane["Total_Flow"]
    Qp = cut_p * Membrane["Total_Flow"]
    CO_ODE_results = x_ret, y_perm, Qr, Qp

    return CO_ODE_results, profile







