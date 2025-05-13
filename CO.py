

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar
import pandas as pd
import math

''' GENERAL INFORMATION:

- Co current model discretised into N elements, with element N being feed/sweep side and element 1 being permeate/retentate side

- Model based on Coker and Freeman (1998)

- Feed and Sweep flow and compositions are known, solving for retentate and permeate compositions and flows

- Solving element by element from N to 1

- Inlcuding pressure drop across the module as an option
'''


def mass_balance_CO(vars):

    Membrane, Component_properties, Fibre_Dimensions = vars

    Total_flow = Membrane["Feed_Flow"] + Membrane["Sweep_Flow"] # Total flow in mol/h
    cut_r_N = Membrane["Feed_Flow"] / Total_flow # Cut ratio at the feed side
    cut_p_N = Membrane["Sweep_Flow"] / Total_flow # Cut ratio at the permeate side

    #Number of elements N
    J = len(Membrane["Feed_Composition"])
    min_elements = [1]  # minimum of 2 elements
    for i in range(J):  # (Coker and Freeman, 1998)
        N_i = (Membrane["Feed_Flow"] * (1 - Membrane["Feed_Composition"][i] + 0.005) * Membrane["Permeance"][i] * Membrane["Pressure_Feed"] * Membrane["Feed_Composition"][i]) / (Membrane["Feed_Flow"] * 0.005)
        min_elements.append(N_i)
    n_elements = min(round(max(min_elements)), 1000)

    DA = Membrane["Area"] / n_elements # Area of each element

    user_vars = DA, J, Total_flow, Membrane["Pressure_Feed"], Membrane["Permeance"]

    '''----------------------------------------------------------###
    ###------------- Mixture Viscosity Calculation --------------###
    ###----------------------------------------------------------'''

    def mixture_visc(composition): #Calculate the viscosity of a mixture using Wilke's method

        y = composition # mole fractions
    
        visc = np.zeros(J)
        for i, (slope, intercept) in enumerate(Component_properties["Viscosity_param"]):
            visc[i] = 1e-6*(slope * Membrane["Temperature"] + intercept)  # Viscosity of pure component - in Pa.s

        Mw = Component_properties["Molar_mass"]  # Molar mass of component in kg/kmol

        phi = np.zeros((J, J))
        for i in range(J):
            for j in range(J):
                if i != j:
                    phi[i][j] = ( ( 1 + ( visc[i]/visc[j] )**0.5 * ( Mw[j]/Mw[i] )**0.25 ) **2 ) / ( ( 8 * ( 1 + Mw[i]/Mw[j] ) )**0.5 ) 
                else:
                    phi[i][j] = 1

        nu=np.zeros(J)
        for i in range(J):
            nu[i] = y[i] * visc [i] / sum(y[i] * phi[i][j] for j in range(J))
    
        visc_mix = sum(nu) # Viscosity of the mixture in Pa.s
        return visc_mix

    '''----------------------------------------------------------###
    ###--------------- Pressure Drop Calculation ----------------###
    ###----------------------------------------------------------'''

    def pressure_drop(composition, Q, P): #change in pressure across the element

        visc_mix = mixture_visc(composition)                                                # Viscosity of the mixture in Pa.s
        D_in = Fibre_Dimensions["D_in"]                                                     # Inner diameter in m
        Q = Q/Fibre_Dimensions['Number_Fibre']                                                                          # Flowrate in fibre in mol/s
        dL = Fibre_Dimensions['Length']/n_elements                                          # Length of the discretised element in m
        R = 8.314                                                                           # J/(mol.K) - gas constant
        dP = 8 * visc_mix / (math.pi * D_in**4) * Q * R * Membrane["Temperature"]/ P * dL   # Pressure drop in Pa
        return dP

    '''----------------------------------------------------------###
    ###-------- Mass Balance Function Across One Element --------###
    ###----------------------------------------------------------'''

    def equations(vars, inputs, user_vars):

        DA, J, Total_flow, P_ret, Perm = user_vars

        # known composition and flowrates connected to element k+1
        x_known = inputs[0:J]
        y_known = inputs[J:2*J]
        cut_r_known = inputs[-3]
        cut_p_known = inputs[-2]

        P_perm = inputs[-1] 
        
        Qr_known = cut_r_known * Total_flow
        Qp_known = cut_p_known * Total_flow

        # Variables to solve for
        x = vars [0:J] # retentate side mole fractions leaving element k - to be exported to next element k-1
        y = vars [J:2*J] # permeate side mole fractions entering element k - to be exported to next element k-1
        cut_r = vars[-2] # retentate flowrate leaving element k - to be exported to next element k-1
        cut_p = vars[-1] # permeate flowrate leaving element k - to be exported to next element k-1

        Qr = cut_r * Total_flow # retentate flowrate leaving element k
        Qp = cut_p * Total_flow
   
        eqs = [0]*(2*J+2) # empty list to store the equations

        #molar fractions summing to unity:
        eqs[0] = sum(x) - 1
        eqs[1] = sum(y) - 1

        #mass balance for each component accros the DA:
        for i in range(J):
            eqs[i+2] = x_known[i] * cut_r_known + y_known[i] * cut_p_known - x[i] * cut_r - y[i] * cut_p #ret in + perm in - ret out - perm out = 0

        #flow across membrane --> chenge in permeate flow is equal to permeation across DA:
        for i in range(J):
            eqs[i+2+J] = (y[i] * Qp -  y_known[i] * Qp_known) - ( Perm[i] * DA * (P_ret * x[i] - P_perm * y[i]) )

        return eqs

    '''-----------------------------------------------------------###
    ###--------- Mass Balance Function Across The Module ---------###
    ###-----------------------------------------------------------'''

    # Create a DataFrame to store the results
    columns = ['Element'] + [f'x{i+1}' for i in range(J)] + [f'y{i+1}' for i in range(J)] + ['cut_r/Qr', 'cut_p/Qp','P_Perm']

    # Preallocate for n_elements
    Solved_membrane_profile = pd.DataFrame(index=range(n_elements), columns=columns)

    # Set the element N with feed known values and guessed permeate value
    Solved_membrane_profile.loc[0] = [n_elements] + list(Membrane["Feed_Composition"]) + list(Membrane["Sweep_Composition"]) + [cut_r_N, cut_p_N, Membrane["Pressure_Permeate"] ]  # element N (Feed/Sweep side)

    for k in range(n_elements - 1):
        # Input vector of known/calculated values from element k+1

        inputs = Solved_membrane_profile.loc[k, Solved_membrane_profile.columns[1:]].values

        # Initial guess for the element k
        guess = [0.5] * (2 * J + 2)
        
        sol_element = least_squares(
            equations,  # function to solve
            guess,  # initial guess
            args=(inputs, user_vars),  # arguments for the function
            method='lm',
            xtol = 1e-8,
            ftol = 1e-8,
            gtol = 1e-8
        )
        
        if not sol_element.success:raise ValueError(f"Mass balance solver failed at element {k}: {sol_element.message}")  
        element_output = sol_element.x
        
        if sol_element.cost > 1e-5:
            print(f'Large mass balance closure error at element {k}; error: {sol_element.cost:.3e}; with residuals {sol_element.fun}')
        
        # Calculate the pressure drop for the permeate side
        y_k = element_output[J:2*J]                     # Permeate composition
        Qp_k = element_output[-1] * Total_flow          # Permeate flowrate
        pP_k = Solved_membrane_profile.loc[k, 'P_Perm'] # Current permeate pressure

        if not Membrane["Pressure_Drop"]:
            pP_new = pP_k  # No pressure drop
            
        else:
            # Calculate the pressure drop
            dP = pressure_drop(y_k, Qp_k, pP_k)

            if dP/pP_k > 1e-4:
                pP_new = Membrane["Pressure_Permeate"] - dP
            else:
                pP_new = pP_k #negligible pressure drop: helps with stability of the solver
    
    
        # Update the DataFrame with the results
        df_element = np.concatenate(([n_elements-1-k], element_output, [pP_new]))
        Solved_membrane_profile.loc[k + 1] = df_element

        #print(f'mass balance closure error: {sol_element.cost:.3e}')
        if sol_element.cost > 1e-5: print(f'with residuals {sol_element.fun}')
    
    x_ret = Solved_membrane_profile.iloc[-1, 1:J+1].values
    y_perm = Solved_membrane_profile.iloc[-1, J+1:2*J+1].values
    cut_r = Solved_membrane_profile.iloc[-1, -3]
    cut_p = Solved_membrane_profile.iloc[-1, -2]
    Qr = cut_r * Total_flow
    Qp = cut_p * Total_flow
    P_perm = Solved_membrane_profile.iloc[-1, -1] * 1e-5  # Permeate pressure in bar at the last element

    CO_results = x_ret, y_perm, Qr, Qp
    
    profile = Solved_membrane_profile.copy()

    return CO_results, profile