import math
import matplotlib.pyplot as plt
from pytest import approx
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import os


def mass_balance_CC_ODE(vars):

    Membrane, Component_properties, Fibre_Dimensions = vars
    Membrane["Total_Flow"] = Membrane["Feed_Flow"] + Membrane["Sweep_Flow"]
    Fibre_Dimensions["Number_Fibre"] = Membrane["Area"] / (Fibre_Dimensions["Length"] * math.pi * Fibre_Dimensions["D_out"])  # number of fibres in the module
    epsilon = 1e-8

    Membrane["Feed_Composition"] = np.array(Membrane["Feed_Composition"])
    Membrane["Sweep_Composition"] = np.array(Membrane["Sweep_Composition"])

    '''---------------------------------------------------------------###
    ###---------- Non discretised solution for initial guess ----------###
    ###---------------------------------------------------------------'''

    '''
    - Approximate solution for overall mass balance. Will be used as an input for the shooting method.
    - Using the log mean partial pressure difference as driving force. Heavier and less stable than dicretisation, but provides a good estimate.
    '''

    def approx_mass_balance(vars):

        J = len(Membrane["Feed_Composition"])

        # known composition and flowrates entering the module
        x_N = Membrane["Feed_Composition"]
        y_0 = Membrane["Sweep_Composition"]
        cut_r_N = Membrane["Feed_Flow"]/Membrane["Total_Flow"]
        cut_p_0 = Membrane["Sweep_Flow"]/Membrane["Total_Flow"]
    
        Qr_N = Membrane["Feed_Flow"] 
        Qp_0 = Membrane["Sweep_Flow"]

        x_0 = vars [0:J] # retentate mole fractions
        y_N = vars [J:2*J] # permeate mole fractions
        cut_r_0 = vars[-2] # retentate flowrate fraction
        cut_p_N= vars[-1] # permeate flowrate fraction

        Qr_0 = Membrane["Total_Flow"] * cut_r_0 # retentate flowrate
        Qp_N = Membrane["Total_Flow"] * cut_p_N

        eqs = [0]*(2*J+2) # empty list to store the equations


        #molar fractions summing to unity:
        eqs[0] = sum(x_0) - 1
        eqs[1] = sum(y_N) - 1

        #mass balance for each component across the module
        for i in range(J):
            eqs[i+2] = ( x_N[i] * cut_r_N - x_0[i] * cut_r_0 + y_0[i] * cut_p_0 - y_N[i] * cut_p_N ) #in ret - out ret + in perm - out perm

        #flow across membrane --> change in permeate flowrate is equal to the permeation across the area
        for i in range (J): 
            pp_diff_in = Membrane["Pressure_Feed"] * x_N[i] - Membrane["Pressure_Permeate"] * y_0[i]
            pp_diff_out = Membrane["Pressure_Feed"] * x_0[i] - Membrane["Pressure_Permeate"] * y_N[i]

            if (pp_diff_in / (pp_diff_out + epsilon) + epsilon) >= 0: #using the log mean partial pressure difference as it is a better approximation when the membrane is not discretise.
                ln_term = math.log((pp_diff_in) / (pp_diff_out + epsilon) + epsilon)  #It is however less stable, hence these expressions to make sure there is no division by zero.
            else:
                ln_term = epsilon 

            dP = (pp_diff_in - pp_diff_out) / ln_term # driving force

            eqs[i+2+J] = 1 - ( Membrane["Area"] * dP * Membrane["Permeance"][i] ) / ( y_N[i] * Qp_N - y_0[i] * Qp_0 +epsilon)# difference in permeate flowrate in/out = permeation across the area

        return eqs

    def approx_shooting_guess():

        J = len(Membrane["Feed_Composition"])
        #solving the approximate mass balance for the module
        approx_guess = [1/J]*J * 2 + [0.5] * 2 #ret and perm compositions + flows

        approx_sol = least_squares(
            approx_mass_balance,
            approx_guess,
            bounds=(0,1),
            xtol=1e-6,
            ftol=1e-6   
            )


        return approx_sol


    def mass_balance_reverse(vars):

        J = len(Membrane["Feed_Composition"])

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
        U_x_0 = vars #input retentate guess for the shooting method 
        U_y_0 = Membrane["Sweep_Composition"] * Membrane["Sweep_Flow"] / Membrane["Total_Flow"]

        boundary = np.concatenate((U_x_0, U_y_0))

        params = (Membrane, Component_properties, Fibre_Dimensions)
    
        t_span = [Fibre_Dimensions['Length'], 0]
        t_eval = np.linspace(t_span[0], t_span[1], 250)

        #solution = solve_ivp(membrane_odes, t_span, y0 = boundary, args=(params,), method='BDF', t_eval=t_eval)
        solution = solve_ivp(lambda z, var: membrane_odes(z, var, params), t_span, y0 = boundary, method='BDF', t_eval=t_eval)

        '''
        z_points = solution.t
        U_x_profile = solution.y[:J, :]
        U_y_profile = solution.y[J:2*J, :]

        # Composition Profiles
        x_profiles = U_x_profile / np.sum(U_x_profile, axis=0)
        y_profiles = U_y_profile / (np.sum(U_y_profile, axis=0) + epsilon)
    
        # Final Results
        final_Fx = solution.y[:J, -1]
        final_Fy = solution.y[J:2*J, -1]
        '''

        return solution


    '''----------------------------------------------------------###
    ###---------- Shooting Method for Overall Solution ----------###
    ###----------------------------------------------------------'''

    def shooting_method():

        J = len(Membrane["Feed_Composition"])
        
        approx_solution = approx_shooting_guess()
        approx_guess = approx_solution.x[0:J].tolist() + [approx_solution.x[-2]] #guess for the retentate composition and normalised flowrate at element 1
        approx_guess = [comp * approx_guess[-1] for comp in approx_guess[0:J]]


        def module_shooting_error(vars):


            guess_solution = mass_balance_reverse(vars)
            guess_Fx_N = guess_solution.y[:J,-1]

            true_Fx_N = [comp * Membrane["Feed_Flow"]/Membrane["Total_Flow"] for comp in Membrane["Feed_Composition"]]

            error_Fy = [abs(guess_Fx_N[i]-true_Fx_N[i]) for i in range(J)]
 
            shooting_error = error_Fy

            return shooting_error

        #least_squares function to solve the overall mass balance
        overall_sol = least_squares(
            module_shooting_error,
            approx_guess,
            method='dogbox',
            bounds=(0,1),
            xtol=1e-8,
            ftol=1e-8,
        )
    
        if overall_sol.cost > 1e-5: 
            print(f'Large mass balance closure error for shooting solution: {overall_sol.cost:.3e}')

        solution = mass_balance_reverse(overall_sol.x)

        z_points = solution.t
        z_points_norm = z_points / np.max(z_points)    
        U_x_profile = solution.y[:J, :]
        U_y_profile = -solution.y[J:2*J, :]

        # Calculate the compositions and flows
        x_profiles = U_x_profile / np.sum(U_x_profile, axis=0)
        y_profiles = U_y_profile / (np.sum(U_y_profile, axis=0) + epsilon)
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

        x_ret = profile.iloc[0, 1:J+1].values
        y_perm = profile.iloc[-1, J+1:2*J+1].values
        cut_r = profile.iloc[0, 2*J+1]
        cut_p = profile.iloc[-1, 2*J+2]
        Qr = cut_r * Membrane["Total_Flow"]
        Qp = cut_p * Membrane["Total_Flow"]
        CC_ODE_results = x_ret, y_perm, Qr, Qp

        return CC_ODE_results, profile

    return shooting_method()





