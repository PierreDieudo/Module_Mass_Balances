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
    Fibre_Dimensions["Number_Fibre"] = Membrane["Area"] / (Fibre_Dimensions["Length"] * math.pi * Fibre_Dimensions["D_out"])
    
    # CHANGE 1: epsilon from 1e-8 to 1e-10
    epsilon = 1e-10
    
    #Number of elements N
    J = len(Membrane["Feed_Composition"])
    min_elements = [100]
    for i in range(J):
        N_i = (Membrane["Area"] * (1 - Membrane["Feed_Composition"][i] + 0.005) * Membrane["Permeance"][i] * Membrane["Pressure_Feed"] * Membrane["Feed_Composition"][i]) / (Membrane["Feed_Flow"] * 0.005)
        min_elements.append(N_i)
    n_elements = min(round(max(min_elements)), 1000)
     
    Membrane["Feed_Composition"] = np.array(Membrane["Feed_Composition"])
    Membrane["Sweep_Composition"] = np.array(Membrane["Sweep_Composition"])
   
   
    '''----------------------------------------------------------###
    ###------------- Mixture Viscosity Calculation --------------###
    ###----------------------------------------------------------'''

    def mixture_visc(composition):

        y = composition
    
        visc = np.zeros(J)
        for i, (slope, intercept) in enumerate(Component_properties["Viscosity_param"]):
            visc[i] = 1e-6*(slope * Membrane["Temperature"] + intercept)

        Mw = Component_properties["Molar_mass"]

        phi = np.zeros((J, J))
        for i in range(J):
            for j in range(J):
                if i != j:
                    phi[i][j] = ( ( 1 + ( visc[i]/visc[j] )**0.5 * ( Mw[j]/Mw[i] )**0.25 ) **2 ) / ( ( 8 * ( 1 + Mw[i]/Mw[j] ) )**0.5 ) 
                else:
                    phi[i][j] = 1

        nu=np.zeros(J)
        for i in range(J):
            nu[i] = y[i] * visc [i] / sum(y[j] * phi[i][j] for j in range(J))
    
        visc_mix = sum(nu)
        return visc_mix

    '''----------------------------------------------------------###
    ###--------------- Pressure Drop Calculation ----------------###
    ###----------------------------------------------------------'''

    def pressure_drop(composition, Q, P):

        visc_mix = mixture_visc(composition)
        D_in = Fibre_Dimensions["D_in"]
        Q = Q/Fibre_Dimensions['Number_Fibre']
        R = 8.314
        nu = (Q * R * Membrane["Temperature"])/P

        dP_dz = (128 * visc_mix) / (math.pi * D_in**4 * P ) * nu
        return dP_dz


    '''---------------------------------------------------------------###
    ###---------- Non discretised solution for initial guess ----------###
    ###---------------------------------------------------------------'''

    def approx_mass_balance(vars):

        J = len(Membrane["Feed_Composition"])

        x_N = Membrane["Feed_Composition"]
        y_0 = Membrane["Sweep_Composition"]
        cut_r_N = Membrane["Feed_Flow"]/Membrane["Total_Flow"]
        cut_p_0 = Membrane["Sweep_Flow"]/Membrane["Total_Flow"]
    
        Qr_N = Membrane["Feed_Flow"] 
        Qp_0 = Membrane["Sweep_Flow"]

        x_0 = vars [0:J]
        y_N = vars [J:2*J]
        cut_r_0 = vars[-2]
        cut_p_N= vars[-1]

        Qr_0 = Membrane["Total_Flow"] * cut_r_0
        Qp_N = Membrane["Total_Flow"] * cut_p_N

        eqs = [0]*(2*J+2)

        eqs[0] = sum(x_0) - 1
        eqs[1] = sum(y_N) - 1

        for i in range(J):
            eqs[i+2] = ( x_N[i] * cut_r_N - x_0[i] * cut_r_0 + y_0[i] * cut_p_0 - y_N[i] * cut_p_N )

        for i in range (J): 
            pp_diff_in = Membrane["Pressure_Feed"] * x_N[i] - Membrane["Pressure_Permeate"] * y_0[i]
            pp_diff_out = Membrane["Pressure_Feed"] * x_0[i] - Membrane["Pressure_Permeate"] * y_N[i]

            if (pp_diff_in / (pp_diff_out + epsilon) + epsilon) >= 0:
                ln_term = math.log((pp_diff_in) / (pp_diff_out + epsilon) + epsilon)
            else:
                ln_term = epsilon 

            dP = (pp_diff_in - pp_diff_out) / ln_term

            eqs[i+2+J] = 1 - ( Membrane["Area"] * dP * Membrane["Permeance"][i] ) / ( y_N[i] * Qp_N - y_0[i] * Qp_0 +epsilon)

        return eqs

    def approx_shooting_guess():

        J = len(Membrane["Feed_Composition"])
        approx_guess = [1/J]*J * 2 + [0.5] * 2

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

            # CHANGE 1: epsilon from 1e-8 to 1e-10
            epsilon = 1e-10
            
            dx_dz = np.zeros(J)
            dy_dz = np.zeros(J)
            
            P_perm= var[-1]
            
            for i in range(J):
                dx_dz[i] = 1/Membrane["Total_Flow"] * ( - Membrane["Permeance"][i] * (Fibre_Dimensions["D_out"] * math.pi * Fibre_Dimensions["Number_Fibre"]) * (Membrane["Pressure_Feed"] * var[i]/(sum(var[:J])+epsilon) - P_perm * var[J+i]/(sum(var[J:2*J])+epsilon)) )
                dy_dz[i] = - dx_dz[i]
                
            if Membrane["Pressure_Drop"]:     
                composition = var[:J] / sum(var[:J])
                Q = sum(var[:J])
                dP_dz = pressure_drop(composition, Q, var[-1])
            else: 
                dP_dz = 0
            
            return np.concatenate((dx_dz, dy_dz, [dP_dz]))

        U_x_L = vars[:J]
        P_y_L = Membrane["Pressure_Permeate"]/vars[-1] if Membrane["Pressure_Drop"] else Membrane["Pressure_Permeate"]
        U_y_L = -Membrane["Sweep_Composition"] * Membrane["Sweep_Flow"] / Membrane["Total_Flow"]

        boundary = np.concatenate((U_x_L, U_y_L, [P_y_L]))
        params = (Membrane, Component_properties, Fibre_Dimensions)
    
        t_span = [Fibre_Dimensions['Length'], 0]
        t_eval = np.linspace(t_span[0], t_span[1], max(250,n_elements))

        solution = solve_ivp(
            lambda z, var: membrane_odes(z, var, params),
            t_span,
            y0 = boundary,
            method='BDF',
            t_eval=t_eval,
            # CHANGE 4: No analytical Jacobian - let BDF compute numerically
            # rtol: Keep default 1e-3 (tighter breaks convergence at high areas!)
            # atol: Keep default 1e-6
            )

        return solution

    '''----------------------------------------------------------###
    ###---------- Shooting Method for Overall Solution ----------###
    ###----------------------------------------------------------'''

    def shooting_method():

        J = len(Membrane["Feed_Composition"])
    
        approx_solution = approx_shooting_guess()

        approx_guess = approx_solution.x[0:J].tolist() + [approx_solution.x[-2]]
        approx_guess = [comp * approx_guess[-1] for comp in approx_guess[0:J]]

        if Membrane["Pressure_Drop"]:
            approx_guess = approx_guess + [0.99]
        
        # CHANGE 5: Check if approximate solution is reasonable
        # If approx_solution has high cost, we might need continuation
        needs_continuation = approx_solution.cost > 1e-3
        
        if needs_continuation and Membrane["Area"] > 15:
            print(f'High area ({Membrane["Area"]} m^2) with poor initial guess - using continuation method')
            
            # Solve at 70% of target area first
            Area_original = Membrane["Area"]
            Membrane["Area"] = Area_original * 0.7
            
            # Get solution at lower area
            intermediate_sol = shooting_method_core(approx_guess)
            
            # Restore original area
            Membrane["Area"] = Area_original
            
            # Use intermediate solution as guess for full area
            if intermediate_sol.cost < 1e-3:
                approx_guess = intermediate_sol.x.tolist()
                print(f'Continuation from {Area_original*0.7:.1f} m^2 successful, now solving at {Area_original} m^2')
            else:
                print(f'Continuation failed, using original guess')

        return shooting_method_core(approx_guess)
    
    def shooting_method_core(initial_guess):
        """Core shooting method that can be called recursively for continuation"""
        J = len(Membrane["Feed_Composition"])

        def module_shooting_error(vars):

            guess_solution = mass_balance_reverse(vars)

            guess_Fx_N = guess_solution.y[:J,-1]

            true_Fx_N = [comp * Membrane["Feed_Flow"]/Membrane["Total_Flow"] for comp in Membrane["Feed_Composition"]]

            error_Fy = []
            for i in range(J):
                denominator = max(abs(true_Fx_N[i]), epsilon * 10)
                rel_error = (guess_Fx_N[i] - true_Fx_N[i]) / denominator
                error_Fy.append(rel_error)

            shooting_error = error_Fy
            
            total_feed = sum(true_Fx_N)
            total_guess = sum(guess_Fx_N)
            mass_balance_error = (total_guess - total_feed) / (total_feed + epsilon)
            shooting_error.append(mass_balance_error)

            if Membrane["Pressure_Drop"]:
                guess_P_y_0 = guess_solution.y[-1,-1]
                true_P_y_0 = Membrane["Pressure_Permeate"]
                error_P = (guess_P_y_0 - true_P_y_0) / true_P_y_0
                shooting_error.append(error_P)

            return shooting_error

        bounds = (0, 1) 

        overall_sol = least_squares(
            module_shooting_error,
            initial_guess,
            method='dogbox',
            bounds=bounds,
            xtol=1e-10,
            ftol=1e-10, 
            max_nfev=500,
        )
    
        if overall_sol.cost > 1e-5: 
            print(f'Large mass balance closure error: {overall_sol.cost:.3e}')
            print(f'Final residuals: {overall_sol.fun}')
            
            solution_test = mass_balance_reverse(overall_sol.x)
            guess_total_in = sum(solution_test.y[:J, -1])
            true_total_in = Membrane["Feed_Flow"] / Membrane["Total_Flow"]
            print(f'Total mole balance error: {abs(guess_total_in - true_total_in)/true_total_in * 100:.2f}%')

        solution = mass_balance_reverse(overall_sol.x)

        z_points = solution.t
        z_points_norm = z_points / np.max(z_points)    
        U_x_profile = solution.y[:J, :]
        U_y_profile = solution.y[J:2*J, :]

        x_profiles = U_x_profile / np.sum(U_x_profile, axis=0)
        y_profiles = U_y_profile / (np.sum(U_y_profile, axis=0) + epsilon)
        Qr_profile = np.sum(U_x_profile, axis=0)
        Qp_profile = -np.sum(U_y_profile, axis=0)
      
        if Membrane["Pressure_Drop"]:
            P_profile = solution.y[-1, :]
            print(f'Pressure drop across the module: {P_profile[0]-P_profile[-1]:.2f} Pa')
            plt.figure(figsize=(8,6))
            plt.plot(z_points_norm, P_profile)
            plt.xlabel('Normalized Length')
            plt.ylabel('Pressure (Pa)')
            plt.title('Pressure Drop Profile Along the Membrane')
            plt.show()
        
        data = {
            "norm_z": z_points_norm,
            **{f"x{i+1}": x_profiles[i, :] for i in range(J)},
            **{f"y{i+1}": y_profiles[i, :] for i in range(J)},
            "cut_r/Qr": Qr_profile,
            "cut_p/Qp": Qp_profile,
        }

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