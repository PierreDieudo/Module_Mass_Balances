

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar
import pandas as pd
import math

''' GENERAL INFORMATION:

- Counter current model discretised into N elements, with element N being feed/permeate side and element 1 being sweep/retentate side

- Model based on Coker and Freeman (1998)

- Feed and Sweep flow and compositions are known, solving for retentate and permeate compositions and flows

- Using shooting method to solve: guess RETENTATE and calculate the mass balance from element 1 to N, then adjust the guess to minimize the error between calculated and known feed

- Mass balance in each element will be using a known sweep and the guessed/calculated retentate of an element k, and determine streams connected to element k+1

- The results and profile are stored in a pandas dataframe.

- Pressure drop can be considered. In the case the shooting method is also solving for the permeate pressure.
'''


def mass_balance_CC(vars):
    
    Membrane, Component_properties, Fibre_Dimensions = vars

    Total_flow = Membrane["Feed_Flow"] + Membrane["Sweep_Flow"] # Total flow in mol/h
    cut_r_N = Membrane["Feed_Flow"] / Total_flow # Cut ratio at the feed side
    cut_p_0 = Membrane["Sweep_Flow"] / Total_flow # Cut ratio at the permeate side

    #Number of elements N
    J = len(Membrane["Feed_Composition"])
    min_elements = [1]  # minimum of 2 elements
    for i in range(J):  # (Coker and Freeman, 1998)
        N_i = (Membrane["Feed_Flow"] * (1 - Membrane["Feed_Composition"][i] + 0.005) * Membrane["Permeance"][i] * Membrane["Pressure_Feed"] * Membrane["Feed_Composition"][i]) / (Membrane["Feed_Flow"] * 0.005)
        min_elements.append(N_i)
    n_elements = min(round(max(min_elements)), 1000)

    DA = Membrane["Area"] / n_elements # Area of each element

    epsilon = 1e-8

    user_vars = DA, J, Total_flow, Membrane["Pressure_Feed"], Membrane["Permeance"], epsilon
    
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

    def mass_balance(vars, inputs, user_vars):

        DA, J, Total_flow, pR, Perm, epsilon = user_vars

        # known composition and flowrates connected to element k-1
        x_known = inputs[0:J]
        y_known = inputs[J:2*J]
        cut_r_known = inputs[-3]
        cut_p_known = inputs[-2] 
        pP = inputs[-1] # Permeate pressure from element k-1
   
        Qr_known = Total_flow * cut_r_known # retentate flowrate entering element k
        Qp_known = Total_flow * cut_p_known

        x = vars [0:J] # retentate side mole fractions leaving element k - to be exported to next element k+1
        y = vars [J:2*J] # permeate side mole fractions entering element k - to be exported to next element k+1
        cut_r = vars[-2] # retentate flowrate leaving element k - to be exported to next element k+1
        cut_p = vars[-1] # permeate flowrate leaving element k - to be exported to next element k+1

        #print(f'Initial guess {vars}')

        Qr = Total_flow * cut_r # retentate flowrate exiting element k
        Qp = Total_flow * cut_p

        eqs = [0]*(2*J+2) # empty list to store the equations


        #molar fractions summing to unity:
        eqs[0] = sum(x) - 1
        eqs[1] = sum(y) - 1

        #mass balance for each component across the module
        for i in range(J):
            eqs[i+2] = ( cut_p_known * y_known[i] + cut_r * x[i] - cut_p * y[i] - cut_r_known * x_known[i] ) #in perm + in ret - out perm - out ret

        #flow across membrane --> change in permeate flowrate is equal to the permeation across DA
        for i in range (J):

            eqs[i+2+J] = ( ( y[i] * Qp - y_known[i] * Qp_known ) - Perm[i] * DA * (pR * x_known[i] - pP * y[i]))

        return eqs


    '''-----------------------------------------------------------###
    ###--------- Mass Balance Function Across The Module ---------###
    ###-----------------------------------------------------------'''

    def module_mass_balance(vars, user_vars):

        # Guessed composition and flowrates for element N permeate
        x_guess = vars[:J]
        cut_r_guess = vars[J]
        pP_0 = Membrane["Pressure_Permeate"] / vars[-1]  # guess for the sweep pressure


        # Create a DataFrame to store the results
        columns = ['Element'] + [f'x{i+1}' for i in range(J)] + [f'y{i+1}' for i in range(J)] + ['cut_r/Qr', 'cut_p/Qp', 'P_perm']

        # Preallocate for n_elements
        df = pd.DataFrame(index=range(n_elements), columns=columns)

        # Set the element 1 with Sweep known values and guessed retentate value
        df.loc[0] = [1] + list(x_guess) + list(Membrane["Sweep_Composition"]) + [cut_r_guess, cut_p_0, pP_0]  # element 1 (retentate/sleep side)


        for k in range(n_elements - 1):
            # Input vector of known/calculated values from element k-1

            inputs = df.loc[k, df.columns[1:]].values

            # Initial guess for the element k
            guess = [0.5] * (2 * J + 2)
        
            sol_element = least_squares(
                mass_balance,  # function to solve
                guess,  # initial guess
                args=(inputs, user_vars),  # arguments for the function
                #bounds= (0, 1),
                method='lm',
                xtol = 1e-8,
                ftol = 1e-8,
                gtol = 1e-8
            )

            element_output = sol_element.x
        
            if sol_element.cost > 1e-5:
                print(f'Large mass balance closure error at element {k}; error: {sol_element.cost:.3e}; with residuals {sol_element.fun}')
        
            # Calculate the pressure drop for the permeate side
            y_k = element_output[J:2*J]  # Permeate composition
            Qp_k = element_output[-1] * Total_flow  # Permeate flowrate
            pP_k = df.loc[k, 'P_perm']  # Current permeate pressure


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
            df_element = np.concatenate(([k+2], element_output, [pP_new]))
            df.loc[k + 1] = df_element


        # Calculate the error between the known sweep and the calculated sweep

        error_x = [abs((df.iloc[-1, df.columns.get_loc(f'x{i+1}')] - Membrane["Feed_Composition"][i])) for i in range(J)]
        error_flow = abs((df.loc[df.index[-1], 'cut_r/Qr'] - cut_r_N))  # difference in flowrate
        if Membrane["Pressure_Drop"]:
            error_pressure = abs((df.loc[df.index[-1], 'P_perm'] - Membrane["Pressure_Permeate"]))
        else :
            error_pressure = 0

        shooting_error = error_x + [error_flow] + [error_pressure]

        return shooting_error, df


    '''---------------------------------------------------------------###
    ###---------- Non discretised solution for initial guess ----------###
    ###---------------------------------------------------------------'''

    '''
    - Approximate solution for overall mass balance. Will be used as an input for the shooting method.
    - Using the log mean partial pressure difference as driving force. Heavier and less stable than dicretisation, but provides a good estimate.
    '''

    def approx_mass_balance(vars, inputs, user_vars):

        DA, J, Total_flow, pR , Perm, epsilon = user_vars

        # known composition and flowrates entering the module
        x_N = inputs[0:J]
        y_0 = inputs[J:2*J]
        cut_r_N = inputs[-2]
        cut_p_0 = inputs[-1] 
    
        Qr_N = Total_flow * cut_r_N # retentate flowrate entering element k
        Qp_0 = Total_flow * cut_p_0

        x_0 = vars [0:J] # retentate mole fractions
        y_N = vars [J:2*J] # permeate mole fractions
        cut_r_0 = vars[-2] # retentate flowrate fraction
        cut_p_N= vars[-1] # permeate flowrate fraction

        Qr_0 = Total_flow * cut_r_0 # retentate flowrate
        Qp_N = Total_flow * cut_p_N

        eqs = [0]*(2*J+2) # empty list to store the equations


        #molar fractions summing to unity:
        eqs[0] = sum(x_0) - 1
        eqs[1] = sum(y_N) - 1

        #mass balance for each component across the module
        for i in range(J):
            eqs[i+2] = ( x_N[i] * cut_r_N - x_0[i] * cut_r_0 + y_0[i] * cut_p_0 - y_N[i] * cut_p_N ) #in ret - out ret + in perm - out perm

        #flow across membrane --> change in permeate flowrate is equal to the permeation across the area
        for i in range (J): 
            pp_diff_in = pR * x_N[i] - Membrane["Pressure_Permeate"] * y_0[i]
            pp_diff_out = pR * x_0[i] - Membrane["Pressure_Permeate"] * y_N[i]

            if (pp_diff_in / (pp_diff_out + epsilon) + epsilon) >= 0: #using the log mean partial pressure difference as it is a better approximation when the membrane is not discretise.
                ln_term = math.log((pp_diff_in) / (pp_diff_out + epsilon) + epsilon)  #It is however less stable, hence these expressions to make sure there is no division by zero.
            else:
                ln_term = epsilon 

            dP = (pp_diff_in - pp_diff_out) / ln_term # driving force

            eqs[i+2+J] = 1 - ( DA * n_elements * dP * Perm[i] ) / ( y_N[i] * Qp_N - y_0[i] * Qp_0 +epsilon)# difference in permeate flowrate in/out = permeation across the area

        return eqs


    #solving the approximate mass balance for the module
    approx_guess = [1/J]*J * 2+ [0.5] * 2
    inputs = np.concatenate((Membrane["Feed_Composition"], Membrane["Sweep_Composition"], np.array([cut_r_N]), np.array([cut_p_0]))) # Convert cut_p_0 to a 1D array

    approx_sol = least_squares(
        approx_mass_balance,
        approx_guess,
        args=(inputs, user_vars),
        method='dogbox',
        bounds=(0,1),
        xtol=1e-8,
        ftol=1e-8   
        )

    #print (f'aprroximate solution used for initial guess: {approx_sol.x}') #guess for the retentate composition and flowrate at element 1
    #print (f'with mass balance error of {approx_sol.cost:.3e}') #mass balance error for the approximate solution
    #print (f'and residuals {[f"{v:.3e}" for v in approx_sol.fun]}')  
    #print()

    '''----------------------------------------------------------###
    ###---------- Shooting Method for Overall Solution ----------###
    ###----------------------------------------------------------'''

    # Initial guess of retentate composition and flowrate at element 1 (feed in the module)
    reten_guess = approx_sol.x[0:J].tolist() + [approx_sol.x[-2]]  # guess for the retentate composition and flowrate at element 1
    pressure_guess = 0.99 #guess for sweep pressure, knowing that P_sweep = pP/guess
    if Membrane["Pressure_Drop"]:
        shooting_guess = reten_guess + [pressure_guess]  # guess for the retentate composition and flowrate at element 1
    else:
        shooting_guess = reten_guess

    def module_mass_balance_error(vars, user_vars):

        vars[0:J] = vars[0:J] / sum(vars[0:J])  # Normalise the first J elements to 1

        shooting_error, _ = module_mass_balance(vars, user_vars)
        return shooting_error

    #least_squares function to solve the overall mass balance
    overall_sol = least_squares(
        module_mass_balance_error,
        shooting_guess,
        args=(user_vars,),
        #bounds=(0,1),
        method='trf',
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8
    )

    shooting_error, Solved_membrane_profile = module_mass_balance(overall_sol.x , user_vars) #Running the membrane mass balance with the solution of the shooting method
    
    if overall_sol.cost > 1e-5: 
        print(f'Large mass balance closure error for overall solution: {overall_sol.cost:.3e}; with residuals {[f"{v:.3e}" for v in shooting_error]}')

    # Display the mass balance closure error of the penultimate element - as an indicator
    penultimate_element = n_elements - 1
    penultimate_inputs = Solved_membrane_profile.loc[penultimate_element, Solved_membrane_profile.columns[1:]].values
    penultimate_guess = [0.5] * (2 * J + 2)
    penultimate_sol = least_squares(
        mass_balance,
        penultimate_guess,
        args=(penultimate_inputs, user_vars),
        method='lm',
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8
    )

    #print(f'Mass balance closure error of penultimate element for CC model: {penultimate_sol.cost:.3e}')
    #print(f'shooting method error: {overall_sol.cost:.3e}')
    #print (Solved_membrane_profile)

    ''' Solved_membrane_profile is a DataFrame (matrix) with N rows and 2J+3 columns, listing x, y, cut_r, cut_p, and permeate pressure for each element'''
    
    x_ret = Solved_membrane_profile.iloc[0, 1:J+1].values
    y_perm = Solved_membrane_profile.iloc[-1, J+1:2*J+1].values
    cut_r = Solved_membrane_profile.iloc[0, -3]
    cut_p = Solved_membrane_profile.iloc[-1, -2]
    Qr = cut_r * Total_flow
    Qp = cut_p * Total_flow
    P_perm = Solved_membrane_profile.iloc[-1, -1] * 1e-5  # Permeate pressure in bar at the last element
    CC_results = x_ret, y_perm, Qr, Qp

    profile = Solved_membrane_profile.copy()


    return CC_results, profile



