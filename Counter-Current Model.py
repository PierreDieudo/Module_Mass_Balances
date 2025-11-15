import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import csv
import os
import math

''' GENERAL INFORMATION:

- Counter current model discretised into N elements, with element N being feed/permeate side and element 1 being sweep/retentate side

- Model based on Coker and Freeman (1998)

- Feed and Sweep flow and compositions are known, solving for retentate and permeate compositions and flows

- Using shooting method to solve: guess RETENTATE and calculate the mass balance from element 1 to N, then adjust the guess to minimize the error between calculated and known feed

- Mass balance in each element will be using a known sweep and the guessed/calculated retentate of an element k, and determine streams connected to element k+1

- Hopefully, the results will be stored in a pands dataframe and exported to a csv file
'''

'''--------------------------------------------------------###
###------------------------ Knowns ------------------------###
###--------------------------------------------------------'''

epsilon = 1e-8 # Small number to avoid division by zero
A = 5e3  # Membrane area in m2
pR = 2 * 1e5
pP_N = 0.2 * 1e5 # Permeate pressure from bar to Pa
Temp = 40  # Temperature in C
Temp = Temp + 273.15  # Convert to K

xN = np.array([0.247758,0.709923,0.030415,0.011904])  # Feed inlet composition (at element N)
y0 = np.array([0, 0, 0, 0])  # Sweep inlet composition (at element 1)
Perm = np.array([3.348e-06, 1.674e-07, 5.022e-08, 3.348e-08]) # 3.348*1e-10  # Permeances from GPU to mol/m2.s.Pa

Feed_flow = 7421/3.6 # Feed flow in mol/s
Sweep_flow = 0
if Sweep_flow == 0: y0 = np.zeros(len(xN))  # If sweep flow is zero, set y0 to zero
Total_flow = Feed_flow + Sweep_flow

if abs(xN.sum() - 1) > 1e-6:    
    raise ValueError("Feed inlet mole fractions must sum to 1")
if abs(y0.sum() - 1) > 1e-6 and Sweep_flow != 0:
    raise ValueError("Sweep inlet mole fractions must sum to 1")
if len(xN) != len(y0) or len(xN) != len(Perm):
    raise ValueError("Data given does not match the number of components")


# For normalisation, molar flows are expressed in term of fractions of the total flow (hence stay between 0 and 1)
cut_r_N = Feed_flow / Total_flow  # at feed side (element N)
cut_p_0 = Sweep_flow / Total_flow   # at sweep side (element 1)

#Number of components J and elements N
J = len(xN)
min_elements = [2]  # minimum of 2 elements

for i in range(J):  # (Coker and Freeman, 1998)
    N_i = (A * (1 - xN[i] + 0.005) * Perm[i] * pR * xN[i]) / (Feed_flow * 0.005)
    min_elements.append(N_i)
n_elements = min(round(max(min_elements)), 1000)

DA = A / n_elements  # discretised area

#user vars to be used in functions
user_vars = DA, J, Total_flow, pR, pP_N, Perm, epsilon

#known sweep for comparison
sweep = np.append(y0,cut_p_0)

'''----------------------------------------------------------###
###-------- Mass Balance Function Across One Element --------###
###----------------------------------------------------------'''

def mass_balance(vars, inputs, user_vars):

    DA, J, Total_flow, pR, pP, Perm, epsilon = user_vars

    # known composition and flowrates connected to element k-1
    x_known = inputs[0:J]
    y_known = inputs[J:2*J]
    cut_r_known = inputs[-2]
    cut_p_known = inputs[-1] 
   
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
    
    J = user_vars[1]
    
    # Guessed composition and flowrates for element N permeate
    x_guess = vars[:J]
    cut_r_guess = vars[J]

    '''
    - Want to create a DataFrame matrix for N elements with 2J+2 columns: x, y, cut_r, cut_p
    - Each row will represent an element k, with the first row being element N and the last being element 1
    - Each element mass balance result will be stored in row k and is using row k-1 as input
    - The first row (element 1) will have known values for sweep and the guess retentate.
    
    NB: the notation of rows is not reflecting the in/out flow direction, but the direction of the shooting method
    '''

    # Create a DataFrame to store the results
    columns = ['Element'] + [f'x{i+1}' for i in range(J)] + [f'y{i+1}' for i in range(J)] + ['cut_r/Qr', 'cut_p/Qp']

    # Preallocate for n_elements
    df = pd.DataFrame(index=range(n_elements), columns=columns)

    # Set the element 1 with Sweep known values and guessed retentate value
    df.loc[0] = [1] + list(x_guess) + list(y0) + [cut_r_guess, cut_p_0]  # element 1 (retentate/sleep side)


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
            xtol = 1e-6,
            ftol = 1e-6,
            gtol = 1e-6
        )

        element_output = sol_element.x
        
        if sol_element.cost > 1e-5:
            print(f'Large mass balance closure error at element {k}; error: {sol_element.cost:.3e}; with residuals {sol_element.fun}')

        # Update the DataFrame with the results
        df_element = np.concatenate(([k+2], element_output))
        df.loc[k + 1] = df_element


    # Calculate the error between the known sweep and the calculated sweep

    error_x = [abs((df.iloc[-1, df.columns.get_loc(f'x{i+1}')] - xN[i])) for i in range(J)]
    error_flow = abs((df.loc[df.index[-1], 'cut_r/Qr'] - cut_r_N))  # difference in flowrate

    shooting_error = error_x + [error_flow]

    return shooting_error, df


'''---------------------------------------------------------------###
###---------- Non discretised solution for intial guess ----------###
###---------------------------------------------------------------'''

'''
- Approximate solution for overall mass balance. Will be used as an input for the shooting method.
- Using the log mean partial pressure difference as driving force. Heavier and less stable than dicretisation, but provides a good estimate.
'''

def approx_mass_balance(vars, inputs, user_vars):

    DA, J, Total_flow, pR, pP, Perm, epsilon = user_vars

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
        pp_diff_in = pR * x_N[i] - pP * y_0[i]
        pp_diff_out = pR * x_0[i] - pP * y_N[i]

        if (pp_diff_in / (pp_diff_out + epsilon) + epsilon) >= 0: #using the log mean partial pressure difference as it is a better approximation when the membrane is not discretise.
            ln_term = math.log((pp_diff_in) / (pp_diff_out + epsilon) + epsilon)  #It is however less stable, hence these expressions to make sure there is no division by zero.
        else:
            ln_term = epsilon 

        dP = (pp_diff_in - pp_diff_out) / ln_term # driving force

        eqs[i+2+J] = 1 - ( DA * n_elements * dP * Perm[i] ) / ( y_N[i] * Qp_N - y_0[i] * Qp_0 )# difference in permeate flowrate in/out = permeation across the area

    return eqs


#solving the approximate mass balance for the module
approx_guess = [1/J]*J * 2+ [0.5] * 2
inputs = np.concatenate((xN, y0, np.array([cut_r_N]), np.array([cut_p_0]))) # Convert cut_p_0 to a 1D array

approx_sol = least_squares(
    approx_mass_balance,
    approx_guess,
    args=(inputs, user_vars),
    bounds=(0,1),
    xtol=1e-6,
    ftol=1e-6   
    )

print (f'aprroximate solution used for intial guess: {approx_sol.x}') #guess for the retentate composition and flowrate at element 1
print (f'with mass balance error of {approx_sol.cost:.3e}') #mass balance error for the approximate solution
print (f'and residuals {[f"{v:.3e}" for v in approx_sol.fun]}')  
print()

'''----------------------------------------------------------###
###---------- Shooting Method for Overall Solution ----------###
###----------------------------------------------------------'''

# Initial guess of retentate composition and flowrate at element 1 (feed in the module)
reten_guess = approx_sol.x[0:J].tolist() + [approx_sol.x[-2]]  # guess for the retentate composition and flowrate at element 1

def module_mass_balance_error(vars, user_vars):

    vars[0:J] = vars[0:J] / sum(vars[0:J])  # Normalize the first J elements to 1

    shooting_error, _ = module_mass_balance(vars, user_vars)
    return shooting_error

#least_squares function to solve the overall mass balance
overall_sol = least_squares(
    module_mass_balance_error,
    reten_guess,
    args=(user_vars,),
    method='lm',
    xtol=1e-8,
    ftol=1e-8
)

shooting_error, Solved_membrane_profile = module_mass_balance(overall_sol.x , user_vars) #Running the membrane mass balance with the solution of the shooting method

shooting_error = np.array(shooting_error)
print()
print(f'shooting method error {overall_sol.cost:.3e} with residuals {overall_sol.fun}')
print(f'Solution found for retentate composition of {overall_sol.x[0:J]} and flowrate fraction of {overall_sol.x[-1]:.3f}')
''' Solved_membrane_profile is a DataFrame (matrix) with N rows and 2J+2 columns, listing x, y, cut_r, and cut_p for each element'''

# Transforming the last two columns of the DataFrame into flowrates:
Solved_membrane_profile['Qr'] = Solved_membrane_profile['cut_r/Qr'] * Total_flow
Solved_membrane_profile['Qp'] = Solved_membrane_profile['cut_p/Qp'] * Total_flow

# Formatting the DataFrame to be exported to a csv file
Solved_membrane_profile['Element'] = Solved_membrane_profile['Element'].astype(int)
for i in range(J):
    Solved_membrane_profile[f'x{i+1}'] = Solved_membrane_profile[f'x{i+1}'].map('{:.4f}'.format)
    Solved_membrane_profile[f'y{i+1}'] = Solved_membrane_profile[f'y{i+1}'].map('{:.4f}'.format)
Solved_membrane_profile['Qr'] = Solved_membrane_profile['Qr'].map('{:.4e}'.format)
Solved_membrane_profile['Qp'] = Solved_membrane_profile['Qp'].map('{:.4e}'.format)

print(Solved_membrane_profile)


# Export to CSV
Solved_membrane_profile.to_csv(r'C:\Users\s1854031\Desktop\ret_guess_membrane_profile.csv', index = None)


print(f'Profile saved to ret_guess_membrane_profile.csv')