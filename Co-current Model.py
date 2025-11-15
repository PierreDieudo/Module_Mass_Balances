
import numpy as np
from scipy.optimize import least_squares
import pandas as pd

''' GENERAL INFORMATION:

- Co current model discretised into N elements, with element N being feed/sweep side and element 1 being permeate/retentate side

- Model based on Coker and Freeman (1998)

- Feed and Sweep flow and compositions are known, solving for retentate and permeate compositions and flows

- Solving element by element from N to 1

- Hopefully, the results will be stored in a pandas dataframe
'''


'''--------------------------------------------------------###
###------------------------ Knowns ------------------------###
###--------------------------------------------------------'''

epsilon = 1e-8 # Small number to avoid division by zero
A = 3.0*1e5  # Membrane area in m2
P_ret = 2 * 1e5
P_perm = 0.2 * 1e5 # Permeate pressure from bar to Pa
Temp = 25  # Temperature in C
Temp = Temp + 273.15  # Convert to K

x_feed = np.array([0.15, 0.76, 0.06, 0.03])  # Feed inlet composition (at element N)
y_sweep = np.array([0.05, 0.05, 0.10, 0.8])  # Sweep inlet composition (at element 1)
Perm = np.array([5000,100,5000/15,5000]) * 3.348*1e-10  # Permeances from GPU to mol/m2.s.Pa

if abs(x_feed.sum() - 1) > 1e-6:    
    raise ValueError("Feed inlet mole fractions must sum to 1")
if abs(y_sweep.sum() - 1) > 1e-6:
    raise ValueError("Sweep inlet mole fractions must sum to 1")
if len(x_feed) != len(y_sweep) or len(x_feed) != len(Perm):
    raise ValueError("Data given does not match the number of components")

Q_feed = 2.0*1e4  # Feed flow in mol/s
Q_sweep = 1e3
if Q_sweep == 0: y_sweep = np.zeros(len(y_sweep))  # If no sweep flow, set y_sweep to zero
Total_flow=Q_feed+Q_sweep

vars = Temp, P_ret, P_perm, A, Perm, Q_sweep, y_sweep, x_feed, Q_feed

def mass_balance_CO(vars):

    print('CO module started')


    Temp, P_ret, P_perm, A, Perm, Q_sweep, y_sweep, x_feed, Q_feed = vars

    epsilon = 1e-8 # Small number to avoid division by zero

    Total_flow = Q_feed + Q_sweep # Total flow in mol/h
    cut_r_N = Q_feed / Total_flow # Cut ratio at the feed side
    cut_p_N = Q_sweep / Total_flow # Cut ratio at the permeate side

    #Number of elements N
    J = len(x_feed)
    min_elements = [2]  # minimum of 2 elements
    for i in range(J):  # (Coker and Freeman, 1998)
        N_i = (A * (1 - x_feed[i] + 0.005) * Perm[i] * P_ret * x_feed[i]) / (Q_feed * 0.005)
        min_elements.append(N_i)
    n_elements = min(round(max(min_elements)), 1000)


    DA = A / n_elements # Area of each element
    print (DA)
    user_vars = DA, J, Total_flow, P_ret, P_perm, Perm, epsilon


    '''----------------------------------------------------------###
    ###-------- Mass Balance Function Across One Element --------###
    ###----------------------------------------------------------'''

    def equations(vars, inputs, user_vars):

        DA, J, Total_flow, P_ret, P_perm, Perm, epsilon = user_vars

        # known composition and flowrates connected to element k+1
        x_known = inputs[0:J]
        y_known = inputs[J:2*J]
        cut_r_known = inputs[-2]
        cut_p_known = inputs[-1]
        
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
    columns = ['Element'] + [f'x{i+1}' for i in range(J)] + [f'y{i+1}' for i in range(J)] + ['cut_r/Qr', 'cut_p/Qp']

    # Preallocate for n_elements
    df = pd.DataFrame(index=range(n_elements), columns=columns)

    # Set the element N with feed known values and guessed permeate value
    df.loc[0] = [1] + list(x_feed) + list(y_sweep) + [cut_r_N, cut_p_N]  # element N (Feed/Sweep side)
    # Set the element N with feed known values and guessed permeate value

    for k in range(n_elements - 1):
        # Input vector of known/calculated values from element k+1

        inputs = df.loc[k, df.columns[1:]].values
        # Initial guess for the element k
        guess = [0.5] * (2 * J + 2)
        
        sol_element = least_squares(
            equations,  # function to solve
            guess,  # initial guess
            args=(inputs, user_vars),  # arguments for the function
            bounds= (0, 1),
            method='dogbox',
            xtol = 1e-8,
            ftol = 1e-8,
            gtol = 1e-8
        )

        element_output = sol_element.x
        #print(element_output)
        #print(f'CO2 permeate of first element: {element_output[J] * element_output[-1] * Total_flow:.3f}')
        #print(f'permeation through first DA: {DA * Perm[0] * (element_output[0]*P_ret-element_output[J]*P_perm)}')
        #print(f'error in corresponding equation: {sol_element.fun[J+2]:.3e}')

        # Update the DataFrame with the results
        df_element = np.concatenate(([k+2], element_output))
        df.loc[k + 1] = df_element
 
        print(f'error in element {k+2}: {sol_element.cost:.3e}')
        if sol_element.cost > 1e-5:
            print(f'high error in element {k+2} with error {sol_element.cost:.3e} residuals {sol_element.fun}')
    
    print(f'mass balance closure error: {sol_element.cost:.3e}')
    
    x_ret = df.iloc[-1, 1:J+1].values
    y_perm = df.iloc[-1, J+1:2*J+1].values
    cut_r = df.iloc[-1, -2]
    cut_p = df.iloc[-1, -1]
    Qr = cut_r * Total_flow * 3.6  # convert back to kmol/h
    Qp = cut_p * Total_flow * 3.6  # convert back to kmol/h

    CO_results = x_ret, y_perm, Qr, Qp
    return(CO_results, df)


CO_results , Solved_membrane_profile = mass_balance_CO(vars)

# Transforming the last two columns of the DataFrame into flowrates:
Solved_membrane_profile['Qr'] = Solved_membrane_profile['cut_r/Qr'] * Total_flow
Solved_membrane_profile['Qp'] = Solved_membrane_profile['cut_p/Qp'] * Total_flow

# Formatting the DataFrame to be exported to a csv file
for i in range(len(Perm)):
    Solved_membrane_profile[f'x{i+1}'] = Solved_membrane_profile[f'x{i+1}'].map('{:.4f}'.format)
    Solved_membrane_profile[f'y{i+1}'] = Solved_membrane_profile[f'y{i+1}'].map('{:.4f}'.format)
Solved_membrane_profile['Qr'] = Solved_membrane_profile['Qr'].map('{:.4e}'.format)
Solved_membrane_profile['Qp'] = Solved_membrane_profile['Qp'].map('{:.4e}'.format)

print(Solved_membrane_profile.head())



# Export to CSV
Solved_membrane_profile.to_csv(r'C:\Users\s1854031\Desktop\cocurrent_membrane_profile.csv', index = None)


print(f'profile saved as cocurrent_membrane_profile.csv')
