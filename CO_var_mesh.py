import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar
import pandas as pd
import math
import matplotlib.pyplot as plt


''' GENERAL INFORMATION:

- Co current model discretised into N elements, with element N being feed/sweep side and element 1 being permeate/retentate side

- Model based on Coker and Freeman (1998)

- The discretised mesh was made variable to allow for maximum mass balance accuracy while ensuring solver convergence.

- The simulation starts with N discretised volumes (>100 for accuracy) and checks the mass balance closure error every time an element gets solved.
  If the error is above a certain threshold (~1e-5), the current element is merged with the next one, repeting until convergence. The following element will start at the default size, repeating this logic until final convergence.

'''
print("Starting varmesh method")

# Example usage (ensure to replace these with actual values)
Membrane = {
    "Feed_Flow": 1,
    "Sweep_Flow": 0,
    "Feed_Composition": np.array([0.2,0.6,0.15,0.05]),
    "Sweep_Composition": np.array([0, 0,0,0 ]),
    "Permeance": np.array([40.024,1.111,0.305,0.06]),
    "Pressure_Feed": 30,
    "Pressure_Permeate": 1.0130,
    "Area": 1000,
    "Pressure_Drop": False,
}

Membrane["Permeance"] = [p * 3.348 * 1e-10 for p in Membrane["Permeance"]]  # Convert from GPU to mol/m2.s.Pa
Membrane["Pressure_Feed"] *= 1e5  # Convert to Pa
Membrane["Pressure_Permeate"] *= 1e5
Membrane["Total_Flow"] = Membrane["Feed_Flow"] + Membrane["Sweep_Flow"]

Component_properties = {
    "Viscosity_param": ([0.0479, 0.6112], [0.0466, 3.8874]), 
    "Molar_mass": [44.009, 28.0134],
}

Fibre_Dimensions = {
    "D_in": 150 * 1e-6,  # Inner diameter in m
    "D_out": 300 * 1e-6,  # Outer diameter in m
    "Length": 1, # Fibre length in m
}
    
vars = Membrane, Component_properties, Fibre_Dimensions
J = len(Membrane["Feed_Composition"])

def mass_balance_CO_varmesh(vars):

    Membrane, Component_properties, Fibre_Dimensions = vars

    Total_flow = Membrane["Total_Flow"] # Total flow in mol/h
    cut_r_N = Membrane["Feed_Flow"] / Membrane["Total_Flow"] # Cut ratio at the feed side
    cut_p_N = Membrane["Sweep_Flow"] / Membrane["Total_Flow"] # Cut ratio at the permeate side

    J = len(Membrane["Feed_Composition"])

    #Maximum number of elements N
    n_elements = 250 #enough to be representative    

    DA = Membrane["Area"] / n_elements # Length of an element

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

        Placeholder_profile, index = inputs #data frame and element number
        
        #print(index)

        if len(inputs) > 1 and isinstance(inputs[1], int):
        
            # Subset the DataFrame up to the element before k
            Subset = Placeholder_profile.loc[:k]

            # Find indices where 'Solved' is True
            Solved_indices = Subset[Subset['Solved']==True].index

            # Get the closest True index smaller than k
            Closest_true = Solved_indices.max() if not Solved_indices.empty else None

            # Calculate n as described
            if Closest_true is not None:
                n = index - Closest_true

        elif len(inputs) > 1 and isinstance(inputs[1], tuple):
            start_k, end_k = inputs[1]
            n = end_k - start_k + 1 # number of merged elements
            index = start_k # so the boundary condition is taken from the element before the merged ones

        # Define Area based on DA and the number of merged elements
        A = DA * n

        # known composition and flowrates connected to previous element k+1 (i.e. index k-1 in DataFrame)
        '''columns = ['Element'] + [f'x{i+1}' for i in range(J)] + [f'y{i+1}' for i in range(J)] + ['cut_r/Qr', 'cut_p/Qp','P_Perm','Error','Solved'] '''
        x_known = Placeholder_profile.loc[index-1].iloc[1:J+1].to_numpy()
        y_known = Placeholder_profile.loc[index-1].iloc[J+1:2*J+1].to_numpy()
        cut_r_known = Placeholder_profile.loc[index-1].iloc[2*J+1]
        cut_p_known = Placeholder_profile.loc[index-1].iloc[2*J+2]
        P_perm = Placeholder_profile.loc[index-1].iloc[-3]
        #print(f'Previous index location (i.e., boundary conditions): {Placeholder_profile.loc[index-1]}')

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
            eqs[i+2+J] = (y[i] * Qp -  y_known[i] * Qp_known) - ( Perm[i] * A * (P_ret * x[i] - P_perm * y[i]) )

        return eqs

    '''-----------------------------------------------------------###
    ###--------- Mass Balance Function Across The Module ---------###
    ###-----------------------------------------------------------'''

    # Create a DataFrame to store the results
    columns = ['Element'] + [f'x{i+1}' for i in range(J)] + [f'y{i+1}' for i in range(J)] + ['cut_r/Qr', 'cut_p/Qp','P_Perm','Error','Solved']

    # Preallocate for n_elements
    Placeholder_profile = pd.DataFrame(index=range(n_elements), columns=columns)

    # Set the element N with feed known values and guessed permeate value
    Placeholder_profile.loc[0] = [n_elements] + list(Membrane["Feed_Composition"]) + list(Membrane["Sweep_Composition"]) + [cut_r_N, cut_p_N, Membrane["Pressure_Permeate"],0, 1 ]  # element N (Feed/Sweep side)
    #print (Placeholder_profile)
    for k in range(n_elements - 1):
        # Input vector of known/calculated values from element k+1

        inputs = Placeholder_profile, k+1

        # Initial guess for the element k
        guess = [0.5] * (2 * J + 2)
        
        sol_element = least_squares(
            equations,  # function to solve
            guess,  # initial guess
            args=(inputs, user_vars),  # arguments for the function
            method='dogbox',
            xtol = 1e-8,
            ftol = 1e-8,
            gtol = 1e-8,
        )
        
        element_output = sol_element.x
        
        if sol_element.cost > 1e-5 or np.any(element_output<-1e-5) or not sol_element.success:
            print(f'Large mass balance closure error at element {k}, merging with next element')

            if not k+1 == n_elements:
                element_output = [0]*(2*J+3) #dummy output to allow merging of elements
                df_element = np.concatenate((
                    np.array([n_elements - 1 - k]),   # Make scalar a 1D array
                    element_output,                  # Already 1D
                    np.array([sol_element.cost]),    # Make scalar a 1D array
                    np.array([False])                 # Make Boolean a 1D array
                ))


            if k+2 == n_elements:
                print(f"Cannot merge element {k} with next element as it is the last element.")
                print("Going back to penultimate solved for element and solving until the end")
                
                solved_indices = Placeholder_profile[Placeholder_profile['Solved'] == True].index
       
                if not solved_indices.empty:
                    Last_solved = solved_indices.max()
        
                    if Last_solved != 0:
                        Penultimate_solved = solved_indices[solved_indices < Last_solved].max()
                    else:
                        Penultimate_solved = 0

                    #need to solved merged element from k=penultimate_solved all the way to k=n_elements-1
                    inputs = Placeholder_profile, (Penultimate_solved+1, n_elements-1)
                    guess = [0.5] * (2 * J + 2)
                    sol_element = least_squares(
                        equations,  # function to solve
                        guess,  # initial guess
                        args=(inputs, user_vars),  # arguments for the function
                        method='dogbox',
                        bounds=(0,1),
                        xtol = 1e-8,
                        ftol = 1e-8,
                        gtol = 1e-8,) 
                    if not sol_element.success:
                        print(Placeholder_profile.loc[Penultimate_solved:])
                        raise ValueError(f"Mass balance solver failed at merged element from {Penultimate_solved+1} to {n_elements-1}: {sol_element.message}")
                    element_output = sol_element.x

                    # Calculate the pressure drop for the permeate side
                    y_k = element_output[J:2*J]                     # Permeate composition
                    Qp_k = element_output[-1] * Total_flow          # Permeate flowrate
                    pP_k = Placeholder_profile.loc[k, 'P_Perm'] # Current permeate pressure
                    if not Membrane["Pressure_Drop"]:
                        pP_new = pP_k  # No pressure drop
                    else: # Calculate the pressure drop
                        dP = pressure_drop(y_k, Qp_k, pP_k) #WARNING : this function needs to be updated to account for merged elements
                        if dP>5: pP_new = pP_k - dP
                        else: pP_new = pP_k #negligible pressure drop: helps with stability of the solver

                    df_element = np.concatenate((
                        np.array([n_elements - 1 - k]),   # Make scalar a 1D array
                        element_output,                  # Already 1D
                        np.array([pP_new]),              # Make scalar a 1D array
                        np.array([sol_element.cost]),    # Make scalar a 1D array
                        np.array([True])                 # Make Boolean a 1D array
                    ))
                else:
                    raise ValueError("No previously solved elements to revert to for merging.")

        else: 
            # Calculate the pressure drop for the permeate side
            y_k = element_output[J:2*J]                     # Permeate composition
            Qp_k = element_output[-1] * Total_flow          # Permeate flowrate
            pP_k = Placeholder_profile.loc[k, 'P_Perm'] # Current permeate pressure

            if not Membrane["Pressure_Drop"]:
                pP_new = pP_k  # No pressure drop
            
            else: # Calculate the pressure drop
                dP = pressure_drop(y_k, Qp_k, pP_k) #WARNING : this function needs to be updated to account for merged elements
                if dP>1:
                    pP_new = pP_k - dP
                else:
                    pP_new = pP_k #negligible pressure drop: helps with stability of the solver
 
            df_element = np.concatenate((
                np.array([n_elements - 1 - k]),   # Make scalar a 1D array
                element_output,                  # Already 1D
                np.array([pP_new]),              # Make scalar a 1D array
                np.array([sol_element.cost]),    # Make scalar a 1D array
                np.array([True])                 # Make Boolean a 1D array
            ))

        # Update the DataFrame with the results
        #print(df_element)
        Placeholder_profile.loc[k + 1] = df_element
        #print(Placeholder_profile)
    
    '''columns = ['Element'] + [f'x{i+1}' for i in range(J)] + [f'y{i+1}' for i in range(J)] + ['cut_r/Qr', 'cut_p/Qp','P_Perm','Error','Solved']'''
    x_ret = Placeholder_profile.iloc[-1,1:J+1].to_numpy()
    y_perm = Placeholder_profile.iloc[-1,J+1:2*J+1].to_numpy()
    Qr = Placeholder_profile.iloc[-1,2*J+1] * Membrane["Total_Flow"]
    Qp = Placeholder_profile.iloc[-1,2*J+2] * Membrane["Total_Flow"]
    CO_results = x_ret, y_perm, Qr, Qp
    
    profile = Placeholder_profile.copy()
    Solved_membrane_profile = profile[profile['Solved']==True].reset_index(drop=True)

    return CO_results, Solved_membrane_profile


CO_results, profile = mass_balance_CO_varmesh(vars)
print("Retentate Composition:", CO_results[0])
print("Permeate Composition:", CO_results[1])
print("Retentate Flowrate (mol/h):", CO_results[2])
print("Permeate Flowrate (mol/h):", CO_results[3])
print()
print(f'Stage cut: {CO_results[3]/(CO_results[2]+CO_results[3])*100:.2f} %')
print(f'CO2 Recovery: {CO_results[1][0] * CO_results[3] / (Membrane["Feed_Flow"] * Membrane["Feed_Composition"][0]) * 100:.2f} %')
print(f'CO2 Purity: {CO_results[1][0] * 100:.2f} %')
print()
print("Membrane Profile:")
print(profile)

elements = profile['Element']
n = elements.max()

# Calculate normalized length
normalized_length = (n - elements) / n

# Plotting
plt.figure(figsize=(14, 6))

# Retentate Molar Fractions Plot
plt.subplot(1, 2, 1)
for i in range(J):
    plt.plot(normalized_length, profile[f'x{i+1}'], label=f'Retentate x{i+1}')
plt.xlabel('Normalized Length')
plt.ylabel('Molar Fraction')
plt.title('Retentate Molar Fractions')
plt.legend(loc='best')
plt.grid()

# Permeate Molar Fractions Plot
plt.subplot(1, 2, 2)
for i in range(J):
    plt.plot(normalized_length, profile[f'y{i+1}'], label=f'Permeate y{i+1}')
plt.xlabel('Normalized Length')
plt.ylabel('Molar Fraction')
plt.title('Permeate Molar Fractions')
plt.legend(loc='best')
plt.grid()

plt.tight_layout()
plt.show()

# Plotting Flow Rates
plt.figure(figsize=(14, 6))

# Retentate Flow Rates Plot
plt.subplot(1, 2, 1)
plt.plot(normalized_length, profile['cut_r/Qr'], label='Retentate Flow')
plt.xlabel('Normalized Length')
plt.ylabel('Flow (mol/h)')
plt.title('Retentate Flow Rates')
plt.legend(loc='best')
plt.grid()

# Permeate Flow Rates Plot
plt.subplot(1, 2, 2)
plt.plot(normalized_length, profile['cut_p/Qp'], label='Permeate Flow')
plt.xlabel('Normalized Length')
plt.ylabel('Flow (mol/h)')
plt.title('Permeate Flow Rates')
plt.legend(loc='best')
plt.grid()

plt.tight_layout()
plt.show()