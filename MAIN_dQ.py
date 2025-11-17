

import profile
import numpy as np
import pandas as pd
import os
from Hub import Hub_Connector


''' General information here: 

    - This script is structured in the same way as the other MAIN code, but serves a different purpose.

    - This file adds a loop to run the simulation over a range of parameters (here, feed flow)

 '''
  
#-----------------------------------------#
#--------- User input parameters ---------#
#-----------------------------------------#

directory = 'C:\\Users\\s1854031\\Desktop\\' #input file path here.

Membrane = {
    "Solving_Method": 'CO_ODE',                     # 'CC' or 'CO' - CC is for counter-current, CO is for co-current
    "Temperature": 50+273.15,                   # Kelvin
    "Feed_Composition": [0.4,0.6], # molar fraction
    "Feed_Flow": 1,                           # mol/s (PS: 1 mol/s = 3.6 kmol/h)
    "Pressure_Feed": 59.7,                         # bar
    "Pressure_Permeate": 1.7,                   # bar
    "Area": 377,                                # m2
    "Permeance": [22.7,0.7],        # GPU
    "Sweep_Option": False,                      # True or False - use a sweep or not
    "Sweep_Source": 'Recycling',                # 'User' or 'Recycling' - where the sweep comes from
    "Recycling_Ratio": 0.1,                     # Fraction of a stream (likely retentate) being sent back as sweep 
    "Pressure_Drop": False,
    "Export_Profile": True,                    # True or False - export the profile to a CSV file        
    "Plot_Profiles": False,                      # True or False - plot the profile of the membrane"
    }
 
Component_properties = {
    "Viscosity_param": ([0.0479,0.6112],[0.0466,3.8874]),#,[0.0558,3.8970], [0.03333, -0.23498]),  # Viscosity parameters for each component: slope and intercept for the viscosity correlation wiht temperature (in K) - from NIST
    "Molar_mass": [44.009, 28.0134],#, 31.999,18.01528],                                           # Molar mass of each component in g/mol
    }

Fibre_Dimensions = {
    "D_in" : 150 * 1e-6, # Inner diameter in m (from µm)
    "D_out" : 300 * 1e-6, # Outer diameter in m (from µm)
    }

User_Sweep = { # Only if Sweep_Option is True and Sweep source is User
    "Sweep_Flow": 0,                            # mol/s 
    "Sweep_Composition": [0, 0],          # molar fraction
    }

Export_to_mass_balance = Membrane, Component_properties, Fibre_Dimensions

def Run_Module():
    
    J = len(Membrane["Permeance"]) #number of components

    if not Membrane["Sweep_Option"]: # sweep deactivated
            
        Membrane["Sweep_Flow"] = 0
        Membrane["Sweep_Composition"] = [0] * J
            
        results, profile = Hub_Connector(Export_to_mass_balance)
        Membrane["Retentate_Composition"],Membrane["Permeate_Composition"],Membrane["Retentate_Flow"],Membrane["Permeate_Flow"] = results
    
    elif Membrane["Sweep_Option"] and Membrane["Sweep_Source"] == 'User': # sweep from user
        
        Membrane["Sweep_Flow"] = User_Sweep["Sweep_Flow"]
        Membrane["Sweep_Composition"] = User_Sweep["Sweep_Composition"]

        results, profile = Hub_Connector(Export_to_mass_balance)
        Membrane["Retentate_Composition"],Membrane["Permeate_Composition"],Membrane["Retentate_Flow"],Membrane["Permeate_Flow"] = results

    else: # sweep from Recycling - needs iteration
        max_iter = 100
        tolerance = 1e-6

        for i in range(max_iter):
            print(f"Sweep iteration {i+1}")

            if i == 0:  # first iteration assuming no sweep
                Membrane["Sweep_Flow"] = 0
                Membrane["Sweep_Composition"] = [0] * J

            else:  # subsequent iterations
                Membrane["Sweep_Composition"] = Membrane["Retentate_Composition"]
                Membrane["Sweep_Flow"] = Membrane["Recycling_Ratio"] * Membrane["Retentate_Flow"]

            results, profile = Hub_Connector(Export_to_mass_balance)
            Membrane["Retentate_Composition"], Membrane["Permeate_Composition"], Membrane["Retentate_Flow"], Membrane["Permeate_Flow"] = results

            if i > 0 and np.all(np.abs(np.array(Membrane["Retentate_Composition"]) - np.array(Membrane["Sweep_Composition"])) < tolerance) and abs( (Membrane["Sweep_Flow"] - Membrane["Retentate_Flow"] * Membrane["Recycling_Ratio"]) / Membrane["Sweep_Flow"]) < tolerance: 
                print(f"Converged after {i+1} iterations.")
                break

            #Need to reset the parameters to go through the general mass balance file formatting again - will think of a smarter way to do this later.
            Membrane["Permeance"] = [p / ( 3.348 * 1e-10 ) for p in Membrane["Permeance"]]  # convert from mol/m2.s.Pa to GPU
            Membrane["Pressure_Feed"] *= 1e-5  #convert to bar
            Membrane["Pressure_Permeate"] *= 1e-5  

        else:
            print("Warning: Sweep iteration did not converge within the maximum number of iterations.")
    
    errors = []
    for i in range(J):
    
        # Calculate comp molar flows
        Feed_Sweep_Mol = Membrane["Feed_Flow"] * Membrane["Feed_Composition"][i] + Membrane["Sweep_Flow"] * Membrane["Sweep_Composition"][i]
        Retentate_Mol = Membrane["Retentate_Flow"] * Membrane["Retentate_Composition"][i]
        Permeate_Mol = Membrane["Permeate_Flow"] * Membrane["Permeate_Composition"][i]
    
        # Calculate and store the error
        error = (Feed_Sweep_Mol - Retentate_Mol - Permeate_Mol)/Feed_Sweep_Mol
        errors.append(error)

    # Calculate the cumulated error
    cumulated_error = sum(errors)
    print(f"Cumulated Component Mass Balance Error: {cumulated_error:.2e}")    

    if np.any(profile<-1e-5) or cumulated_error>1e-5:
        print(f'Cumulated Component Mass Balance Error: {cumulated_error:.2e} with array {errors}')
        print(profile)
        raise ValueError("Mass Balance Error: Check Profile") #check for negative values in the profile
        

    Recovery = Membrane["Permeate_Composition"][0] * Membrane["Permeate_Flow"] / (Membrane["Feed_Flow"] * Membrane["Feed_Composition"][0]) * 100
    Purity = Membrane["Permeate_Composition"][0] * 100
    
    print(f'Simulation finished with Recovery: {Recovery:.2f} % and Purity: {Purity:.2f} %')

    Membrane["Recovery"] = Recovery
    Membrane["Purity"] = Purity

    #print(profile)

    return profile

def plot_composition_profiles(profile):  
   num_components = len(Membrane["Permeance"])  
   Norm_length = (max(profile['Element']) - profile['Element']) / (max(profile['Element']) - 1)  

   fig, axes = plt.subplots(1, 2, figsize=(16, 5))  

   # Retentate composition plot  
   for j in range(num_components):  
       col = f'x{j+1}'  
       if col in profile.columns:  
           axes[0].plot(Norm_length, profile[col] * 100, label=f'Component {j+1}')  
   axes[0].set_xlabel('Normalised Length')  
   axes[0].set_ylabel('Retentate Composition (%)')  
   axes[0].set_title('Retentate Composition Profile')  
   axes[0].legend()  
   axes[0].grid(True)  

   # Permeate composition plot  
   if not Membrane["Sweep_Option"]:   # Filter out the sweep entry based on configuration if no sweep (to avoid composition jump from zero)
       if Membrane["Solving_Method"] == 'CC':  
           # Ignore element 1  
           permeate_profile = profile[profile['Element'] != 1]  
           norm_length_perm = (max(profile['Element']) - permeate_profile['Element']) / (max(profile['Element']) - 2)  
       elif Membrane["Solving_Method"] == 'CO':  
           # Ignore last element  
           N = max(profile['Element'])  
           permeate_profile = profile[profile['Element'] != N]  
           norm_length_perm = (N - permeate_profile['Element']) / (N - 2)  
   else:  
       permeate_profile = profile  
       norm_length_perm = Norm_length  

   for j in range(num_components):  
       col = f'y{j+1}'  
       if col in permeate_profile.columns:  
           axes[1].plot(norm_length_perm, permeate_profile[col] * 100, label=f'Component {j+1}')  
   axes[1].set_xlabel('Normalised Length')  
   axes[1].set_ylabel('Permeate Composition (%)')  
   axes[1].set_title('Permeate Composition Profile')  
   axes[1].legend()  
   axes[1].grid(True)  

   plt.tight_layout()  
   plt.show()



# Setup range of parameters to iterate over
Q_feed = np.linspace(4, 36, 50)  # mol/s
J = len(Membrane["Permeance"])  # Number of components
columns = ['Feed_Flow'] + [f'x{i+1}' for i in range(J)] + [f'y{i+1}' for i in range(J)] + ['Qr', 'Qp','Recovery','Purity']
param_screening_df = pd.DataFrame(columns=columns)  # Start with an empty DataFrame

def param_screening(Q):
    Membrane["Feed_Flow"] = Q
    profile = Run_Module()
    return profile

for idx, Q in enumerate(Q_feed):
    print()
    print(f"Running simulation for Feed Flow: {Q:.2f} mol/s")
    Membrane["Feed_Flow"] = Q
    profile = param_screening(Q)
    param_screening_df.loc[idx] = [
           Q,
           *[Membrane["Retentate_Composition"][i] for i in range(J)],
           *[Membrane["Permeate_Composition"][i] for i in range(J)],
           Membrane["Retentate_Flow"],
           Membrane["Permeate_Flow"],
           Membrane["Recovery"],
           Membrane["Purity"]
       ]

    #Resets the parameters to go through the general mass balance file formatting again - will think of a smarter way to do this later.
    Membrane["Permeance"] = [p / ( 3.348 * 1e-10 ) for p in Membrane["Permeance"]]  # convert from mol/m2.s.Pa to GPU
    Membrane["Pressure_Feed"] *= 1e-5  #convert to bar
    Membrane["Pressure_Permeate"] *= 1e-5

csv_name = os.path.join(directory, 'dQ.csv')
param_screening_df.to_csv(csv_name, index=None)
print(f'Parameter screening results exported to {csv_name}')


if Membrane["Export_Profile"]: # Export the profile to a CSV file in the same directory as the Unisim file
    csv_name = os.path.join(directory, 'membrane_profile.csv')  
    profile.to_csv(csv_name, index=None)  
    print(f'Profile exported to {csv_name}')


if Membrane["Plot_Profiles"]:
    import matplotlib.pyplot as plt
    plot_composition_profiles(profile)


print("Done - probably")
