

import profile
import numpy as np
import pandas as pd
import os
from Hub import Hub_Connector


''' General information here: 

    Hello me or Pete or whoever that is. Good luck.

    - The aim of this Solution is to perform the simulation of a polymeric membrane module.

    - For now, the membrane is isothermal, with either co-current or counter-current configurations.

    - The aim of this script is to serve as the user input file as the actual mass balance of the membrane will be done in other scripts.

    - There is an option to export the profile of the membrane to a CSV file, which is useful for debugging and checking the results.

    - Mass balance errors are displayed in the console. Anything over 1e-5 is considered large. It is likely that one of the components' driving force is too low at some point in the module. Try decreasing Area.

    Please refer to me (s1854031@ed.ac.uk) for any questions or issues until February 2027 - except if I got fired from the phd before that for being too cheeky.
    xxx
    Pierre
 '''
  
#-----------------------------------------#
#--------- User input parameters ---------#
#-----------------------------------------#

directory = 'C:\\Users\\s1854031\\Desktop\\' #input file path here.

Membrane = {
    "Solving_Method": 'CC',                     # 'CC' or 'CO' - CC is for counter-current, CO is for co-current
    "Temperature": 35+273.15,                   # Kelvin
    "Feed_Composition": [0.2,0.6,0.15,0.05], # molar fraction
    "Feed_Flow": 1,                           # mol/s (PS: 1 mol/s = 3.6 kmol/h)
    "Pressure_Feed": 30,                         # bar
    "Pressure_Permeate": 1.013,                   # bar
    "Area": 25,                                # m2
    "Permeance": [40.024,1.111,0.305,0.06],        # GPU
    "Sweep_Option": False,                  # True or False - use a sweep or not
    "Sweep_Source": 'User',                # 'User' or 'Recycling' - where the sweep comes from
    "Recycling_Ratio": 0.1,                # Fraction of a stream (likely retentate) being sent back as sweep 
    "Pressure_Drop": False,
    "Export_Profile": True,               # True or False - export the profile to a CSV file        
    "Plot_Profiles": True,                 # True or False - plot the profile of the membrane"
    }

Component_properties = {
    "Viscosity_param": ([0.0479,0.6112],[0.0466,3.8874],[0.0558,3.8970], [0.03333, -0.23498]),  # Viscosity parameters for each component: slope and intercept for the viscosity correlation wiht temperature (in K) - from NIST
    "Molar_mass": [44.009, 28.0134, 31.999,18.01528],                                           # Molar mass of each component in g/mol
    }

Fibre_Dimensions = {
    "D_in" : 150 * 1e-6, # Inner diameter in m (from um)
    "D_out" : 300 * 1e-6, # Outer diameter in m (from um)
    }

User_Sweep = { # Only if Sweep_Option is True and Sweep source is User
    "Sweep_Flow": 3.5,                            # mol/s 
    "Sweep_Composition": [0, 1,0,0],          # molar fraction
    }

#--------------------------------------#
#--------- End of User Inputs ---------#
#--------------------------------------#

Export_to_mass_balance = Membrane, Component_properties, Fibre_Dimensions

def Run_Module():
    
    print("Running Simulation...")

    global J
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


    print(f"Overall mass balance error: Feed + Sweep  - Retentate - Permeate = {abs(Membrane["Feed_Flow"] + Membrane["Sweep_Flow"] - Membrane["Retentate_Flow"] - Membrane["Permeate_Flow"]):.3e}")
        
    if np.any(profile<-1e-5):
        print(profile)
        raise ValueError("Negative values in the membrane profile") #check for negative values in the profile
        

    Recovery = Membrane["Permeate_Composition"][0] * Membrane["Permeate_Flow"] / (Membrane["Feed_Flow"] * Membrane["Feed_Composition"][0]) * 100
    Purity = Membrane["Permeate_Composition"][0] * 100
    Stage_cut = Membrane["Permeate_Flow"] / (Membrane["Feed_Flow"]+Membrane["Sweep_Flow"]) * 100
    print(f'Simulation finished with Recovery: {Recovery:.2f}%, Purity: {Purity:.2f}%, and a stage cut of {Stage_cut:.2f}%')

    print(profile)

    return profile

def plot_composition_profiles(profile):  

    df = profile.copy()
    global J
    z = df["norm_z"]

    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 5))  

    # Retentate composition plot  
    for j in range(J):  
        col = f'x{j+1}'  
        if col in profile.columns:  
            axes1[0].plot(z, profile[col] * 100, label=f'Component {j+1}')  
    axes1[0].set_xlabel('Normalised Length')  
    axes1[0].set_ylabel('Retentate Composition (%)')  
    axes1[0].set_title('Retentate Composition Profile')  
    axes1[0].legend()  
    axes1[0].grid(True)  

    # Permeate composition plot  
    for j in range(J):  
        col = f'y{j+1}'  
        if col in profile.columns:  
            axes1[1].plot(z, profile[col] * 100, label=f'Component {j+1}')  
    axes1[1].set_xlabel('Normalised Length')  
    axes1[1].set_ylabel('Permeate Composition (%)')  
    axes1[1].set_title('Permeate Composition Profile')  
    axes1[1].legend()  
    axes1[1].grid(True)  

    plt.tight_layout()  
    plt.show(block=False)

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))  

    # Retentate component flows plot
    for j in range(J):  
        col = f'x{j+1}'  
        if col in profile.columns:  
            # Multiply component fraction by normalised retentate flow
            axes2[0].plot(z, profile[col] * profile['cut_r/Qr'], label=f'Component {j+1}')  

    axes2[0].set_xlabel('Normalised Length')  
    axes2[0].set_ylabel('Retentate Normalised Component Flow (-)')  
    axes2[0].set_title('Retentate Flow Profile')  
    axes2[0].legend()  
    axes2[0].grid(True)  

    # Permeate component flows plot
    for j in range(J):  
        col = f'y{j+1}'  
        if col in profile.columns:  
            # Multiply component fraction by normalised permeate flow
            axes2[1].plot(z, profile[col] * profile['cut_p/Qp'], label=f'Component {j+1}')  

    axes2[1].set_xlabel('Normalised Length')  
    axes2[1].set_ylabel('Permeate Normalised Component Flow (-)')  
    axes2[1].set_title('Permeate Flow Profile')  
    axes2[1].legend()  
    axes2[1].grid(True)  

    plt.tight_layout()  
    plt.show()


profile = Run_Module()

if Membrane["Export_Profile"]: # Export the profile to a CSV file in the same directory as the Unisim file
    csv_name = os.path.join(directory, 'membrane_profile.csv')  
    profile.to_csv(csv_name, index=None)  
    print(f'Profile exported to {csv_name}')


if Membrane["Plot_Profiles"]:
    import matplotlib.pyplot as plt
    plot_composition_profiles(profile)

print("Done - probably")
