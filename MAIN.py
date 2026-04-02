import profile
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("TkAgg")          # GUI backend - windows survive after the terminal closes
import matplotlib.pyplot as plt
from datetime import datetime
from Hub import Hub_Connector
import warnings

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

directory = 'C:\\Users\\s1854031\\Desktop\\'  # input file path here.
Run_Name  = 'test2'  # Optional name for this run (e.g. 'high_pressure_test').
                # Leave as '' to use the timestamp only as the subfolder name.

Membrane = {
    "Solving_Method": 'CC_ODE_BVP',                     # 'CC' or 'CO' - CC is for counter-current, CO is for co-current
    "Temperature": 30+273.15,                   # Kelvin
    "Feed_Composition": [0.25,0.045,0.705], # molar fraction
    "Feed_Flow": 11.8,                           # mol/s (PS: 1 mol/s = 3.6 kmol/h)
    "Pressure_Feed": 5,                         # bar
    "Pressure_Permeate": 1,                   # bar
    "Area": 300,                                # m2
    "Permeance": [7700,140,210],              # GPU
    "Sweep_Option": False,                    # True or False - use a sweep or not
    "Sweep_Source": 'User',                   # 'User' or 'Recycling' - where the sweep comes from
    "Recycling_Ratio": 0,                     # Fraction of a stream (likely retentate) being sent back as sweep 
    "Pressure_Drop": True, 
    "Export_Profile": False,                    # True or False - export the profile to a CSV file        
    "Plot_Profiles": True,                      # True or False - plot the profile of the membrane"
    }

Component_properties = {
    "Viscosity_param": ([0.0479,0.6112],[0.0466,3.8874],[0.0558,3.8970]),  # Viscosity parameters for each component: slope and intercept for the viscosity correlation wiht temperature (in K) - from NIST
    "Molar_mass": [44.009, 28.0134, 31.999]                                 # Molar mass of each component in g/mol
    }

Fibre_Dimensions = {
    "D_in" : 600 * 1e-6, # Inner diameter in m (from mm)
    "D_out" : 800 * 1e-6, # Outer diameter in m (from mm)
    "D_Module" : 0.375*2, # Diameter of the module in m
    "Length": 0.5, # Length of the module in m
    "D_hydraulic": 7.98403e-4, # Hydraulic diameter in m (per module - careful with flowrate in pressure drop function)
    "A_module": 1257, # Cross-sectional area of a module in m2
    }

User_Sweep = { # Only if Sweep_Option is True and Sweep source is User
    "Sweep_Flow": 1,                         # mol/s 
    "Sweep_Composition": [0,1,0],          # molar fraction
    }

#--------------------------------------#
#--------- End of User Inputs ---------#
#--------------------------------------#

if Membrane["Plot_Profiles"] or Membrane["Export_Profile"]:
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if Run_Name.strip():
        run_folder = Run_Name.strip()       # e.g. high_pressure_test
    else:
        run_folder = f"run_{run_timestamp}" # e.g. run_2026-04-02_14-30-05

    output_directory = os.path.join(directory, "simulation_outputs", run_folder)
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output folder: {output_directory}")
   

Export_to_mass_balance = Membrane, Component_properties, Fibre_Dimensions

def Run_Module():

    print("Running Simulation...")

    global J
    J = len(Membrane["Permeance"])  # number of components

    if not Membrane["Sweep_Option"]:  # sweep deactivated
            
        Membrane["Sweep_Flow"] = 0
        Membrane["Sweep_Composition"] = [0] * J
            
        results, profile = Hub_Connector(Export_to_mass_balance)
        Membrane["Retentate_Composition"],Membrane["Permeate_Composition"],Membrane["Retentate_Flow"],Membrane["Permeate_Flow"] = results
    
    elif Membrane["Sweep_Option"] and Membrane["Sweep_Source"] == 'User':  # sweep from user
        
        Membrane["Sweep_Flow"] = User_Sweep["Sweep_Flow"]
        Membrane["Sweep_Composition"] = User_Sweep["Sweep_Composition"]

        results, profile = Hub_Connector(Export_to_mass_balance)
        Membrane["Retentate_Composition"],Membrane["Permeate_Composition"],Membrane["Retentate_Flow"],Membrane["Permeate_Flow"] = results

    else:  # sweep from Recycling - needs iteration
        max_iter = 100
        tolerance = 1e-5

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

            # Need to reset the parameters to go through the general mass balance file formatting again
            Membrane["Permeance"] = [p / ( 3.348 * 1e-10 ) for p in Membrane["Permeance"]]  # convert from mol/m2.s.Pa to GPU
            Membrane["Pressure_Feed"] *= 1e-5   # convert to bar
            Membrane["Pressure_Permeate"] *= 1e-5  

        else:
            print("Warning: Sweep iteration did not converge within the maximum number of iterations.")

    errors = []
    for i in range(J):    
        Feed_Sweep_Mol = Membrane["Feed_Flow"] * Membrane["Feed_Composition"][i] + Membrane["Sweep_Flow"] * Membrane["Sweep_Composition"][i]
        Retentate_Mol  = Membrane["Retentate_Flow"] * Membrane["Retentate_Composition"][i]
        Permeate_Mol   = Membrane["Permeate_Flow"]  * Membrane["Permeate_Composition"][i]
        error = abs((Feed_Sweep_Mol - Retentate_Mol - Permeate_Mol) / Feed_Sweep_Mol)
        errors.append(error)

    cumulated_error = sum(errors)
    print(f"Cumulated Component Mass Balance Error: {cumulated_error:.2e}")    

    composition_cols = [f"x{i+1}" for i in range(J)] + [f"y{i+1}" for i in range(J)]

    Recovery  = Membrane["Permeate_Composition"][0] * Membrane["Permeate_Flow"] / (Membrane["Feed_Flow"] * Membrane["Feed_Composition"][0]) * 100
    Purity    = Membrane["Permeate_Composition"][0] * 100
    Stage_cut = Membrane["Permeate_Flow"] / (Membrane["Feed_Flow"] + Membrane["Sweep_Flow"]) * 100
    print(f'Simulation finished with Recovery: {Recovery:.2f}%, Purity: {Purity:.2f}%, and a stage cut of {Stage_cut:.2f}%')

    return profile


def plot_composition_profiles(profile):
    """Save composition and flow profile figures to output_directory and open them
    with the default Windows image viewer (independent of Python/VS Code)."""
    df = profile.copy()
    global J
    z = df["norm_z"]

    # --- Figure 1: Composition profiles -------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 5))

    for j in range(J):
        col = f'x{j+1}'
        if col in profile.columns:
            axes1[0].plot(z, profile[col] * 100, label=f'Component {j+1}')
    axes1[0].set_xlabel('Normalised Length')
    axes1[0].set_ylabel('Retentate Composition (%)')
    axes1[0].set_title('Retentate Composition Profile')
    axes1[0].legend()
    axes1[0].grid(True)

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
    fig1_path = os.path.join(output_directory, f"composition_profile_{run_folder}.png")
    fig1.savefig(fig1_path, dpi=150)
    print(f"Composition profile saved to {fig1_path}")

    # --- Figure 2: Flow profiles ---------
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))

    for j in range(J):
        col = f'x{j+1}'
        if col in profile.columns:
            axes2[0].plot(z, profile[col] * profile['Qr'], label=f'Component {j+1}')
    axes2[0].set_xlabel('Normalised Length')
    axes2[0].set_ylabel('Retentate Normalised Component Flow (-)')
    axes2[0].set_title('Retentate Flow Profile')
    axes2[0].legend()
    axes2[0].grid(True)

    for j in range(J):
        col = f'y{j+1}'
        if col in profile.columns:
            axes2[1].plot(z, profile[col] * profile['Qp'], label=f'Component {j+1}')
    axes2[1].set_xlabel('Normalised Length')
    axes2[1].set_ylabel('Permeate Normalised Component Flow (-)')
    axes2[1].set_title('Permeate Flow Profile')
    axes2[1].legend()
    axes2[1].grid(True)

    plt.tight_layout()
    fig2_path = os.path.join(output_directory, f"flow_profile_{run_folder}.png")
    fig2.savefig(fig2_path, dpi=150)
    print(f"Flow profile saved to {fig2_path}")

    plt.close('all')  # no matplotlib window - images open via Windows viewer instead

    # Open both images with the default Windows image viewer
    os.startfile(fig1_path)
    os.startfile(fig2_path)


# --- Main execution ----------------
profile = Run_Module()

if Membrane["Export_Profile"]:
    csv_path = os.path.join(output_directory, "membrane_profile.csv")
    profile.to_csv(csv_path, index=None)
    print(f"Profile exported to {csv_path}")

if Membrane["Plot_Profiles"]:
    plot_composition_profiles(profile)

print("Done - probably")