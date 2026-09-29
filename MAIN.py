import numpy as np
import pandas as pd
import os
import math
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
Run_Name  = ''  # Optional name for this run (e.g. 'high_pressure_test').
                # Leave as '' to use the timestamp only as the subfolder name.

Membrane = {
    "Solving_Method": 'CC_ODE',                     # 'CC_ODE' or 'CO_ODE' - CC is for counter-current, CO is for co-current. Chiara for counter current with variable permeance.
    "Temperature": 25+273.15,                   # Kelvin
    "Feed_Composition": [0.2,0.6,0.2], # molar fraction
    "Feed_Flow": 100,                           # mol/s (PS: 1 mol/s = 3.6 kmol/h)
    "Pressure_Feed": 2,                         # bar
    "Pressure_Permeate": 0.22,                   # bar
    "Area": 1000,                                # m2 ; a membrane module is about 251 m2
    "Permeance": [1000,50,100],              # GPU
    "fac": [1,0,0,0],                        # factor for permeance variation along the module - for Chiara method only
    "Sweep_Option": True,                    # True or False - use a sweep or not
    "Sweep_Source": 'User',                   # 'User' or 'Recycling' - where the sweep comes from
    "Recycling_Ratio": 0,                     # Fraction of a stream (likely retentate) being sent back as sweep
    "Pressure_Drop": True,
    "Export_Profile": False,                    # True or False - export the profile to a CSV file
    "Plot_Profiles": False,                      # True or False - plot the profile of the membrane (also shows the pressure profile when Pressure_Drop is True)
    }

#print(Membrane)

Component_properties = {
    "Viscosity_param": ([0.0479,0.6112],[0.0466,3.8874],[0.0558,3.8970]),  # Viscosity parameters for each component: slope and intercept for the viscosity correlation wiht temperature (in K) - from NIST
    "Molar_mass": [44.009, 28.0134, 31.999],                                           # Molar mass of each component in g/mol
    }

Fibre_Dimensions = {
    "D_in" : 600 * 1e-6, # Inner diameter in m (from mm)
    "D_out" : 800 * 1e-6, # Outer diameter in m (from mm)
    "Volume_Packing": 0.3, # (m3/m3) Volume packing of the fibres in the module
    "Fibre_per_Module": 10000, # Number of fibres in a module
    "Length": 0.3, # Length of the module in m
    }

# Calculate module dimensions based on the fibre dimensions and packing
D_Module = 2 * math.sqrt((Fibre_Dimensions["D_out"]/2)**2*Fibre_Dimensions["Fibre_per_Module"]/Fibre_Dimensions["Volume_Packing"]) # Diameter of the module in m
D_hydraulic = Fibre_Dimensions["D_out"] * (1/ Fibre_Dimensions["Volume_Packing"] - 1)# Hydraulic diameter in m
A_module = Fibre_Dimensions["Fibre_per_Module"] * Fibre_Dimensions["Length"] * math.pi * Fibre_Dimensions["D_out"] # Membrane area of a module in m2 (recomputed in Hub from the final Length)

# Update the fibre dimensions with the calculated module dimensions
Fibre_Dimensions["D_Module"] = D_Module
Fibre_Dimensions["D_hydraulic"] = D_hydraulic
Fibre_Dimensions["A_module"] = A_module
#print(Fibre_Dimensions)

User_Sweep = { # Only if Sweep_Option is True and Sweep source is User
    "Sweep_Flow": 10,                         # mol/s
    "Sweep_Composition": [0,1,0],          # molar fraction
    }

# Calculate Q/A ratio as an idicator
Membrane["Q_A_ratio"] = (Membrane["Feed_Flow"] * 0.0224  * 3600) / Membrane["Area"]  # (in m3(stp)/m2.hr)
#print(Membrane["Q_A_ratio"])
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

    recycling = Membrane["Sweep_Option"] and Membrane["Sweep_Source"] != 'User'

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

            # [CORRECTED] Relative flow check no longer divides by zero when Sweep_Flow is 0 (e.g. Recycling_Ratio = 0)
            composition_converged = np.all(np.abs(np.array(Membrane["Retentate_Composition"]) - np.array(Membrane["Sweep_Composition"])) < tolerance)
            flow_residual = abs(Membrane["Sweep_Flow"] - Membrane["Retentate_Flow"] * Membrane["Recycling_Ratio"])
            flow_converged = flow_residual <= tolerance * max(abs(Membrane["Sweep_Flow"]), 1e-12)

            if i > 0 and composition_converged and flow_converged:
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
        error = abs((Feed_Sweep_Mol - Retentate_Mol - Permeate_Mol) / (Feed_Sweep_Mol+1e-10))
        errors.append(error)

    cumulated_error = sum(errors)
    print(f"Cumulated Component Mass Balance Error: {cumulated_error:.2e}")

    # [CORRECTED] Performance indicators at system level
    # User sweep: the sweep is an external input, so its flow (and any component 1 it carries) is removed.
    # Recycling:  the sweep is an internal loop, the system outputs are the net retentate (1 - ratio) * Qr and the permeate Qp.
    if recycling:
        external_sweep_flow  = 0
        external_sweep_comp1 = 0
        Membrane["Net_Retentate_Flow"] = (1 - Membrane["Recycling_Ratio"]) * Membrane["Retentate_Flow"]
        print(f"Net retentate flow leaving the system: {Membrane['Net_Retentate_Flow']:.3f} mol/s")
    else:
        external_sweep_flow  = Membrane["Sweep_Flow"]
        external_sweep_comp1 = Membrane["Sweep_Flow"] * Membrane["Sweep_Composition"][0]

    Recovery  = (Membrane["Permeate_Composition"][0] * Membrane["Permeate_Flow"] - external_sweep_comp1) / (Membrane["Feed_Flow"] * Membrane["Feed_Composition"][0]) * 100
    Purity    = Membrane["Permeate_Composition"][0] * 100
    Stage_cut = (Membrane["Permeate_Flow"] - external_sweep_flow) / (Membrane["Feed_Flow"]) * 100
    print(f'Simulation finished with Recovery: {Recovery:.2f}%, Purity: {Purity:.2f}%, and a stage cut of {Stage_cut:.2f}%')
    print()
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


    # --- Figure 3: Permeance profiles (only if available) ---
    perm_cols = [f"perm{j+1} (GPU)" for j in range(J)]
    if all(col in profile.columns for col in perm_cols):
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        for j in range(J):
            ax3.plot(z, profile[perm_cols[j]], label=f'Component {j+1}')
        ax3.set_xlabel('Normalised Length')
        ax3.set_ylabel('Permeance (GPU)')
        ax3.set_title('Permeance Profile Along the Module')
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
        fig3_path = os.path.join(output_directory, f"permeance_profile_{run_folder}.png")
        fig3.savefig(fig3_path, dpi=150)

    plt.close('all')  # no matplotlib window - images open via Windows viewer instead
    print(f"Figures saved to {output_directory}")

    # Open images with the default Windows image viewer
    os.startfile(fig1_path)
    os.startfile(fig2_path)
    os.startfile(fig3_path) if all(col in profile.columns for col in perm_cols) else None

# --- Main execution ----------------
profile = Run_Module()

if Membrane["Export_Profile"]:
    csv_path = os.path.join(output_directory, "membrane_profile.csv")
    profile.to_csv(csv_path, index=None)
    print(f"Profile exported to {csv_path}")

if Membrane["Plot_Profiles"]:
    plot_composition_profiles(profile)

print()
print("Done - probably")