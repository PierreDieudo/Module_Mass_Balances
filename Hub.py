import math
from scipy.optimize import minimize_scalar
import numpy as np

''' General Information:

This script runs the mass balance of the membrane being defined in MAIN.

It takes feed information and membrane parameters from the user and inputs them into a separate python script for the corresponding mass balance method.

It is designed to be used for any membrane, no matter the number of components.

'''

def Hub_Connector(Export_to_mass_balance): #general because it will call the corresponding mass balance desired by the user.

    # Unpacking inlet variables and membrane parameters
    Membrane, Component_properties, Fibre_Dimensions = Export_to_mass_balance

    # Unpacking and transforming inlet variables and membrane parameters
    
    Membrane["Permeance"] = [p * 3.348 * 1e-10 for p in Membrane["Permeance"]]  # convert from GPU to mol/m2.s.Pa
    Membrane["Pressure_Feed"] *= 1e5  #convert to Pa
    Membrane["Pressure_Permeate"] *= 1e5  
    Membrane["Total_Flow"]  = Membrane["Feed_Flow"]+Membrane["Sweep_Flow"]

    #number of components
    J = len(Membrane["Feed_Composition"])

    if not J == len(Membrane["Feed_Composition"]) == len(Membrane["Sweep_Composition"]) == len(Component_properties["Viscosity_param"]):
        raise ValueError("Number of components does not match data provided")

    #Checks the input data for inconsistency
    if abs(sum(Membrane["Feed_Composition"]) - 1) > 1e-8:
        raise ValueError("Initial mole fractions do not sum to 1")
    if Membrane["Sweep_Flow"]!=0 and abs(sum(Membrane["Sweep_Composition"]) - 1) > 1e-8:
        raise ValueError(f"Initial mole fractions do not sum to 1 ({(sum(Membrane["Sweep_Composition"])):.3e})")
  

    #Determines Module length and number of fibers to minimise pressure drop (Shao, Huang, 2006)
    def module_length_calc(L):
            R = 8.314 # J/(mol.K) - gas constant
            Delta = [0] * J
          
            for i in range(J):
                slope, intercept = Component_properties["Viscosity_param"][i]
                visc = 1e-6*(slope * Membrane["Temperature"] + intercept) # Viscosity in Pa.s (from trend obtain from NIST)
                Delta[i] = 8 * math.sqrt(2 * R * Membrane["Temperature"] * Fibre_Dimensions["D_out"] * visc *  Membrane["Permeance"][i] * (L**2) / (Fibre_Dimensions["D_out"]**4 * Membrane["Pressure_Permeate"]))
     
            return max(Delta)

    def objective(L):
            
        max_delta = module_length_calc(L)
       
        # Penalise values of max_delta that exceed 0.4
        if max_delta > 0.4:
            return max_delta - 0.4 #penalty
        # Minimise the difference between max_delta and 0.4
        return 0.4 - max_delta

    result = minimize_scalar(objective, bounds=(3e-1, 5), method='bounded')
    if result.success:
        Fibre_Dimensions['Length'] = float(result.x)
        #print(f'Optimised module length: {Fibre_Dimensions['Length']:.4f} m')
    else:
        print("Optimisation failed to find a suitable module length")
        Fibre_Dimensions['Length'] = 0.1 #m - module length

    fibre_area = math.pi * Fibre_Dimensions['Length'] * Fibre_Dimensions["D_out"] #m2
    Fibre_Dimensions["Number_Fibre"] =  Membrane["Area"] / fibre_area #number of fibres in the module

    #Solving the mass balance (for now humid conditions are not considered)
    vars = Membrane, Component_properties, Fibre_Dimensions

    if Membrane["Solving_Method"] == 'CO':
        from CO import mass_balance_CO
        return mass_balance_CO(vars) # tuple containing [x_ret, y_perm, Qr, Qp] and the data frame with the module profile

    elif Membrane["Solving_Method"] == 'CC':
        from CC import mass_balance_CC
        return mass_balance_CC(vars)
    elif Membrane["Solving_Method"] == 'CO_Molten':
        from CO_Molten import mass_balance_CO_Molten
        return mass_balance_CO_Molten(vars)
    elif Membrane["Solving_Method"] == 'CO_ODE':
        from CO_ODE import mass_balance_CO_ODE
        return mass_balance_CO_ODE(vars)
    elif Membrane["Solving_Method"] == 'CC_ODE':
        from CC_ODE import mass_balance_CC_ODE
        return mass_balance_CC_ODE(vars)
    else:
        raise ValueError("Solving_Method not recognised")


