import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import least_squares
import numpy as np
import pandas as pd

def mass_balance_CC_ODE_BVP_dP(vars):

    Membrane, Component_properties, Fibre_Dimensions = vars
    Membrane["Total_Flow"] = Membrane["Feed_Flow"] + Membrane["Sweep_Flow"]
    Fibre_Dimensions["Number_Fibre"] = Membrane["Area"] / (Fibre_Dimensions["Length"] * math.pi * Fibre_Dimensions["D_out"])

    epsilon = 1e-10

    J = len(Membrane["Feed_Composition"])   

    n_elements = 250

    L = Fibre_Dimensions["Length"]

    Membrane["Feed_Composition"]  = np.array(Membrane["Feed_Composition"])
    Membrane["Sweep_Composition"] = np.array(Membrane["Sweep_Composition"])


    '''----------------------------------------------------------###
    ###------------- Mixture Viscosity Calculation --------------###
    ###----------------------------------------------------------'''

    def mixture_visc(composition): #vectorised for 2D arrays of compositions, returns array of viscosities for each point in the profile
        y = composition #shape (J, n_points)
        params = np.array(Component_properties["Viscosity_param"])
        visc = 1e-6 * (params[:, 0] * Membrane["Temperature"] + params[:, 1])  # shape (J,)
        Mw = np.array(Component_properties["Molar_mass"])  # shape (J,)
        phi = np.zeros((J, J))  # shape (J, J)
        # Calculate phi_ij for all pairs of components using broadcasting
        visc_ratio = np.sqrt(visc[:, None] / visc[None, :])  # shape (J, J)
        Mw_ratio = (Mw[:, None] / Mw[None, :]) ** 0.25  # shape (J, J)
        phi = ( ( 1 + visc_ratio * Mw_ratio ) **2 ) / ( ( 8 * ( 1 + Mw[:, None]/Mw[None, :]) )**0.5 )  # shape (J, J)

        denominator = (phi @ y).T #matrix multiplication, shape (n_points, J)
        nu = (y.T * visc) / (denominator + epsilon)  # shape (n_points, J)
        visc_mix = np.sum(nu, axis=1)  # shape (n_points,)

        return visc_mix


    '''    def mixture_visc(composition):
            y = composition
            visc = np.zeros(J)
            params = np.array(Component_properties["Viscosity_param"])
            visc = 1e-6 * (params[:, 0] * Membrane["Temperature"] + params[:, 1]) 
            Mw = Component_properties["Molar_mass"]
            phi = np.zeros((J, J))
            for i in range(J):
                for j in range(J):
                    if i != j:
                        phi[i][j] = ( ( 1 + ( visc[i]/visc[j] )**0.5 * ( Mw[j]/Mw[i] )**0.25 ) **2 ) / ( ( 8 * ( 1 + Mw[i]/Mw[j] ) )**0.5 )
                    else:
                        phi[i][j] = 1
            nu = np.zeros(J)
            for i in range(J):
                nu[i] = y[i] * visc[i] / sum(y[j] * phi[i][j] for j in range(J))
            return sum(nu)
    '''

    '''----------------------------------------------------------###
    ###--------------- Pressure Drop Calculation ----------------###
    ###----------------------------------------------------------'''

    def pressure_drop_permeate(composition, Q, P):
        # composition shape: (J, n_pts)
        # Q shape: (n_pts,)
        # P shape: (n_pts,)
        visc_mix = mixture_visc(composition)   # shape (n_pts,)
        D_in = Fibre_Dimensions["D_in"]
        Q_per_fibre = Q / Fibre_Dimensions['Number_Fibre']
        R = 8.314
        nu = (Q_per_fibre * R * Membrane["Temperature"]) / P
        dP_dz = (128 * visc_mix) / (math.pi * D_in**4 * P) * nu
        return dP_dz   # shape (n_pts,)

    def pressure_drop_retentate(composition, Q, P):
        visc_mix = mixture_visc(composition)   # shape (n_pts,)
        D_hyd = Fibre_Dimensions["D_hydraulic"]
        Q_per_module = Q / Fibre_Dimensions['Number_Module']
        R = 8.314
        nu = (Q_per_module * R * Membrane["Temperature"]) / P
        dP_dz = (128 * visc_mix) / (math.pi * D_hyd**4 * P) * nu
        return dP_dz   # shape (n_pts,)
  
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
    

    '''----------------------------------------------------------###
    ###--------------------- BVP Solver ------------------------###
    ###----------------------------------------------------------'''

    def membrane_odes(z, var):
        u_x = np.maximum(var[:J], 1e-10)
        u_y = np.minimum(var[J:2*J], -1e-10) 
        P_perm = var[2*J]      # permeate pressure, varies with z
        P_ret = var[2*J+1]    # retentate pressure, varies with z

        A      = Fibre_Dimensions["D_out"] * math.pi * Fibre_Dimensions["Number_Fibre"]
        Ttot   = Membrane["Total_Flow"]
    
        permeance = np.array(Membrane["Permeance"])

        sum_ux = np.sum(u_x, axis=0)
        sum_uy = np.sum(u_y, axis=0)

        # safe mole fractions
        x = np.zeros_like(u_x)
        y = np.zeros_like(u_y)
    
        safe_x = np.abs(sum_ux) > 1e-6
        safe_y = np.abs(sum_uy) > 1e-6
    
        x[:, safe_x] = u_x[:, safe_x] / sum_ux[safe_x]
        y[:, safe_y] = u_y[:, safe_y] / sum_uy[safe_y]

        driving_force = P_ret * x - P_perm * y
    
        # suppress permeation when retentate is nearly depleted
        sum_ux = np.sum(u_x, axis=0)
        depleted = sum_ux < 1e-4   # shape (n_points,)
        driving_force[:, depleted] = 0.0
    
        du_x_dz = -(permeance[:, None] * A / Ttot) * driving_force
        du_y_dz = -du_x_dz

        # retentate pressure drops in +z direction
        dP_feed_dz = -pressure_drop_retentate(x, sum_ux * Ttot, P_ret)

        # permeate pressure rises in +z direction (flows in -z)
        dP_perm_dz = +pressure_drop_permeate(np.abs(y), -sum_uy * Ttot, P_perm)


        return np.concatenate([du_x_dz, du_y_dz, 
                               dP_feed_dz[None, :], 
                               dP_perm_dz[None, :]], axis=0)
    
    def bc(ya, yb):
        feed_norm        =  Membrane["Feed_Composition"]  * Membrane["Feed_Flow"]  / Membrane["Total_Flow"]
        sweep_norm       = -Membrane["Sweep_Composition"] * Membrane["Sweep_Flow"] / Membrane["Total_Flow"]
        feed_pressure    =  Membrane["Pressure_Feed"]
        perm_pressure    =  Membrane["Pressure_Permeate"]
        return np.concatenate([
            ya[:J]    - feed_norm,                    # feed flow BC at z=0
            yb[J:2*J] - sweep_norm,                   # sweep flow BC at z=L
            [ya[2*J]   - perm_pressure],              # permeate pressure at z=0
            [ya[2*J+1] - feed_pressure],              # retentate pressure at z=0
        ])

    sol = approx_shooting_guess() #conducts a simplified mass balance to get an initial guess
    x_L_approx = sol.x[:J]
    y_0_approx = sol.x[J:2*J]
    cut_r_L    = sol.x[-2]
    cut_p_0    = sol.x[-1]
    U_x_z0_approx = x_L_approx * cut_r_L #approx 
    U_y_zL_approx  = -y_0_approx * cut_p_0

    U_x_feed_norm  =  Membrane["Feed_Composition"]  * Membrane["Feed_Flow"]  / Membrane["Total_Flow"]
    U_y_sweep_norm = -Membrane["Sweep_Composition"] * Membrane["Sweep_Flow"] / Membrane["Total_Flow"]

    # Initialise the solution on a coarse grid, using the approximated values from the simplified mass balance
    x_init = np.linspace(0, Fibre_Dimensions["Length"], 10)
    n_init = x_init.shape[0]

    U_x_init = np.zeros((J, n_init))
    U_y_init = np.zeros((J, n_init))
    for i in range(J):
        U_x_init[i, :] = np.linspace(U_x_feed_norm[i], U_x_z0_approx[i], n_init) # initial guess of linear profile between feed and retentate outlet
        U_y_init[i, :] = np.linspace(U_y_zL_approx[i], U_y_sweep_norm[i], n_init)

    P_perm_init = np.full(n_init, Membrane["Pressure_Permeate"]) #initial guess of constant permeate pressure, will be updated by solver to account for pressure drop
    P_feed_init = np.full(n_init, Membrane["Pressure_Feed"])

    y_init = np.vstack([U_x_init, U_y_init, P_perm_init, P_feed_init])

    ### BVP SOLVER ###
    import warnings                                                                                                                                                                                                                                                                                                                                                                                                             
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                              module='scipy.integrate._bvp')
        bvp_sol = solve_bvp(membrane_odes, bc, x_init, y_init, tol=1e-4, max_nodes=1000)

    y_sol = bvp_sol.y                          # shape (2J, n_points) on adaptive mesh
    z_adaptive = bvp_sol.x                     # adaptive mesh chosen by solver

    U_x_profile = y_sol[:J, :]  * Membrane["Total_Flow"]
    U_y_profile = y_sol[J:2*J, :] * Membrane["Total_Flow"]
    
    if Membrane["Sweep_Flow"] == 0:
        threshold = 1e-4
    else:
        threshold = 1e-8

    x_profiles = np.zeros_like(U_x_profile)
    y_profiles = np.zeros_like(U_y_profile)

    sum_ux_prof = np.sum(U_x_profile, axis=0)
    sum_uy_prof = np.sum(U_y_profile, axis=0)

    safe_x = np.abs(sum_ux_prof) > threshold
    safe_y = np.abs(sum_uy_prof) > threshold

    x_profiles[:, safe_x] = U_x_profile[:, safe_x] / sum_ux_prof[safe_x]
    y_profiles[:, safe_y] = U_y_profile[:, safe_y] / sum_uy_prof[safe_y]

    Qr_profile =  np.sum(U_x_profile, axis=0)
    Qp_profile = -np.sum(U_y_profile, axis=0)

    P_perm_profile = y_sol[2*J, :]     # permeate pressure profile
    P_feed_profile = y_sol[2*J+1, :]   # retentate pressure profile

    z_norm = z_adaptive / Fibre_Dimensions["Length"]

    depletion_mask = Qr_profile > 1e-4 * Membrane["Total_Flow"]
    if not np.all(depletion_mask):
        last_valid = np.argmax(~depletion_mask)  # first index where depleted
        z_adaptive  = z_adaptive[:last_valid]
        U_x_profile = U_x_profile[:, :last_valid]
        U_y_profile = U_y_profile[:, :last_valid]
        sum_ux_prof = sum_ux_prof[:last_valid]
        sum_uy_prof = sum_uy_prof[:last_valid]
        Qr_profile  = Qr_profile[:last_valid]
        Qp_profile  = Qp_profile[:last_valid]
        x_profiles  = x_profiles[:, :last_valid]
        y_profiles  = y_profiles[:, :last_valid]
        z_norm      = z_adaptive / L
        print(f"Retentate depleted at norm_z={z_norm[-1]:.3f} ; equivalent to an area of {z_norm[-1]*Membrane["Area"]:.2f} m2")

    data = {
        "norm_z":   z_norm,
        **{f"x{i+1}": x_profiles[i, :] for i in range(J)},
        **{f"y{i+1}": y_profiles[i, :] for i in range(J)},
        "Qr":       Qr_profile,
        "Qp":       Qp_profile,
        "P_feed":   P_feed_profile,
        "P_perm":   P_perm_profile,
    }
   
    print(f"Retentate pressure drop: {abs(P_feed_profile[0] - P_feed_profile[-1]):.2f} Pa")
    print(f"Permeate pressure drop: {abs(P_perm_profile[-1] - P_perm_profile[0]):.2f} Pa")


    profile = pd.DataFrame(data)

    x_ret  = profile.iloc[-1][[f"x{i+1}" for i in range(J)]].values
    y_perm = profile.iloc[0][[f"y{i+1}" for i in range(J)]].values
    Qr     = profile.iloc[-1]["Qr"]
    Qp     = profile.iloc[0]["Qp"]

    '''
    #plot pressure profiles on two y-axes
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(profile["norm_z"], profile["P_feed"], label="Retentate Pressure (Pa)", color='red')
    ax1.set_xlabel("Normalized Module Length")
    ax1.set_ylabel("Retentate Pressure (Pa)", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2 = ax1.twinx()
    ax2.plot(profile["norm_z"], profile["P_perm"], label="Permeate Pressure (Pa)", color='blue')
    ax2.set_ylabel("Permeate Pressure (Pa)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title("Pressure Profiles Along the Module")
    plt.show()
    '''
    

    '''
    #plot driving for component 1 accross the module
    plt.figure(figsize=(8, 5))
    plt.plot(profile["norm_z"], Membrane["Pressure_Feed"] * profile["x1"] - Membrane["Pressure_Permeate"] * profile["y1"], label="Driving Force (Pa)", color='blue')
    '''

    '''
    if Membrane["Pressure_Drop"]:
        P_profile = y_eval[2*J, :]
        print(f'Pressure drop across the module: {P_profile[-1] - P_profile[0]:.2f} Pa')
    '''
    CC_ODE_results = x_ret, y_perm, Qr, Qp
    return CC_ODE_results, profile