import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import warnings


def mass_balance_CO_ODE(vars):

    Membrane, Component_properties, Fibre_Dimensions = vars

    Membrane["Total_Flow"] = Membrane["Feed_Flow"] + Membrane["Sweep_Flow"]
    Fibre_Dimensions["Number_Fibre"] = Membrane["Area"] / (
        Fibre_Dimensions["Length"] * math.pi * Fibre_Dimensions["D_out"]
    )

    epsilon = 1e-10
    J = len(Membrane["Feed_Composition"])
    L = Fibre_Dimensions["Length"]

    Membrane["Feed_Composition"]  = np.array(Membrane["Feed_Composition"])
    Membrane["Sweep_Composition"] = np.array(Membrane["Sweep_Composition"])

    # ------------------------------------------------------------------ #                                
    #  var[0:J]     = x[i]    retentate mole fractions                   #
    #  var[J:2J]    = y[i]    permeate  mole fractions                   #
    #  var[2J]      = Q_ret   total retentate flow (normalised)          #
    #  var[2J+1]    = Q_perm  total permeate  flow (normalised)          #
    # ------------------------------------------------------------------ #
    def membrane_odes(z, var):

        x     = var[:J]
        y     = var[J:2*J]
        Q_ret = max(var[2*J],   epsilon)
        Q_perm= max(var[2*J+1], epsilon)

        P_perm = Membrane["Pressure_Permeate"]
        Pf     = Membrane["Pressure_Feed"]
        A      = Fibre_Dimensions["D_out"] * math.pi * Fibre_Dimensions["Number_Fibre"] #total membrane area
        Ttot   = Membrane["Total_Flow"]

        permeance = np.array(Membrane["Permeance"])

        # component permeation flux (normalised)
        J_perm = (permeance * A / Ttot) * (Pf * x - P_perm * y)  # shape (J,)

        # total flow derivatives
        dQ_ret_dz  = -np.sum(J_perm)
        dQ_perm_dz =  np.sum(J_perm)

        # mole fraction derivatives
        # d(x[i])/dz = (-J_perm[i] - x[i]*dQ_ret_dz) / Q_ret
        dx_dz = (-J_perm - x * dQ_ret_dz) / Q_ret

        # d(y[i])/dz = ( J_perm[i] - y[i]*dQ_perm_dz) / Q_perm
        dy_dz = ( J_perm - y * dQ_perm_dz) / Q_perm

        return np.concatenate([dx_dz, dy_dz, [dQ_ret_dz, dQ_perm_dz]])

    # ------------------------------------------------------------------ #
    #  Initial conditions at z=0                                          #
    # ------------------------------------------------------------------ #
    Q_feed  = Membrane["Feed_Flow"]  / Membrane["Total_Flow"]
    Q_sweep = Membrane["Sweep_Flow"] / Membrane["Total_Flow"]

    x0 = Membrane["Feed_Composition"].copy()
    y0 = Membrane["Sweep_Composition"].copy()

    # guard against zero sweep
    if Q_sweep < epsilon:
        y0 = x0.copy()   # dummy composition, will not affect results
        Q_sweep = epsilon

    boundary = np.concatenate([x0, y0, [Q_feed, Q_sweep]])

    # ------------------------------------------------------------------ #
    #  Solve                                                               #
    # ------------------------------------------------------------------ #
    t_span = [0, L]
    t_eval = np.linspace(0, L, 500)

    def retentate_exhausted(z, var): # event to stop integration if retentate flow goes to zero (full permeation)
        return var[2*J] - 1e-4  # Q_ret normalised

    retentate_exhausted.terminal = True
    retentate_exhausted.direction = -1

    solution = solve_ivp(
        membrane_odes, t_span, y0=boundary,
        method='BDF', t_eval=t_eval,
        rtol=1e-6, atol=1e-8,
        events=retentate_exhausted
    )

    if not solution.success:
        print(f"Warning: {solution.message}")

    # ------------------------------------------------------------------ #
    #  Extract profiles                                                    #
    # ------------------------------------------------------------------ #
    z_points      = solution.t
    z_norm        = z_points / L

    x_profiles    = solution.y[:J,    :]   # mole fractions, shape (J, n_pts)
    y_profiles    = solution.y[J:2*J, :]
    Qr_profile    = solution.y[2*J,   :] * Membrane["Total_Flow"]   # unnormalise
    Qp_profile    = solution.y[2*J+1, :] * Membrane["Total_Flow"]

    # clip compositions to [0, 1] — small numerical violations possible
    x_profiles = np.clip(x_profiles, 0, 1)
    y_profiles = np.clip(y_profiles, 0, 1)

    # renormalise rows to sum to 1
    x_profiles = x_profiles / (np.sum(x_profiles, axis=0) + epsilon)
    y_profiles = y_profiles / (np.sum(y_profiles, axis=0) + epsilon)

    # ------------------------------------------------------------------ #
    #  Build profile DataFrame                                             #
    # ------------------------------------------------------------------ #
    data = {
        "norm_z": z_norm,
        **{f"x{i+1}": x_profiles[i, :] for i in range(J)},
        **{f"y{i+1}": y_profiles[i, :] for i in range(J)},
        "Qr": Qr_profile,
        "Qp": Qp_profile,
    }
    profile = pd.DataFrame(data)

    # ------------------------------------------------------------------ #
    #  Outlet values                                                       #
    # ------------------------------------------------------------------ #
    x_ret  = profile.iloc[-1][[f"x{i+1}" for i in range(J)]].values
    y_perm = profile.iloc[-1][[f"y{i+1}" for i in range(J)]].values
    Qr     = profile.iloc[-1]["Qr"]
    Qp     = profile.iloc[-1]["Qp"]

    CO_ODE_results = x_ret, y_perm, Qr, Qp
    return CO_ODE_results, profile