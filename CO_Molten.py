import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar
import pandas as pd
import math

def mass_balance_CO_Molten(vars):
    print('test')
    Membrane, Component_properties, Fibre_Dimensions = vars

    Total_flow = Membrane["Feed_Flow"] + Membrane["Sweep_Flow"]
    cut_r_N = Membrane["Feed_Flow"] / Total_flow
    cut_p_N = Membrane["Sweep_Flow"] / Total_flow

    J = len(Membrane["Feed_Composition"])
    min_elements = [3]
    for i in range(J):
        N_i = (Membrane["Feed_Flow"] * (1 - Membrane["Feed_Composition"][i] + 0.005) * Membrane["Permeance"][i] * Membrane["Pressure_Feed"] * Membrane["Feed_Composition"][i]) / (Membrane["Feed_Flow"] * 0.005)
        min_elements.append(N_i)
    n_elements = min(round(max(min_elements)), 1000)

    DA = Membrane["Area"] / n_elements

    user_vars = DA, J, Total_flow, Membrane["Pressure_Feed"], Membrane["Permeance"]

    def mixture_visc(composition):
        y = composition
        visc = np.zeros(J)
        for i, (slope, intercept) in enumerate(Component_properties["Viscosity_param"]):
            visc[i] = 1e-6 * (slope * Membrane["Temperature"] + intercept)

        Mw = Component_properties["Molar_mass"]
        phi = np.zeros((J, J))
        for i in range(J):
            for j in range(J):
                if i != j:
                    phi[i][j] = ((1 + (visc[i]/visc[j])**0.5 * (Mw[j]/Mw[i])**0.25)**2) / ((8 * (1 + Mw[i]/Mw[j]))**0.5)
                else:
                    phi[i][j] = 1

        nu = np.zeros(J)
        for i in range(J):
            nu[i] = y[i] * visc[i] / sum(y[i] * phi[i][j] for j in range(J))

        visc_mix = sum(nu)
        return visc_mix

    def pressure_drop(composition, Q, P):
        visc_mix = mixture_visc(composition)
        D_in = Fibre_Dimensions["D_in"]
        Q = Q / Fibre_Dimensions['Number_Fibre']
        dL = Fibre_Dimensions['Length'] / n_elements
        R = 8.314
        dP = 8 * visc_mix / (math.pi * D_in**4) * Q * R * Membrane["Temperature"] / P * dL
        return dP

    def equations(vars, inputs, user_vars):
        DA, J, Total_flow, P_ret, Perm = user_vars

        x_known = inputs[0:J]
        y_known = inputs[J:2*J]
        cut_r_known = inputs[-3]
        cut_p_known = inputs[-2]
        P_perm = inputs[-1]

        Qr_known = cut_r_known * Total_flow
        Qp_known = cut_p_known * Total_flow

        x = vars[0:J]
        y = vars[J:2*J]
        cut_r = vars[-2]
        cut_p = vars[-1]

        Qr = cut_r * Total_flow
        Qp = cut_p * Total_flow

        eqs = [0]*(2*J+2)
        eqs[0] = sum(x) - 1
        eqs[1] = sum(y) - 1

        for i in range(J):
            eqs[i+2] = x_known[i] * cut_r_known + y_known[i] * cut_p_known - x[i] * cut_r - y[i] * cut_p

        Temperature = Membrane["Temperature"]

        for i in range(J):
            print(i)
            if i==0:
                print('Test1')
                P_CO2_ret = P_ret * x[i]
                P_CO2_perm = P_perm * y[i]
                if P_CO2_ret > 0 and P_CO2_perm > 0 and P_CO2_ret > P_CO2_perm:
                    J_CO2 = (3732.61 / Temperature) * math.exp(-9730 / Temperature) * math.log(P_CO2_ret /( P_CO2_perm + 1e-8 ))
                    print(J_CO2)
                else:
                    J_CO2 = 0
                eqs[i + 2 + J] = (y[i] * Qp - y_known[i] * Qp_known) - (J_CO2 * DA)
            else:
                eqs[i + 2 + J] = (y[i] * Qp - y_known[i] * Qp_known)

        print()
        return eqs

    columns = ['Element'] + [f'x{i+1}' for i in range(J)] + [f'y{i+1}' for i in range(J)] + ['cut_r/Qr', 'cut_p/Qp','P_Perm']
    Solved_membrane_profile = pd.DataFrame(index=range(n_elements), columns=columns)
    Solved_membrane_profile.loc[0] = [n_elements] + list(Membrane["Feed_Composition"]) + list(Membrane["Sweep_Composition"]) + [cut_r_N, cut_p_N, Membrane["Pressure_Permeate"]]

    for k in range(n_elements - 1):
        inputs = Solved_membrane_profile.loc[k, Solved_membrane_profile.columns[1:]].values
        guess = [0.5] * (2 * J + 2)

        sol_element = least_squares(
            equations,
            guess,
            args=(inputs, user_vars),
            method='trf',
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8
        )

        if not sol_element.success:
            raise ValueError(f"Mass balance solver failed at element {k}: {sol_element.message}")
        element_output = sol_element.x

        if sol_element.cost > 1e-5:
            print(f'Large mass balance closure error at element {k}; with residuals {sol_element.fun}')

        y_k = element_output[J:2*J]
        Qp_k = element_output[-1] * Total_flow
        pP_k = Solved_membrane_profile.loc[k, 'P_Perm']

        if not Membrane["Pressure_Drop"]:
            pP_new = pP_k
        else:
            dP = pressure_drop(y_k, Qp_k, pP_k)
            if dP / pP_k > 1e-4:
                pP_new = Membrane["Pressure_Permeate"] - dP
            else:
                pP_new = pP_k

        df_element = np.concatenate(([n_elements-1-k], element_output, [pP_new]))
        Solved_membrane_profile.loc[k + 1] = df_element

    x_ret = Solved_membrane_profile.iloc[-1, 1:J+1].values
    y_perm = Solved_membrane_profile.iloc[-1, J+1:2*J+1].values
    cut_r = Solved_membrane_profile.iloc[-1, -3]
    cut_p = Solved_membrane_profile.iloc[-1, -2]
    Qr = cut_r * Total_flow
    Qp = cut_p * Total_flow
    P_perm = Solved_membrane_profile.iloc[-1, -1] * 1e-5

    CO_results = x_ret, y_perm, Qr, Qp
    profile = Solved_membrane_profile.copy()

    return CO_results, profile

