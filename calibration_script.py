import matplotlib.pyplot as plt
import numpy as np
import copy

from JINAPyCEE import omega_plus

def get_value(array, time_array, time = None):
    '''Interpolate array at time "time"'''
    if time is None:
        print("Please, provide an age for calculation. Returning 0")
        return 0

    # Search for the time
    for ii in range(len(array)):
        if time_array[ii] >= time:
            time1 = time_array[ii - 1]
            time2 = time_array[ii]

            val1 = array[ii - 1]
            val2 = array[ii]

            break

    # Interpolate linearly
    val = (time - time1)*(val2 - val1)/(time2 - time1) + val1
    return val

def run_omega(kwargs, param_vals, param_norms, time = 8.4e9):
    '''Run omega with all the parameters'''

    # Recover parameters
    a1 = param_vals["a1"]*param_norms["a1"]
    b1 = param_vals["b1"]*param_norms["b1"]
    imf_yield_top = param_vals["imf_yield_top"]*param_norms["imf_yield_top"]
    sfe = param_vals["sfe"]*param_norms["sfe"]
    mass_loading = param_vals["mass_loading"]*param_norms["mass_loading"]

    # Define the inflow rates
    # [norm, t_max, timescale]
    exp_infall = [[a1, 0.0, 0.68e9], [b1, 1.0e9, 7.0e9]] # this is good
    kwargs["imf_yields_range"] = [1, imf_yield_top]
    kwargs["sfe"] = sfe
    kwargs["mass_loading"] = mass_loading
    kwargs["exp_infall"] = exp_infall

    # Running omega
    op = omega_plus.omega_plus(**kwargs)

    # Extract values
    sfr = op.inner.history.sfr_abs[-1]
    inflow_rate = op.inner.m_inflow_t[-1]/op.inner.history.timesteps[-1]
    m_gas = np.sum(op.inner.ymgal[-1])
    cc_sne_rate = op.inner.sn2_numbers[-1]/op.inner.history.timesteps[-1]
    Ia_sne_rate = op.inner.sn1a_numbers[-1]/op.inner.history.timesteps[-1]
    metallicity = get_value(op.inner.history.metallicity, op.inner.history.age,
                            time = time)

    time_arr, feh_arr = op.inner.plot_spectro(solar_norm='Asplund_et_al_2009', return_x_y=True)
    FeH = get_value(feh_arr, time_arr, time = time)

    # Store values and return
    values = {}
    values["sfr"] = sfr
    values["inflow_rate"] = inflow_rate
    values["m_gas"] = m_gas
    values["cc_sne_rate"] = cc_sne_rate
    values["Ia_sne_rate"] = Ia_sne_rate
    values["FeH"] = FeH
    values["metallicity"] = metallicity

    return op, values

def copy_yields(kwargs, op):
    '''Copy the yields from the op simulation to the kwargs dictionary'''

    kwargs_yields = copy.deepcopy(kwargs)

    kwargs_yields["input_yields"] = True
    kwargs_yields["isotopes_in"] = op.inner.history.isotopes
    kwargs_yields["ytables_in"] = op.inner.ytables
    kwargs_yields["ytables_1a_in"] = op.inner.ytables_1a
    kwargs_yields["ytables_nsmerger_in"] = op.inner.ytables_nsmerger
    kwargs_yields["inter_Z_points"] = op.inner.inter_Z_points
    kwargs_yields["nb_inter_Z_points"] = op.inner.nb_inter_Z_points
    kwargs_yields["y_coef_M"] = op.inner.y_coef_M
    kwargs_yields["y_coef_M_ej"] = op.inner.y_coef_M_ej
    kwargs_yields["y_coef_Z_aM"] = op.inner.y_coef_Z_aM
    kwargs_yields["y_coef_Z_bM"] = op.inner.y_coef_Z_bM
    kwargs_yields["y_coef_Z_bM_ej"] = op.inner.y_coef_Z_bM_ej
    kwargs_yields["tau_coef_M"] = op.inner.tau_coef_M
    kwargs_yields["tau_coef_M_inv"] = op.inner.tau_coef_M_inv
    kwargs_yields["tau_coef_Z_aM"] = op.inner.tau_coef_Z_aM
    kwargs_yields["tau_coef_Z_bM"] = op.inner.tau_coef_Z_bM
    kwargs_yields["tau_coef_Z_aM_inv"] = op.inner.tau_coef_Z_aM_inv
    kwargs_yields["tau_coef_Z_bM_inv"] = op.inner.tau_coef_Z_bM_inv
    kwargs_yields["y_coef_M_pop3"] = op.inner.y_coef_M_pop3
    kwargs_yields["y_coef_M_ej_pop3"] = op.inner.y_coef_M_ej_pop3
    kwargs_yields["tau_coef_M_pop3"] = op.inner.tau_coef_M_pop3
    kwargs_yields["tau_coef_M_pop3_inv"] = op.inner.tau_coef_M_pop3_inv
    kwargs_yields["inter_lifetime_points_pop3"] = op.inner.inter_lifetime_points_pop3
    kwargs_yields["inter_lifetime_points_pop3_tree"] = op.inner.inter_lifetime_points_pop3_tree
    kwargs_yields["nb_inter_lifetime_points_pop3"] = op.inner.nb_inter_lifetime_points_pop3
    kwargs_yields["inter_lifetime_points"] = op.inner.inter_lifetime_points
    kwargs_yields["inter_lifetime_points_tree"] = op.inner.inter_lifetime_points_tree
    kwargs_yields["nb_inter_lifetime_points"] = op.inner.nb_inter_lifetime_points
    kwargs_yields["nb_inter_M_points_pop3"] = op.inner.nb_inter_M_points_pop3
    #kwargs_yields["inter_M_points_pop3"] = op.inner.inter_M_points_pop3
    #kwargs_yields["inter_M_points_pop3_tree"] = op.inner.inter_M_points_pop3_tree
    kwargs_yields["nb_inter_M_points"] = op.inner.nb_inter_M_points
    kwargs_yields["inter_M_points"] = op.inner.inter_M_points
    #kwargs_yields["inter_M_points_tree"] = op.inner.inter_M_points_tree
    kwargs_yields["y_coef_Z_aM_ej"] = op.inner.y_coef_Z_aM_ej

    return kwargs_yields

def run_calibration(kwargs, kwargs_yields, weights, values, param_vals,\
        param_norms, sol_ranges, fix_params, threshold=2e-2,\
        learning_factor=1e0, max_iter=10, time=8.4e9):
    '''Calibrate omega using gradient descent'''

    # Get the target solutions and deltas_deriv
    solutions = {}; deltas_deriv = {}
    for key in sol_ranges:
        solutions[key] = np.mean(sol_ranges[key])
    for key in param_vals:
        deltas_deriv[key] = learning_factor*1e-1

    best_solution = None
    best_parameters = None
    smallest_error = None

    # Perform gradient descent
    ii = 0
    while True:
        try:
            # Print current solution
            print("----------")
            print("Current solution")
            for key, val in values.items():
                print(f"{key}: {val:.2e}")

            print()
            print("Current parameters")
            for key in param_vals:
                val = param_vals[key]*param_norms[key]
                print(f"{key}: {val:.2e}")

            # Check the solution to see if it's good enough
            rel_error = {}
            sum_err = 0; sum_weights = 0
            for key in values:
                rel_error[key] = (values[key] - solutions[key])

                # Because the solution of FeH is 0, we do not divide by it.
                if key == "FeH":
                    sol = 1
                else:
                    sol = solutions[key]

                rel_error[key] *= weights[key]/sol

                sum_err += rel_error[key]**2
                sum_weights += weights[key]

            error = sum_err/sum_weights
            if smallest_error is None or error < smallest_error:
                smallest_error = error
                best_solution = copy.copy(values)
                best_parameters = copy.copy(param_vals)

            print()
            print(f"Current error = {error:.3f}; threshold = {threshold:.3f}")
            print()
            if error < threshold:
                print("----------")
                print()
                print("Error threshold achieved")
                break

            # If it is not good enough, calculate the derivatives
            derivatives = {}; maxDeriv = 0
            param_cpy = copy.copy(param_vals)
            for key in param_vals:
                print(f"Derivating parameter {key}")

                # Change only one parameter
                param_cpy[key] += deltas_deriv[key]
                if key == "imf_yield_top":
                    op, new_values = run_omega(kwargs, param_cpy,\
                                               param_norms, time=time)
                else:
                    op, new_values = run_omega(kwargs_yields, param_cpy,\
                                               param_norms, time=time)

                # Calculate derivative
                # This array holds the derivative of all the values (sfr, inflow...)
                # with respect to the ii-th parameter (a1, b1, imf_yield_top, sfe...)
                derivs = []
                for key2 in values:
                    # If this parameter is fixed, do not change it
                    if fix_params[key]:
                        derivs = [0]
                        break

                    # Because the solution of FeH is 0, we do not divide by it.
                    if key2 == "FeH":
                        sol = 1
                    else:
                        sol = solutions[key2]

                    der = 2*rel_error[key2]*weights[key2]/sol
                    der *= (new_values[key2] - values[key2])/deltas_deriv[key]
                    derivs.append(der)

                # We are interested in the average of the derivative
                derivs = np.array(derivs)
                derivatives[key] = np.mean(derivs)
                maxDeriv = max([max(abs(derivs)), maxDeriv])

                # Restore the previous value
                param_cpy[key] = param_vals[key]
            # Now make sure that the total change in parameter space is lower than
            # the learning_factor
            norm = 0
            for key in derivatives:
                norm += derivatives[key]**2
            norm = np.sqrt(norm)
            max_move = learning_factor*error/maxDeriv
            if norm > max_move:
                for key in derivatives:
                    derivatives[key] *= max_move/norm

            # Also make sure that no parameter goes below zero
            move_no_zero = min(max_move, norm)
            for key in derivatives:
                if param_vals[key] - derivatives[key] < 0:
                    if param_vals[key] < move_no_zero:
                        move_no_zero = param_vals[key]
            for key in derivatives:
                derivatives[key] *= move_no_zero/min(max_move, norm)

            # Re-scale the deltas_deriv
            for key in derivatives:
                deltas_deriv[key] = derivatives[key]*1e-1

            # Now finally calculate the new parameters and do the new run
            print("Calculating next solution")
            for key in derivatives:
                param_vals[key] -= derivatives[key]
            op, values = run_omega(kwargs, param_vals, param_norms)
            kwargs_yields = copy_yields(kwargs, op)

            ii += 1
            if (ii > max_iter):
                print("----------")
                print()
                print("Maximum of iterations reached")
                break

        except KeyboardInterrupt:
            break
        except:
            raise

    # Print best solution
    print()
    print("----------")
    print(f"Best solution with error: {smallest_error:.4f}")
    values = best_solution
    for key, val in values.items():
        print(f"{key}: {val:.2e}")

    print()
    print("Best parameters")
    param_vals = best_parameters
    for key in param_vals:
        val = param_vals[key]*param_norms[key]
        print(f"{key}: {val:.2e}")

# Parameters for the machine-learning
threshold = 1e-2 # Maximum relative error
learning_factor = 1e0 # Controls the maximum parameter step (does not need to be < 1)
max_iter = 200 # Maximum iterations before the program exits

# Define omega arguments
table = 'yield_tables/AK_stable.txt'
#table = 'yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt'
table = 'yield_tables/agb_and_massive_stars_K10_K06_0.5HNe.txt'

# Immutable
kwargs = {}
kwargs["Z_trans"] = -1
kwargs["t_star"] = 1.0
kwargs["table"] = table
kwargs["mgal"] = 1.0
kwargs["m_DM_0"] = 1.0e12
kwargs["sn1a_rate"] = 'power_law'
kwargs["print_off"] = True

kwargs["special_timesteps"] = 150
#kwargs["dt"] = 5e8

# Weight dictionary
weights = {}
weights["sfr"] = 1
weights["inflow_rate"] = 0.5
weights["m_gas"] = 1
weights["cc_sne_rate"] = 1
weights["Ia_sne_rate"] = 1
weights["FeH"] = 0
weights["metallicity"] = 10

# Define ranges of solutions. The solution must fall inside these ranges
sol_ranges = {}
sol_ranges["sfr"] = [0.65, 3.00]
sol_ranges["inflow_rate"] = [0.6, 1.6]
sol_ranges["m_gas"] = [3.6e9, 1.26e10]
sol_ranges["cc_sne_rate"] = [1e-2, 3e-2]
sol_ranges["Ia_sne_rate"] = [2e-3, 6e-3]
sol_ranges["FeH"] = [-0.001, 0.001]
sol_ranges["metallicity"] = [0.0135, 0.0145]

# Define ranges of parameters. These give an idea to the code on the scale
# of the changes it should expect, but they are not hard limits.
param_ranges = {}
param_ranges["a1"] = [0, 150]
param_ranges["b1"] = [0, 15]
param_ranges["imf_yield_top"] = [20, 50]
param_ranges["sfe"] = [1e-10, 1e-9]
param_ranges["mass_loading"] = [0, 2]

# Initial values (guess for parameter values)
param_vals = {}
param_vals["a1"] = 22.9
param_vals["b1"] = 6.86
param_vals["imf_yield_top"] = 47.8
param_vals["sfe"] = 1.94e-10
param_vals["mass_loading"] = 0.671

# Whether to fix a parameter so it does not change
fix_params = {}
fix_params["a1"] = False
fix_params["b1"] = False
fix_params["imf_yield_top"] = False
fix_params["sfe"] = False
fix_params["mass_loading"] = False

# -------------- Do not change anything below this line -------------------

# Define array of parameters
param_norms = {}
for key in param_ranges:
    lst = param_ranges[key]
    param_norms[key] = lst[1] - lst[0]
    param_vals[key] /= param_norms[key]

# Run initial omega
op, values = run_omega(kwargs, param_vals, param_norms)

# Before continuing, copy the yield tables
kwargs_yields = copy_yields(kwargs, op)

# Now calibrate:
time = 8.4e9
run_calibration(kwargs, kwargs_yields, weights, values, param_vals,\
        param_norms, sol_ranges, fix_params, threshold = threshold,\
        learning_factor = learning_factor, max_iter = max_iter, time = time)
