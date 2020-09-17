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

def run_omega(kwargs, param_vals, param_norms):
    '''Run omega with all the parameters'''

    # Recover parameters
    a1 = param_vals["a1"]*param_norms["a1"]
    b1 = param_vals["b1"]*param_norms["b1"]
    range2 = param_vals["range2"]*param_norms["range2"]
    sfe = param_vals["sfe"]*param_norms["sfe"]
    mass_loading = param_vals["mass_loading"]*param_norms["mass_loading"]

    # Define the inflow rates
    # [norm, t_max, timescale]
    exp_infall = [[a1, 0.0, 0.68e9], [b1, 1.0e9, 7.0e9]] # this is good
    kwargs["imf_yields_range"] = [1, range2]
    kwargs["sfe"] = sfe
    kwargs["mass_loading"] = mass_loading
    kwargs["exp_infall"] = exp_infall

    # Running omega
    op = omega_plus.omega_plus(**kwargs)

    # Extract values
    sfr = op.inner.sfr_abs[-1]
    inflow_rate = op.inner.m_inflow_t[-1]/op.inner.history.timesteps[-1]
    m_gas = np.sum(op.inner.ymgal[-1])
    cc_sne_rate = op.inner.sn2_numbers[-1]/op.inner.history.timesteps[-1]
    Ia_sne_rate = op.inner.sn1a_numbers[-1]/op.inner.history.timesteps[-1]
    metallicity = get_value(op.inner.history.metallicity, op.inner.history.age,
                            time = 8.4e9)

    values = {}
    values["sfr"] = sfr
    values["inflow_rate"] = inflow_rate
    values["m_gas"] = m_gas
    values["cc_sne_rate"] = cc_sne_rate
    values["Ia_sne_rate"] = Ia_sne_rate
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
    kwargs_yields["inter_M_points_pop3"] = op.inner.inter_M_points_pop3
    kwargs_yields["inter_M_points_pop3_tree"] = op.inner.inter_M_points_pop3_tree
    kwargs_yields["nb_inter_M_points"] = op.inner.nb_inter_M_points
    kwargs_yields["inter_M_points"] = op.inner.inter_M_points
    kwargs_yields["inter_M_points_tree"] = op.inner.inter_M_points_tree
    kwargs_yields["y_coef_Z_aM_ej"] = op.inner.y_coef_Z_aM_ej

    return kwargs_yields

def run_calibration(kwargs, kwargs_yields, weights, values, param_vals,\
        param_norms, sol_ranges, threshold = 3e-2, learning_factor = 8e-1,\
        max_iter = 10):
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
            # Current solution
            print("----------")
            print("Current solution")
            for key, val in values.items():
                print(key + ": {:.2e}".format(val))

            print()
            print("Current parameters")
            for key in param_vals:
                val = param_vals[key]*param_norms[key]
                print(key + ": {:.2e}".format(val))

            # Check the solution to see if it's good enough
            rel_error = {}
            sum_err = 0; sum_weights = 0
            for key in values:
                rel_error[key] = (values[key] - solutions[key])
                rel_error[key] *= weights[key]/solutions[key]

                sum_err += rel_error[key]**2
                sum_weights += weights[key]

            error = sum_err/sum_weights
            if smallest_error is None or error < smallest_error:
                smallest_error = error
                best_solution = copy.copy(values)
                best_parameters = copy.copy(param_vals)

            print()
            print("Current error = {}; threshold = {}".format(error, threshold))
            print()
            if error < threshold:
                break

            # If it is not good enough, calculate the derivatives
            derivatives = {}; maxDeriv = 0
            for key in param_vals:
                print("Derivating parameter {}".format(key))

                # Change only one parameter
                param_cpy = copy.copy(param_vals)
                param_cpy[key] += deltas_deriv[key]
                if key == "range2":
                    op, new_values = run_omega(kwargs, param_cpy,\
                                               param_norms)
                else:
                    op, new_values = run_omega(kwargs_yields, param_cpy,\
                                               param_norms)

                # Calculate derivative
                # This array holds the derivative of all the values (sfr, inflow...)
                # with respect to the ii-th parameter (a1, b1, range2, sfe...)
                derivs = []
                for key2 in values:
                    der = 2*rel_error[key2]*weights[key2]/solutions[key2]
                    der *= (new_values[key2] - values[key2])/deltas_deriv[key]
                    derivs.append(der)

                # We are interested in the average of the derivative
                derivs = np.array(derivs)
                derivatives[key] = np.mean(derivs)
                maxDeriv = max([max(abs(derivs)), maxDeriv])

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
                break

        except KeyboardInterrupt:
            break
        except:
            raise

    # Best solution
    print()
    print("----------")
    print("Best solution with error: {:.4f}".format(smallest_error))
    values = best_solution
    for key, val in values.items():
        print(key + ": {:.2e}".format(val))

    print()
    print("Best parameters")
    param_vals = best_parameters
    for key in param_vals:
        val = param_vals[key]*param_norms[key]
        print(key + ": {:.2e}".format(val))

# Parameters for the machine-learning
threshold = 3e-2 # Maximum relative error
learning_factor = 2e0 # Controls the maximum parameter step (does not need to be < 1)
max_iter = 100 # Maximum iterations before the program exits

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
weights["inflow_rate"] = 0.1
weights["m_gas"] = 2
weights["cc_sne_rate"] = 1
weights["Ia_sne_rate"] = 1
weights["metallicity"] = 5

# Define ranges of solutions. The solution must fall inside these ranges
sol_ranges = {}
sol_ranges["sfr"] = [0.65, 3.00]
sol_ranges["inflow_rate"] = [0.6, 1.6]
sol_ranges["m_gas"] = [3.6e9, 12.6e9]
sol_ranges["cc_sne_rate"] = [1e-2, 3e-2]
sol_ranges["Ia_sne_rate"] = [2e-3, 6e-3]
sol_ranges["metallicity"] = [0.0135, 0.0145]

# Define ranges of parameters. These give an idea to the code on the scale
# of the changes it should expect, but they are not hard limits.
param_ranges = {}
param_ranges["a1"] = [0, 150]
param_ranges["b1"] = [0, 15]
param_ranges["range2"] = [20, 50]
param_ranges["sfe"] = [1e-10, 1e-9]
param_ranges["mass_loading"] = [0, 2]

# Initial values (guess for parameter values)
param_vals = {}
param_vals["a1"] = 46
param_vals["b1"] = 5.4
param_vals["range2"] = 50
param_vals["sfe"] = 2.5e-10
param_vals["mass_loading"] = 0.5

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
run_calibration(kwargs, kwargs_yields, weights, values, param_vals,\
        param_norms, sol_ranges, threshold = threshold,\
        learning_factor = learning_factor, max_iter = max_iter)
