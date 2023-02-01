"""
© (or copyright) 2023. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.

2.Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.

3.Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import copy
from random import random

from JINAPyCEE import omega_plus

def linear_interp(array, time_array, time):
    '''
    Interpolate array at time "time"
    '''

    err_msg = f"Interpolation time {time:.2e}"
    err_msg2 = "than time_array"

    # Time greater than time_arr[-1]
    if time_array[-1] < time:
        s = err_msg + " greater " + err_msg2
        s += f"[-1], {time_array[-1]:.2e}"
        raise Exception(s)

    # Time lower than time_arr[0]
    if time_array[0] > time:
        s = err_msg + " lower " + err_msg2
        s += f"[0], {time_array[0]:.2e}"
        raise Exception(s)

    # Search for the time
    for ii in range(len(array)):
        if time_array[ii] >= time:
            time1 = time_array[ii - 1]
            time2 = time_array[ii]

            val1 = array[ii - 1]
            val2 = array[ii]

            # Interpolate linearly
            val = (time - time1)*(val2 - val1)/(time2 - time1) + val1
            return val

def run_omega(kwargs, param_vals, param_norms, time=8.4e9):
    '''Run omega with all the parameters'''

    # Recover parameters
    kwargs_alt = {}
    for key in param_vals:
        if key not in ["a1", "b1", "imf_yield_top"]:
            kwargs[key] = param_vals[key] * param_norms[key]
        else:
            kwargs_alt[key] = param_vals[key] * param_norms[key]

    # Define the inflow rates
    # [norm, t_max, timescale]
    exp_infall = [[kwargs_alt["a1"], 0.0, 0.68e9],
                  [kwargs_alt["b1"], 1.0e9, 7.0e9]] # this is good
    kwargs["exp_infall"] = exp_infall
    kwargs["imf_yields_range"] = [1, kwargs_alt["imf_yield_top"]]

    # Running omega
    op = omega_plus.omega_plus(**kwargs)

    # Extract values
    time_arr = op.inner.history.age
    sfr = op.inner.history.sfr_abs[-1]
    inflow_rate = op.inner.m_inflow_t[-1]/op.inner.history.timesteps[-1]
    m_gas = np.sum(op.inner.ymgal[-1])
    cc_sne_rate = op.inner.sn2_numbers[-1]/op.inner.history.timesteps[-1]
    Ia_sne_rate = op.inner.sn1a_numbers[-1]/op.inner.history.timesteps[-1]

    # Get the metallicity (mass fraction)
    metallicity = np.zeros(op.inner.nb_timesteps + 1)
    for i_t in range(op.inner.nb_timesteps + 1):
        m_Z = 0.0
        for iso, y in zip(op.inner.history.isotopes, op.inner.ymgal[i_t]):
            if iso.split("-")[0] not in ["H", "He", "Li"]:
                m_Z += y
        metallicity[i_t] = m_Z / np.sum(op.inner.ymgal[i_t])

    metallicity = linear_interp(metallicity, time_arr, time=time)

    # Get stellar mass
    m_star_lost = np.sum(np.sum(op.inner.mdot))
    stellar_mass = np.sum(op.inner.history.m_locked) - m_star_lost

    # Get abundances at time of sun formation
    masses = linear_interp(op.inner.ymgal, time_arr, time=time)
    m_tot = np.sum(masses)

    # Grab the iron
    name = "Fe"
    indices = [ii for ii, x in enumerate(op.inner.history.isotopes)
                    if x.split("-")[0] == name]
    FeMass = np.sum(masses[indices])
    XFe = FeMass/m_tot

    # Store values and return
    values = {}
    values["sfr"] = sfr
    values["stellar_mass"] = stellar_mass
    values["inflow_rate"] = inflow_rate
    values["m_gas"] = m_gas
    values["cc_sne_rate"] = cc_sne_rate
    values["Ia_sne_rate"] = Ia_sne_rate
    values["XFe"] = XFe
    values["metallicity"] = metallicity

    return op, values

def copy_yields(kwargs, op):
    '''
    Copy the yields from the op simulation to the kwargs dictionary
    '''

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
        lf=1e0, momentum=0.5, max_iter=10, time=8.4e9, max_lf_f=2,
        min_lf_f=1, period_lf=20):

    '''
    Calibrate omega using gradient descent
    '''

    # Save the learning factor
    lf0 = lf

    # Get the target solutions and deltas_deriv
    solutions = {}; deltas_deriv = {}
    for key in sol_ranges:
        solutions[key] = np.mean(sol_ranges[key])
    for key in param_vals:
        deltas_deriv[key] = lf * 1e-1

    best_solution = None
    best_parameters = None
    smallest_error = None
    prev_changes = {key: None for key in param_vals}

    # Perform gradient descent
    ii = 0
    while True:
        try:
            # Print current solution
            print("----------")
            print("Current solution")
            for key, val in values.items():
                sol = solutions[key]
                print(f"{key}: {val:.2e} - {sol:.2e}")

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
                rel_error[key] *= weights[key]/solutions[key]

                sum_err += rel_error[key]**2
                sum_weights += weights[key]**2

            error = sum_err/sum_weights
            if smallest_error is None or error < smallest_error:
                smallest_error = error
                best_solution = copy.copy(values)
                best_parameters = copy.copy(param_vals)

            print()
            print(f"Current error = {error:.4f}; threshold = {threshold:.4f}")
            print()
            if error < threshold:
                print("----------")
                print()
                print("Error threshold achieved")
                break

            # Open derivatives file
            deriv_file = "derivatives.txt"
            fwrite = open(deriv_file, "w")

            # If it is not good enough, calculate the derivatives
            param_cpy = copy.copy(param_vals)
            norm_gradient = 0
            derivs = {}
            for key in param_vals:
                derivs[key] = 0

                # If this parameter is fixed, do not change it
                if fix_params[key]:
                    continue

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
                for key2 in values:
                    # Do parametric derivative
                    der = (new_values[key2] - values[key2])/deltas_deriv[key]

                    # Save
                    val = der/param_norms[key]
                    s = f"Derivative of {key2} with {key} = {val:.2e}\n"
                    fwrite.write(s)

                    # Continue with relative error derivative
                    der *= rel_error[key2]*weights[key2]/solutions[key2]
                    derivs[key] += der
                derivs[key] *= 2
                norm_gradient += derivs[key]**2

                # Restore the previous value
                param_cpy[key] = param_vals[key]
                fwrite.write("=========\n")

            # Close derivatives file
            fwrite.close()

            # And multiply by factors
            changes = {}
            for key in derivs:
                changes[key] = derivs[key] * lf
                if prev_changes[key] is not None:
                    changes[key] *= (1 - momentum)

            # Re-calculate deltas
            for key in deltas_deriv:
                deltas_deriv[key] = abs(changes[key]) * lf * 1e-1
                deltas_deriv[key] = max(deltas_deriv[key],
                        abs(1e-1 * lf * param_vals[key]))

            # Now calculate the new parameters and do the new run
            print("Calculating next solution")
            for key in derivs:
                # calculate change with momentum
                if prev_changes[key] is not None:
                    changes[key] += momentum * prev_changes[key]

                # Apply change
                param_vals[key] -= changes[key]

            # Store change
            prev_changes = changes

            # Run omega with the new parameters
            op, values = run_omega(kwargs, param_vals, param_norms)
            kwargs_yields = copy_yields(kwargs, op)

            # Change learning factor with the period given by the user
            if ii%period_lf < period_lf/2:
                lf = 2*(max_lf_f - min_lf_f)/period_lf*(ii%period_lf) + min_lf_f
            else:
                lf = 2*(min_lf_f - max_lf_f)/period_lf*(ii%period_lf - period_lf/2) + max_lf_f
            lf *= lf0

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
    with open("output.txt", "w") as fwrite:
        print()
        print("----------")
        s = f"Best solution with error: {smallest_error:.4f}"
        print(s)
        fwrite.write(s + "\n")

        values = best_solution
        for key, val in values.items():
            s = f"{key} = {val:.2e}"

            print(s)
            fwrite.write(s + "\n")

        print()
        s = "Best parameters"
        print(s)
        fwrite.write("\n" + s + "\n")

        param_vals = best_parameters
        param_vals["t_sun"] = time
        param_norms["t_sun"] = 1
        for key in param_vals:
            val = param_vals[key]*param_norms[key]
            s = f"{key} = {val:.2e}"

            print(s)
            fwrite.write(s + "\n")

        print()
        print(f"Iterations: {ii}")

# Parameters for the machine-learning

# Maximum relative error
threshold = 5e-4 # Maximum relative error

# Maximum parameter step when error = 1
learning_factor = 1e-4

# Momentum of the descent
momentum = 0.90

# Maximum iterations before the program exits
max_iter = 10

# Define omega arguments
#table = 'yield_tables/AK_stable.txt'
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

kwargs["special_timesteps"] = 300
#kwargs["dt"] = 5e8

# Weight dictionary
weights = {}
weights["sfr"] = 1
weights["stellar_mass"] = 1
weights["inflow_rate"] = 1
weights["m_gas"] = 1
weights["cc_sne_rate"] = 1
weights["Ia_sne_rate"] = 1
weights["XFe"] = 1
weights["metallicity"] = 4

# Define ranges of solutions. The solution must fall inside these ranges
sol_ranges = {}
sol_ranges["sfr"] = [2.5, 3.5]
sol_ranges["stellar_mass"] = [3.0e10, 4.0e10]
sol_ranges["inflow_rate"] = [0.1, 1.1]
sol_ranges["m_gas"] = [1.21e10, 1.31e10]
sol_ranges["cc_sne_rate"] = [2.5e-2, 3.5e-2]
sol_ranges["Ia_sne_rate"] = [5e-3, 7e-3]
sol_ranges["XFe"] = [1.28e-3, 1.32e-3]
sol_ranges["metallicity"] = [0.0152, 0.0154]

# Define ranges of parameters. These give an idea to the code on the scale
# of the changes it should expect, but they are not hard limits.
param_ranges = {}
param_ranges["a1"] = [0, 150]
param_ranges["b1"] = [0, 15]
param_ranges["imf_yield_top"] = [20, 50]
param_ranges["sfe"] = [1e-10, 1e-9]
param_ranges["mass_loading"] = [0, 2]
param_ranges["nb_1a_per_m"] = [5e-4, 2e-3]

# Initial values (guess for parameter values)
param_vals = {}
param_vals["a1"] = 52.3
param_vals["b1"] = 3.47
param_vals["imf_yield_top"] = 36.0
param_vals["sfe"] = 1.65e-10
param_vals["mass_loading"] = 0.170
param_vals["nb_1a_per_m"] = 1.59e-3

# Whether to fix a parameter so it does not change
fix_params = {}
fix_params["a1"] = False
fix_params["b1"] = False
fix_params["imf_yield_top"] = False
fix_params["sfe"] = False
fix_params["mass_loading"] = False
fix_params["nb_1a_per_m"] = False

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
        param_norms, sol_ranges, fix_params, threshold=threshold,\
        lf=learning_factor, momentum=momentum, max_iter=max_iter, time=time,\
        max_lf_f=2, min_lf_f=0.5, period_lf=20)
