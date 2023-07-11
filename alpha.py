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
from scipy import integrate
import os
import copy
import sys

# TODO
from JINAPyCEE import omega
import matplotlib.pyplot as plt

nupy_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
nupy_path = os.path.join(nupy_path, "NuPyCEE")

# To-do list:

# TODO Include more fine-grained control over which instances interact
# depending on what we need

class Alpha(object):

    '''
    This object controls the material flow between
    different omega and omega_plus instances

    '''

    def __init__(self, omega_list, omega_p=None, min_half_life=1000,
                 max_half_life=1e14):

        '''
        -omega_list is the list of omega instances
        -omega_p is the omega_plus instance, optional

        The following parameters are overriden by omega_p if the instance
        exists:

        -min_half_life is the minimum half-life considered for decays
        (lower than that is instantaneous)
        -max_half_life is the maximum half-life considered for decaus
        (larger than that is stable)

        '''

        # Grab the omega_list
        self.omega_list = omega_list

        # Create an omega example
        try:
            self.omega_ex = omega_list[0]
        except IndexError:
            s = "Alpha instance cannot be initialized with an empty"
            s += " omega_list."

            raise RuntimeError(s)

        # TODO Only one omega instance implemented
        self.one_instance_error()

        # And omega plus
        self.omega_p = omega_p

        # Create timestep array
        self.timesteps = None
        self.create_timestep_array()

        # Check that the omega instances are equal where it matters
        attr_tuple = (
                      "imf_rnd_sampling",
                      "m_pop_max",
                     )
        self.are_omegas_equal(attr_tuple)

        # Initialize rest of arrays
        self.sources = None
        self.mdots = None
        self.sources_outer = None
        self.sources_radio = None
        self.mdots_radio = None
        self.sources_outer_radio = None

        # Define use_radio
        self.use_radio = self.omega_ex.len_decay_file > 0
        self.use_radio = self.use_radio or self.omega_ex.use_decay_module

        if self.omega_p is not None:
            self.yr_in_s = self.omega_p.yr_in_s
        else:
            self.yr_in_s = 3.15569e7

        if self.omega_p is not None:
            self.min_half_life = self.omega_p.min_half_life
            self.max_half_life = self.omega_p.max_half_life
        else:
            self.min_half_life = min_half_life
            self.max_half_life = max_half_life

    # TODO
    def one_instance_error(self):
        '''
        Return a NotImplementedError for more than one omega instance
        '''

        # Right now omega_list must have only one omega instance
        if len(self.omega_list) > 1:
            s = "More than one omega instance not implemented yet"
            raise NotImplementedError(s)

    ##############################################
    #      check if all omega instances have     #
    #            the same attributes             #
    ##############################################
    def are_omegas_equal(self, attr_tuple):

        '''
        Some attributes must be the same among the omega instances.

        Return true if that is the case

        '''

        are_all_equal = True
        s = []
        for key in attr_tuple:
            if key not in self.omega_ex.__dict__:
                continue

            key_equal = all(self.omega_ex.__dict__[key] == x.__dict__[key]
                            for x in self.omega_list)

            if not key_equal:
                s.append(f"Omega instances are not equal in {key}")

            are_all_equal = are_all_equal and key_equal

        if not are_all_equal:
            raise RuntimeError("\n".join(s))

    ##############################################
    #         define a timestep array            #
    ##############################################
    def create_timestep_array(self):

        '''
        Create the timestep array and check that timesteps are equal
        accross all omega instances
        '''

        if self.timesteps is not None:
            return

        # Make sure they are all equal
        if len(self.omega_list) > 1:

            # Extract all timesteps
            timesteps = [x.history.timesteps for x in self.omega_list]

            same_timesteps = all((np.array_equal(timesteps[0], x) for x in timesteps))
            if not same_timesteps:
                s = "All omega instances must have the same timesteps"
                raise Exception(s)

            # If all are equal we can take the just one
            self.timesteps = timesteps[0]

        else:

            self.timesteps = self.omega_list[0].history.timesteps

    ##############################################
    #         Create reaction network            #
    ##############################################
    def __create_reac_dictionary(self):

        '''
        This function creates the network for the decays. It depends on which
        options are active for omega.

        '''

        # Reaction network dictionary
        self.reac_dictionary = {}

        # If there is not decay, return now
        if not self.use_radio:
            return

        # Dictionary to store all isotopes
        self.all_isotopes_names = []

        # This dictionary stores all stable isotopes that can
        # be modified by the present yields
        self.stable_decayed_isotopes = []

        # Dictionary from name to (z, n) and other direction
        zn_to_name = {}

        # Shorten the name for self.omega_ex.history.isotopes
        hist_isotopes = self.omega_ex.history.isotopes

        # Define secondary products depending on the reaction
        self.decay_secondary = [
                [], [], ["Nn-1"], ["H-1"], ["He-4"], ["Nn-1"], ["H-1"], ["He-4"],
                ["He-4"], ["Nn-1", "Nn-1"], ["H-1", "H-1"], ["He-4", "He-4"],
                ["He-4", "He-4"], ["Nn-1", "Nn-1"], ["Nn-1", "Nn-1", "Nn-1"],
                ["Nn-1", "Nn-1", "Nn-1", "Nn-1"], ["H-1", "H-1"],
                ["Nn-1", "He-4"], ["H-1", "He-4"], [], ["C-12"], []
                ]

        # Copy of radio_iso, to resize it
        cpy_radio_iso = list(self.omega_ex.radio_iso)

        # Load the multichannel or single channel network if needed
        # This is done by creating a list of all the decay channels for each of
        # the radioactive isotopes considered here
        if self.omega_ex.use_decay_module:

            # Buld zn <-> name dictionaries
            f_network = os.path.join("decay_data", self.omega_ex.f_network)
            # TODO change the way this is read
            with open(os.path.join(nupy_path, f_network), "r") as fread:
                # Skip header line
                fread.readline()

                # Now store the data
                for line in fread:
                    lnlst = line.split()
                    zz = int(lnlst[0]); aa = int(lnlst[1]); name = lnlst[2]

                    # Create name
                    name = name + "-" + lnlst[1]

                    nn = aa - zz
                    zn_to_name[(zz, nn)] = name

            # Get the decay information from the decay module
            # Store all the isotopic information
            prevZZ, prevNN = None, None
            for ii in range(len(self.omega_ex.decay_module.iso.reactions)):
                # Z and n of every isotope in the module
                zz = self.omega_ex.decay_module.iso.z[ii]
                nn = self.omega_ex.decay_module.iso.n[ii]

                # If not an element, break
                if (zz + nn) == 0:
                    break

                # Store name for these zz, nn
                name = zn_to_name[(zz, nn)]

                # Check if isomer
                if zz == prevZZ and nn == prevNN:
                    name = "*" + name
                    while name in self.all_isotopes_names:
                        name = "*" + name

                self.all_isotopes_names.append(name)
                prevZZ, prevNN = zz, nn

            # At this point we need to choose which isotopes to follow
            # We also store the reactions
            self.__choose_network(cpy_radio_iso, hist_isotopes)

            # We do not need to use the decay_module anymore, so let's
            # delete it for pickling and multiprocessing purposes
            for omega_i in self.omega_list:
                del(omega_i.decay_module)

        elif self.omega_ex.len_decay_file > 0:

            # The information stored in decay_info is...
            # decay_info[nb_radio_iso][0] --> Unstable isotope
            # decay_info[nb_radio_iso][1] --> Stable isotope where it decays
            # decay_info[nb_radio_iso][2] --> Mean-life (half-life/ln2)[yr]

            # Build the network
            for element in self.omega_ex.decay_info:

                # Get names for reaction
                target = elem[0]
                product = elem[1]
                rate = 1 / elem[2]

                # Add reaction
                reaction = Reaction(target, product, rate)

                if target in self.reac_dictionary:
                    self.reac_dictionary[target].append(reaction)
                else:
                    self.reac_dictionary[target] = [reaction]

                # Store stable product self.stable_decayed_isotopes
                if product in hist_isotopes:
                    if product not in self.stable_decayed_isotopes:
                        self.stable_decayed_isotopes.append(product)

        # Restore the (maybe) modified arrays, even if we are declaring them twice
        len_iso = len(cpy_radio_iso)
        zeroArr = np.zeros((self.omega_ex.nb_timesteps + 1, len_iso))

        for omega_i in self.omega_list:
            omega_i.radio_iso = np.array(cpy_radio_iso)
            omega_i.nb_radio_iso = len_iso

            omega_i.ymgal_radio = zeroArr.copy()
            omega_i.ymgal_massive_radio = zeroArr.copy()
            omega_i.ymgal_agb_radio = zeroArr.copy()
            omega_i.ymgal_1a_radio = zeroArr.copy()
            omega_i.ymgal_nsm_radio = zeroArr.copy()
            for ii in range(omega_i.nb_delayed_extra_radio):
                omega_i.ymgal_delayed_extra_radio[ii] = zeroArr.copy()

            omega_i.mdot_radio = zeroArr.copy()
            omega_i.mdot_massive_radio = zeroArr.copy()
            omega_i.mdot_agb_radio = zeroArr.copy()
            omega_i.mdot_1a_radio = zeroArr.copy()
            omega_i.mdot_nsm_radio = zeroArr.copy()
            for ii in range(self.omega_ex.nb_delayed_extra_radio):
                omega_i.mdot_delayed_extra_radio[ii] = zeroArr.copy()

        if self.omega_p is not None:
            self.omega_p.ymgal_outer_radio = zeroArr.copy()
            self.omega_p.ymgal_outer_massive_radio = zeroArr.copy()
            self.omega_p.ymgal_outer_agb_radio = zeroArr.copy()
            self.omega_p.ymgal_outer_1a_radio = zeroArr.copy()
            self.omega_p.ymgal_outer_nsm_radio = zeroArr.copy()
            self.omega_p.ymgal_outer_extra_radio = []
            for ii in range(self.omega_ex.nb_delayed_extra_radio):
                self.omega_p.ymgal_outer_extra_radio.append(zeroArr.copy())

        # Initialize the decay arrays
        nb_radio = self.omega_ex.nb_radio_iso
        nb_iso = len(self.stable_decayed_isotopes)
        self.decay_from_radio = np.zeros(nb_radio)           # dd_radio
        self.decay_to_radio = np.zeros((nb_radio, nb_radio)) # pp_radio
        self.decay_to_stable = np.zeros((nb_iso, nb_radio))  # pp

        # Now for each reaction, add to the decay arrays
        for target in self.reac_dictionary:

            # Get target index
            targ_index = cpy_radio_iso.index(target)

            # Get the inverse of the atomic mass of the target
            targ_inv_AA = 1/float(target.split("-")[1])

            for reac in self.reac_dictionary[target]:

                # Add rate for dd_radio
                self.decay_from_radio[targ_index] += reac.rate

                # Modify rate by dividing for target mass fraction
                rate_mod_1 = reac.rate * targ_inv_AA

                # Select products list
                products = reac.products

                # Rate for pp_radio or pp
                for ii, product in enumerate(products):
                    product = products[ii]
                    prodAA = float(product.split("-")[1])

                    # We have to divide the rate by the atomic mass of the
                    # target and multiply by the atomic mass of the product
                    # to conserve the total mass
                    rate_mod_2 = rate_mod_1 * prodAA

                    if product in cpy_radio_iso:
                        prod_index = cpy_radio_iso.index(product)
                    else:
                        prod_index = self.stable_decayed_isotopes.index(product)

                    if product in cpy_radio_iso:
                        self.decay_to_radio[prod_index][targ_index] += rate_mod_2
                    else:
                        self.decay_to_stable[prod_index][targ_index] += rate_mod_2

    ##############################################
    #        Chose the unstable and stable       #
    #       isotopes to follow based on the      #
    #                 yield tables               #
    ##############################################
    def __choose_network(self, cpy_radio_iso, hist_isotopes, can_skip=True):

        '''
        This function populates the unstable and stable
        isotopes based on the yield tables

        '''

        not_yet_followed = cpy_radio_iso[:]
        skipped_elements = {}
        while len(not_yet_followed) > 0:
            # Get the target
            target = not_yet_followed.pop()

            # Put the target in cpy_radio_iso
            if target in hist_isotopes:
                if target not in self.stable_decayed_isotopes:
                    self.stable_decayed_isotopes.append(target)
                continue

            elif target not in cpy_radio_iso:
                cpy_radio_iso.append(target)

            # Get its decay index
            targ_index = self.all_isotopes_names.index(target)

            # Retrieve the number of reactions
            n_reacts = self.omega_ex.decay_module.iso.reactions[targ_index][1]

            # Decay rate in 1/year
            rate = self.omega_ex.decay_module.iso.decay_constant[targ_index][0]
            rate *= self.yr_in_s

            half_life = np.log(2) / rate

            # Try to skip reaction if too short
            skip_elem = False
            if half_life < self.min_half_life and can_skip:
                if target not in self.omega_ex.radio_iso:
                    skip_elem = True

            # For each reaction, store the products in not_yet_followed if they
            # are not in cpy_radio_iso
            for jj in range(n_reacts):
                prod_list = []
                react_indx = self.omega_ex.decay_module.iso.reactions[targ_index][jj + 2] - 1
                react_type = str(self.omega_ex.decay_module.iso.reaction_types[react_indx])

                # Apply the probability for this branch
                rate_jj = rate * self.omega_ex.decay_module.iso.decay_constant[targ_index][jj + 1]
                half_life_jj = np.log(2) / rate_jj

                # Try to skip reaction if too long
                if half_life_jj > self.max_half_life and n_reacts > 1:
                    s = "Reaction of type {}".format(react_type)
                    s += "for element {} too slow. Skipping.".format(target)
                    print(s)
                    continue

                # Get the product index and name
                prod_index = self.omega_ex.decay_module.iso.product_isomer[targ_index][jj] - 1
                if prod_index == targ_index and "SF" not in react_type:
                    s = "Warning: {} decaying into itself ".format(target)
                    s += "However, the module does not currently track "
                    s += "an isomer of {}. Skipping decay.".format(target)
                    print(s)
                    continue

                prod_name = self.all_isotopes_names[prod_index]
                prod_list.append(prod_name)

                # Now get all side products
                prod_list += self.decay_secondary[react_indx]

                # Treat fissions
                fiss_prods = []; fiss_rates = []
                if "SF" in react_type:
                    # Never skip a fission if we arrive here
                    skip_elem = False

                    fission_index = self.omega_ex.decay_module.iso.reactions[targ_index][0]
                    fiss_vect = self.omega_ex.decay_module.iso.s_fission_vector[fission_index]
                    for kk in range(len(self.all_isotopes_names)):
                        if fiss_vect[kk] > 0:
                            fiss_prods.append(self.all_isotopes_names[kk])
                            fiss_rates.append(fiss_vect[kk])

                # Store this reaction unless we are skipping it
                if skip_elem:
                    # Store the products and the probability of each channel
                    if target in skipped_elements:
                        skipped_elements[target].append([prod_list,\
                                self.omega_ex.decay_module.iso.decay_constant[targ_index][jj + 1]])
                    else:
                        skipped_elements[target] = [[prod_list,\
                                self.omega_ex.decay_module.iso.decay_constant[targ_index][jj + 1]]]
                else:
                    # If the reaction is not skipped, just add it
                    if len(fiss_prods) > 0:
                        reac_list = []
                        # If in a fission, just add every product as a new reaction
                        for kk in range(len(fiss_prods)):
                            reac_list.append(Reaction(target,\
                                    [fiss_prods[kk]], rate_jj*fiss_rates[kk]))
                    else:
                        reac_list = [Reaction(target, prod_list, rate_jj)]

                    if target in self.reac_dictionary:
                        self.reac_dictionary[target] += reac_list
                    else:
                        self.reac_dictionary[target] = reac_list

                # Put products in the list!
                for elem in prod_list + fiss_prods:
                    if elem not in cpy_radio_iso and elem not in not_yet_followed:
                        if elem not in self.stable_decayed_isotopes:
                            not_yet_followed.append(elem)

        # Deal with the skipped elements

        # First, eliminate them from cpy_radio_iso
        for elem in skipped_elements:
            if elem in cpy_radio_iso:
                cpy_radio_iso.remove(elem)

        # Now substitute the skipped elements amongst themselves
        for elem in skipped_elements:
            for elem2 in skipped_elements:
                if elem == elem2:
                    continue

                cpy_reacs = copy.deepcopy(skipped_elements[elem2])
                for reac in cpy_reacs:
                    if elem in reac[0]:

                        # Store probability and other products
                        prob = reac[1]
                        other_prods = []
                        for product in reac[0]:
                            if product != elem:
                                other_prods.append(product)

                        # Eliminate this reaction
                        skipped_elements[elem2].remove(reac)

                        # Now add all reactions for elem
                        reacs = skipped_elements[elem]
                        for reac2 in reacs:
                            # Make a copy
                            cpy = copy.deepcopy(reac2)

                            # Modify this reaction
                            cpy[0] += other_prods
                            cpy[1] *= prob
                            skipped_elements[elem2].append(cpy)

        # Finally, remove the skipped elements from reactions
        for elem in skipped_elements:
            for target in self.reac_dictionary:
                cpy_reacs = copy.copy(self.reac_dictionary[target])
                for reac in cpy_reacs:
                    if elem in reac.products:
                        # Copy rate of reaction
                        rate = reac.rate

                        # Copy all other products
                        keep_prods = []
                        for product in reac.products:
                            if product != elem:
                                keep_prods.append(product)

                        # Eliminate reac
                        self.reac_dictionary[target].remove(reac)

                        # Now add all the reactions of the skipped element
                        for reac2 in skipped_elements[elem]:
                            product = reac2[0] + keep_prods
                            this_rate = reac2[1] * rate
                            reaction = Reaction(target, product, this_rate)
                            self.reac_dictionary[target].append(reaction)

    ##############################################
    #        Integrate the entire evolution      #
    ##############################################
    def integrate(self):

        '''
        Integrate different omega instances according to the interactions. Uses
        the algorithm described in Yagüe López et al. (2022)

        We assume that the omega_plus instance interacts with all omega instances.

        '''

        # Initialize the simulation

        # Load decay network if it exists
        self.__create_reac_dictionary()

        # Define the end of active period (depends on whether a galaxy merger occurs)
        if self.omega_p.t_merge> 0.0:
            i_up_temp = self.omega_ex.i_t_merger + 1
        else:
            i_up_temp = self.omega_ex.nb_timesteps

        # Reset the inflow and outflow rates to zero
        for omega_i in self.omega_list:
            omega_i.m_outflow_t = np.zeros(self.omega_ex.nb_timesteps)
            omega_i.m_inflow_t = np.zeros(self.omega_ex.nb_timesteps)

        # Create primordial arrays
        primordial_init = [copy.copy(x.ymgal[0]) for x in self.omega_list]
        primordial_init = np.array(primordial_init)
        if self.omega_p is not None:
            primordial_outer_init = copy.copy(self.omega_p.ymgal_outer[0])
            primordial_outer_init = np.array(primordial_outer_init)

        # Initialize sources arrays
        self.sources = []
        self.mdots = []
        self.sources_radio = []
        self.mdots_radio = []

        # Initialize depending on the number of sources
        if not self.omega_ex.pre_calculate_SSPs:

            if self.omega_p is not None:
                # Create a sources and mdots arrays
                self.sources_outer = [
                                      self.omega_p.ymgal_outer_massive,
                                      self.omega_p.ymgal_outer_agb,
                                      self.omega_p.ymgal_outer_1a,
                                      self.omega_p.ymgal_outer_nsm,
                                     ]
                self.sources_outer += [x for x in self.omega_p.ymgal_outer_extra]

                # For radiactive sources
                if self.use_radio:
                    self.sources_outer_radio = [
                                                self.omega_p.ymgal_outer_massive_radio,
                                                self.omega_p.ymgal_outer_agb_radio,
                                                self.omega_p.ymgal_outer_1a_radio,
                                                self.omega_p.ymgal_outer_nsm_radio,
                                               ]
                    self.sources_outer_radio += [x for x in
                                            self.omega_p.ymgal_outer_extra_radio]
                else:
                    self.sources_outer_radio = [[0.] for x in self.sources_outer]

            # For the omega instances:
            for omega_i in self.omega_list:

                # For stable sources
                self.sources.append([
                                     omega_i.ymgal_massive,
                                     omega_i.ymgal_agb,
                                     omega_i.ymgal_1a,
                                     omega_i.ymgal_nsm,
                                    ])
                self.sources[-1] += [x for x in omega_i.ymgal_delayed_extra]
                self.mdots.append([
                                   omega_i.mdot_massive,
                                   omega_i.mdot_agb,
                                   omega_i.mdot_1a,
                                   omega_i.mdot_nsm,
                                  ])
                self.mdots[-1] += [x for x in omega_i.mdot_delayed_extra]

                # For radiactive sources
                if self.use_radio:
                    self.sources_radio.append([
                                               omega_i.ymgal_massive_radio,
                                               omega_i.ymgal_agb_radio,
                                               omega_i.ymgal_1a_radio,
                                               omega_i.ymgal_nsm_radio,
                                              ])
                    self.sources_radio[-1] += [x for x in
                                               omega_i.ymgal_delayed_extra_radio]
                    self.mdots_radio.append([
                                             omega_i.mdot_massive_radio,
                                             omega_i.mdot_agb_radio,
                                             omega_i.mdot_1a_radio,
                                             omega_i.mdot_nsm_radio,
                                            ])
                    self.mdots_radio[-1] += [x for x in
                                             omega_i.mdot_delayed_extra_radio]
                else:
                    self.sources_radio.append([[0.] for x in self.sources[-1]])
                    self.mdots_radio.append([[0.] for x in self.mdots[-1]])

        else:

            # Create a sources and mdots arrays
            self.sources_outer = [self.ymgal_outer]
            if self.use_radio:
                self.sources_outer_radio = [self.ymgal_outer_radio]
            else:
                self.sources_outer_radio = [[0.] for x in self.sources_outer]

            # For the omega instances:
            for omega_i in self.omega_list:
                # For stable sources
                self.sources.append([omega_i.ymgal])
                self.mdots.append([omega_i.mdot])

                # For radiactive sources
                if self.use_radio:
                    self.sources_radio.append([omega_i.ymgal_radio])
                    self.mdots_radio.append([omega_i.mdot_radio])
                else:
                    self.sources_radio.append([[0.] for x in self.sources[-1]])
                    self.mdots_radio.append([[0.] for x in self.mdots[-1]])

        # Transform into numpy arrays the mdot lists, because
        # they are not changed here
        self.mdots = np.array(self.mdots)
        self.mdots_radio = np.array(self.mdots_radio)

        # Here perform the integrations
        for i_step_OMEGA, dt in enumerate(self.timesteps):

            # Copy initial values for safekeeping before integration
            mgal_init_split = np.array(self.sources)[:, :, i_step_OMEGA, :]
            if self.omega_ex.pre_calculate_SSPs:
                mgal_init_split[:, 0, :] -= primordial_init
            mgal_radio_init_split = np.array(self.sources_radio)
            mgal_radio_init_split = mgal_radio_init_split[:, :, i_step_OMEGA, :]

            if self.omega_p is not None:
                mcgm_init_split = np.array(self.sources_outer)[:, i_step_OMEGA, :]
                if self.omega_ex.pre_calculate_SSPs:
                    mcgm_init_split[0, :] -= primordial_outer_init
                mcgm_radio_init_split = np.array(self.sources_outer_radio)
                mcgm_radio_init_split = mcgm_radio_init_split[:, i_step_OMEGA, :]

            # Define the total timestep
            totDt = dt

            # Initialize
            final_sfr = 0.;  total_m_added = 0.;  total_m_lost = 0.
            HH = totDt; newHH = HH

            substeps = self.omega_p.substeps

            while totDt > 0:
                converged = True

                # Run the patankar algorithm for the substeps

                # Inner arrays
                t_m_prim = []
                t_m_gal = []
                t_m_gal_radio = []

                # CGM arrays
                t_m_prim_out = []
                t_m_cgm = []
                t_m_cgm_radio = []

                # TODO inner or CGM?
                # TODO probably keep track by individual omega instance
                self.one_instance_error()
                t_total_sfr = []
                t_m_added = []
                t_m_lost = []

                # Error list
                err = []

                for ii, nn in enumerate(substeps):

                    fnn = float(nn)
                    htm = HH / fnn

                    # Store initial values
                    isot_prim = primordial_init.copy()
                    isot_mgal = mgal_init_split.copy()
                    isot_mgal_radio = mgal_radio_init_split.copy()

                    # Initial outer values
                    isot_prim_out = primordial_outer_init.copy()
                    isot_mcgm = mcgm_init_split.copy()
                    isot_mcgm_radio = mcgm_radio_init_split.copy()

                    # Tracking values
                    i_m_added = 0
                    i_m_lost = 0
                    i_sfr = 0

                    # Introduce the yields for all isotopes
                    yield_rate = self.mdots[:, :, i_step_OMEGA, :] / HH
                    yield_rate_radio = self.mdots_radio[:, :, i_step_OMEGA, :]
                    yield_rate_radio /= HH
                    if self.use_radio:
                        # Increase the size of the array if needed
                        diff = self.omega_ex.nb_radio_iso - yield_rate_radio.shape[-1]
                        if diff > 0:

                            # Get the shape in a list
                            new_shape = list(yield_rate_radio.shape)

                            # Save previous size
                            old_size = new_shape[-1]

                            # Introduce new size
                            new_shape[-1] = self.omega_ex.nb_radio_iso

                            # Create new array and broadcast
                            new_yield_rate_radio = np.zeros(new_shape)
                            new_yield_rate_radio[:, :, 0:old_size] = yield_rate_radio
                            yield_rate_radio = new_yield_rate_radio

                    for jj in range(nn):

                        all_p = []
                        all_d = []

                        # Initialize all_p and all_d

                        # Primordial gas (one per omega instance)
                        all_p.append(np.zeros(primordial_init.shape))
                        all_d.append(np.zeros(primordial_init.shape))

                        # ymgal (one per source per omega instance)
                        all_p.append(np.zeros(isot_mgal.shape))
                        all_d.append(np.zeros(isot_mgal.shape))

                        # ymgal_radio (same as ymgal but only unstable isotopes)
                        all_p.append(np.zeros(isot_mgal_radio.shape))
                        all_d.append(np.zeros(isot_mgal_radio.shape))

                        # CGM primordial gas (one)
                        all_p.append(np.zeros(primordial_outer_init.shape))
                        all_d.append(np.zeros(primordial_outer_init.shape))

                        # ymgal_outer (one per source)
                        all_p.append(np.zeros(isot_mcgm.shape))
                        all_d.append(np.zeros(isot_mcgm.shape))

                        # ymgal_outer_radio (same as ymgal_outer
                        # but only unstable isotopes)
                        all_p.append(np.zeros(isot_mcgm_radio.shape))
                        all_d.append(np.zeros(isot_mcgm_radio.shape))

                        all_p += [0, 0, 0]
                        all_d += [0, 0, 0]

                        # Total timestep so far
                        dtt = jj * htm

                        for mm, omega_i in enumerate(self.omega_list):

                            # TODO
                            # For each omega instance, find the
                            # production and destruction
                            # That is, update all_p and all_d from 0 to 2

                            # TODO single instance
                            self.one_instance_error()
                            self.omega_p.inner = omega_i

                            # Integrate
                            values = self.omega_p.get_p_and_d(i_step_OMEGA,
                                          isot_prim[mm], isot_prim_out,
                                          isot_mgal[mm], yield_rate[mm],
                                          isot_mcgm, isot_mgal_radio[mm],
                                          yield_rate_radio[mm], isot_mcgm_radio,
                                          htm, dtt, nn, self.decay_from_radio,
                                          self.decay_to_radio,
                                          self.decay_to_stable,
                                          self.stable_decayed_isotopes)

                            # Unpack
                            for kk, val in enumerate(values):
                                if kk < 3:
                                    all_p[kk][mm] += val[0]
                                    all_d[kk][mm] += val[1]
                                else:
                                    all_p[kk] += val[0]
                                    all_d[kk] += val[1]

                            # Modify inner values
                            isot_prim[mm] += all_p[0][mm] * htm
                            isot_prim[mm] /= 1 + all_d[0][mm] * htm

                            isot_mgal[mm] += all_p[1][mm] * htm
                            isot_mgal[mm] /= 1 + all_d[1][mm] * htm

                            isot_mgal_radio[mm] += all_p[2][mm] * htm
                            isot_mgal_radio[mm] /= 1 + all_d[2][mm] * htm

                        # Modify outer values
                        isot_prim_out += all_p[3] * htm
                        isot_prim_out /= 1 + all_d[3] * htm

                        isot_mcgm += all_p[4] * htm
                        isot_mcgm /= 1 + all_d[4] * htm

                        isot_mcgm_radio += all_p[5] * htm
                        isot_mcgm_radio /= 1 + all_d[5] * htm

                        # Modify mass tracking values
                        i_sfr += all_p[6] * htm
                        i_sfr /= 1 + all_d[6] * htm

                        i_m_added += all_p[7] * htm
                        i_m_added /= 1 + all_d[7] * htm

                        i_m_lost += all_p[8] * htm
                        i_m_lost /= 1 + all_d[8] * htm

                    # Extrapolate according to Deuflhard 1983 but with some
                    # modifications to account for different convergence speed
                    t_m_prim.append([isot_prim])
                    t_m_gal.append([isot_mgal])
                    t_m_gal_radio.append([isot_mgal_radio])
                    t_m_prim_out.append([isot_prim_out])
                    t_m_cgm.append([isot_mcgm])
                    t_m_cgm_radio.append([isot_mcgm_radio])
                    t_total_sfr.append([i_sfr])
                    t_m_added.append([i_m_added])
                    t_m_lost.append([i_m_lost])

                    # Generic extrapolation array:
                    t_extrap = [t_m_prim, t_m_gal, t_m_gal_radio, t_m_prim_out,\
                                t_m_cgm, t_m_cgm_radio, t_total_sfr, t_m_added,\
                                t_m_lost]

                    # Calculate mean relative error
                    if ii > 0:
                        for kk in range(len(t_m_gal) - 1):
                            for tt in t_extrap:
                                new = tt[-1][kk] - tt[-2][kk]
                                new /= fnn / substeps[ii - kk - 1] - 1
                                new += tt[-1][kk]

                                tt[-1].append(new)

                        err.append(0)
                        for tt in t_extrap:
                            err[-1] += np.mean(np.abs(tt[-1][-2] - tt[-1][-1])\
                                    / np.abs(tt[-1][-2] + self.omega_p.min_val))

                        if err[-1] < self.omega_p.tolerance:
                            break

                # Take solution
                if len(err) > 0:
                    if err[-1] < self.omega_p.tolerance:
                        primordial_init = np.abs(t_m_prim[-1][-2])
                        mgal_init_split = np.abs(t_m_gal[-1][-2])
                        mgal_radio_init_split = np.abs(t_m_gal_radio[-1][-2])
                        primordial_outer_init = np.abs(t_m_prim_out[-1][-2])
                        mcgm_init_split = np.abs(t_m_cgm[-1][-2])
                        mcgm_radio_init_split = np.abs(t_m_cgm_radio[-1][-2])
                        final_sfr += np.abs(t_total_sfr[-1][-2])
                        total_m_added += np.abs(t_m_added[-1][-2])
                        total_m_lost += np.abs(t_m_lost[-1][-2])
                        converged = True
                    else:
                        converged = False

                    # Get the root error
                    for ii in range(len(err)):
                        err[ii] /= self.omega_p.tolerance
                        err[ii] **= 1. / (ii + 2)

                    hhcoef = min(err)

                    # Calculate newHH
                    if hhcoef <= 0:
                        newHH = totDt
                    elif not converged:
                        newHH = HH * 0.1
                    else:
                        newHH = HH/hhcoef

                # Update totDt and HH
                if converged:
                    totDt -= HH

                # TODO maybe this acceleration is not necessary
                HH = 2 * newHH

                # Check that HH remains below totDt
                if totDt < HH * 1.1:
                    HH = totDt

            # TODO assuming a single omega instance
            # for total_m_lost, total_m_added and final_sfr
            self.one_instance_error()

            # Update all values

            # CGM quantities
            self.omega_p.ymgal_outer[i_step_OMEGA + 1] += primordial_outer_init
            for ii in range(len(self.sources)):
                self.sources_outer[ii][i_step_OMEGA + 1] += mcgm_init_split[ii]
                if not omega_i.pre_calculate_SSPs:
                    self.omega_p.ymgal_outer[i_step_OMEGA + 1] += mcgm_init_split[ii]

                if self.use_radio:
                    self.sources_outer_radio[ii][i_step_OMEGA + 1] += mcgm_radio_init_split[ii]
                    if not omega_i.pre_calculate_SSPs:
                        self.omega_p.ymgal_outer_radio[i_step_OMEGA + 1] += mcgm_radio_init_split[ii]

            # Omega quantities
            for mm, omega_i in enumerate(self.omega_list):
                # Keep the lost and added values in memory
                # TODO probably have to change this...
                omega_i.m_outflow_t[i_step_OMEGA] = total_m_lost
                omega_i.m_inflow_t[i_step_OMEGA] = total_m_added

                # Now that we are out of it, update final values
                omega_i.ymgal[i_step_OMEGA + 1] += primordial_init[mm]
                for ii in range(len(self.sources)):
                    self.sources[mm][ii][i_step_OMEGA + 1] += mgal_init_split[mm][ii]
                    if not omega_i.pre_calculate_SSPs:
                        omega_i.ymgal[i_step_OMEGA + 1] += mgal_init_split[mm][ii]

                # Update the final values for the radioisotopes
                if self.use_radio:
                    for ii in range(len(self.sources_radio)):
                        self.sources_radio[mm][ii][i_step_OMEGA + 1] += mgal_radio_init_split[mm][ii]
                        if not omega_i.pre_calculate_SSPs:
                            omega_i.ymgal_radio[i_step_OMEGA + 1] += mgal_radio_init_split[mm][ii]

                # Update original arrays
                # TODO probably have to change this as well
                omega_i.history.sfr_abs[i_step_OMEGA] = final_sfr
                omega_i.history.sfr_abs[i_step_OMEGA] /= omega_i.history.timesteps[i_step_OMEGA]
                omega_i.m_outflow_t[i_step_OMEGA] = total_m_lost
                omega_i.m_locked = final_sfr

                # Get the new metallicity of the gas and update history class
                omega_i.zmetal = omega_i._getmetallicity(i_step_OMEGA)
                omega_i._update_history(i_step_OMEGA)


##############################################
#               Reaction CLASS               #
##############################################
class Reaction():

    '''
    Class for reactions

    '''

    #############################
    #        Constructor        #
    #############################
    def __init__(self, target, products, rate):

        '''
        Initialize the reaction. It takes a target, a single
        product or a list of products and a rate.

        '''

        self.target = target
        self.products = products if type(products) is list else [products]
        self.rate = rate


    #############################
    #      __str__ method       #
    #############################
    def __str__(self):

        '''
        __str__ method for the class, for when using "print"

        '''
        s = "{} -> {}".format(self.target, self.products[0])
        for prod in self.products[1:]:
            s += " + {}".format(prod)
        s += "; rate = {} 1/y".format(self.rate)

        return s

##### INTRODUCED BY KATE ######

def get_value_from_arr(y_arr, x_arr=None, xval=None, i_x=None):

    """

    Grab the value of y_arr indexed in x_arr for either
    i_t or current_time

    input parameters:
    =================

    y_arr: n,m-sized array or list of floats
    value array

    x_arr: n-sized array or list of floats
    indexing array, must be specified if xval is

    xval: float
    value for interpolation if i_x is None.
    NOTE: If provided, given priority

    i_x: int
    integer index to be used if xval is None

    returns: m-sized array
    estimated value of y_arr in x_val

    """

    # Interpolated return (priority)
    if xval is not None:
        if x_val is None:
            s = "In get_value_from_arr, if x-value given,"
            s += " an x_arr must be specified"
            sys.exit(s)

        # pass values to log
        xlog = np.log(np.array(x_arr) + 1e-30)
        ylog = np.log(np.array(y_arr) + 1e-30)

        # interpolate
        f_interp = scipy.interpolate.interp1d(xlog, ylog, assume_sorted=true)
        y_val = np.exp(f_interp(x_val))

        return yval

    # Now the simple return
    if i_x is not None:
        return y_arr[i_x]

    if xval is None:
        s = "In get_value_from_arr, must specify either x-index or x-value"
        sys.exit(s)

def make_zones(mass, start_radius , end_radius, n_zones, kwargs_list=[{}]):

    """

    Make the number of omega zones as required.

    Input Parameters:
    =================

    mass: float
    Initial mass of the gas in the galaxy [Msun]
    TODO maybe if this is a list, then we just
    assign mass to each bin indexed by the list

    start_radius: float
    Inside radius of the galaxy [kpc]

    end_radius: float:
    Outside radius of the galaxy [kpc]

    n_zones: float
    Number of zones required

    kwargs_list: dictionary
    Properties of the chemical evolution model (see chem_evol, sygma, omega)

    """

    zone_radii = np.linspace(start_radius, end_radius, n_zones + 1)

    # List for all of the bins
    bins = []

    # Calculate the physical parameters for each zone: mass, inner radius,
    # outer radius, centre...
    for i in range(n_zones):

        # Inner and outer radius for each bin by index
        rin = zone_radii[i]
        rout = zone_radii[i + 1]

        # Split the initial mass evenly across the zones
        # TODO Maybe we can add an user option
        zone_mass = mass / n_zones

        # Get the centre of the bin (linear distance)
        centre = (rout + rin) * 0.5

        # Put the bin properties in the gal_bin list
        gal_bin = {
                   "index": i,
                   "rin": rin,
                   "rout": rout,
                   "centre": centre,
                   "zone_mass": zone_mass
                   }
        bins.append(gal_bin)

    # Version number
    vers = []

    # Dictionary of all omega instances, each corresponding to a zone
    omegas = {}

    for i in range(len(bins)):
        ver = f'omega{i}'
        vers.append(ver)

        if len(kwargs_list) <= 1:
            # Make an omega for each zone, all with the same kwargs
            omegas[ver] = omega.omega(external_control=True,
                                      in_out_control=True,
                                      mgal=bins[i]["zone_mass"],
                                      **kwargs_list[0])
        if len(kwargs_list) > 1:
            # Custom kwargs for each zone
            omegas[ver] = omega.omega(external_control=True,
                                      in_out_control=True,
                                      mgal=bins[i]["zone_mass"],
                                      **kwargs_list[i])

    return vers, omegas, bins

def fr(r, h):

    """

    Radially dependent part of the normalisation integral for the gas infall.

    r: float
    Radius of the galaxy [kpc]

    h: float
    Disk scale height [kpc]

    """

    return np.exp(-r / h)

def ft(r, t_end, a=-1.267, b=1.033):

    """

    Time dependent part of the normalisation integral for the gas infall.

    r: float
    Radius of the galaxy [kpc]

    t_end: float
    Age of the galaxy at the end of the simulation [Gyr]

    a: float
    Infall timescale at the galaxy centre [Gyr]
    Default: a = -1.267

    b: float
    Gradient of the infall timescale [Gyr kpc-1]
    Default: b = 1.033

    Return val_t [Gyr]

    """

    val_t, err_t = integrate.quad(lambda t: np.exp(-t / (a + b * r)), 0, t_end)

    return val_t

def get_area(index, bins):

    """

    Find area of a zone.

    index: int
    Index of the zone

    bins:list
    List of bins containing the galaxy properties

    """

    # Integrate area from centre
    Ar = np.pi * bins[index]["rout"] ** 2

    # Remove inner areas if we are not in central bin
    if index > 0:
        Ar -= np.pi * bins[index]["rin"] ** 2

    return Ar

def inflows(i_t, index, bin_radius, omega_zone, minf, bins, a=-1.267, b=1.033,
            get_rates=False, current_time=None):

    """

    Calculate the gas infall onto each zone. Returns mass of the infalling
    gas [Msun kcp-2], chemical composision of the infalling gas in [Msun kpc-2]
    for each isotope and the inflow rate of the gas [Mo yr-1 kpc-2]

    Input Parameters:
    =================

    i_t: float
    Timestep index for the timestep of interest

    index:int
    Index of the zone

    bin_radius: float
    Mid-point radius of the zone of interest [kpc]

    omega_zone: dictionary entry
    OMEGA instance of interest

    minf: float
    Total mass of gas infalling throughout the simulation [Mo]

    bins:list
    List of bins

    a: float
    Infall timescale at the galaxy centre [Gyr]
    Default: a = -1.267

    b: float
    Gradient of the infall timescale [Gyr kpc-1]
    Default: b = 1.033

    get_rates: bool
    Get rates instead of total quantities for the integration
    Default: False

    current_time: float, range [age[i_t], age[i_t + 1]]
    If provided, interpolate into current_time

    """

    # kpc from Palla et al (which was from Spitoni, Gioanna and Matteucci 2017)
    h = 3.5

    # Define the function to integrate here (Chiappini 2001)
    def normalisation_integral(r):

        result = 2 * np.pi * r
        result *= fr(r, h)
        result *= ft(r, omega_zone.history.tend * 1e-9, a, b)

        return result

    # Integrate in radius
    A = minf / integrate.quad(normalisation_integral, bins[0]["rin"],
                              bins[-1]["rout"])[0]

    # In Gyr
    tau = a + b * bin_radius

    # Msun / (Gyr * kpc ** 2)
    i0 = A * np.exp(-bin_radius / h)

    # Current time in Gyr
    if current_time is None:
        age = omega_zone.history.age[i_t]
    else:
        age = current_time

    # Infall rate in Msun / (yr * kpc ** 2)
    inf_density_rate = i0 * np.exp(-age * 1e-9 / tau) * 1e-9

    # Get the dt
    if i_t < len(omega_zone.history.age) - 1:
        dt = omega_zone.history.age[i_t + 1] - age
    else:
        dt = 0

    # Mass rate of gas infalling in Msun
    m_gas_inf_rate = inf_density_rate * get_area(index, bins)

    # Chemical composition of infalling gas
    ym_inf_rate = omega_zone.prim_comp.get(quantity='Yields', Z=0.0)
    ym_inf_rate *= m_gas_inf_rate

    # Return the mass of gas, the chemical composition and inflow rate
    if get_rates:
        return m_gas_inf_rate, ym_inf_rate, inf_density_rate
    else:
        return m_gas_inf_rate * dt, ym_inf_rate * dt, inf_density_rate

def outflows(mass_loading, sfr, omega_zone, i_t, index, bins,
             get_rates=False, current_time=None):

    """

    Calculate the gas outflow from each zone. Returns mass of the outflowing gas [Mo kcp-2],
    chemical composision of the outflowing gas and the outflow rate of the gas [Mo Gyr-1 kpc-2]

    mass_loading:float
    Dimensionless ratio between the outflow rate and the star formation rate

    sfr: float
    Star formation rate of the zone at the timestep you are interested in [Mo yr-1 kcp-2]

    omega_zone: dictonary entry
    OMEGA instance of interest

    i_t: float
    Timestep index for the timestep of interest

    get_rates: bool
    Get rates instead of total quantities for the integration
    Default: False

    current_time: float, range [age[i_t], age[i_t + 1]]
    If provided, interpolate into current_time

    """

    # Outflow rate in Msun / (yr * kpc ** 2)
    out_density_rate = mass_loading * sfr

    if current_time is None:
        age = omega_zone.history.age[i_t]
    else:
        age = current_time

    # Get the dt
    if i_t < len(omega_zone.history.age) - 1:
        dt = omega_zone.history.age[i_t + 1] - age
    else:
        dt = 0

    # Mass of gas going out in Msun
    m_gas_out_rate = out_density_rate * get_area(index, bins)

    # Interpolate ymgal
    ymgal = get_value_from_arr(omega_zone.ymgal, x_arr=omega_zone.history.age,
                               xval=current_time, i_x=i_t)

    # Composition of the gas going out
    # Fraction of the zone that is the outflowing gas
    frac_out_rate = m_gas_out_rate / np.sum(ymgal)

    # Calculating the chemical composition of that fraction
    ym_out_rate = ymgal * frac_out_rate

    # Return the mass of gas, the chemical composition and outflow rate
    if get_rates:
        return m_gas_out_rate, ym_out_rate, out_density_rate
    else:
        return m_gas_out_rate * dt, ym_out_rate * dt, out_density_rate

def migration(index, vers, omegas, bins, fstar, i_t, coeff, minf, mass_loading,
              a=-1.267, b=1.033, KS_pow=1, get_rates=False, current_time=None):

    """

    Function to move the gas in and out of each zone via inflow, outflow
    and radial flow.

    Input Parameters:
    =================

    index: int
    Index of the omega zone from 0 to number of zones-1

    vers: list
    List of the omega zone version names

    omegas: dictionary
    Dictionary of the omega instances createed for each zone

    bins:list
    List of bins

    f_star: float
    Star formation efficiency [yr-1]

    i_t: float
    Timestep index through the evolution when the migration is occurring

    coeff: float
    Constant coefficient for the migration of gas, as a percentage of total
    mass in the zone
    (Default = 0, i.e. no radial flows)

    minf: float
    Total mass of gas infalling throughout the simulation [Mo]

    mass_loading:float
    Dimensionless ratio between the outflow rate and the star formation rate

    a: float
    Infall timescale at the galaxy centre [Gyr]
    Default: a = -1.267

    b: float
    Gradient of the infall timescale [Gyr kpc-1]
    Default: a = 1.033

    KS_pow: float
    Value of exponent if using modified Kennicutt-Schmdit SF law
    (Default = 1)

    get_rates: bool
    Get rates instead of total quantities for the integration
    Default: False

    current_time: float, range [age[i_t], age[i_t + 1]]
    If provided, interpolate into current_time

    """

    # Grab the current and next ymgal
    ymgal = get_value_from_arr(omegas[vers[index]].ymgal,
                               x_arr=omegas[vers[index]].history.age,
                               xval=current_time, i_x=i_t)

    # Get the next one. The 0 value is to avoid having
    # the same conditional check later
    if index < len(vers) - 1:
        ymgal_p1 = get_value_from_arr(omegas[vers[index + 1]].ymgal,
                                      x_arr=omegas[vers[index + 1]].history.age,
                                      xval=current_time, i_x=i_t)
    else:
        ymgal_p1 = 0

    # Calculation for gas surface density for the zone
    # and setting a threshold of gas surface density needed to form stars
    mass_sum = np.sum(ymgal)

    # In Msun / pc ** 2
    gas_surf_density = mass_sum / get_area(index, bins) * 1e6

    # TODO why 7? This should be a variable somewhere
    # TODO from chiappini. stellar_form_thresh = 7 MSun / pc
    if gas_surf_density < 7:
        sfr = 0
    else:
        # Star formation rate in Msun / (yr * kpc ** 2)
        sfr = (fstar * mass_sum / get_area(index, bins)) ** KS_pow

    # Calculate the gas gained and lost
    gas_gained = inflows(i_t, index, bins[index]["centre"],
                         omegas[vers[index]], minf, bins, a, b,
                         get_rates=get_rates, current_time=current_time)[1]
    gas_lost = outflows(mass_loading, sfr, omegas[vers[index]], i_t, index,
                        bins, get_rates=get_rates, current_time=current_time)[0]

    # Now correct them depending where we are

    # If we are everywhere but the outermost zone
    # then correct by adding the inflow from the outer zone
    radial_gained = coeff * ymgal_p1
    gas_gained += coeff * ymgal_p1

    # If we are everywhere but the innermost zone
    # then correct by adding the outflow to the inner zone
    if index > 0:
        radial_lost = np.sum(coeff * ymgal)
        gas_lost += np.sum(coeff * ymgal)
    else:
        radial_lost = 0

    # Star formation rate, gas lost and gas gained for this zone
    return sfr, gas_lost, gas_gained

def multizone(n_zones=10, mass=1e-12, start_radius=4, end_radius=16.5,
              fstar=2.3e-10, coeff=0, minf=1e11, mass_loading=0,
              kwargs_list=[{}], a=-1.267, b=1.033, KS_pow=1,
              give_return_info=False):

    """

    Function which combines the capailities of make_zones, inflows and
    migration in order to make the required number of omega zones and move
    gas between them at each timestep.

    Input Parameters:
    =================

    n_zones:float
    Number of zones required for the model

    mass: float
    Initial mass of the gas in the galaxy [Mo]

    start_radius: float
    Inside radius of the galaxy [kpc]

    end_radius: float:
    Outside radius of the galaxy [kpc]

    fstar: float
    Star formation efficiency [yr-1]

    coeff: float
    Constant coefficient for the migration of gas, as a percentage of total
    mass in the zone

    minf: float
    Total mass of gas infalling throughout the simulation [Mo]

    a: float
    Infall timescale at the galaxy centre [Gyr]

    b: float
    Gradient of the infall timescale [Gyr kpc-1]

    KS_pow: float
    Value of exponent if using modified Kennicutt-Schmdit SF law
    (Default = 1)

    mass_loading: float
    Dimensionless ratio between the outflow rate and the star formation rate

    kwargs_list: dictionary
    Properties of the chemical evolution model (see chem_evol, sygma, omega)

    give_return_info: boolean
    Return the migration information: star formation rate [Mo yr-1 kpc-2],
    mass of gas lost [Mo], mass of gas gained [Mo for each isotope]
    Default Value: False, to activate give_return_info = True

    """

    vers, omegas, bins = make_zones(mass, start_radius, end_radius, n_zones)

    # This may need to be altered with parameters that dictate length of
    # timestep etc.
    # This is the support omega instance, just to have the dummy variables
    o_default = omega.omega(**kwargs_list[0])

    # The variable migr_info provides information about total gas gained
    # and lost through all possible gas migration processes (so between zones
    # and the CGM)
    # It does not contain only information about radial flows.

    # TODO NOT FINISHED. OVERHAUL WHEN INTRODUCED IN ALPHA

    # TODO Return somewhere information about only radial flows

    migr_info = []
    for t in range(o_default.nb_timesteps):
        migr_info.append([])

        # Add migration information for every timestep
        for i in range(n_zones):
            migr_info[t].append(migration(i, vers, omegas, bins, fstar, t,
                                          coeff, minf, mass_loading, a, b,
                                          KS_pow))

            # Sanity check for mass.
            # TODO maybe put inside of omega?
            if np.sum(omegas[vers[i]].ymgal[t]) <= 0:
                error = f"Negative mass in {vers[i]} at time {t} - end run"
                sys.exit(error)

        for i in range(n_zones):

            # Run a step for this omega instance
            omegas[vers[i]].run_step(t + 1, migr_info[t][i][0],
                                     m_lost=migr_info[t][i][1],
                                     m_added=migr_info[t][i][2])

    if give_return_info:
        return vers, omegas, bins, migr_info

    return vers, omegas, bins

def get_radii(bins):

    """

    Returns the mid-point radius of each zone

    """

    radii = [x["centre"] for x in bins]

    return radii

def get_radial_SFR(vers, omegas, bins, fstar, i_t, KS_pow=1, current_time=None):

    """

    Get the SFR [Mo yr-1 kpc-2] as a function of radius (inside out)
    for a given timestep

    """

    rad_sfr = []

    for i in range(len(omegas)):

        # Grab ymgal
        ymgal = get_value_from_arr(omegas[vers[i]].ymgal,
                                   x_arr=omegas[vers[i]].history.age,
                                   xval=current_time, i_x=i_t)

        # Calculation for gas surface density for the zone:
        mass_sum = np.sum(ymgal)

        # In Msun / pc ** 2
        gas_surf_density = (mass_sum / get_area(i, bins)) * 1e6

        # TODO Same as before put the stellar_form_thresh
        if gas_surf_density < 7:
            star_form_rate = 0
        else:
            star_form_rate = (fstar * (mass_sum / get_area(i, bins))) ** KS_pow

        rad_sfr.append(star_form_rate)

    rad_sfr = np.array(rad_sfr)

    return rad_sfr

def get_outflow_rate(vers, omegas, bins, fstar, mass_loading, i_t, KS_pow=1,
                     current_time=None):

    """

    Find outflow [Mo yr-1kpc-2] rate as a function of radius for a given timestep

    """

    # This is the radial SFR at a given timestep i_t
    sfr_at_i_t = get_radial_SFR(vers, omegas, bins, fstar, i_t, KS_pow=KS_pow,
                                current_time=current_time)

    # This is the radial outflow (SFR times mass loading)
    rad_outflow = mass_loading * sfr_at_i_t

    return rad_outflow

def get_radial_mass(vers, omegas, bins, i_t, current_time=None):

    """

    Find the mass of the gas in each zone [Mo] as a function of radius
    for a given timestep

    """

    radial_masses = [np.sum(get_value_from_arr(omegas[vers[i]].ymgal,
                                               x_arr=omegas[vers[i]].history.age,
                                               xval=current_time, i_x=i_t))
                    for i in range(len(omegas))]

    return radial_masses

def get_radial_surface_mass(vers, omegas, bins, i_t, current_time=None):

    """

    Find the surface gas density in each zone [Mo kpc-2] as a function of
    radius for a given timestep

    """

    rad_surface_mass = []
    rad_mass_sum = get_radial_mass(vers, omegas, bins, i_t,
                                   current_time=current_time)

    for i in range(len(omegas)):
        surf_mass = rad_mass_sum[i] / get_area(i, bins)
        rad_surface_mass.append(surf_mass)

    return rad_surface_mass

def plot_vs_radius(y_arr, ylabel, omegas, bins, i_t, current_time=None):

    """

    Plot y_arr vs radius with common plot parameters

    """

    radii = get_radii(bins)
    xlabel = "Radius (kpc)"

    # TODO label would be better as age but leave for now
    age = omegas["omega0"].history.age[i_t] * 1e-9
    label = f"{age:.2f} Gyr"

    plt.plot(radii, y_arr, marker='o', label=label)

    plt.legend(fontsize=10, loc='upper right')

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)

def plot_inflow_rate(vers, omegas, bins, i_t, minf, a=-1.267, b=1.033):

    """

    Plots the inflow rate as a function of radius for a given timestep

    vers: list
    List of the omega zone version names

    omegas: dictionary
    Dictionary of the omega instances createed for each zone

    bins:list
    List of bins created by the initial omega run

    """

    ylabel = 'Inflow Rate (M$_{\odot}$yr$^{-1}$kpc$^{-2}$)'
    in_rate = [inflows(i_t, i, x[0]["centre"], omegas[x[1]], minf, bins, a, b)[2]
               for i, x in enumerate(zip(bins, omegas))]

    plot_vs_radius(in_rate, ylabel, omegas, bins, i_t)

def plot_radial_sfr(vers, omegas, bins, fstar, i_t):

    """

    Plot star formation rate as a function of radius for a given timestep

    """

    ylabel = 'SFR (M$_{\odot}$yr$^{-1}$kpc$^{-2}$)'
    radial_SFR = get_radial_SFR(vers, omegas, bins, fstar, i_t)

    plot_vs_radius(radial_SFR, ylabel, omegas, bins, i_t)

def plot_outflow_rate(vers, omegas, bins, fstar, mass_loading, i_t):

    """

    Plot outflow rate as a function of radius for a given timestep

    """

    ylabel = 'Outflow Rate (M$_{\odot}$yr$^{-1}$kpc$^{-2}$)'
    outflow_rate = get_outflow_rate(vers, omegas, bins, fstar, mass_loading, i_t)

    plot_vs_radius(outflow_rate, ylabel, omegas, bins, i_t)

def plot_radial_mass(vers, omegas, bins, i_t):

    """

    Plot mass in each zone as a function of radius for a given timestep

    """

    ylabel = 'Zone mass (M$_{\odot}$)'
    radial_mass = get_radial_mass(vers, omegas, bins, i_t)

    plot_vs_radius(radial_mass, ylabel, omegas, bins, i_t)

def plot_radial_surface_mass(vers, omegas, bins, i_t):

    """

    Plot the gas surface density of each zone as a function of radius for a
    given timestep

    """

    ylabel = 'Zone surface mass(gas) density (M$_{\odot}$ kpc$^{-2}$)'
    radial_surface_mass = get_radial_surface_mass(vers, omegas, bins, i_t)

    plot_vs_radius(radial_surface_mass, ylabel, omegas, bins, i_t)

def get_exp_scale_length(vers, omegas, bins):

    """

    Find the exponential scale length of the model at the final timestep

    """

    # Gas mass as a function of radius for the final timestep
    radial_mass_final = np.array(get_radial_mass(vers, omegas, bins, -1))

    # Assuming that the surface density rho goes as
    # rho = rho0 * exp(-r / h)
    # then the mass between ri and rj is given by:
    # mij = 2 * pi * h * rho0 * ((h + ri) * exp(-ri / h) - (h + rj) * exp(-rj / h))
    # So we can extract h by dividing consecutive masses and solving with NR

    # Define the mass function and derivative
    def mass_func(r1, r2, h, rho0=1):
        '''
        Mass function assuming that surface density rho goes as
        rho = rho0 * exp(-r / h)

        If rho0 is not given, is assumed to be 1
        '''

        mass = (h + r1) * np.exp(-r1 / h) - (h + r2) * np.exp(-r2 / h)
        mass *= 2 * np.pi * h * rho0

        return mass

    def mass_deriv(r1, r2, h, rho0=1):
        '''
        Derivative of the mass function with h

        If rho0 is not given, is assumed to be 1
        '''

        # Derivative of parenthesis
        mass_der = (1 + r1 / h ** 2 * (h + r1)) * np.exp(-r1 / h)
        mass_der -= (1 + r2 / h ** 2 * (h + r2)) * np.exp(-r2 / h)
        mass_der *= 2 * np.pi * h * rho0

        # Derivative of factor before parenthesis
        mass_der += mass_func(r1, r2, h, rho0=rho0) / h

        return mass_der

    def find_hscale_for_masses(mass1, mass2, bin1, bin2):
        '''
        Newton rhapson to find the hscale between mass1 and mass2
        '''

        ratio = mass1 / mass2
        h0 = 1
        dif = None

        # Do the newton-rhapson step
        while dif is None or dif > 1e-3:

            r1 = bin1["rin"]
            r2 = bin1["rout"]
            r3 = bin2["rin"]
            r4 = bin2["rout"]

            ratio_fun = mass_func(r1, r2, h0) / mass_func(r3, r4, h0) - ratio
            ratio_der = mass_deriv(r1, r2, h0) * mass_func(r3, r4, h0)
            ratio_der -= mass_func(r1, r2, h0) * mass_deriv(r3, r4, h0)
            ratio_der /= mass_func(r3, r4, h0) ** 2

            hnew = h0 - ratio_fun / ratio_der

            dif = np.abs(h0 - hnew)
            h0 = hnew

        return h0

    exp_scale_length = []
    for i in range(len(radial_mass_final) - 1):
        mass1 = radial_mass_final[i]
        mass2 = radial_mass_final[i + 1]
        bin1 = bins[i]
        bin2 = bins[i + 1]

        if i == 0:
            bin1["rin"] = 0

        hscale = find_hscale_for_masses(mass1, mass2, bin1, bin2)
        exp_scale_length.append(hscale)
    exp_scale_length = np.array(exp_scale_length)

    if np.max(np.abs(exp_scale_length - np.mean(exp_scale_length))) > 1e-2:
        print(exp_scale_length)
        sys.exit("The h-scale is not constant!")

    return exp_scale_length[0]

def radial_plot_spectro(vers, omegas, bins, i_t, yaxis='[Fe/H]',
                        return_x_y=False):

    """

    Plot spectroscopic ratio versus radius

    """

    # For time zero just give the minimum value of -30
    if i_t == 0:
        spec_ratio = [-30 for x in omegas]

    # For any other time, grab the second argument of plot_spectro (spec_abund)
    # and extract the values at i_t - 1 for each omega (if i_t == 1, then this
    # the same as taking the 0th value from spec_abund)
    else:

        spec_ratio = [omegas[x].plot_spectro(yaxis=yaxis, return_x_y=True,
                                             plot_for_radial=True)[1][i_t - 1]
                      for x in omegas]

    # Either return or plot
    if return_x_y:
        return get_radii(bins), spec_ratio
    else:
        plot_vs_radius(spec_ratio, yaxis, omegas, bins, i_t)

# TODO
vers_test, omegas_test, bins_test = multizone()
ts = [0, 7, 15, 22, 25, 28, 30]
plt.figure(2, (10,6))

# TODO
#get_exp_scale_length(vers_test, omegas_test, bins_test)

for i in ts:
    plot_radial_surface_mass(vers_test, omegas_test, bins_test, i)
    #plot_inflow_rate(vers_test, omegas_test, bins_test, i, 1e11)
    #plot_outflow_rate(vers_test, omegas_test, bins_test, 0, 1e11, i)

plt.show()

plt.figure(10, (10,6))
for i in ts:
    radial_plot_spectro(vers_test, omegas_test, bins_test, i, yaxis = '[Fe/H]')
    plt.ylim(-5,0)

plt.show()
