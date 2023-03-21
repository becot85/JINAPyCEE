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
import os
import copy

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
