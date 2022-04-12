from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

'''

GAMMA (Galaxy Assembly with Merger trees for Modeling Abundances)

 - May2017: B. Cote

Definitions

    Halo : One "halo" at a specific redshift
    Primordial halo : Halo that formed naturally, not from a merger
    Branch : Evolution of dark matter without merger (series of halos)
    Merger : Merger of two or more branches

    A branch typically contains many halos, since the "same" halo has
    different identification for each redshift.. even if no merger occured.

Each branch is represented by an OMEGA_SAM or OMEGA simulation:

    OMEGA (One-zone Model for the Evolution of GAlaxies) module

    OMEGA_SAM is a 2-zone models with OMEGA at the center surrounded by halo gas

'''

# Standard packages
import numpy as np
import time as t_module
import os
import multiprocessing as mp
import copy

# Define where is the working directory
# This is where the NuPyCEE code will be extracted
nupy_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
nupy_path = os.path.join(nupy_path, "NuPyCEE")

# Import NuPyCEE and JINAPyCEE codes
import NuPyCEE.read_yields as ry
import NuPyCEE.omega as omega
import JINAPyCEE.omega_plus as omega_plus


#####################
# Class Declaration #
#####################

class gamma():

    # Initialisation function
    def __init__(self, tree_trunk_ID=-1, mvir_sf_tresh=-1, n_proc = 1, \
                 table='yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt', \
                 pop3_table='yield_tables/popIII_heger10.txt', \
                 br_r_vir=[], redshifts=[], times=[], br_halo_ID=[], br_age=[], \
                 br_z=[], br_t_merge=[], br_ID_merge=[], br_m_halo=[], \
                 br_is_prim=[], br_is_SF=[], br_sfe_t=[], br_sfh=[], \
                 br_is_sub=[], br_is_SF_t=[], sne_L_feedback=[], **kwargs):

        # Check if we have the trunk ID
        if tree_trunk_ID < 0:
            print ('Error - GAMMA needs the tree_trunk_ID parameter.')
            return

        # Announce the beginning of the simulation
        print ('GAMMA run in progress..')
        start_time = t_module.time()
        self.start_time = start_time

        # Keep the OMEGA parameters in memory
        self.kwargs = kwargs
        self.kwargs["table"] = table
        self.kwargs["pop3_table"] = pop3_table
        self.kwargs["sne_L_feedback"] = sne_L_feedback

        # Keep the GAMMA parameters in memory
        self.kwargs = kwargs
        self.tree_trunk_id = tree_trunk_ID
        self.redshifts = redshifts
        self.times = times
        self.br_halo_ID = br_halo_ID
        self.br_age = br_age
        self.br_z = br_z
        self.br_t_merge = br_t_merge
        self.br_ID_merge = br_ID_merge
        self.br_m_halo = br_m_halo
        self.br_is_prim = br_is_prim
        self.br_is_SF = br_is_SF
        self.br_is_SF_t = br_is_SF_t
        self.br_sfe_t = br_sfe_t
        self.br_sfh = br_sfh
        self.len_br_is_SF = len(br_is_SF)
        self.len_br_sfe_t = len(br_sfe_t)
        self.len_br_is_SF_t = len(br_is_SF_t)
        self.len_br_sfh = len(br_sfh)
        self.br_r_vir = br_r_vir
        self.br_is_sub = br_is_sub
        self.mvir_sf_tresh = mvir_sf_tresh
        self.n_proc = n_proc
        self.sne_L_feedback = sne_L_feedback
        self.len_sne_L_feedback = len(sne_L_feedback)

        # Return if inputs not ok
        if self.len_br_is_SF_t > 0 and self.len_br_is_SF == 0:
            print ('Error - br_is_SF_t cannot be used without br_is_SF.')
            return

        # Initialisation of the parameters
        self.__initialisation()

        # Run the simulation
        self.__start_simulation()

        # Announce the end of the simulation
        print ('   GAMMA run completed -',self.__get_time())

    ##############################################
    #                  Get Time                  #
    ##############################################
    def __get_time(self):

        out = 'Run time: ' + \
        str(round((t_module.time() - self.start_time),2))+"s"
        return out


    ##############################################
    #               Initialisation               #
    ##############################################
    def __initialisation(self):

        '''
        Read the merger tree and declare and fill arrays.

        '''

        # Run an OMEGA simulation in order to copy basic arrays
        self.o_ini = omega.omega(table=self.kwargs["table"], \
                                 pop3_table=self.kwargs["pop3_table"],\
                                 special_timesteps=2, cte_sfr=0.0, \
                                 mgal=1e10, print_off=True)

        # Add NuPyCEE kwargs arguments to speed-up the initialization of branches
        self.kwargs["input_yields"] = True
        self.kwargs["ytables_in"] = self.o_ini.ytables
        self.kwargs["isotopes_in"] = self.o_ini.history.isotopes
        self.kwargs["ytables_1a_in"] = self.o_ini.ytables_1a
        self.kwargs["inter_Z_points"] = self.o_ini.inter_Z_points
        self.kwargs["nb_inter_Z_points"] = self.o_ini.nb_inter_Z_points
        self.kwargs["y_coef_M"] = self.o_ini.y_coef_M
        self.kwargs["y_coef_M_ej"] = self.o_ini.y_coef_M_ej
        self.kwargs["y_coef_Z_aM"] = self.o_ini.y_coef_Z_aM
        self.kwargs["y_coef_Z_bM"] = self.o_ini.y_coef_Z_bM
        self.kwargs["y_coef_Z_bM_ej"] = self.o_ini.y_coef_Z_bM_ej
        self.kwargs["tau_coef_M"] = self.o_ini.tau_coef_M
        self.kwargs["tau_coef_M_inv"] = self.o_ini.tau_coef_M_inv
        self.kwargs["tau_coef_Z_aM"] = self.o_ini.tau_coef_Z_aM
        self.kwargs["tau_coef_Z_bM"] = self.o_ini.tau_coef_Z_bM
        self.kwargs["tau_coef_Z_aM_inv"] = self.o_ini.tau_coef_Z_aM_inv
        self.kwargs["tau_coef_Z_bM_inv"] = self.o_ini.tau_coef_Z_bM_inv
        self.kwargs["y_coef_M_pop3"] = self.o_ini.y_coef_M_pop3
        self.kwargs["y_coef_M_ej_pop3"] = self.o_ini.y_coef_M_ej_pop3
        self.kwargs["tau_coef_M_pop3"] = self.o_ini.tau_coef_M_pop3
        self.kwargs["tau_coef_M_pop3_inv"] = self.o_ini.tau_coef_M_pop3_inv
        self.kwargs["inter_lifetime_points_pop3"] = self.o_ini.inter_lifetime_points_pop3
        self.kwargs["inter_lifetime_points_pop3_tree"] = self.o_ini.inter_lifetime_points_pop3_tree
        self.kwargs["nb_inter_lifetime_points_pop3"] = self.o_ini.nb_inter_lifetime_points_pop3
        self.kwargs["inter_lifetime_points"] = self.o_ini.inter_lifetime_points
        self.kwargs["inter_lifetime_points_tree"] = self.o_ini.inter_lifetime_points_tree
        self.kwargs["nb_inter_lifetime_points"] = self.o_ini.nb_inter_lifetime_points
        self.kwargs["nb_inter_M_points_pop3"] = self.o_ini.nb_inter_M_points_pop3
        self.kwargs["inter_M_points_pop3_tree"] = self.o_ini.inter_M_points_pop3_tree
        self.kwargs["nb_inter_M_points"] = self.o_ini.nb_inter_M_points
        self.kwargs["inter_M_points"] = self.o_ini.inter_M_points
        self.kwargs["y_coef_Z_aM_ej"] = self.o_ini.y_coef_Z_aM_ej

        # Calculate the number of redshifts
        self.nb_redshifts = len(self.redshifts)

        # Declare the galaxy instance array
        self.galaxy_inst = [0]*self.nb_redshifts
        for i_i in range(0,self.nb_redshifts):
           self.galaxy_inst[i_i] = [0]*len(self.br_m_halo[i_i])

        # Declare the extra initial baryonic mass added to galaxies
        self.dm_bar_added_iso = [0.0]*self.nb_redshifts
        for i_i in range(0,self.nb_redshifts):
           self.dm_bar_added_iso[i_i] = np.zeros(\
               (len(self.br_m_halo[i_i]), self.o_ini.nb_isotopes))

        # Set the final redshift
        self.kwargs["redshift_f"] = min(self.redshifts)

        # Get the primordial composition (mass fraction)
        iniabu_table = 'yield_tables/iniabu/iniab_bb_walker91.txt'
        ytables_bb = ry.read_yields_Z( \
            os.path.join(nupy_path, iniabu_table), self.o_ini.history.isotopes)
        self.prim_x_frac = np.array(ytables_bb.get(Z=0.0, quantity='Yields'))
        del ytables_bb

        # Define the information of whether branches are sub-halo or not
        if len(self.br_is_sub) > 0:
            self.is_sub_info = True
        else:
            self.is_sub_info = False


    ##############################################
    #              Start Simulation              #
    ##############################################
    def __start_simulation(self):

        '''
        Run all galaxy instance on each of the tree branch.

        '''

        # Define multiprocessing queue
        if self.n_proc > 1:
            queue = mp.Queue()

        # For each redshift ...
        for i_z_ss in range(0,self.nb_redshifts):

            # Check if each branch is primordial or not and save tuples
            branches = []; tot_branch = len(self.br_m_halo[i_z_ss])
            for i_br_ss in range(tot_branch):

                # If it's a primordial branch ...
                if self.br_is_prim[i_z_ss][i_br_ss]:
                    arguments = (self, i_z_ss, i_br_ss)

                # If the branch is the results of a merger ...
                else:

                    # Get the stellar ejecta and the ISM of all parents
                    mdot, mdot_t, ism, outer, dm, dmo, dmo_t = \
                        self.__get_mdot_parents(i_z_ss, i_br_ss)

                    arguments = (self, i_z_ss, i_br_ss, dm, mdot, mdot_t, \
                            ism, outer, dmo, dmo_t)

                # Create the Omega_Branch instance
                branches.append(Omega_Branch(arguments))

            # Now compute each galaxy
            if self.n_proc == 1 or tot_branch == 1:

                # Avoid the overhead if only one case or one process
                for i_br_ss in range(tot_branch):
                    galaxy = branches[i_br_ss].get_galaxy()
                    self.galaxy_inst[i_z_ss][galaxy[0]] = galaxy[1]

            else:

                # Divide the processes in chunks of n_proc
                i_br_ss = 0
                while i_br_ss < tot_branch:

                    # Calculate how many processes to use
                    use_proc = min(self.n_proc, tot_branch - i_br_ss)

                    # Define and run the processes
                    for ii in range(use_proc):
                        process = mp.Process(\
                            target = lambda q, o: o.get_galaxy(q), \
                            args = (queue, branches[i_br_ss + ii]))
                        process.start()

                    # Now get values
                    for ii in range(use_proc):
                        galaxy = queue.get()
                        self.galaxy_inst[i_z_ss][galaxy[0]] = galaxy[1]

                    # Increase i_br_ss
                    i_br_ss += use_proc


    ##############################################
    #              Get Mdot Parents              #
    ##############################################
    def __get_mdot_parents(self, i_z_ss, i_br_ss):

        '''
        Create an array of the mass ejected by stars as a function of
        time for all the parent branches that merge to yeild the current
        branch.  Combine the ISM of all parents.  Combine the delayed
        outflowing mass becaused of SNe feedback

        Arguments
        =========

          i_z_ss : Redshift index
          i_br_ss : Branch index

        '''

        # Find the indexes of all the parents
        i_z_par, i_br_par = self.__find_parents(i_z_ss, i_br_ss)
        nb_parents = len(i_z_par)

        # Declare the mdot arrays to be returned
        mdot = [0]*nb_parents
        mdot_t = [0]*nb_parents
        if self.len_sne_L_feedback > 0:
            dmo = [0]*nb_parents
            dmo_t = [0]*nb_parents
        ism = np.zeros(self.o_ini.nb_isotopes)
        outer = np.zeros(self.o_ini.nb_isotopes)
        dm = 0.0

        # For every parent ...
        for i_gmp in range(0,nb_parents):

            # Copy the indexes
            iz = i_z_par[i_gmp]
            ibr = i_br_par[i_gmp]

            # Add the dark matter halo mass
            dm += self.galaxy_inst[iz][ibr].inner.m_DM_t[\
               self.galaxy_inst[iz][ibr].inner.i_t_merger+1]

            # Create the parent mdot array (and delayed mass outflow if needed)
            array_len = self.galaxy_inst[iz][ibr].inner.nb_timesteps - \
                        (self.galaxy_inst[iz][ibr].inner.i_t_merger+1)
            mdot[i_gmp] = [0]*array_len
            mdot_t[i_gmp] = [0.0]*(array_len+1)
            if self.len_sne_L_feedback > 0:
                dmo[i_gmp] = [0]*array_len
                dmo_t[i_gmp] = [0.0]*(array_len+1)

            # Add time zero
            mdot_t[i_gmp][0] = 0.0
            if self.len_sne_L_feedback > 0:
                dmo_t[i_gmp][0] = 0.0

            # Combine the inner gas
            ism += self.galaxy_inst[iz][ibr].inner.ymgal[\
               self.galaxy_inst[iz][ibr].inner.i_t_merger+1]

            # Combine the outer gas
            outer += self.galaxy_inst[iz][ibr].ymgal_outer[\
               self.galaxy_inst[iz][ibr].inner.i_t_merger+1]

            # For every step starting from the merging point ...
            for i_step_om in range(0,array_len):

                # Timestep for the inner region
                i_inn = i_step_om + self.galaxy_inst[iz][ibr].inner.i_t_merger+1

                # Add the mdot value (including all isotopes)
                mdot[i_gmp][i_step_om] = self.galaxy_inst[iz][ibr].inner.mdot[i_inn]
                if self.len_sne_L_feedback > 0:
                    dmo[i_gmp][i_step_om] = \
                        self.galaxy_inst[iz][ibr].delayed_m_outflow[i_inn]

                # Add the current time since the merging point.
                # This represents the upper time limit of the current step
                mdot_t[i_gmp][i_step_om+1] = mdot_t[i_gmp][i_step_om] + \
                    self.galaxy_inst[iz][ibr].inner.history.timesteps[i_inn]
                if self.len_sne_L_feedback > 0:
                    dmo_t[i_gmp][i_step_om+1] = dmo_t[i_gmp][i_step_om] + \
                        self.galaxy_inst[iz][ibr].inner.history.timesteps[i_inn]

        # Return the stellar ejecta and the associated time
        if self.len_sne_L_feedback > 0:
            return mdot, mdot_t, ism, outer, dm, dmo, dmo_t
        else:
            return mdot, mdot_t, ism, outer, dm, np.array([]), np.array([])


    ##############################################
    #                Find Parents                #
    ##############################################
    def __find_parents(self, i_z_ss, i_br_ss):

        '''
        Find the i_z and i_br indexes of all parents of the current branch

        Arguments
        =========

          i_z_ss : Redshift index
          i_br_ss : Branch index

        '''

        # Declare the index arrays to be returned
        i_z_par = []
        i_br_par = []

        # For each previous redshift ...
        for i_z_fp in range(0,i_z_ss):

          # For each branch formed in that redshift ...
          for i_br_fp in range(0,len(self.br_ID_merge[i_z_fp])):

            # If the branch is one of the parents ...
            if self.br_ID_merge[i_z_fp][i_br_fp] == \
               self.br_halo_ID[i_z_ss][i_br_ss][0]:

                # Add the indexes in the array
                i_z_par.append(i_z_fp)
                i_br_par.append(i_br_fp)

        # Return the indexes
        return i_z_par, i_br_par


#####################
# Class Declaration #
#####################

class Omega_Branch():
    '''
    This class handles the creation of a branch for the merger tree.

    '''

    def __init__(self, args):
        '''
        This initialization class will simply create a copy of gammas' arguments.

        '''

        # Copy all gamma values
        self.__dict__ = copy.copy(args[0].__dict__)
        self.arguments = args[1:]


    ##############################################
    #                  Get galaxy                #
    ##############################################
    def get_galaxy(self, queue = None):
        '''
        Wrapper class for create_branch, it allows to call
        the class from outside without modifying it.

        '''

        gal = self.__create_branch(*self.arguments)
        if queue is not None:
            queue.put(gal)
        else:
            return gal


    ##############################################
    #               Create Branch                #
    ##############################################
    def __create_branch(self, i_z_ss, i_br_ss, dm=-1, mdot_ini=np.array([]), \
                        mdot_ini_t=np.array([]), ism_ini=np.array([]), \
                        ymgal_outer_ini=np.array([]), dmo_ini=np.array([]), \
                        dmo_ini_t=np.array([])):

        '''
        Create OMEGA to represent a specific branch from a merger tree, until
        the branch merges.

        Arguments
        =========

          i_z_ss: Redshift index
          i_br_ss: Branch index
          dm: Combined dark matter mass of the progenitors
          mdot_ini: Future stellar ejecta of the merged stellar populations
          mdot_ini_t: Times associated with the future ejecta
          ism_ini: Mass and composition of the merged inner gas reservoir
          ymgal_outer_ini: Mass and composition of the merged outer gas reservoir
          dmo_ini: Future outflow ejecta of the merged SNe (feedback)
          dmo_ini_t: Times associated with the future outflow ejecta

        '''

        # Assign the combined SSP, ISM gas, outflow (needs to be in create_omega)
        self.kwargs["mdot_ini"] = mdot_ini
        self.kwargs["mdot_ini_t"] = mdot_ini_t
        self.kwargs["ism_ini"] = ism_ini
        self.kwargs["ymgal_outer_ini"] = ymgal_outer_ini
        self.kwargs["dmo_ini"] = dmo_ini
        self.kwargs["dmo_ini_t"] = dmo_ini_t

        # Calculate the duration of the OMEGA instance
        self.kwargs["tend"] = self.times[-1] - self.times[i_z_ss] + \
                   (self.times[-1] - self.times[-2]) # Extra step for the trunk

        # Assign the DM array
        DM_array = []
        for i_cb in range(0,len(self.br_age[i_z_ss][i_br_ss])):
            DM_array.append([0.0]*2)
            DM_array[i_cb][0] = self.br_age[i_z_ss][i_br_ss][i_cb]
            DM_array[i_cb][1] = self.br_m_halo[i_z_ss][i_br_ss][i_cb]
        self.kwargs["DM_array"] = np.array(DM_array)

        # Assign the R_vir array
        r_vir_array = []
        for i_cb in range(0,len(self.br_age[i_z_ss][i_br_ss])):
            r_vir_array.append([0.0]*2)
            r_vir_array[i_cb][0] = self.br_age[i_z_ss][i_br_ss][i_cb]
            r_vir_array[i_cb][1] = self.br_r_vir[i_z_ss][i_br_ss][i_cb]
        self.kwargs["r_vir_array"] = np.array(r_vir_array)

        # Assign whether or not the branch will be a sub-halo at some point
        is_sub_array = np.array([])
        if self.is_sub_info:
            for i_cb in range(0,len(self.br_is_sub[i_z_ss][i_br_ss])):
                is_sub_array.append([0.0]*2)
                is_sub_array[i_cb][0] = self.br_age[i_z_ss][i_br_ss][i_cb]
                is_sub_array[i_cb][1] = self.br_is_sub[i_z_ss][i_br_ss][i_cb]
        self.kwargs["is_sub_array"] = is_sub_array

        # Add mass depending on the dark matter mass ratio.
        # This is because the sum of dark matter masses from the progenitors
        # sometime does not equal the initial dark matter mass of the new branch..
        if dm > 0.0:
            self.__correct_initial_state(i_z_ss, i_br_ss, \
                dm, self.br_m_halo[i_z_ss][i_br_ss][0])

        # Create a galaxy instance with external_control = True
        return self.__create_galaxy(i_z_ss, i_br_ss)


    ##############################################
    #            Correct Initial State           #
    ##############################################
    def __correct_initial_state(self, i_z_ss, i_br_ss, dm_comb, dm_ini):

        '''
        Add primordial gas if their is more dark matter than
        the sum of all pregenitors' dark matter mass. Gas is
        removed if less dark matter.

        Arguments
        =========

          i_z_ss: Redshift index
          i_br_ss: Branch index
          dm_comb : Combined dark matter mass
          dm_ini  : Initial dark matter mass according to the merger tree

        '''

        # Calculate the extra dark matter added (or removed if negative)
        dm_added = dm_ini - dm_comb

        # If we need to add primordial gas ..
        if dm_added > 0.0:

            # Calculate the baryonic mass added
            dm_bar_added = dm_added * self.o_ini.omega_b_0 / self.o_ini.omega_0
            self.dm_bar_added_iso[i_z_ss][i_br_ss] = self.prim_x_frac * dm_bar_added

            # Add or remove the mass to the halo
            self.kwargs["ymgal_outer_ini"] += self.dm_bar_added_iso[i_z_ss][i_br_ss]

        # If we need to remove gas ..
        else:

            # Calculate the fraction of halo mass that needs to stay
            f_keep_temp = dm_ini / dm_comb

            # Make sure not to remove more than available
            if f_keep_temp < 0.0:
                if np.sum(self.kwargs["ymgal_outer_ini"]) < abs(dm_added):
                    print ('Warning - Not enough outer gas for removal.', f_keep_temp)
                    f_keep_temp = 0.0

            # Correct the gas halo mass
            self.kwargs["ymgal_outer_ini"] *= f_keep_temp


    ##############################################
    #               Create Galaxy                #
    ##############################################
    def __create_galaxy(self, i_z_ss, i_br_ss):

        '''
        Create an OMEGA or OMEGA_SAM instance.

        Arguments
        =========

          i_z_ss : Redshift index
          i_br_ss : Branch index

        '''

        # Define whether the branch will form stars
        if self.len_br_is_SF > 0:
            br_is_SF_temp = self.br_is_SF[i_z_ss][i_br_ss]
        else:
            if self.kwargs["DM_array"][-1][1] >= self.mvir_sf_tresh:
                br_is_SF_temp = True
            else:
                br_is_SF_temp = False

        # Assigned pre-defined star formation timescale
        # and efficiency .. if provided
        br_is_SF_t_temp = np.array([])
        if self.len_br_is_SF_t > 0:
            br_is_SF_t_temp = self.br_is_SF_t[i_z_ss][i_br_ss]
        br_sfe_t_temp = np.array([])
        if self.len_br_sfe_t > 0:
            max_sfe = max(self.br_sfe_t[i_z_ss][i_br_ss])
            if max_sfe > 0.0:
                br_sfe_t_temp = [max_sfe]*self.len_br_sfe_t

        # Assigned pre-defined star formation history .. if provided
        br_sfh_temp = np.array([])
        if self.len_br_sfh > 0:
            br_sfh_temp = self.br_sfh[i_z_ss][i_br_ss]

        # Create an OMEGA+ instance (GAMMA branch)
        self.kwargs["is_SF_t"] = br_is_SF_t_temp
        self.kwargs["sfe_t"] = br_sfe_t_temp
        self.kwargs["sfh_with_sfe"] = br_sfh_temp
        self.kwargs["is_SF"] = br_is_SF_temp
        self.kwargs["t_merge"] = self.br_t_merge[i_z_ss][i_br_ss]
        return (i_br_ss, omega_plus.omega_plus(**self.kwargs))
