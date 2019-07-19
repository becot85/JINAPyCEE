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
    def __init__(self, in_out_control=False, SF_law=False, DM_evolution=False, \
                 Z_trans=1e-20, f_dyn=0.1, sfe=0.1, outflow_rate=-1.0, \
                 inflow_rate=-1.0, rand_sfh=0.0, cte_sfr=1.0, m_DM_0=1.0e11, \
                 mass_loading=1.0, t_star=-1.0, in_out_ratio=1.0, stellar_mass_0=-1.0, \
                 z_dependent=True, exp_ml=2.0, imf_type='kroupa', alphaimf=2.35, \
                 imf_bdys=[0.1,100], sn1a_rate='power_law', iniZ=0.0, dt=1e6, \
                 special_timesteps=30, tend=13e9, mgal=-1, transitionmass=8, ini_alpha=True, \
                 table='yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt', \
                 hardsetZ=-1, sn1a_on=True, t_nsm_coal=-1.0, \
                 sn1a_table='yield_tables/sn1a_t86.txt', ns_merger_on=False, \
                 f_binary=1.0, f_merger=0.0008, t_merger_max=1.0e10, m_ej_nsm = 2.5e-02, \
                 nsmerger_table = 'yield_tables/r_process_rosswog_2014.txt', iniabu_table='', \
                 pop3_table='yield_tables/popIII_heger10.txt', \
                 imf_bdys_pop3=[0.1,100], imf_yields_range_pop3=[10,30], \
                 beta_pow=-1.0, gauss_dtd=[1e9,6.6e8], exp_dtd=2e9, nb_1a_per_m=1.0e-3, \
                 t_merge=-1.0, imf_yields_range=[1,30], exclude_masses=[], \
                 skip_zero=False, redshift_f=0.0, print_off=False, \
                 long_range_ref=False, calc_SSP_ej=False, input_yields=False, \
                 popIII_info_fast=True, t_sf_z_dep = 1.0, m_crit_on=False, norm_crit_m=8.0e+09, \
                 mass_frac_SSP=0.5, imf_rnd_sampling=False, cte_m_gas = -1.0, \
                 omega_dur=-1.0, tree_trunk_ID=-1, halo_in_out_on=True, \
                 pre_calculate_SSPs=False, gal_out_index=1.0, \
                 epsilon_sne_halo=0.0, nb_ccsne_per_m=0.01, epsilon_sne_gal=-1, \
                 sfe_m_index=1.0, halo_out_index=1.0, sfe_m_dep=False, \
                 DM_outflow_C17=False, m_cold_flow_tresh=-1, C17_eta_z_dep=True, \
                 f_halo_to_gal_out=-1, beta_crit=1.0, Grackle_on=True, \
                 f_t_ff=1.0, mvir_sf_tresh=-1, t_inflow=-1.0, t_ff_index=1.0, \
                 substeps = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384],\
                 tolerance = 1e-5, min_val = 1e-20,\
                 br_is_SF_t=np.array([]), br_r_vir=np.array([]), \
                 dt_in_SSPs=np.array([]), SSPs_in=np.array([]), \
                 M_array=np.array([]), ytables_in=np.array([]), \
                 isotopes_in=np.array([]), \
                 ytables_pop3_in=np.array([]), \
                 ytables_1a_in=np.array([]), ytables_nsmerger_in=np.array([]),\
                 dt_in=np.array([]), dt_split_info=np.array([]), ej_massive=np.array([]), \
                 ej_agb=np.array([]), ej_sn1a=np.array([]), ej_massive_coef=np.array([]),\
                 ej_agb_coef=np.array([]), ej_sn1a_coef=np.array([]), dt_ssp=np.array([]), \
                 m_trans_in=np.array([]), poly_fit_dtd_5th=np.array([]), \
                 poly_fit_range=np.array([]), nb_1a_ssp=np.array([]), \
                 redshifts=np.array([]), times=np.array([]), br_halo_ID=np.array([]), \
                 br_age=np.array([]), br_z=np.array([]), br_t_merge=np.array([]), \
                 br_ID_merge=np.array([]), br_m_halo=np.array([]), br_is_prim=np.array([]),\
                 br_is_SF=np.array([]), sne_L_feedback=np.array([]),\
                 br_sfe_t=np.array([]), br_sfh=np.array([]),
                 br_is_sub=np.array([])):

        # Check if we have the trunk ID
        if tree_trunk_ID < 0:
            print ('Error - GAMMA needs the tree_trunk_ID parameter.')
            return

        # Announce the beginning of the simulation
        print ('GAMMA run in progress..')
        start_time = t_module.time()
        self.start_time = start_time

        # Keep the OMEGA parameters in memory
        self.in_out_control = in_out_control
        self.SF_law = SF_law
        self.DM_evolution = DM_evolution
        self.Z_trans = Z_trans
        self.f_dyn = f_dyn
        self.sfe = sfe
        self.outflow_rate = outflow_rate
        self.inflow_rate = inflow_rate
        self.rand_sfh = rand_sfh
        self.cte_sfr = cte_sfr
        self.m_DM_0 = m_DM_0
        self.mass_loading = mass_loading
        self.t_star = t_star
        self.in_out_ratio = in_out_ratio
        self.stellar_mass_0 = stellar_mass_0
        self.z_dependent = z_dependent
        self.exp_ml = exp_ml
        self.imf_type = imf_type
        self.alphaimf = alphaimf
        self.imf_bdys = imf_bdys
        self.sn1a_rate = sn1a_rate
        self.iniZ = iniZ
        self.dt = dt
        self.special_timesteps = special_timesteps
        self.tend = tend
        self.mgal = mgal
        self.transitionmass = transitionmass
        self.ini_alpha = ini_alpha
        self.table = table
        self.hardsetZ = hardsetZ
        self.sn1a_on = sn1a_on
        self.sn1a_table = sn1a_table
        self.ns_merger_on = ns_merger_on
        self.f_binary = f_binary
        self.f_merger = f_merger
        self.t_merger_max = t_merger_max
        self.t_nsm_coal = t_nsm_coal
        self.m_ej_nsm = m_ej_nsm
        self.nsmerger_table = nsmerger_table
        self.iniabu_table = iniabu_table
        self.pop3_table = pop3_table
        self.imf_bdys_pop3 = imf_bdys_pop3
        self.imf_yields_range_pop3 = imf_yields_range_pop3
        self.beta_pow = beta_pow
        self.gauss_dtd = gauss_dtd
        self.exp_dtd = exp_dtd
        self.nb_1a_per_m = nb_1a_per_m
        self.t_merge = t_merge
        self.imf_yields_range = imf_yields_range
        self.exclude_masses = exclude_masses
        self.skip_zero = skip_zero
        self.redshift_f = redshift_f
        self.print_off = print_off
        self.long_range_ref = long_range_ref
        self.calc_SSP_ej = calc_SSP_ej
        self.input_yields = input_yields
        self.popIII_info_fast = popIII_info_fast
        self.t_sf_z_dep = t_sf_z_dep
        self.m_crit_on = m_crit_on
        self.norm_crit_m = norm_crit_m
        self.mass_frac_SSP = mass_frac_SSP
        self.imf_rnd_sampling = imf_rnd_sampling
        self.cte_m_gas = cte_m_gas
        self.omega_dur = omega_dur
        self.ytables_in = ytables_in
        self.isotopes_in = isotopes_in
        self.ytables_pop3_in = ytables_pop3_in
        self.ytables_1a_in = ytables_1a_in
        self.ytables_nsmerger_in = ytables_nsmerger_in
        self.dt_in = dt_in
        self.dt_split_info = dt_split_info
        self.ej_massive = ej_massive
        self.ej_agb = ej_agb
        self.ej_sn1a = ej_sn1a
        self.ej_massive_coef = ej_massive_coef
        self.ej_agb_coef = ej_agb_coef
        self.ej_sn1a_coef = ej_sn1a_coef
        self.m_trans_in = m_trans_in
        self.poly_fit_dtd_5th = poly_fit_dtd_5th
        self.poly_fit_range = poly_fit_range
        self.pre_calculate_SSPs = pre_calculate_SSPs
        self.dt_in_SSPs = dt_in_SSPs
        self.SSPs_in = SSPs_in
        self.epsilon_sne_halo = epsilon_sne_halo
        self.nb_ccsne_per_m = nb_ccsne_per_m
        self.epsilon_sne_gal = epsilon_sne_gal
        self.sfe_m_index = sfe_m_index
        self.sfe_m_dep = sfe_m_dep
        self.sne_L_feedback = sne_L_feedback
        self.len_sne_L_feedback = len(sne_L_feedback)
        self.halo_out_index = halo_out_index
        self.gal_out_index = gal_out_index
        self.f_halo_to_gal_out = f_halo_to_gal_out
        self.beta_crit = beta_crit
        self.DM_outflow_C17 = DM_outflow_C17
        self.m_cold_flow_tresh = m_cold_flow_tresh
        self.C17_eta_z_dep = C17_eta_z_dep
        self.Grackle_on = Grackle_on
        self.f_t_ff = f_t_ff
        self.t_inflow = t_inflow
        self.t_ff_index = t_ff_index
        self.substeps = substeps
        self.tolerance = tolerance
        self.min_val = min_val

        # Keep the GAMMA parameters in memory
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
        self.halo_in_out_on = halo_in_out_on
        self.br_sfe_t = br_sfe_t
        self.br_sfh = br_sfh
        self.len_br_is_SF = len(br_is_SF)
        self.len_br_sfe_t = len(br_sfe_t)
        self.len_br_is_SF_t = len(br_is_SF_t)
        self.len_br_sfh = len(br_sfh)
        self.br_r_vir = br_r_vir
        self.br_is_sub = br_is_sub
        self.mvir_sf_tresh = mvir_sf_tresh

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
    #               Initialisation               #
    ##############################################
    def __initialisation(self):

        '''
        Read the merger tree and declare and fill arrays.

        '''

        # Run an OMEGA simulation in order to copy basic arrays
        #print ('Should add the r-process tables, extra_yields_table, etc...')
        self.o_ini = omega.omega(table=self.table, pop3_table=self.pop3_table,\
                                 special_timesteps=2, cte_sfr=0.0, mgal=1e10,\
                                 print_off=self.print_off)

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

        # Get the final redshift
        self.redshift_f = min(self.redshifts)

        # Get the primordial composition (mass fraction)
        iniabu_table = 'yield_tables/iniabu/iniab_bb_walker91.txt'
        ytables_bb = ry.read_yield_sn1a_tables( \
            os.path.join(nupy_path, iniabu_table), self.o_ini.history.isotopes)
        self.prim_x_frac = np.array(ytables_bb.get(quantity='Yields'))
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

        # For each redshift ...
        for i_z_ss in range(0,self.nb_redshifts):

          # For new each branch ...
          for i_br_ss in range(0,len(self.br_m_halo[i_z_ss])):

            # If it's a primordial branch ...
            if self.br_is_prim[i_z_ss][i_br_ss]:

                # Create an OMEGA instance without merger
                self.__create_branch(i_z_ss, i_br_ss)

            # If the branch is the results of a merger ...
            else:

                # Get the stellar ejecta and the ISM of all parents
                mdot, mdot_t, ism, outer, dm, dmo, dmo_t = \
                    self.__get_mdot_parents(i_z_ss, i_br_ss)

                # Create an OMEGA instance with merger
                self.__create_branch(i_z_ss, i_br_ss, dm, mdot, mdot_t, \
                    ism, outer, dmo, dmo_t)


    ##############################################
    #               Create Branch                 #
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
        self.mdot_ini = mdot_ini
        self.mdot_ini_t = mdot_ini_t
        self.ism_ini = ism_ini
        self.ymgal_outer_ini = ymgal_outer_ini
        self.dmo_ini = dmo_ini
        self.dmo_ini_t = dmo_ini_t

        # Calculate the duration of the OMEGA instance
        self.tend = self.times[-1] - self.times[i_z_ss] + \
                   (self.times[-1] - self.times[-2]) # Extra step for the trunk

        # Assign the DM array
        self.DM_array = []
        for i_cb in range(0,len(self.br_age[i_z_ss][i_br_ss])):
            self.DM_array.append([0.0]*2)
            self.DM_array[i_cb][0] = self.br_age[i_z_ss][i_br_ss][i_cb]
            self.DM_array[i_cb][1] = self.br_m_halo[i_z_ss][i_br_ss][i_cb]
        self.DM_array = np.array(self.DM_array)

        # Assign the R_vir array
        self.r_vir_array = []
        for i_cb in range(0,len(self.br_age[i_z_ss][i_br_ss])):
            self.r_vir_array.append([0.0]*2)
            self.r_vir_array[i_cb][0] = self.br_age[i_z_ss][i_br_ss][i_cb]
            self.r_vir_array[i_cb][1] = self.br_r_vir[i_z_ss][i_br_ss][i_cb]
        self.r_vir_array = np.array(self.r_vir_array)

        # Assign whether or not the branch will be a sub-halo at some point
        self.is_sub_array = np.array([])
        if self.is_sub_info:
            for i_cb in range(0,len(self.br_is_sub[i_z_ss][i_br_ss])):
                self.is_sub_array.append([0.0]*2)
                self.is_sub_array[i_cb][0] = self.br_age[i_z_ss][i_br_ss][i_cb]
                self.is_sub_array[i_cb][1] = self.br_is_sub[i_z_ss][i_br_ss][i_cb]

        # Add mass depending on the dark matter mass ratio.
        # This is because the sum of dark matter masses from the progenitors
        # sometime does not equal the initial dark matter mass of the new branch..
        if dm > 0.0:
            self.__correct_initial_state(i_z_ss, i_br_ss, \
                dm, self.br_m_halo[i_z_ss][i_br_ss][0])

        # This is to print information when testing the code
        #if not self.print_off:
        #    print (' ')
        #    print ('Branch index :',i_z_ss, i_br_ss)
        #    print ('   Branch first ID :', self.br_halo_ID[i_z_ss][i_br_ss][0])
        #    print ('   Branch last ID :',self.br_ID_merge[i_z_ss][i_br_ss])
        #    print ('   M_DM_f :', self.br_m_halo[i_z_ss][i_br_ss][-1], 'Msun')

        # Create a galaxy instance with external_control = True
        self.__create_galaxy(i_z_ss, i_br_ss)


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
            self.ymgal_outer_ini += self.dm_bar_added_iso[i_z_ss][i_br_ss]

        # If we need to remove gas ..
        else:

            # Calculate the fraction of halo mass that needs to stay
            f_keep_temp = dm_ini / dm_comb

            # Make sure not to remove more than available
            if f_keep_temp < 0.0:
                if np.sum(self.ymgal_outer_ini) < abs(dm_added):
                    print ('OH GOD', f_keep_temp)
                    f_keep_temp = 0.0

            # Correct the gas halo mass
            self.ymgal_outer_ini *= f_keep_temp


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
            #if np.sum(self.galaxy_inst[iz][ibr].inner.history.m_locked) <= 0.0:
            #    ism += self.galaxy_inst[iz][ibr].ymgal_outer[\
            #       self.galaxy_inst[iz][ibr].inner.i_t_merger+1]
            #else:
            #    outer += self.galaxy_inst[iz][ibr].ymgal_outer[\
            #       self.galaxy_inst[iz][ibr].inner.i_t_merger+1]

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

        # If we are at the trunk ...
        if self.br_ID_merge[i_z_ss][i_br_ss] == self.tree_trunk_id:

            # No merger
            # The '-1.0' is because i_t_merger in OMEGA needs to
            # be the last timestep.  The calculation of i_t_merge
            # workds by scanning 't' until 't' > t_merge.  To avoid
            # index out of bound error, the '-1.0' is needed.
            self.t_merge = self.tend - 1.0
            self.t_merge = -1

        # Define whether the branch will form stars
        if self.len_br_is_SF > 0:
            br_is_SF_temp = self.br_is_SF[i_z_ss][i_br_ss]
        else:
            if self.DM_array[-1][1] >= self.mvir_sf_tresh:
                br_is_SF_temp = True
            else:
                br_is_SF_temp = False

        # Assigned pre-defined star formation timescale
        # and efficiency .. if provided
        if self.len_br_is_SF_t > 0:
            br_is_SF_t_temp = self.br_is_SF_t[i_z_ss][i_br_ss]
        else:
            br_is_SF_t_temp = np.array([])
        if self.len_br_sfe_t > 0:
            max_sfe = max(self.br_sfe_t[i_z_ss][i_br_ss])
            if max_sfe > 0.0:
                br_sfe_t_temp = [max_sfe]*self.len_br_sfe_t
            else:
                br_sfe_t_temp = np.array([])
        else:
            br_sfe_t_temp = np.array([])

        # Assigned pre-defined star formation history .. if provided
        if self.len_br_sfh > 0:
            br_sfh_temp = self.br_sfh[i_z_ss][i_br_ss]
        else:
            br_sfh_temp = np.array([])

        # Create an OMEGA or OMEGA_SAM instance
        self.galaxy_inst[i_z_ss][i_br_ss] = omega_plus.omega_plus(Z_trans=self.Z_trans, \
            f_dyn=self.f_dyn, sfe=self.sfe, mass_loading=self.mass_loading, \
            t_star=self.t_star, m_DM_0=self.m_DM_0, z_dependent=self.z_dependent,\
            exp_ml=self.exp_ml, imf_type=self.imf_type, alphaimf=self.alphaimf,\
            imf_bdys=self.imf_bdys, sn1a_rate=self.sn1a_rate, iniZ=self.iniZ,\
            dt=self.dt, special_timesteps=self.special_timesteps, tend=self.tend,\
            mgal=self.mgal, transitionmass=self.transitionmass, ini_alpha=self.ini_alpha,\
            table=self.table, hardsetZ=self.hardsetZ, sn1a_on=self.sn1a_on,\
            sn1a_table=self.sn1a_table, ns_merger_on=self.ns_merger_on,\
            f_binary=self.f_binary, f_merger=self.f_merger, \
            halo_in_out_on=self.halo_in_out_on, ymgal_outer_ini=self.ymgal_outer_ini, \
            t_merger_max=self.t_merger_max, m_ej_nsm=self.m_ej_nsm, \
            nsmerger_table=self.nsmerger_table, iniabu_table=self.iniabu_table,\
            pop3_table=self.pop3_table, imf_bdys_pop3=self.imf_bdys_pop3,\
            imf_yields_range_pop3=self.imf_yields_range_pop3,\
            beta_pow=self.beta_pow, gauss_dtd=self.gauss_dtd, exp_dtd=self.exp_dtd,\
            nb_1a_per_m=self.nb_1a_per_m, t_merge=self.br_t_merge[i_z_ss][i_br_ss],\
            imf_yields_range=self.imf_yields_range, exclude_masses=self.exclude_masses,\
            skip_zero=self.skip_zero, redshift_f=self.redshift_f,\
            pre_calculate_SSPs=self.pre_calculate_SSPs, dt_in_SSPs=self.dt_in_SSPs,\
            SSPs_in=self.SSPs_in, halo_out_index=self.halo_out_index,\
            print_off=self.print_off, long_range_ref=self.long_range_ref,\
            calc_SSP_ej=self.calc_SSP_ej, input_yields=True, \
            gal_out_index=self.gal_out_index, f_halo_to_gal_out=self.f_halo_to_gal_out, \
            popIII_info_fast=self.popIII_info_fast, t_sf_z_dep=self.t_sf_z_dep, \
            m_crit_on=self.m_crit_on, norm_crit_m=self.norm_crit_m,\
            t_nsm_coal=self.t_nsm_coal, imf_rnd_sampling=self.imf_rnd_sampling,\
            DM_array=self.DM_array, ism_ini=self.ism_ini, mdot_ini=self.mdot_ini, \
            mdot_ini_t=self.mdot_ini_t, ytables_in=self.o_ini.ytables, \
            isotopes_in=self.o_ini.history.isotopes,\
            ytables_1a_in=self.o_ini.ytables_1a, dt_in=self.dt_in,\
            dt_split_info=self.dt_split_info, ej_massive=self.ej_massive,\
            ej_agb=self.ej_agb, ej_sn1a=self.ej_sn1a, ej_massive_coef=self.ej_massive_coef,\
            ej_agb_coef=self.ej_agb_coef, ej_sn1a_coef=self.ej_sn1a_coef,\
            m_trans_in=self.m_trans_in, poly_fit_dtd_5th=self.poly_fit_dtd_5th,\
            poly_fit_range=self.poly_fit_range, is_SF=br_is_SF_temp,\
            epsilon_sne_halo=self.epsilon_sne_halo, nb_ccsne_per_m=self.nb_ccsne_per_m,\
            epsilon_sne_gal=self.epsilon_sne_gal, sfe_m_index=self.sfe_m_index,\
            sfe_m_dep=self.sfe_m_dep, sne_L_feedback=self.sne_L_feedback, \
            is_SF_t=br_is_SF_t_temp, sfe_t=br_sfe_t_temp, sfh_with_sfe=br_sfh_temp, \
            beta_crit=self.beta_crit, DM_outflow_C17=self.DM_outflow_C17, \
            m_cold_flow_tresh=self.m_cold_flow_tresh, C17_eta_z_dep=self.C17_eta_z_dep, \
            r_vir_array=self.r_vir_array, dmo_ini=self.dmo_ini, dmo_ini_t=self.dmo_ini_t, \
            f_t_ff=self.f_t_ff, Grackle_on=self.Grackle_on, t_inflow=self.t_inflow, \
            t_ff_index=self.t_ff_index, is_sub_array=self.is_sub_array, \
            inter_Z_points=self.o_ini.inter_Z_points, nb_inter_Z_points=self.o_ini.nb_inter_Z_points,\
            y_coef_M=self.o_ini.y_coef_M, y_coef_M_ej=self.o_ini.y_coef_M_ej,\
            y_coef_Z_aM=self.o_ini.y_coef_Z_aM, y_coef_Z_bM=self.o_ini.y_coef_Z_bM,\
            y_coef_Z_bM_ej=self.o_ini.y_coef_Z_bM_ej, tau_coef_M=self.o_ini.tau_coef_M,\
            tau_coef_M_inv=self.o_ini.tau_coef_M_inv, tau_coef_Z_aM=self.o_ini.tau_coef_Z_aM,\
            tau_coef_Z_bM=self.o_ini.tau_coef_Z_bM, tau_coef_Z_aM_inv=self.o_ini.tau_coef_Z_aM_inv,\
            tau_coef_Z_bM_inv=self.o_ini.tau_coef_Z_bM_inv, y_coef_M_pop3=self.o_ini.y_coef_M_pop3,\
            y_coef_M_ej_pop3=self.o_ini.y_coef_M_ej_pop3, tau_coef_M_pop3=self.o_ini.tau_coef_M_pop3,\
            tau_coef_M_pop3_inv=self.o_ini.tau_coef_M_pop3_inv,\
            inter_lifetime_points_pop3=self.o_ini.inter_lifetime_points_pop3,\
            inter_lifetime_points_pop3_tree=self.o_ini.inter_lifetime_points_pop3_tree,\
            nb_inter_lifetime_points_pop3=self.o_ini.nb_inter_lifetime_points_pop3,\
            inter_lifetime_points=self.o_ini.inter_lifetime_points,\
            inter_lifetime_points_tree=self.o_ini.inter_lifetime_points_tree,\
            nb_inter_lifetime_points=self.o_ini.nb_inter_lifetime_points,\
            nb_inter_M_points_pop3=self.o_ini.nb_inter_M_points_pop3,\
            inter_M_points_pop3_tree=self.o_ini.inter_M_points_pop3_tree,\
            nb_inter_M_points=self.o_ini.nb_inter_M_points,\
            inter_M_points=self.o_ini.inter_M_points,\
            substeps=self.substeps,tolerance=self.tolerance,\
            min_val=self.min_val,y_coef_Z_aM_ej=self.o_ini.y_coef_Z_aM_ej)

    ##############################################
    #                  Get Time                  #
    ##############################################
    def __get_time(self):

        out = 'Run time: ' + \
        str(round((t_module.time() - self.start_time),2))+"s"
        return out
