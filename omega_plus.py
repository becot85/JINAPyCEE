from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

'''
test
OMEGA Plus, 2-zone model with a central galaxy sourrounded
by a circumgalactic medium. 

 - Feb2017: B. Cote


Definitions

    Inner : Galaxy (or the center of the dark matter halo)
    Outer : Gas surrounding the inner part (or halo gas)


Sequential steps for each timestep (computing procedure)

    1 - Star formation
    2 - Stellar ejecta
    3 - Galactic inflow
    4 - Galactic outflow
    5 - Halo inflow
    6 - Halo outflo


The inner region is represented by an OMEGA simulation:

    OMEGA (One-zone Model for the Evolution of GAlaxies) module

'''


# Standard packages
import numpy as np
from imp import *
from pylab import * 
import time as t_module
import copy
import math
import os
import re
import imp

global notebookmode
notebookmode=True

# Define where is the working directory
# This is where the NuPyCEE code will be extracted
global global_path
try:
    if os.environ['SYGMADIR']:
        global_path = os.environ['SYGMADIR']
except KeyError:
    global_path=os.getcwd()
global_path=global_path+'/'

# This is where the JINAPyCEE code will be extracted
global global_path_jinapycee
try:
    if os.environ['JINAPYDIR']:
        global_path_jinapycee = os.environ['JINAPYDIR']
except KeyError:
    global_path_jinapycee=os.getcwd()
global_path_jinapycee=global_path_jinapycee+'/'

# Import NuPyCEE codes
ry = imp.load_source('read_yields', global_path+'read_yields.py')
omega = imp.load_source('omega', global_path+'omega.py')


#####################
# Class Declaration #
#####################

class omega_plus():

    # Initialisation function
    def __init__(self, Z_trans=1e-20, f_dyn=0.1, sfe=0.01, \
                 m_DM_0=1.0e12, mass_loading=1.0, t_star=-1, \
                 z_dependent=True, exp_ml=2.0, imf_type='kroupa', \
                 alphaimf=2.35, imf_bdys=[0.1,100], sn1a_rate='power_law', \
                 iniZ=0.0, dt=1e6, special_timesteps=60, tend=13e9, \
                 mgal=1.0, transitionmass=8, ini_alpha=True, \
                 table='yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt', \
                 table_radio='', decay_file='', sn1a_table_radio='',\
                 bhnsmerger_table_radio='', nsmerger_table_radio='',\
                 nsmerger_table = 'yield_tables/r_process_arnould_2007.txt', \
                 sn1a_table='yield_tables/sn1a_t86.txt', radio_refinement=1, \
                 pop3_table='yield_tables/popIII_heger10.txt', \
                 hardsetZ=-1, sn1a_on=True, t_nsm_coal=-1.0, nb_nsm_per_m=-1,\
                 ns_merger_on=False, f_binary=1.0, f_merger=0.0008, \
                 t_merger_max=1.0e10, m_ej_nsm = 2.5e-02, iniabu_table='', \
                 imf_bdys_pop3=[0.1,100], imf_yields_range_pop3=[10,30], \
                 beta_pow=-1.0, gauss_dtd=[1e9,6.6e8], exp_dtd=2e9, \
                 nb_1a_per_m=1.0e-3, t_merge=-1.0, imf_yields_range=[1,30], \
                 exclude_masses=[], skip_zero=False, eta_norm=-1, redshift_f=0.0, \
                 print_off=False, long_range_ref=False, calc_SSP_ej=False, \
                 input_yields=False, popIII_info_fast=True, t_sf_z_dep=1.0, m_crit_on=False, \
                 norm_crit_m=8.0e+09, mass_frac_SSP=0.5, imf_rnd_sampling=False, \
                 halo_in_out_on=True, pre_calculate_SSPs=False, m_outer_ini=-1.0, \
                 epsilon_sne_halo=-1, nb_ccsne_per_m=0.01, epsilon_sne_gal=-1, \
                 sfe_m_index=1.0, halo_out_index=1.0, is_SF=True, sfe_m_dep=False, \
                 gal_out_index=1.0, f_halo_to_gal_out=-1, beta_crit=1.0, \
                 DM_outflow_C17=False, m_cold_flow_tresh=-1, C17_eta_z_dep=True, \
                 Grackle_on=False, f_t_ff=1.0, t_inflow=-1.0, t_ff_index=1.0, \
                 use_decay_module=False, yield_tables_dir='',\
                 delayed_extra_log=False, delayed_extra_yields_log_int=False, \
                 r_vir_array=np.array([]), nsm_dtd_power=np.array([]), \
                 dt_in_SSPs=np.array([]), SSPs_in=np.array([]), is_SF_t=np.array([]), \
                 DM_array=np.array([]), ism_ini=np.array([]), ism_ini_radio=np.array([]), \
                 mdot_ini=np.array([]), mdot_ini_t=np.array([]), ytables_in=np.array([]), \
                 zm_lifetime_grid_nugrid_in=np.array([]), isotopes_in=np.array([]), \
                 ytables_pop3_in=np.array([]), zm_lifetime_grid_pop3_in=np.array([]), \
                 ytables_1a_in=np.array([]), ytables_nsmerger_in=np.array([]),\
                 dt_in=np.array([]), dt_split_info=np.array([]), ej_massive=np.array([]), \
                 ej_agb=np.array([]), ej_sn1a=np.array([]), ej_massive_coef=np.array([]),\
                 ej_agb_coef=np.array([]), ej_sn1a_coef=np.array([]), m_trans_in=np.array([]),\
                 mass_sampled=np.array([]), scale_cor=np.array([]), \
                 poly_fit_dtd_5th=np.array([]), poly_fit_range=np.array([]), \
                 outer_ini_f=np.array([]), ymgal_outer_ini=np.array([]), \
                 sne_L_feedback=np.array([]), sfe_t=np.array([]), \
                 sfh_with_sfe=np.array([]),\
                 delayed_extra_dtd=np.array([]), delayed_extra_dtd_norm=np.array([]), \
                 delayed_extra_yields=np.array([]), delayed_extra_yields_norm=np.array([]), \
                 dmo_ini=np.array([]), dmo_ini_t=np.array([]),\
                 delayed_extra_yields_radio=np.array([]), \
                 delayed_extra_yields_norm_radio=np.array([]), \
                 ytables_radio_in=np.array([]), radio_iso_in=np.array([]), \
                 ytables_1a_radio_in=np.array([]), ytables_nsmerger_radio_in=np.array([]),\
                 test_clayton=np.array([]), exp_infall=np.array([]), m_inflow_in=np.array([]),\
                 is_sub_array=np.array([])):

        # Announce the beginning of the simulation 
        if not print_off:
            print ('OMEGA+ run in progress..')
        start_time = t_module.time()
        self.start_time = start_time

        # Set the initial mass of the inner reservoir
        if mgal > 0.0:
            the_mgal = mgal
        else:
            the_mgal = 1.0

        # Declare the inner region (OMEGA instance)
        self.inner = omega.omega(in_out_control=True, SF_law=False, DM_evolution=False, \
            sfe=sfe, t_star=t_star, mass_loading=mass_loading, external_control=True, \
            Z_trans=Z_trans, imf_type=imf_type, alphaimf=alphaimf, imf_bdys=imf_bdys, \
            sn1a_rate=sn1a_rate, iniZ=iniZ, dt=dt, special_timesteps=special_timesteps, \
            tend=tend, mgal=the_mgal, transitionmass=transitionmass, table=table, \
            sn1a_on=sn1a_on, sn1a_table=sn1a_table, ns_merger_on=ns_merger_on, \
            table_radio=table_radio, decay_file=decay_file,\
            sn1a_table_radio=sn1a_table_radio, nb_nsm_per_m=nb_nsm_per_m,\
            bhnsmerger_table_radio=bhnsmerger_table_radio,\
            nsmerger_table_radio=nsmerger_table_radio, ism_ini_radio=ism_ini_radio,\
            f_binary=f_binary, f_merger=f_merger, t_merger_max=t_merger_max, \
            m_ej_nsm=m_ej_nsm, nsmerger_table=nsmerger_table, iniabu_table=iniabu_table, \
            pop3_table=pop3_table, imf_bdys_pop3=imf_bdys_pop3, exp_ml=exp_ml, \
            m_crit_on=m_crit_on, norm_crit_m=norm_crit_m, beta_crit=beta_crit, \
            imf_yields_range_pop3=imf_yields_range_pop3, beta_pow=beta_pow, \
            gauss_dtd=gauss_dtd, exp_dtd=exp_dtd, nb_1a_per_m=nb_1a_per_m, t_merge=t_merge, \
            imf_yields_range=imf_yields_range, exclude_masses=exclude_masses, \
            skip_zero=skip_zero, redshift_f=redshift_f, print_off=print_off, \
            calc_SSP_ej=calc_SSP_ej, input_yields=input_yields, popIII_info_fast=popIII_info_fast, \
            t_sf_z_dep=t_sf_z_dep, mass_frac_SSP=mass_frac_SSP, t_nsm_coal=t_nsm_coal, \
            imf_rnd_sampling=imf_rnd_sampling, ism_ini=ism_ini, mdot_ini=mdot_ini, \
            mdot_ini_t=mdot_ini_t, ytables_in=ytables_in, DM_array=DM_array, \
            zm_lifetime_grid_nugrid_in=zm_lifetime_grid_nugrid_in, r_vir_array=r_vir_array,\
            isotopes_in=isotopes_in, ytables_pop3_in=ytables_pop3_in,\
            zm_lifetime_grid_pop3_in=zm_lifetime_grid_pop3_in, ytables_1a_in=ytables_1a_in, \
            ytables_nsmerger_in=ytables_nsmerger_in, dt_in=dt_in, dt_split_info=dt_split_info, \
            delayed_extra_log=delayed_extra_log,\
            delayed_extra_yields_log_int=delayed_extra_yields_log_int,\
            ej_massive=ej_massive, ej_agb=ej_agb, ej_sn1a=ej_sn1a, \
            ej_massive_coef=ej_massive_coef, ej_agb_coef=ej_agb_coef, ej_sn1a_coef=ej_sn1a_coef, \
            mass_sampled=mass_sampled, scale_cor=scale_cor, m_DM_0=m_DM_0,\
            poly_fit_dtd_5th=poly_fit_dtd_5th, poly_fit_range=poly_fit_range,\
            pre_calculate_SSPs=pre_calculate_SSPs, dt_in_SSPs=dt_in_SSPs, SSPs_in=SSPs_in,\
            use_decay_module=use_decay_module, yield_tables_dir=yield_tables_dir, \
            delayed_extra_dtd=delayed_extra_dtd, delayed_extra_dtd_norm=delayed_extra_dtd_norm, \
            delayed_extra_yields=delayed_extra_yields, delayed_extra_yields_norm=delayed_extra_yields_norm, \
            delayed_extra_yields_radio=delayed_extra_yields_radio,\
            delayed_extra_yields_norm_radio=delayed_extra_yields_norm_radio,\
            ytables_radio_in=ytables_radio_in, radio_iso_in=radio_iso_in,\
            ytables_1a_radio_in=ytables_1a_radio_in,\
            ytables_nsmerger_radio_in=ytables_nsmerger_radio_in,\
            test_clayton=test_clayton, radio_refinement=radio_refinement,\
            nsm_dtd_power=nsm_dtd_power)

        # Parameters associated with OMEGA+
        self.m_outer_ini = m_outer_ini
        self.outer_ini_f = outer_ini_f
        self.ymgal_outer_ini = ymgal_outer_ini
        self.halo_in_out_on = halo_in_out_on
        self.t_merge = t_merge
        self.is_SF = is_SF
        self.is_SF_t = is_SF_t
        self.sne_L_feedback = sne_L_feedback
        self.len_sne_L_feedback = len(sne_L_feedback)
        self.G_cgs = 6.6743e-8 # [cm^3 g^-1 s^-2]
        self.E_51 = 1.0
        self.epsilon_sne_halo = epsilon_sne_halo
        self.nb_ccsne_per_m = nb_ccsne_per_m
        self.epsilon_sne_gal = epsilon_sne_gal
        self.sfe_m_dep = sfe_m_dep
        self.sfe_m_index = sfe_m_index
        self.halo_out_index = halo_out_index
        self.gal_out_index = gal_out_index
        self.f_halo_to_gal_out = f_halo_to_gal_out
        self.sfe_t = sfe_t
        self.sfh_with_sfe = sfh_with_sfe
        self.DM_outflow_C17 = DM_outflow_C17
        self.m_cold_flow_tresh = m_cold_flow_tresh
        self.C17_eta_z_dep = C17_eta_z_dep
        self.Grackle_on = Grackle_on
        self.dmo_ini = dmo_ini
        self.dmo_ini_t = dmo_ini_t
        self.f_t_ff = f_t_ff
        self.t_inflow = t_inflow
        self.t_ff_index = t_ff_index
        self.exp_infall = exp_infall
        self.nb_exp_infall = len(exp_infall)
        self.m_inflow_in = m_inflow_in
        self.len_m_inflow_in = len(m_inflow_in)
        self.is_sub_array = is_sub_array

        # If the sub-halo information is provided ..
        if len(self.is_sub_array) > 0:

            # Synchonize the timesteps
            self.__copy_is_sub_input()

        # If the sub-halo information is not provided ..
        else:
            self.is_sub = [False]*(self.inner.nb_timesteps+1)

        # If the dark matter mass is constant ..
        if len(DM_array) == 0:

            # Assign a constant mass to all timesteps
            for i_step_OMEGA in range(0,self.inner.nb_timesteps+1):
                self.inner.m_DM_t[i_step_OMEGA] = m_DM_0

        # If the dark matter mass evolves ..
        else:

            # Use the input array and synchronize the timesteps
            self.inner.copy_DM_input()

        # Set physical constants
        self.G = 6.674e-8         # [cm3 g-1 s-2]
        self.m_p = 1.6726231e-24  # [g]
        self.mu = 0.6             # should be between 0.6 and 1.22
        self.k_b = 1.380658e-16   # [erg K-1]
        self.yr_in_s = 3.15569e7
        self.km_to_Mpc = 3.24077929e-20

        # Calculate the primordial composition (mass fraction) for inflows ..
        iniabu_table = 'yield_tables/iniabu/iniab_bb_walker91.txt'
        ytables_bb = ry.read_yield_sn1a_tables( \
            global_path+iniabu_table, self.inner.history.isotopes)
        self.prim_x_frac = ytables_bb.get(quantity='Yields')
        del ytables_bb

        # Calculate the baryonic fraction
        self.f_b_temp = self.inner.omega_b_0 / self.inner.omega_0

        # Calculate the SFE array
        self.__calculate_SFE()

        # Calculate the redshift for each timestep
        self.inner.calculate_redshift_t()

        # Calculate the virial radius and velocity for each timestep
        self.inner.calculate_virial()

        # Calculate the star formation timescale
        self.__calculate_SF_t()

        # Calculate the virial temperature
        self.__calculate_T_vir_t()

        # Calculate the average gas density within R_500
        self.__calculate_rho_500_t()

        # Initialize Grackle for gas cooling
        self.t_cool = np.zeros(self.inner.nb_timesteps)
        #if self.Grackle_on:
        #    self.__initialize_Grackle()

        # Declare the outer region
        self.__declare_outer()

        # Declare the mass ejected by delayed outflows .. if needed ..
        if self.len_sne_L_feedback > 0:
            self.delayed_m_outflow = [0.0]*self.inner.nb_timesteps
            if len(self.dmo_ini) > 0:
                self.__add_ext_dmo()
            self.dt_sne = self.sne_L_feedback[1] - self.sne_L_feedback[0]

        # Declare the number of CC SNe for the halo feedback
        self.nb_ccsne = [0.0]*self.inner.nb_timesteps
        self.gamma_cte = 7.792e8 * self.E_51 / self.G_cgs

        # If active SF is only allowed for specific timeframes ..
        self.SF_allowed_t = [True]*self.inner.nb_timesteps
        self.treat_sfh_with_sfe = False
        if len(self.sfe_t) > 0:
            print ('sfe_t option is not yet implemented.')
            self.treat_sfe_t = False
            #self.treat_sfe_t = True
            #self.__define_SF_timeframes()
        else:
            self.treat_sfe_t = False

        # Calculate the mass-loading constant if needed ..
        if self.DM_outflow_C17:
            if C17_eta_z_dep:
                self.eta_norm = self.inner.mass_loading * \
                    self.inner.m_DM_0**(self.inner.exp_ml*0.33333)* \
                        (1.0+self.inner.redshift_t[-1])**(0.5*self.inner.exp_ml)
            else:
                self.eta_norm = self.inner.mass_loading * \
                    self.inner.m_DM_0**(self.inner.exp_ml*0.33333)

        # Calculate the inflow timescale constant
        t_ff_final = self.f_t_ff * 0.1 * (1.0 + \
            self.inner.redshift_t[-1])**((-1.5)) / self.inner.H_0 * 9.7759839e11
        self.t_ff_cte = t_ff_final / (self.f_t_ff * 0.1 * (1.0 + \
            self.inner.redshift_t[-1])**((-1.5)*self.t_ff_index) / \
                self.inner.H_0 * 9.7759839e11)

        # Run the simulation
        self.__start_simulation()

        # Delete Grackle stuff ..
        #if self.Grackle_on:
        #    del self.my_chemistry

        # Announce the end of the simulation
        if not print_off:
            print ('   OMEGA+ run completed -',self.__get_time())


    ##############################################
    #                  Get SFE                   #
    ##############################################
    def __calculate_SFE(self):

        '''
        Calculate the SFE at each timestep

        Arguments
        =========

          i_step_OMEGA : Current timestep index of the OMEGA instance

        '''

        # Create the SFE array
        self.sfe = [0.0]*self.inner.nb_timesteps

        # If the SFE depends on the dark matter mass ..
        if self.sfe_m_dep:

            # Calculate the SFE with the DM mass in OMEGA
            m_DM_inv = 1.0 / self.inner.m_DM_0
            for i_sfe in range(0,self.inner.nb_timesteps):
                self.sfe[i_sfe] = self.inner.sfe * \
                    (self.inner.m_DM_t[i_sfe] * m_DM_inv)**(self.sfe_m_index)

        # If the SFE is constant with time ..
        else:

            # Use the value of OMEGA
            for i_sfe in range(0,self.inner.nb_timesteps):
                self.sfe[i_sfe] = self.inner.sfe


    ##############################################
    #               Calculate SF t               #
    ##############################################
    def __calculate_SF_t(self):

        '''
        Calculate the star formation timescale at each timestep

        '''

        # Calculate the real constant (without t_sf_z_dep)
        t_SF_t_final = self.inner.f_dyn * 0.1 * (1.0 + \
            self.inner.redshift_t[-1])**((-1.5)) / self.inner.H_0 * 9.7759839e11
        the_constant = t_SF_t_final / (self.inner.f_dyn * 0.1 * (1.0 + \
            self.inner.redshift_t[-1])**((-1.5)*self.inner.t_sf_z_dep) / \
                self.inner.H_0 * 9.7759839e11)

        # For each timestep ..
        for i_cSFt in range(0,self.inner.nb_timesteps+1):

            # Calculate the timescale
            if self.inner.t_star > 0.0:
                self.inner.t_SF_t[i_cSFt] = self.inner.t_star
            else:
                self.inner.t_SF_t[i_cSFt] = the_constant* self.inner.f_dyn * 0.1 * (1.0 + \
                    self.inner.redshift_t[i_cSFt])**((-1.5)*self.inner.t_sf_z_dep) / \
                        self.inner.H_0 * 9.7759839e11


    ##############################################
    #             Calculate T_vir t              #
    ##############################################
    def __calculate_T_vir_t(self):

        '''
        Calculate the virial temperature [K] at each timestep

        '''

        # Declare the array and the normalization constant
        self.T_vir_t = np.zeros(self.inner.nb_timesteps+1)
        cte_T_vir_t = self.mu * self.m_p / (2.0 * self.k_b) * 100000**2

        # For each timestep ..
        for i_cSFt in range(0,self.inner.nb_timesteps+1):

            # Calculate the virial temperature [K]
            self.T_vir_t[i_cSFt] = cte_T_vir_t * self.inner.v_vir_DM_t[i_cSFt]**2


    ##############################################
    #            Calculate rho_500 t             #
    ##############################################
    def __calculate_rho_500_t(self):

        '''
        Calculate the average gas density [g cm-3] within R_500 at each timestep

        '''

        # Declare the array and the normalization constant
        self.rho_500_t = np.zeros(self.inner.nb_timesteps+1)
        cte_rho_500_t = 3.0 / (8.0 * 3.1416 * self.G) * self.km_to_Mpc**2

        # For each timestep ..
        for i_cSFt in range(0,self.inner.nb_timesteps+1):

            # Calculate the Hubble parameter squared [km s-1 Mpc-1]**2
            H_squared = self.inner.H_0**2 * (self.inner.omega_0 *\
                (1+self.inner.redshift_t[i_cSFt])**3 + self.inner.lambda_0)

            # Calculate 500 times the critical density
            self.rho_500_t[i_cSFt] = 500.0 * cte_rho_500_t * H_squared


    ##############################################
    #             Initialize Grackle             #
    ##############################################
#    def __initialize_Grackle(self):

#        '''
#        Set the cooling option for the Grackle code
#
#        '''

#        # Set solver parameters
#        self.my_chemistry = chemistry_data()
#        self.my_chemistry.use_grackle = 1
#        self.my_chemistry.with_radiative_cooling = 1
#        self.my_chemistry.primordial_chemistry = 0 # specie
#        self.my_chemistry.metal_cooling = 1
#        self.my_chemistry.UVbackground = 0
#        self.my_chemistry.self_shielding_method = 0
#        self.grackle_dir = '/Users/benoitcote/Grackle/grackle/'

#        # Set units
#        self.my_chemistry.comoving_coordinates = 0 # proper units
#        self.my_chemistry.a_units = 1.0
#        self.my_chemistry.density_units = mass_hydrogen_cgs # rho = 1.0 is 1.67e-24 g
#        self.my_chemistry.length_units = cm_per_mpc         # 1 Mpc in cm
#        self.my_chemistry.time_units = sec_per_Myr          # 1 Myr in s

#        # Create array to keep track of the cooling rates
#        self.cooling_rate = np.zeros(self.inner.nb_timesteps)

#        # Choose Cloudy data
#        self.my_chemistry.grackle_data_file = os.sep.join(
#            [self.grackle_dir, 'input', 'CloudyData_noUVB.h5'])


    ##############################################
    #               Declare Outer                #
    ##############################################
    def __declare_outer(self):

        '''
        Create the external gas reservoir (outer/halo gas)

        '''

        # Assign the total composition mass if provided ..
        if len(self.ymgal_outer_ini) > 0:
            self.outer_ini_f = self.ymgal_outer_ini
            self.m_outer_ini = 1.0

        # Assume an primordial composition (mass fraction) if not provided ..
        if len(self.outer_ini_f) == 0:
            self.outer_ini_f = copy.deepcopy(self.prim_x_frac)

        # Convert into NumPy arrayay
        self.ymgal_outer = []
        self.ymgal_outer.append(np.array([]))
        self.ymgal_outer[0] = np.array(self.outer_ini_f)

        # If the total mass of the outer region is not provided ..
        if self.m_outer_ini <= 0.0:

            # Use the cosmological baryonic fraction
            self.m_outer_ini = self.inner.m_DM_t[0] * self.f_b_temp

        # Scale the composition currently in mass fraction
        self.ymgal_outer[0] *= self.m_outer_ini

        # Create the next timesteps (+1 to get the final state of the last dt)
        for i_do in range(1, self.inner.nb_timesteps+1):
            self.ymgal_outer.append(np.array([0.0]*self.inner.nb_isotopes))

        # Declare the total mass array
        self.sum_ymgal_outer = [0.0] * (self.inner.nb_timesteps+1)
        self.sum_ymgal_outer[0] = np.sum(self.ymgal_outer[0])

        # Declare the metallicity array
        self.outer_Z = [0.0]*(self.inner.nb_timesteps+1)
        self.__calculate_outer_Z(0)

        # Declare the mass lost from the halo
        self.m_lost_t = [0.0]*(self.inner.nb_timesteps)


    ##############################################
    #                Add Ext DMO                 #
    ##############################################
    def __add_ext_dmo(self):

        '''
        This function adds the delayed mass outflow (DMO) of external galaxies that
        just merged. This function synchronize the timesteps to the current galaxy.

        Notes
        =====

            i_ext : Step index in the "external" merging mdot array
            i_cur : Step index in the "current" galaxy mdot array
            t_cur_prev : Lower time limit in the current i_cur bin
            t_cur : Upper time limit in the current i_cur bin

            dmo_ini has an extra slot in the isotopes for the time,
            which is t = 0.0 for i_ext = 0.
         
        '''

        # For every merging galaxy (every branch of a merger tree)
        for i_merg in range(0,len(self.dmo_ini)):

            # Initialisation of the local variables
            i_ext = 0
            i_cur = 0
            t_cur_prev = 0.0
            t_cur = self.inner.history.timesteps[0]
            t_ext_prev = 0.0
            t_ext = self.dmo_ini_t[i_merg][i_ext+1]

            # While the external ejecta has not been fully transfered...
            len_dmo_ini_i_merg = len(self.dmo_ini[i_merg])
            while i_ext < len_dmo_ini_i_merg and i_cur < self.inner.nb_timesteps:

                # While we need to change the external time bin ...
                while t_ext <= t_cur:

                    # Calculate the overlap time between ext. and cur. bins
                    dt_trans = t_ext - max([t_ext_prev, t_cur_prev])

                    # Calculate the mass fraction that needs to be transfered
                    f_dt = dt_trans / (t_ext - t_ext_prev)

                    # Transfer the mass in the current mdot array
                    self.delayed_m_outflow[i_cur] += \
                        self.dmo_ini[i_merg][i_ext] * f_dt

                    # Move to the next external bin
                    i_ext += 1
                    if i_ext == (len_dmo_ini_i_merg):
                        break
                    t_ext_prev = t_ext
                    t_ext = self.dmo_ini_t[i_merg][i_ext+1]

                # Quit the loop if all external bins have been considered
                if i_ext == (len_dmo_ini_i_merg):
                    break

                # While we need to change the current time bin ...
                while t_cur < t_ext:

                    # Calculate the overlap time between ext. and cur. bins
                    dt_trans = t_cur - max([t_ext_prev, t_cur_prev]) 

                    # Calculate the mass fraction that needs to be transfered
                    f_dt = dt_trans / (t_ext - t_ext_prev)

                    # Transfer all isotopes in the current mdot array
                    self.delayed_m_outflow[i_cur] += \
                        self.dmo_ini[i_merg][i_ext] * f_dt

                    # Move to the next current bin
                    i_cur += 1
                    if i_cur == self.inner.nb_timesteps:
                        break
                    t_cur_prev = t_cur
                    t_cur += self.inner.history.timesteps[i_cur]


    ##############################################
    #            Define SF Timeframes            #
    ##############################################
#    def __define_SF_timeframes(self):

#        '''
#        Define the timesteps where SF is allowed. When it is not
#        allowed, there is no star formation (but there is inflow). Stars
#        still releases their ejecta and the halo still gets corrected
#        by the DM mass evolution. The is_SF_t needs to start with t=0.
#
#        '''

#        for iii in range(self.inner.nb_timesteps):
#            self.sfe[iii] = self.sfe_t[0]

        # Declare the current OMEGA age
#        t_O_up = 0.0

        # Declare current input timeframe step
#        i_in_temp = 0

        # Add an input timestep to cover all OMEGA timesteps
#        self.is_SF_t[0].append(self.inner.history.tend*1.1)
#        self.is_SF_t[1].append(self.is_SF_t[1][-1])
#        if self.treat_sfe_t:
#            self.sfe_t.append(self.sfe_t[-1])
#        if self.treat_sfh_with_sfe:
#            self.sfh_with_sfe.append(self.sfh_with_sfe[-1])

        # For each OMEGA timestep ..
#        for i_O_temp in range(0,self.inner.nb_timesteps-1):

            # Copy the timeframe (upper time boundary)
#            t_O_up += self.inner.history.timesteps[i_O_temp]

            # Count the number of input entries incluced in the OMEGA timestep
#            nb_counts = 0
#            SF_temp = False
#            if self.treat_sfe_t:
#                SFE_list = []
#            if self.treat_sfh_with_sfe:
#                SFH_with_list = []
#            while self.is_SF_t[0][i_in_temp] <= t_O_up:
#                if self.is_SF_t[1][i_in_temp]:
#                    SF_temp = True
#                    if self.treat_sfe_t:
#                        SFE_list.append(self.sfe_t[i_in_temp])
#                    if self.treat_sfh_with_sfe:
#                        SFH_with_list.append(self.sfh_with_sfe[i_in_temp])
#                nb_counts += 1
#                i_in_temp += 1

            # Define whether or not there is SF in this OMEGA timestep
            # No "i_in_temp+1" when nb_counts >=1 since the while loop already add a +1
#            if nb_counts >= 1:
#                self.SF_allowed_t[i_O_temp+1] = \
#                    (SF_temp or self.is_SF_t[1][i_in_temp] or self.is_SF_t[1][i_in_temp-1])
#                if self.treat_sfe_t:
#                    SFE_list.append(self.sfe_t[i_in_temp])
#                    SFE_list.append(self.sfe_t[i_in_temp-1])
#                if self.treat_sfh_with_sfe:
#                    SFH_with_list.append(self.sfh_with_sfe[i_in_temp])
#                    SFH_with_list.append(self.sfh_with_sfe[i_in_temp-1])
#            else:
#                self.SF_allowed_t[i_O_temp+1] = \
#                    (self.is_SF_t[1][i_in_temp-1] or self.is_SF_t[1][i_in_temp])
#                if self.treat_sfe_t:
#                    SFE_list.append(self.sfe_t[i_in_temp])
#                    SFE_list.append(self.sfe_t[i_in_temp-1])
#                if self.treat_sfh_with_sfe:
#                    SFH_with_list.append(self.sfh_with_sfe[i_in_temp])
#                    SFH_with_list.append(self.sfh_with_sfe[i_in_temp-1])

            # Select the maximum SFE and SFR and overwrite the current SFE
#            if self.treat_sfe_t:
#                self.sfe[i_O_temp+1] = max(SFE_list)
#            if self.treat_sfh_with_sfe:
#                self.sfh_t[i_O_temp+1] = max(SFH_with_list)

        # Treat the first OMEGA timestep
#        if self.treat_sfe_t:
#            self.sfe[0] = self.sfe[1]
#        if self.treat_sfh_with_sfe:
#            self.sfh_t[0] = self.sfh_t[1]
#        self.SF_allowed_t[0] = self.SF_allowed_t[1]


    ##############################################
    #              Start Simulation              #
    ##############################################
    def __start_simulation(self):


        # Define the end of active period (depends on whether a galaxy merger occur)
        if self.t_merge > 0.0:
            i_up_temp = self.inner.i_t_merger+1
        else:
            i_up_temp = self.inner.nb_timesteps

        # Reset the inflow and outflow rates to zero
        self.inner.m_outflow_t = np.zeros(self.inner.nb_timesteps)
        self.inner.m_inflow_t = np.zeros(self.inner.nb_timesteps)

        # For each timestep (defined by the OMEGA instance) ...
        for i_step_OMEGA in range(0,i_up_temp):

            # Calculate the total current gas mass in the inner region
            self.sum_inner_ymgal_cur = np.sum(self.inner.ymgal[i_step_OMEGA])

            # Calculate the star formation rate [Msun/yr]
            sfr_temp = self.__get_SFR(i_step_OMEGA)

            # Calculate the galactic outflow rate [Msun/yr]
            or_temp = self.__get_outflow_rate(i_step_OMEGA, sfr_temp)

            # Calculate the galactic inflow rate [Msun/yr] for all isotopes
            ir_iso_temp = self.__get_inflow_rate(i_step_OMEGA)
            #ir_iso_temp[-1] = 2.0e-3 * np.sum(ir_iso_temp)

            # Convert rates into total masses (except for SFR)
            m_lost = or_temp * self.inner.history.timesteps[i_step_OMEGA]
            m_added = ir_iso_temp *  self.inner.history.timesteps[i_step_OMEGA]
            sum_m_added = np.sum(m_added)
            if self.f_halo_to_gal_out >= 0.0:
                self.m_lost_for_halo = copy.deepcopy(m_lost)

            # Keep the values in memory
            self.inner.m_outflow_t[i_step_OMEGA] = m_lost
            self.inner.m_inflow_t[i_step_OMEGA] = sum_m_added

            # Limit the inflow rate if needed (the outflow rate is considered in OMEGA)
            if sum_m_added > np.sum(self.ymgal_outer[i_step_OMEGA]):
                m_added = copy.deepcopy(self.ymgal_outer[i_step_OMEGA])

            # If the IMF must be sampled ...
            m_stel_temp = sfr_temp * self.inner.history.timesteps[i_step_OMEGA]
            if self.inner.imf_rnd_sampling and self.inner.m_pop_max >= m_stel_temp:
                
                # Get the sampled masses
                mass_sampled = self.inner._get_mass_sampled(\
                    sfr_temp * self.inner.history.timesteps[i_step_OMEGA])

            # No mass sampled if using the full IMF ...
            else:
                mass_sampled = np.array([])
            if m_stel_temp < 1.0:
                sfr_temp = 0.0


            # Evolve the inner region for one step.  The 'i_step_OMEGA + 1'
            # is because OMEGA update the quantities in the next upcoming timestep
            self.inner.run_step(i_step_OMEGA+1, sfr_temp, mass_sampled=mass_sampled, \
                m_added=m_added, m_lost=m_lost, no_in_out=True)

            # Evolve the outer region for one step
            self.__evolve_outer(m_lost, m_added, i_step_OMEGA)

            # If the inner gas reservoir needs to be modified to recover
            # a pre-defined SFR for the next timestep ..
            if self.treat_sfh_with_sfe:
                if i_step_OMEGA < (self.inner.nb_timesteps-1) and \
                   self.sfh_t[i_step_OMEGA+1] > 0.0:

                    # Correct the mass of the inner region
                    self.__correct_inner_for_sfh(i_step_OMEGA)

        # Evolve the stellar population only .. if a galaxy merger occured
        if self.t_merge > 0.0:
            for i_step_last in range(i_step_OMEGA+1,self.inner.nb_timesteps):
                self.inner.run_step(i_step_last+1, 0.0, no_in_out=True)


    ##############################################
    #                  Get SFR                   #
    ##############################################
    def __get_SFR(self, i_step_OMEGA):

        '''
        Calculate and return the star formation rate of the inner region

        Arguments
        =========

          i_step_OMEGA : Current timestep index of the OMEGA instance

        '''

        # Use the classical SF law (SFR = f_s / t_s * M_gas)
        if self.is_SF:
            if self.SF_allowed_t[i_step_OMEGA]:
                if self.inner.m_crit_on:
                    if self.sum_inner_ymgal_cur <= self.inner.m_crit_t[i_step_OMEGA]:
                        m_gas_temp = 0.0
                    else:
                        #m_gas_temp = self.sum_inner_ymgal_cur - self.inner.m_crit_t[i_step_OMEGA]
                        m_gas_temp = self.sum_inner_ymgal_cur
                else:
                    m_gas_temp = copy.deepcopy(self.sum_inner_ymgal_cur)
                if self.treat_sfe_t:
                    return self.sfe[i_step_OMEGA] * m_gas_temp  
                else:
                    return self.sfe[i_step_OMEGA] * m_gas_temp / \
                           self.inner.t_SF_t[i_step_OMEGA]
            else:
                return 0.0
        else:
            return 0.0


    ##############################################
    #              Get Outflow Rate              #
    ##############################################
    def __get_outflow_rate(self, i_step_OMEGA, sfr_temp):

        '''
        Calculate and return the galactic outflow rate of the inner region

        Arguments
        =========

          i_step_OMEGA : Current timestep index of the OMEGA instance
          sfr_temp : Star formation rate of the current timestep [Msun/yr]

        '''

        # If we use the DM_evolution option from Cote et al. (2017) ..
        if self.DM_outflow_C17:
 
            # Calculate the mass-loading factor
            if self.C17_eta_z_dep:
                mass_loading_gor = self.eta_norm * \
                    self.inner.m_DM_t[i_step_OMEGA]**((-0.3333)*self.inner.exp_ml) * \
                        (1+self.inner.redshift_t[i_step_OMEGA])**(-(0.5)*self.inner.exp_ml)
            else:
                mass_loading_gor = self.eta_norm * \
                    self.inner.m_DM_t[i_step_OMEGA]**((-0.3333)*self.inner.exp_ml)

        # If the mass-loading follows Crosby et al. (2015) ..
        elif self.epsilon_sne_gal >= 0.0:

            # Calculate the mass-loading factor
            mass_loading_gor = self.gamma_cte * self.nb_ccsne_per_m * self.epsilon_sne_gal * \
                (self.inner.r_vir_DM_t[i_step_OMEGA]*0.001 / \
                    self.inner.m_DM_t[i_step_OMEGA])**self.gal_out_index

        # If the mass-loading is constant ..
        else:

            # Take the value from OMEGA
            mass_loading_gor = self.inner.mass_loading

        # Keep the mass-loading parameter in memory
        self.inner.eta_outflow_t[i_step_OMEGA] = mass_loading_gor

        # If the outflow follows the SFR ..
        if self.len_sne_L_feedback == 0:

            # Calculate the number of CC SNe in this timestep
            self.nb_ccsne[i_step_OMEGA] = sfr_temp * self.nb_ccsne_per_m * \
                self.inner.history.timesteps[i_step_OMEGA]

            # Use the OMEGA mass loading factor
            return mass_loading_gor * sfr_temp

        # If the outflow follows the SNe energy (delayed outflows) ..
        else:

            # Calculate the SSP mass
            SSP_mass_temp = sfr_temp * self.inner.history.timesteps[i_step_OMEGA]

            # Calculate the total mass ejected by this SSP
            m_ej_SSP_temp = mass_loading_gor * SSP_mass_temp

            # Calculate the normalization constant for the outflow rate
            # Here L_SNe is not present because it cancels out when integrating
            A_temp = m_ej_SSP_temp / self.dt_sne

            # Calculate the number of CC SNe per unit of time
            A_CC_temp = SSP_mass_temp * self.nb_ccsne_per_m / self.dt_sne

            # For all upcoming timesteps (including this one) ..
            age_SSP_temp = 0.0
            for i_gor in range(i_step_OMEGA, self.inner.nb_timesteps):

                # Copy the lower and upper SSP age boundaries for this timestep
                age_low = age_SSP_temp
                age_up = age_SSP_temp + self.inner.history.timesteps[i_gor]

                # Stop if no more SN will occur
                if (age_low >= self.sne_L_feedback[1]):
                    break

                # If SNe are occuring
                elif (age_up > self.sne_L_feedback[0]):

                    # Calculate the time fraction covered by SNe in this timestep
                    t_covered = min(age_up, self.sne_L_feedback[1]) - \
                                max(age_low, self.sne_L_feedback[0])

                    # Add the mass ejected to the global outflow array
                    self.delayed_m_outflow[i_gor] += A_temp * t_covered

                    # Add the number of CC SNe
                    self.nb_ccsne[i_step_OMEGA] += A_CC_temp * t_covered

                # Update the SSP age
                age_SSP_temp = age_SSP_temp + self.inner.history.timesteps[i_gor]

            # Return the outflow rate at i_step_OMEGA
            return self.delayed_m_outflow[i_step_OMEGA] / \
                self.inner.history.timesteps[i_step_OMEGA]


    ##############################################
    #              Get Inflow Rate               #
    ##############################################
    def __get_inflow_rate(self, i_step_OMEGA):

        '''
        Calculate and return the galactic inflow rate of the inner region
        for all isotopes

        Arguments
        =========

          i_step_OMEGA : Current timestep index of the OMEGA instance

        '''

        # If this is a star forming galaxy ..
        if self.is_SF and np.sum(self.ymgal_outer[i_step_OMEGA]) > 0.0:

          # If input exponential infall laws ..
          # For each infall episode, exp_infall --> [Norm, t_max, timescale]
          if self.nb_exp_infall > 0:
            cooling_rate = 0.0
            for i_in in range(self.nb_exp_infall):
                cooling_rate += self.exp_infall[i_in][0] * \
                    np.exp(-(self.inner.t - self.exp_infall[i_in][1]) / \
                        self.exp_infall[i_in][2])
            # Calculate the isotope cooling rates
            iso_rate_temp = np.zeros(self.inner.nb_isotopes)
            sum_ymgal_outer_temp = np.sum(self.ymgal_outer[i_step_OMEGA])
            if sum_ymgal_outer_temp > 0.0:
                m_tot_inv = 1.0 / sum_ymgal_outer_temp
                for j_gir in range(0,self.inner.nb_isotopes):
                    iso_rate_temp[j_gir] = cooling_rate * m_tot_inv * \
                       self.ymgal_outer[i_step_OMEGA][j_gir]

            # Return the inflow rate of all isotopes
            return np.array(iso_rate_temp)

          # If an input inflow mass is provided ..
          elif self.len_m_inflow_in > 0:
            cooling_rate = self.m_inflow_in[i_step_OMEGA]/\
                           self.inner.history.timesteps[i_step_OMEGA]
            # Calculate the isotope cooling rates
            iso_rate_temp = np.zeros(self.inner.nb_isotopes)
            sum_ymgal_outer_temp = np.sum(self.ymgal_outer[i_step_OMEGA])
            if sum_ymgal_outer_temp > 0.0:
                m_tot_inv = 1.0 / sum_ymgal_outer_temp
                for j_gir in range(0,self.inner.nb_isotopes):
                    iso_rate_temp[j_gir] = cooling_rate * m_tot_inv * \
                       self.ymgal_outer[i_step_OMEGA][j_gir]

            # Return the inflow rate of all isotopes
            return np.array(iso_rate_temp)

          else:

            # Calculate the free-fall timescale [yr]
            if self.t_inflow > 0.0:
                t_ff_temp = self.t_inflow
            else:
                t_ff_temp = self.t_ff_cte * self.f_t_ff * 0.1 * (1.0 + \
                    self.inner.redshift_t[i_step_OMEGA])**((-1.5)*self.t_ff_index) / \
                        self.inner.H_0 * 9.7759839e11
                # Constant is 1.1107 * 3.086e16 / 3.154e7

            # If Grackle is used to calculate the cooling timescale ..
            if False:
                dummy = 1.0
#            if self.Grackle_on:

#                # Choose Cloudy data
#                # Somehow, this line needs to be at the same place than the fc
#                # declaration.. otherwise the chemistry cannot be initialized .. weird
#                self.my_chemistry.grackle_data_file = os.sep.join(
#                    [self.grackle_dir, 'input', 'CloudyData_noUVB.h5'])

#                # Set the scale factor
#                self.my_chemistry.a_value = 1.0 / \
#                    (1.0 + self.inner.redshift_t[i_step_OMEGA]) / \
#                        self.my_chemistry.a_units
#                self.my_chemistry.velocity_units = \
#                    (self.my_chemistry.length_units / self.my_chemistry.time_units)

#                # Set the fluid container
# !! IF I use rho_500
#                fc = setup_fluid_container(self.my_chemistry, \
#                    temperature=self.T_vir_t[i_step_OMEGA], \
#                        density=self.rho_500_t[i_step_OMEGA], \
#                            metal_mass_fraction=self.outer_Z[i_step_OMEGA], converge=True)
# !! IF I use average halo gas density [g cm-3]
#                av_density = np.sum(self.ymgal_outer[i_step_OMEGA]) * 2.83489281e-31 / \
#                              self.inner.r_vir_DM_t[i_step_OMEGA]**3
#                fc = setup_fluid_container(self.my_chemistry, \
#                    temperature=self.T_vir_t[i_step_OMEGA], \
#                        density=av_density, \
#                            metal_mass_fraction=self.outer_Z[i_step_OMEGA], converge=True)

#                # Calculate the cooling time [yr]
#                fc.calculate_temperature()
#                fc.calculate_cooling_time()
#                cooling_time = fc["cooling_time"] * \
#                    self.my_chemistry.time_units / self.yr_in_s * -1.0

#                # Calculate the cooling rate
#                mu = fc.calculate_mean_molecular_weight()
#                density_proper = fc["density"] / \
#                    (self.my_chemistry.a_units * self.my_chemistry.a_value)**(3*\
#                        self.my_chemistry.comoving_coordinates)
#                e_CGS = fc["energy"] * self.my_chemistry.velocity_units**2
#                rho_CGS = density_proper * self.my_chemistry.density_units
#                n_CGS = rho_CGS / (mu * mass_hydrogen_cgs)
#                t_cool_CGS = np.abs(fc["cooling_time"] * self.my_chemistry.time_units)
#                self.cooling_rate[i_step_OMEGA] = e_CGS * rho_CGS / (t_cool_CGS * n_CGS**2)
#                del fc

#                # If the cooling time is negative (heating) ..
#                if cooling_time <= 0.0:                
#                    return np.zeros(self.inner.nb_isotopes)

#                # Choose free-fall time if the cooling time is shorter 
#                if cooling_time < t_ff_temp:
#                    cooling_time = t_ff_temp
#                self.t_cool[i_step_OMEGA] = copy.deepcopy(cooling_time)

            # If Grackle is not used ..
            else:

                # Use free-fall time
                cooling_time = t_ff_temp
                #cooling_time = t_ff_temp * (544507042.254/self.inner.m_DM_t[i_step_OMEGA])**0.5
                self.t_cool[i_step_OMEGA] = copy.deepcopy(cooling_time)

            # Get the total mass of the halo gas
            sum_ymgal_outer_temp = np.sum(self.ymgal_outer[i_step_OMEGA])

            # Calculate the total cooling rate [Msun/yr]
            cooling_rate = sum_ymgal_outer_temp / cooling_time

            # Calculate the isotope cooling rates
            iso_rate_temp = np.zeros(self.inner.nb_isotopes)
            if sum_ymgal_outer_temp > 0.0:
                m_tot_inv = 1.0 / sum_ymgal_outer_temp
                for j_gir in range(0,self.inner.nb_isotopes):
                    iso_rate_temp[j_gir] = cooling_rate * m_tot_inv * \
                       self.ymgal_outer[i_step_OMEGA][j_gir]

            # Return the inflow rate of all isotopes
            return np.array(iso_rate_temp)

        # If this is not a star forming galaxy, there is not inflow
        else:
            # Calculate the free-fall timescale [yr]
            if self.t_inflow > 0.0:
                t_ff_temp = self.t_inflow
            else:
                t_ff_temp = self.t_ff_cte * self.f_t_ff * 0.1 * (1.0 + \
                    self.inner.redshift_t[i_step_OMEGA])**((-1.5)*self.t_ff_index) / \
                        self.inner.H_0 * 9.7759839e11
            self.t_cool[i_step_OMEGA] = copy.deepcopy(t_ff_temp)
            return np.zeros(self.inner.nb_isotopes)


    ##############################################
    #                Evolve Outer                #
    ##############################################
    def __evolve_outer(self, m_lost, m_added, i_step_OMEGA):

        '''
        Evolve the outer region by adding and removing mass

        Arguments
        =========

          m_lost :  Total mass transfered from the inner to the outer region
          m_added : Mass transfered from the outer to the inner region
          i_step_OMEGA : Timestep index

        Notes
        =====

          The m_added parameter can either be total mass, or isotope masses.

          A - In OMEGA, inflows are applied before outflows. We thus need
          to remove gas from the outer region before adding the galactic
          inflows coming from the inner region.

          B - Outflows are the last process treated in an OMEGA timestep.
          Therefore, we can use the modified composition of the inner region
          (once star formation, stellar ejecta, and inflows are considered).
          This refers to the composition of the (i_step_OMEGA+1) timestep.

        '''

        # Convert the m_added total mass into individual isotopes .. if needed
        sum_ymgal_outer_temp =  np.sum(self.ymgal_outer[i_step_OMEGA])
        if type(m_added) == float:
            if sum_ymgal_outer_temp <= 0.0:
                f_added = 0.0
                m_added = np.zeros(self.inner.nb_isotopes)
            else:
                f_added = m_added / sum_ymgal_outer_temp
                if f_added > 1.0:
                    print ('!!Warning, inflows calculated by OMEGA exceed mass in outer region!!')
                    f_added = 1.0
                m_added = f_added * self.ymgal_outer[i_step_OMEGA]

        # Remove gas from the outer region (modify the state of the next timestep)
        self.ymgal_outer[i_step_OMEGA+1] = self.ymgal_outer[i_step_OMEGA] - m_added
        if np.sum(self.ymgal_outer[i_step_OMEGA+1]) < 0.0:
            self.ymgal_outer[i_step_OMEGA+1] *= 0.0

        # Calculate the mass fraction of each isotopes that will be added to the
        # outer region, relative to the total mass of the inner at dt+1
        sum_inner_ymgal_temp = np.sum(self.inner.ymgal[i_step_OMEGA+1])
        if sum_inner_ymgal_temp > 0.0:

            # Add gas added in the outer region (lost by the inner)
            f_add_temp = self.inner.m_outflow_t[i_step_OMEGA] / sum_inner_ymgal_temp
            self.ymgal_outer[i_step_OMEGA+1] +=  \
                f_add_temp * np.array(self.inner.ymgal[i_step_OMEGA+1])

        # If the outer region's mass follow the dark matter mass ..
        if self.halo_in_out_on:

            # Correct for the dark matter mass change
            self.__correct_outer_for_DM_variation(i_step_OMEGA)

        # Eject mass from the halo into the intergalactic medium
        self.__eject_halo(i_step_OMEGA)

        # If the halo is a sub-halo ..
        if self.is_sub[i_step_OMEGA]:

            # Remove all the gas of inside the CGM
            self.ymgal_outer[i_step_OMEGA+1] *= 0.0

        # Calculate the total gas mass of the outer region
        self.sum_ymgal_outer[i_step_OMEGA+1] = np.sum(self.ymgal_outer[i_step_OMEGA+1])

        # Calculate the metallicity of the outer region
        self.__calculate_outer_Z(i_step_OMEGA+1)
 

    ##############################################
    #       Correct Outer for DM Variation       #
    ##############################################
    def __correct_outer_for_DM_variation(self, i_step_OMEGA):

        '''
        Add gas to the outer region if the dark matter mass increases, or
        remove gas from the outer region if the dark matter mass decreases.

        Arguments
        =========

          i_step_OMEGA : Timestep index

        '''

        # Calculate the mass of dark matter added or removed
        dm_dm = self.inner.m_DM_t[i_step_OMEGA+1] - self.inner.m_DM_t[i_step_OMEGA]

        # If gas needs to be removed ..
        if dm_dm < 0.0:

            # Calculate the fraction of the halo that is not stripped.
            #f_keep = 1.0 - dm_dm / self.inner.m_DM_t[i_step_OMEGA]
            f_keep = self.inner.m_DM_t[i_step_OMEGA+1] / self.inner.m_DM_t[i_step_OMEGA]

            # Correct the outer gas for the stripping. Here we use i_step_OMEGA+1,
            # since this occurs once galactic inflows and outflows have occured.
            self.ymgal_outer[i_step_OMEGA+1] *= f_keep

        # If gas needs to be added ..
        elif not self.is_sub[i_step_OMEGA]:

            # Calculate the mass added in the outer gas
            dm_m_outer = dm_dm * self.f_b_temp

            # Add mass to the outer region with the same composition that
            # its initial composition. Here we use i_step_OMEGA+1, since
            # this occurs once galactic inflows and outflows have occured.
            # - Inner or outer region based on a M_DM threshold
            iso_add = self.prim_x_frac * dm_m_outer
            if self.inner.m_DM_t[i_step_OMEGA] > self.m_cold_flow_tresh:
                self.ymgal_outer[i_step_OMEGA+1] += iso_add
                #sum_temp = np.sum(self.ymgal_outer[i_step_OMEGA+1])
                #f_temp = (dm_m_outer + sum_temp) / sum_temp
                #self.ymgal_outer[i_step_OMEGA+1] *= f_temp
            else:
                self.inner.ymgal[i_step_OMEGA+1] += iso_add


    ##############################################
    #                 Eject Halo                 #
    ##############################################
    def __eject_halo(self, i_step_OMEGA):

        '''
        Remove gas from the halo and makes it disapear from the system.
        Ejection into the intergalactic medium. Eq (10) of Crosby et al. (2015)

        Arguments
        =========

          i_step_OMEGA : Timestep index

        '''

        # If the halo outflows is following the galactic outflow rate..
        if self.f_halo_to_gal_out >= 0.0:

            # Use the galactic outflow rate times the input factor
            m_lost = self.m_lost_for_halo * self.f_halo_to_gal_out

        # Calculate the mass ejected from the halo (Crosby et al. 2015)
        else:

            # In OMEGA, R_vir is in [kpc] while in Crosby+15 R_vir is in [Mpc]
            m_lost = self.gamma_cte * self.nb_ccsne[i_step_OMEGA] * self.epsilon_sne_halo * \
                (self.inner.r_vir_DM_t[i_step_OMEGA]*0.001 / \
                    self.inner.m_DM_t[i_step_OMEGA])**self.halo_out_index

        # Remove the mass from the halo (make sure to limit the mass ejected)
        if m_lost > 0.0:
            if m_lost > np.sum(self.ymgal_outer[i_step_OMEGA+1]):
                if not self.inner.print_off:
                    print ('Halo outflow empties the halo!!')
                self.ymgal_outer[i_step_OMEGA+1] *= 0.0
            else:
                f_remnant = 1.0 - m_lost/np.sum(self.ymgal_outer[i_step_OMEGA+1])
                self.ymgal_outer[i_step_OMEGA+1] *= f_remnant

        # Keep the mass loss in memory
        self.m_lost_t[i_step_OMEGA] = m_lost


    ##############################################
    #              Calculate Outer Z             #
    ##############################################
    def __calculate_outer_Z(self, i_step_OMEGA):

        '''
        Calculate the metallicity of the outer gas region

        Arguments
        =========

          i_step_OMEGA : Timestep index

        '''

        # Calculate the total mass metals
        mmetal = 0.0
        nonmetals = ['H-1','H-2','H-3','He-3','He-4','Li-6','Li-7']
        for k_coZ in range(0,self.inner.nb_isotopes): 
            if not self.inner.history.isotopes[k_coZ] in nonmetals:
                mmetal += self.ymgal_outer[i_step_OMEGA][k_coZ]

        # In the case where there is no gas left
        if self.sum_ymgal_outer[i_step_OMEGA] <= 0.0:
            self.outer_Z[i_step_OMEGA] = 0.0

        # If gas left, calculate the mass fraction of metals
        else:
            self.outer_Z[i_step_OMEGA] = mmetal / self.sum_ymgal_outer[i_step_OMEGA]


    ##############################################
    #            Correct Inner for SFH           #
    ##############################################
    def __correct_inner_for_sfh(self, i_step_OMEGA):

        '''
        Add/remove gas in the inner region is some is missing/too_much
        with respect to an input SFR. If not enough gas is in the halo,
        primordial gas is added with all the gas halo.

        Arguments
        =========

          i_step_OMEGA : Timestep index

        '''

        # Calculate the total current gas mass in the inner region
        self.sum_inner_ymgal_cur = np.sum(self.inner.ymgal[i_step_OMEGA+1])

        # Get the SFR of the next timestep if no gas is added
        sfr_next = self.__get_SFR(i_step_OMEGA+1)

        # Calculate the star formation rate ratio (input / what would happen)
        if sfr_next <= 0.0:
            empty_inner = True
            ratio_SFR = 100.0 # Dummy value to enter the "else" statement
        else:
            empty_inner = False
            ratio_SFR = self.sfh_t[i_step_OMEGA+1] / sfr_next

        # If gas needs to be removed ..
        if ratio_SFR <= 1.0:

            # Calculate the mass (isotopes) of gas that needs to be removed
            f_rem_temp = 1.0 - ratio_SFR
            m_rem_iso_temp = f_rem_temp * self.inner.ymgal[i_step_OMEGA+1]

            # Add the mass to the halo, and remove it from the inner region
            # "+1" for inner because we want to modify the next timestep
            self.inner.ymgal[i_step_OMEGA+1] -= m_rem_iso_temp
            self.ymgal_outer[i_step_OMEGA+1] += m_rem_iso_temp

        # If gas needs to be added ..
        else:

            # Calculate the mass of gas that needs to be added
            if empty_inner:
                if self.treat_sfe_t:
                    m_add_temp = self.sfh_t[i_step_OMEGA+1] / self.sfe[i_step_OMEGA+1]  
                else:
                    m_add_temp = self.sfh_t[i_step_OMEGA+1] * \
                          self.inner.t_SF_t[i_step_OMEGA+1] / self.sfe[i_step_OMEGA+1]
            else:
                m_add_temp = (ratio_SFR - 1) * self.sum_inner_ymgal_cur

            # Calculate the total gas mass in the halo
            m_gas_halo_temp = np.sum(self.ymgal_outer[i_step_OMEGA+1])

            # If enough gas is in the halo ..
            if m_add_temp <= m_gas_halo_temp:

                # Calculate the mass (isotopes) of the halo gas that needs to be added
                f_add_temp = m_add_temp / m_gas_halo_temp
                m_add_iso_temp = f_add_temp * self.ymgal_outer[i_step_OMEGA+1]

                # Remove the mass from the halo, and add it to the inner region
                self.inner.ymgal[i_step_OMEGA+1] += m_add_iso_temp
                self.ymgal_outer[i_step_OMEGA+1] -= m_add_iso_temp

            # If not enough gas is in the halo ..
            else:

                # Remove all the mass from the halo, and add it to the inner region
                self.inner.ymgal[i_step_OMEGA+1] += self.ymgal_outer[i_step_OMEGA+1]
                self.ymgal_outer[i_step_OMEGA+1] *= 0.0

                # Calculate the mass of primordial gas and add it in the inner region
                m_add_prim_temp = m_add_temp - m_gas_halo_temp
                self.inner.ymgal[i_step_OMEGA+1] += m_add_prim_temp * self.prim_x_frac

        # Calculate the total gas mass of the outer region
        self.sum_ymgal_outer[i_step_OMEGA+1] = np.sum(self.ymgal_outer[i_step_OMEGA+1])

        # Calculate the metallicity of the outer region
        self.__calculate_outer_Z(i_step_OMEGA+1)


    ##############################################
    #             Copy Is Sub Input              #
    ##############################################
    def __copy_is_sub_input(self):

        # Declare the boolean array saying whether the halo is a sub-halo
        self.is_sub = [False]*(self.inner.nb_timesteps+1)

        # Variable to keep track of the OMEGA's timestep
        i_dt_csa = 0
        t_csa = 0.0
        nb_dt_csa = self.inner.nb_timesteps

        # For every timestep given in the array
        for i_csa in range(1,len(self.is_sub_array)):

            # While we stay in the same time bin ...
            while t_csa <= self.is_sub_array[i_csa][0]:

                # Assign the input value
                self.is_sub[i_dt_csa] = self.is_sub_array[i_csa-1][1]

                # Exit the loop if the array is full
                if i_dt_csa >= nb_dt_csa:
                    break

                # Calculate the new time
                t_csa += self.inner.history.timesteps[i_dt_csa]
                i_dt_csa += 1

            # Exit the loop if the array is full
            if i_dt_csa >= nb_dt_csa:
                break

        # If the array has been read completely, but the OMEGA+ array is
        # not full, fil the rest of the array with the last input value
        while i_dt_csa < nb_dt_csa+1:
            self.is_sub[i_dt_csa] = self.is_sub_array[-1][1]
            i_dt_csa += 1


    ##############################################
    #                  Get Time                  #
    ##############################################
    def __get_time(self):

        out = 'Run time: ' + \
        str(round((t_module.time() - self.start_time),2))+"s"
        return out

