# coding=utf-8
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

'''

OMEGA Plus, 2-zone model with a central galaxy sourrounded
by a circumgalactic medium.

FEB2017: B. Cote
- Creation of the code

FEB2019: A. Yagüe, B. Cote
- Optimized to code to run faster

JAN2020: A. Yagüe
- Added tracking of enrichment sources in cgm

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
import time as t_module
import copy
import os

# Define where is the working directory
# This is where the NuPyCEE code will be extracted
nupy_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
nupy_path = os.path.join(nupy_path, "NuPyCEE")

# Import NuPyCEE codes
import NuPyCEE.read_yields as ry
import NuPyCEE.omega as omega
import JINAPyCEE.alpha as alpha


#####################
# Class Declaration #
#####################

class omega_plus():

    # Initialisation function
    def __init__(self, mgal=1.0, print_off=False, halo_in_out_on=True, \
                 m_outer_ini=-1.0, epsilon_sne_halo=0.0, nb_ccsne_per_m=0.01, \
                 epsilon_sne_gal=-1, sfe_m_index=1.0, halo_out_index=1.0, \
                 is_SF=True, sfe_m_dep=False, gal_out_index=1.0, f_halo_to_gal_out=-1, \
                 DM_outflow_C17=False, m_cold_flow_tresh=-1, C17_eta_z_dep=True, \
                 Grackle_on=False, f_t_ff=1.0, t_inflow=-1.0, t_ff_index=1.0, \
                 max_half_life=1e14, min_half_life=1000, t_merge=-1.0, \
                 substeps = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384],\
                 tolerance = 1e-5, min_val = 1e-20, print_param=False,\
                 is_SF_t=np.array([]), outer_ini_f=np.array([]), ymgal_outer_ini=np.array([]), \
                 sne_L_feedback=np.array([]), sfe_t=np.array([]), sfh_with_sfe=np.array([]),\
                 dmo_ini=np.array([]), dmo_ini_t=np.array([]), exp_infall=np.array([]), \
                 m_inflow_in=np.array([]), is_sub_array=np.array([]), **kwargs):

        # Not implemented yet
        if len(sne_L_feedback) > 0:
            print('The sne_L_feedback option is currently not available.')
            print('Simulation aborded.')
            return

        # Announce the beginning of the simulation
        if not print_off:
            print ('OMEGA+ run in progress..')
        start_time = t_module.time()
        self.start_time = start_time

        # Print parameters if asked for ..
        if print_param:
            dicto = locals()
            for key in dicto.keys():
                try:
                    if len(dicto[key]) > 0:
                        pass
                except TypeError:
                    print(key,'=',dicto[key])
                except:
                    raise

        # Reset the initial mass of the inner reservoir if needed
        if mgal <= 0:
            mgal = 1.0

        # Declare the inner region (OMEGA instance)
        kwargs["in_out_control"] = True
        kwargs["SF_law"] = False
        kwargs["DM_evolution"] = False
        kwargs["external_control"] = True
        kwargs["use_external_integration"] = True
        kwargs["mgal"] = mgal
        kwargs["print_off"] = print_off
        kwargs["t_merge"] = t_merge
        self.inner = omega.omega(**kwargs)

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
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.substeps = substeps
        self.tolerance = tolerance
        self.min_val = min_val

        # Get inflow rate if input array, and calculate the interpolation coefficients
        if self.len_m_inflow_in > 0:
            self.m_inflow_in_rate = np.zeros(self.inner.nb_timesteps)
            self.m_inflow_in_rate_coef = np.zeros((self.inner.nb_timesteps,2))
            for i_t in range(self.inner.nb_timesteps):
                self.m_inflow_in_rate[i_t] = self.m_inflow_in[i_t]/\
                           self.inner.history.timesteps[i_t]
            for i_t in range(self.inner.nb_timesteps-1):
                self.m_inflow_in_rate_coef[i_t][0] = (self.m_inflow_in_rate[i_t+1] - \
                    self.m_inflow_in_rate[i_t]) / self.inner.history.timesteps[i_t]
                self.m_inflow_in_rate_coef[i_t][1] = self.m_inflow_in_rate[i_t] - \
                    self.m_inflow_in_rate_coef[i_t][0] * self.inner.history.age[i_t]
            self.m_inflow_in_rate_coef[-1][0] = self.m_inflow_in_rate_coef[-2][0]
            self.m_inflow_in_rate_coef[-1][1] = self.m_inflow_in_rate_coef[-2][1]

        # If the sub-halo information is provided ..
        if len(self.is_sub_array) > 0:

            # Synchonize the timesteps
            self.__copy_is_sub_input()

        # If the sub-halo information is not provided ..
        else:
            self.is_sub = [False]*(self.inner.nb_timesteps+1)

        # If the dark matter mass is constant ..
        if len(self.inner.DM_array) == 0:

            # Assign a constant mass to all timesteps
            for i_step_OMEGA in range(0,self.inner.nb_timesteps+1):
                self.inner.m_DM_t[i_step_OMEGA] = self.inner.m_DM_0

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
        iniabu_table = os.path.join("yield_tables", "iniabu", "iniab_bb_walker91.txt")
        ytables_bb = ry.read_yields_Z( \
            os.path.join(nupy_path, iniabu_table), isotopes=self.inner.history.isotopes)
        self.prim_x_frac = ytables_bb.get(quantity='Yields', Z=0.0)
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
        self.SF_allowed_t = [True]*(self.inner.nb_timesteps+1)
        self.treat_sfh_with_sfe = False
        if len(self.sfe_t) > 0:
            print ('sfe_t option is not yet implemented.')
            self.treat_sfe_t = False
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

        self.__start_simulation()

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
        self.sfe = [0.0]*(self.inner.nb_timesteps+1)

        # If the SFE depends on the dark matter mass ..
        if self.sfe_m_dep:

            # Calculate the SFE with the DM mass in OMEGA
            m_DM_inv = 1.0 / self.inner.m_DM_0
            for i_sfe in range(0,self.inner.nb_timesteps+1):
                self.sfe[i_sfe] = self.inner.sfe * \
                    (self.inner.m_DM_t[i_sfe] * m_DM_inv)**(self.sfe_m_index)

        # If the SFE is constant with time ..
        else:

            # Use the value of OMEGA
            for i_sfe in range(0,self.inner.nb_timesteps+1):
                self.sfe[i_sfe] = self.inner.sfe

        # Create the interpolation coefficients
        # sfe = self.sfe_coef[0] * t + self.sfe_coef[1]
        self.sfe_coef = np.zeros((self.inner.nb_timesteps+1,2))
        for i_cmdt in range(self.inner.nb_timesteps):
            self.sfe_coef[i_cmdt][0] = (self.sfe[i_cmdt+1] - \
                self.sfe[i_cmdt]) / self.inner.history.timesteps[i_cmdt]
            self.sfe_coef[i_cmdt][1] = self.sfe[i_cmdt] - \
                self.sfe_coef[i_cmdt][0] * self.inner.history.age[i_cmdt]
        self.sfe_coef[-1][0] = self.sfe_coef[-2][0]
        self.sfe_coef[-1][1] = self.sfe_coef[-2][1]


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

        # Create the interpolation coefficients
        # t_SF = self.inner.t_SF_t_coef[0] * t + self.inner.t_SF_t_coef[1]
        self.inner.t_SF_t_coef = np.zeros((self.inner.nb_timesteps,2))
        for i_cmdt in range(0, self.inner.nb_timesteps):
            self.inner.t_SF_t_coef[i_cmdt][0] = (self.inner.t_SF_t[i_cmdt+1] - \
                self.inner.t_SF_t[i_cmdt]) / self.inner.history.timesteps[i_cmdt]
            self.inner.t_SF_t_coef[i_cmdt][1] = self.inner.t_SF_t[i_cmdt] - \
                self.inner.t_SF_t_coef[i_cmdt][0] * self.inner.history.age[i_cmdt]


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

        # Create the interpolation coefficients
        # T_vir = self.T_vir_t_coef[0] * t + self.T_vir_t_coef[1]
        self.T_vir_t_coef = np.zeros((self.inner.nb_timesteps,2))
        for i_cmdt in range(0, self.inner.nb_timesteps):
            self.T_vir_t_coef[i_cmdt][0] = (self.T_vir_t[i_cmdt+1] - \
                self.T_vir_t[i_cmdt]) / self.inner.history.timesteps[i_cmdt]
            self.T_vir_t_coef[i_cmdt][1] = self.T_vir_t[i_cmdt] - \
                self.T_vir_t_coef[i_cmdt][0] * self.inner.history.age[i_cmdt]


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

        # Create the interpolation coefficients
        # rho_500_t = self.rho_500_t_coef[0] * t + self.rho_500_t_coef[1]
        self.rho_500_t_coef = np.zeros((self.inner.nb_timesteps,2))
        for i_cmdt in range(0, self.inner.nb_timesteps):
            self.rho_500_t_coef[i_cmdt][0] = (self.rho_500_t[i_cmdt+1] - \
                self.rho_500_t[i_cmdt]) / self.inner.history.timesteps[i_cmdt]
            self.rho_500_t_coef[i_cmdt][1] = self.rho_500_t[i_cmdt] - \
                self.rho_500_t_coef[i_cmdt][0] * self.inner.history.age[i_cmdt]


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
            self.outer_ini_f = copy.copy(self.prim_x_frac)

        # Convert into NumPy array
        self.ymgal_outer = []
        self.ymgal_outer.append(np.array([]))
        self.ymgal_outer[0] = np.array(self.outer_ini_f)

        # Create ymgal_outer_radio
        self.ymgal_outer_radio = []
        if self.inner.len_decay_file > 0 or self.inner.use_decay_module:
            self.ymgal_outer_radio.append(np.array([0.0]*self.inner.nb_radio_iso))
        else:
            self.ymgal_outer_radio.append(np.array([0.0]))

        # If the total mass of the outer region is not provided ..
        if self.m_outer_ini <= 0.0:

            # Use the cosmological baryonic fraction
            self.m_outer_ini = self.inner.m_DM_t[0] * self.f_b_temp

        # Scale the composition currently in mass fraction
        self.ymgal_outer[0] *= self.m_outer_ini

        # Create the next timesteps (+1 to get the final state of the last dt)
        for i_do in range(1, self.inner.nb_timesteps+1):
            self.ymgal_outer.append(np.array([0.0]*self.inner.nb_isotopes))
            if self.inner.len_decay_file > 0 or self.inner.use_decay_module:
                self.ymgal_outer_radio.append(np.array([0.0] * \
                        self.inner.nb_radio_iso))
            else:
                self.ymgal_outer_radio.append(np.array([0.0]))

        # Declare all stable outer arrays
        self.ymgal_outer_massive = [arr*0. for arr in self.ymgal_outer]
        self.ymgal_outer_agb = [arr*0. for arr in self.ymgal_outer]
        self.ymgal_outer_1a = [arr*0. for arr in self.ymgal_outer]
        self.ymgal_outer_nsm = [arr*0. for arr in self.ymgal_outer]
        self.ymgal_outer_extra = []
        for extra in self.inner.ymgal_delayed_extra:
            self.ymgal_outer_extra.append([arr*0. for arr in self.ymgal_outer])

        # Now the radio ones
        self.ymgal_outer_massive_radio = [arr*0. for arr in self.ymgal_outer_radio]
        self.ymgal_outer_agb_radio = [arr*0. for arr in self.ymgal_outer_radio]
        self.ymgal_outer_1a_radio = [arr*0. for arr in self.ymgal_outer_radio]
        self.ymgal_outer_nsm_radio = [arr*0. for arr in self.ymgal_outer_radio]
        self.ymgal_outer_extra_radio = []
        for extra in self.inner.ymgal_delayed_extra:
            self.ymgal_outer_extra_radio.append(\
                                    [arr*0. for arr in self.ymgal_outer_radio])

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
    #              Start Simulation              #
    ##############################################
    def __start_simulation(self):

        alpha_instance = alpha.Alpha([self.inner], omega_p=self)
        alpha_instance.integrate()

        # TODO One omega run
        alpha_instance.one_instance_error()

        # Announce the end of the simulation
        print ('   OMEGA run completed -',self.inner._gettime())


    ##############################################
    #    Get the production and destruction      #
    #            for this substep                #
    ##############################################
    def get_p_and_d(self, i_step_OMEGA, isot_prim, isot_prim_out, isot_mgal,
                          yield_rate, isot_mcgm,isot_mgal_radio,
                          yield_rate_radio, isot_mcgm_radio, htm, dtt, nn,
                          decay_from_radio, decay_to_radio, decay_to_stable,
                          stable_decayed_isotopes):

        '''
        This function runs the patankar algorithm for nn substeps.

        '''

        # Calculate the total current gas mass in the inner region
        current_mgal = np.sum(isot_prim) + np.sum(isot_mgal)
        inv_mass = 1 / (current_mgal + self.min_val)

        # Calculate the total current gas mass in the outer region
        current_mcgm = np.sum(isot_prim_out) + np.sum(isot_mcgm)
        inv_mass_cgm = 1 / (current_mcgm + self.min_val)

        # Calculate the star formation rate [Msun/yr]
        sfr_temp = self.__get_SFR(i_step_OMEGA, current_mgal, dtt)
        isot_sfr_prim = sfr_temp * isot_prim * inv_mass
        isot_sfr_split = sfr_temp * isot_mgal * inv_mass
        isot_sfr_split_radio = sfr_temp * isot_mgal_radio * inv_mass

        # Calculate the galactic outflow rate [Msun/yr]
        or_temp = self.__get_outflow_rate(i_step_OMEGA, sfr_temp, dtt)
        isot_or_prim = or_temp * isot_prim * inv_mass
        isot_or_split = or_temp * isot_mgal * inv_mass
        isot_or_split_radio = or_temp * isot_mgal_radio * inv_mass

        # Calculate the galactic inflow rate [Msun/yr]
        ir_iso_temp = self.__get_inflow_rate(i_step_OMEGA, current_mcgm, dtt)
        ir_iso_prim = ir_iso_temp * isot_prim_out * inv_mass_cgm
        ir_iso_split = ir_iso_temp * isot_mcgm * inv_mass_cgm
        ir_iso_split_radio = ir_iso_temp * isot_mcgm_radio * inv_mass_cgm

        # Modify inner gas:
        # Get production factors for primordial and split, stable and radio
        pp_prim = ir_iso_prim
        pp_split = ir_iso_split + yield_rate
        pp_split_radio = ir_iso_split_radio + yield_rate_radio

        # Get destruction factors for primordial and split, stable and radio
        dd_prim = (isot_or_prim + isot_sfr_prim) / (isot_prim + self.min_val)
        dd_split = (isot_or_split + isot_sfr_split) / (isot_mgal + self.min_val)
        dd_split_radio = (isot_or_split_radio + isot_sfr_split_radio)\
                       / (isot_mgal_radio + self.min_val)

        # Modify pp_split, pp_split_radio, and dd_split_radio due to decays
        if self.inner.len_decay_file > 0 or self.inner.use_decay_module:
            pp_split, pp_split_radio, dd_split_radio = \
                self.__get_radio_pp_dd(pp_split, pp_split_radio,
                     dd_split_radio, isot_mgal_radio, decay_from_radio,
                     decay_to_radio, decay_to_stable, stable_decayed_isotopes)

        # Get rates for intergalactic to circumgalactic flows
        added_cgm, removed_cgm = self.__get_rates_for_DM_variation(i_step_OMEGA,\
                                      current_mcgm, dtt)
        m_out_cgm = self.__get_halo_outflow_rate(i_step_OMEGA, dtt, sfr_temp)
        isot_added_cgm = added_cgm * self.prim_x_frac
        isot_removed_cgm_prim = removed_cgm * isot_prim_out * inv_mass_cgm
        isot_removed_cgm_split = removed_cgm * isot_mcgm * inv_mass_cgm
        isot_removed_cgm_split_radio = removed_cgm * isot_mcgm_radio * inv_mass_cgm
        isot_m_out_cgm_prim = m_out_cgm * isot_prim_out * inv_mass_cgm
        isot_m_out_cgm_split = m_out_cgm * isot_mcgm * inv_mass_cgm
        isot_m_out_cgm_split_radio = m_out_cgm * isot_mcgm_radio * inv_mass_cgm

        # Get production factors for ymgal_outer and ymgal_outer_radio
        pp_prim_out = isot_or_prim + isot_added_cgm
        pp_split_out = isot_or_split
        pp_split_radio_out = isot_or_split_radio

        # Get destruction factors for ymgal_outer and ymgal_outer_radio
        dd_prim_out = (ir_iso_prim + isot_removed_cgm_prim +
            isot_m_out_cgm_prim) / (isot_prim_out + self.min_val)
        dd_split_out = (ir_iso_split + isot_removed_cgm_split +
            isot_m_out_cgm_split) / (isot_mcgm + self.min_val)
        dd_split_radio_out = (ir_iso_split_radio + isot_removed_cgm_split_radio
            + isot_m_out_cgm_split_radio) / (isot_mcgm_radio + self.min_val)

        # Modify pp_split_out, pp_split_radio_out, and dd_split_radio_out
        # due to decays
        if self.inner.len_decay_file > 0 or self.inner.use_decay_module:
            pp_split_out, pp_split_radio_out, dd_split_radio_out =\
                self.__get_radio_pp_dd(pp_split_out, pp_split_radio_out,
                     dd_split_radio_out, isot_mcgm_radio, decay_from_radio,
                     decay_to_radio, decay_to_stable, stable_decayed_isotopes)

        # Increments (changes)
        values = (
                  (pp_prim, dd_prim),                       # Primordial gas
                  (pp_split, dd_split),                     # ymgal
                  (pp_split_radio, dd_split_radio),         # ymgal_radio
                  (pp_prim_out, dd_prim_out),               # CGM primordial gas
                  (pp_split_out, dd_split_out),             # ymgal_outer
                  (pp_split_radio_out, dd_split_radio_out), # ymgal_outer_radio
                  (sfr_temp, sfr_temp * 0),                 # total_sfr
                  (ir_iso_temp, ir_iso_temp * 0),           # m_added
                  (or_temp, or_temp * 0),                   # m_lost
                 )

        # Return the values
        return values

    ##############################################
    #           Get pp, dd from reactions        #
    ##############################################
    def __get_radio_pp_dd(self, pp, pp_radio, dd_radio, isot_mass_radio,
                          decay_from_radio, decay_to_radio, decay_to_stable,
                          stable_decayed_isotopes):

        '''
        This function updates pp, pp_radio and dd_radio due
        to radioactive decays and other reactions from the reactions
        dictionary.

        '''

        # Simply add the values
        dd_radio += decay_from_radio

        # For radioactive isotopes
        pp_radio_t = np.transpose(pp_radio)
        for ii in range(self.inner.nb_radio_iso):
            pp_radio_t[ii] += np.sum(decay_to_radio[ii] * isot_mass_radio, axis=1)
        pp_radio = np.transpose(pp_radio_t)

        # For stable isotopes
        pp = np.transpose(pp)
        for ii in range(len(stable_decayed_isotopes)):
            isot = stable_decayed_isotopes[ii]
            indx = self.inner.history.isotopes.index(isot)
            pp[indx] += np.sum(decay_to_stable[ii] * isot_mass_radio, axis=1)
        pp = np.transpose(pp)

        return pp, pp_radio, dd_radio

    ##############################################
    #                  Get SFR                   #
    ##############################################
    def __get_SFR(self, i_step_OMEGA, m_gas_SFR, dtt=0):

        '''
        Calculate and return the star formation rate of the inner region

        Arguments
        =========

          i_step_OMEGA : Current timestep index of the OMEGA instance
          m_gas_SFR: Current mass of gas
          dtt: Time delay off the current time of timestep i_step_OMEGA
               This is used in case of refinement timesteps during system integration

        '''

        # Use the classical SF law (SFR = f_s / t_s * M_gas)
        if self.is_SF:
            if self.SF_allowed_t[i_step_OMEGA]:
                if self.inner.m_crit_on:
                    if dtt > 0:
                        m_crit_t_temp = self.inner.m_crit_t_coef[i_step_OMEGA][0] * \
                            (self.inner.history.age[i_step_OMEGA]+dtt) + \
                                self.inner.m_crit_t_coef[i_step_OMEGA][1]
                    else:
                        m_crit_t_temp = self.inner.m_crit_t[i_step_OMEGA]

                    if m_gas_SFR <= m_crit_t_temp:
                        m_gas_temp = 0.0
                    else:
                        m_gas_temp = m_gas_SFR - m_crit_t_temp
                        #m_gas_temp = m_gas_SFR

                else:
                    m_gas_temp = m_gas_SFR

                # Interpolate SFE
                if dtt > 0:
                    sfe_temp = self.sfe_coef[i_step_OMEGA][0] * \
                        (self.inner.history.age[i_step_OMEGA]+dtt) + \
                            self.sfe_coef[i_step_OMEGA][1]
                else:
                    sfe_temp = self.sfe[i_step_OMEGA]

                # Return star formation rate
                if self.treat_sfe_t:
                    return sfe_temp * m_gas_temp
                else:
                    if dtt > 0:
                        t_SF_t_temp = self.inner.t_SF_t_coef[i_step_OMEGA][0] * \
                            (self.inner.history.age[i_step_OMEGA]+dtt) + \
                                self.inner.t_SF_t_coef[i_step_OMEGA][1]
                    else:
                        t_SF_t_temp = self.inner.t_SF_t[i_step_OMEGA]
                    return sfe_temp * m_gas_temp / t_SF_t_temp
            else:
                return 0.0
        else:
            return 0.0


    ##############################################
    #              Get Outflow Rate              #
    ##############################################
    def __get_outflow_rate(self, i_step_OMEGA, sfr_temp, dtt=0):

        '''
        Calculate and return the galactic outflow rate of the inner region

        Arguments
        =========

          i_step_OMEGA : Current timestep index of the OMEGA instance
          sfr_temp : Star formation rate of the current timestep [Msun/yr]
          dtt: Time delay off the current time of timestep i_step_OMEGA
               This is used in case of refinement timesteps during system integration


        '''

        # If we use the DM_evolution option from Cote et al. (2017) ..
        if self.DM_outflow_C17:

            # Interpolate
            if dtt > 0:
                m_DM_t_temp = self.inner.m_DM_t_coef[i_step_OMEGA][0] * \
                    (self.inner.history.age[i_step_OMEGA]+dtt) + \
                        self.inner.m_DM_t_coef[i_step_OMEGA][1]
            else:
                m_DM_t_temp = self.inner.m_DM_t[i_step_OMEGA]

            # Calculate the mass-loading factor
            if self.C17_eta_z_dep:

                # Interpolate
                if dtt > 0:
                    redshift_t_temp = self.inner.redshift_t_coef[i_step_OMEGA][0] * \
                        (self.inner.history.age[i_step_OMEGA]+dtt) + \
                            self.inner.redshift_t_coef[i_step_OMEGA][1]
                else:
                    redshift_t_temp = self.inner.m_DM_t[i_step_OMEGA]

                mass_loading_gor = self.eta_norm * \
                    m_DM_t_temp**((-0.3333)*self.inner.exp_ml) * \
                        (1+redshift_t_temp)**(-(0.5)*self.inner.exp_ml)
            else:
                mass_loading_gor = self.eta_norm * m_DM_t_temp**((-0.3333)*self.inner.exp_ml)

        # If the mass-loading follows Crosby et al. (2015) ..
        elif self.epsilon_sne_gal >= 0.0:

            # Interpolate
            if dtt > 0:
                m_DM_t_temp = self.inner.m_DM_t_coef[i_step_OMEGA][0] * \
                    (self.inner.history.age[i_step_OMEGA]+dtt) + \
                        self.inner.m_DM_t_coef[i_step_OMEGA][1]
                r_vir_DM_t_temp = self.inner.r_vir_DM_t_coef[i_step_OMEGA][0] * \
                    (self.inner.history.age[i_step_OMEGA]+dtt) + \
                        self.inner.r_vir_DM_t_coef[i_step_OMEGA][1]
            else:
                m_DM_t_temp = self.inner.m_DM_t[i_step_OMEGA]
                r_vir_DM_t_temp = self.inner.r_vir_DM_t[i_step_OMEGA]


            # Calculate the mass-loading factor
            mass_loading_gor = self.gamma_cte * self.nb_ccsne_per_m * self.epsilon_sne_gal * \
                (r_vir_DM_t_temp*0.001 / m_DM_t_temp)**self.gal_out_index

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

            return -1

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
    def __get_inflow_rate(self, i_step_OMEGA, sum_ymgal_outer, dtt=0):

        '''
        Calculate and return the galactic inflow rate of the inner region
        for all isotopes

        Arguments
        =========

          i_step_OMEGA : Current timestep index of the OMEGA instance
          sum_ymgal_outer: Current total gas of the CGM

        '''

        # Initialize cooling_rate
        cooling_rate = 0.0

        # If this is a star forming galaxy ..
        if self.is_SF and sum_ymgal_outer > 0.0:

          # If input exponential infall laws ..
          # For each infall episode, exp_infall --> [Norm, t_max, timescale]
          if self.nb_exp_infall > 0:
            for i_in in range(self.nb_exp_infall):
                cooling_rate += self.exp_infall[i_in][0] * \
                    np.exp(-((self.inner.history.age[i_step_OMEGA] + dtt) - \
                    self.exp_infall[i_in][1]) / self.exp_infall[i_in][2])

          # If an input inflow rate is provided ..
          elif self.len_m_inflow_in > 0:

            # Interpolate
            if dtt > 0:
                cooling_rate = self.m_inflow_in_rate_coef[i_step_OMEGA][0] * \
                    (self.inner.history.age[i_step_OMEGA] + dtt) + \
                        self.m_inflow_in_rate_coef[i_step_OMEGA][1]
            else:
                cooling_rate = self.m_inflow_in_rate[i_step_OMEGA]

          # If the inflow rate needs to be calculated ..
          else:

            # Assign an input inflow timescale for the free-fall timescale [yr]
            if self.t_inflow > 0.0:
                t_ff_temp = self.t_inflow

            # Calculate the free-fall timescale [yr]
            else:

                # Interpolate
                if dtt > 0:
                    redshift_temp = self.inner.redshift_t_coef[i_step_OMEGA][0] * \
                        (self.inner.history.age[i_step_OMEGA] + dtt) + \
                            self.inner.redshift_t_coef[i_step_OMEGA][1]
                else:
                    redshift_temp = self.inner.redshift_t[i_step_OMEGA]

                # Calculate interpolated free-fall
                t_ff_temp = self.t_ff_cte * self.f_t_ff * 0.1 * (1.0 + \
                    redshift_temp)**((-1.5)*self.t_ff_index) / \
                        self.inner.H_0 * 9.7759839e11
                # Constant is 1.1107 * 3.086e16 / 3.154e7

            # Use free-fall time
            cooling_time = t_ff_temp

            #cooling_time = t_ff_temp * (544507042.254/self.inner.m_DM_t[i_step_OMEGA])**0.5
            self.t_cool[i_step_OMEGA] = cooling_time

            # Calculate the total cooling rate [Msun/yr]
            cooling_rate = sum_ymgal_outer / cooling_time

        # If this is not a star forming galaxy, there is not inflow
        else:

            # Calculate the free-fall timescale [yr]
            if self.t_inflow > 0.0:
                t_ff_temp = self.t_inflow
            else:
                t_ff_temp = self.t_ff_cte * self.f_t_ff * 0.1 * (1.0 + \
                    self.inner.redshift_t[i_step_OMEGA])**((-1.5)*self.t_ff_index) / \
                        self.inner.H_0 * 9.7759839e11
            self.t_cool[i_step_OMEGA] = t_ff_temp

        # Calculate the isotope cooling rates
        iso_rate_temp = cooling_rate

        # If the rate is too big, return a constant rate
        dtBig = self.inner.history.timesteps[i_step_OMEGA]
        if iso_rate_temp * dtBig > sum_ymgal_outer:
            iso_rate_temp = sum_ymgal_outer / dtBig

        return iso_rate_temp

    ##############################################
    #         Get Rates for DM variation         #
    ##############################################
    def __get_rates_for_DM_variation(self, i_step_OMEGA, sum_ymgal_outer, dtt=0):

        '''
        Add gas to the outer region if the dark matter mass increases, or
        remove gas from the outer region if the dark matter mass decreases.
        This function only returns the rates in [Msun/yr]

        Arguments
        =========

          i_step_OMEGA : Timestep index
          sum_ymgal_outer: Current isotopic composition of the CGM

        '''

        # Declare rates [Msun/yr]
        dm_m_outer_remove = 0.0
        dm_m_outer_add = 0.0

        # If there is an interaction/correction based on the
        # dark matter halo mass ..
        if self.halo_in_out_on:

            # Interpolate
            if dtt > 0:
                m_DM_temp = self.inner.m_DM_t_coef[i_step_OMEGA][0] * \
                    (self.inner.history.age[i_step_OMEGA]+dtt) + \
                        self.inner.m_DM_t_coef[i_step_OMEGA][1]
                m_DM_temp_up = self.inner.m_DM_t_coef[i_step_OMEGA][0] * \
                    (self.inner.history.age[i_step_OMEGA]+2.0*dtt) + \
                        self.inner.m_DM_t_coef[i_step_OMEGA][1]
                timestep_temp = dtt
            else:
                m_DM_temp = self.inner.m_DM_t[i_step_OMEGA]
                m_DM_temp_up = self.inner.m_DM_t[i_step_OMEGA+1]
                timestep_temp = self.inner.history.timesteps[i_step_OMEGA]

            # Calculate the mass of dark matter added or removed
            dm_dm = m_DM_temp_up - m_DM_temp

            # If gas needs to be removed ..
            if dm_dm < 0.0:

                # Calculate the fraction of the halo that is not stripped.
                #f_keep = 1.0 - dm_dm / self.inner.m_DM_t[i_step_OMEGA]
                f_keep = m_DM_temp_up / m_DM_temp

                # Correct the outer gas for the stripping. Here we use i_step_OMEGA+1,
                # since this occurs once galactic inflows and outflows have occured.
                dm_m_outer_remove = sum_ymgal_outer * (1.0-f_keep)

            # If gas needs to be added ..
            elif not self.is_sub[i_step_OMEGA]:

                # Calculate the mass added in the outer gas
                dm_m_outer_add = dm_dm * self.f_b_temp

        # Return the rates [Msun/yr]
        return dm_m_outer_add / timestep_temp, \
               dm_m_outer_remove / timestep_temp


    ##############################################
    #            Get Halo Outflow Rate           #
    ##############################################
    def __get_halo_outflow_rate(self, i_step_OMEGA, dtt = 0, sfr_temp = 0):

        '''
        Return rate to remove gas from the halo. [Msun/yr]

        Arguments
        =========

          i_step_OMEGA : Timestep index

        '''

        # If the halo outflows is following the galactic outflow rate..
        if self.f_halo_to_gal_out >= 0.0:

            # Use the galactic outflow rate times the input factor
            if dtt > 0:
                return self.f_halo_to_gal_out * self.__get_outflow_rate(i_step_OMEGA, sfr_temp, dtt=dtt)
            else:
                m_lost = self.m_lost_for_halo * self.f_halo_to_gal_out

        # Calculate the mass ejected from the halo (Crosby et al. 2015)
        else:

            # Interpolate
            if dtt > 0:
                m_DM_temp = self.inner.m_DM_t_coef[i_step_OMEGA][0] * \
                    (self.inner.history.age[i_step_OMEGA]+dtt) + \
                        self.inner.m_DM_t_coef[i_step_OMEGA][1]
                r_vir_temp = self.inner.r_vir_DM_t_coef[i_step_OMEGA][0] * \
                    (self.inner.history.age[i_step_OMEGA]+dtt) + \
                        self.inner.r_vir_DM_t_coef[i_step_OMEGA][1]
            else:
                m_DM_temp = self.inner.m_DM_t[i_step_OMEGA]
                r_vir_temp = self.inner.r_vir_DM_t[i_step_OMEGA]


            # In OMEGA, R_vir is in [kpc] while in Crosby+15 R_vir is in [Mpc]
            if self.epsilon_sne_halo < 0:
                m_lost = 0
            else:
                m_lost = self.gamma_cte * self.nb_ccsne[i_step_OMEGA] * \
                    self.epsilon_sne_halo * \
                    (r_vir_temp*0.001 / m_DM_temp)**self.halo_out_index

        # Return the halo outflow rate [Msun/yr]
        if dtt > 0:
            return m_lost / dtt
        else:
            return m_lost / self.inner.history.timesteps[i_step_OMEGA]


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

        print('Warning - This is not ready for integration scheme.')

        # Calculate the total current gas mass in the inner region
        self.sum_inner_ymgal_cur = np.sum(self.inner.ymgal[i_step_OMEGA+1])

        # Get the SFR of the next timestep if no gas is added
        sfr_next = self.__get_SFR(i_step_OMEGA+1, self.sum_inner_ymgal_cur)

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
    #              Get Isolation Time            #
    ##############################################
    def get_isolation_time(self, isotope, value, time_sun):

        '''
        Wrapper for get_isolation_time in NuPyCEE/omega.py

        Parameters
        ----------

        '''

        return self.inner.get_isolation_time(isotope = isotope, value = value,
                time_sun = time_sun, reac_dictionary = self.reac_dictionary)


    ##############################################
    #                  Get Time                  #
    ##############################################
    def __get_time(self):

        out = 'Run time: ' + \
        str(round((t_module.time() - self.start_time),2))+"s"
        return out
