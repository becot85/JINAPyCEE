'''

Script to extract information from a consistent tree for
generating the input arrays used in GAMMA.

BCOTE: July 2018

'''

# Import python modules
import ytree
import numpy as np
import math
import copy

# Add descendent to the merger tree
def add_descendents(arbor):

    # loop over all trees in the arbor
    for tree in arbor:

        # iterate over all halos in the tree
        # use tree.pwalk() to iterate over just the main progenitors
        for node in tree.twalk():

            # "ancestors" will be either None or a list of nodes
            if node.ancestors is None:
                continue
            for anc in node.ancestors:
                anc.descendent = node

# Set cosmological parameters - as in Wise et al. 2012
omega_0   = 0.266   # Current mass density parameter
omega_b_0 = 0.0449  # Current baryonic mass density parameter
lambda_0  = 0.734   # Current dark energy density parameter
H_0       = 71.0    # Hubble constant [km s^-1 Mpc^-1]


##############################################
#                  Get Times                 #
##############################################
def get_times(redshifts):

    '''
    Calculate and return the age of the tree based on the redshifts.

    Argument
    ========

      redshifts: List of redshifts in cecreasing order (from past to present)

    Return
    ======

      List of tree ages

    '''

    # Declare the list of times
    times = []

    # Calculate the age of Universe at the beginning of simulation
    age_ini = get_t_from_z(redshifts[0])

    # Calculate the tree age associated with each redshift
    for i_gt in range(0,len(redshifts)):
        times.append( get_t_from_z(redshifts[i_gt]) - age_ini )

    # Return the times (tree ages)
    return times


##############################################
#               Get t From z                 #
##############################################
def get_t_from_z(z_gttfz):

    '''
    This function returns the age of the universe at a given redshift.

    Argument
    ========

      z_gttfz : Redshift that needs to be converted into age.

    Return
    ======

      Age of the universe.

    '''

    # Return the age of the Universe
    temp_var = math.sqrt((lambda_0/omega_0)/(1+z_gttfz)**3)
    x_var = math.log( temp_var + math.sqrt( temp_var**2 + 1 ) )
    return 2 / ( 3 * H_0 * math.sqrt(lambda_0)) * \
           x_var * 9.77793067e11


##############################################
#               Get z From t                 #
##############################################
def get_z_from_t(t_gtzft):

    '''
    This function returns the redshift of a given Universe age.

    Argument
    ========

      t_gtzft : Age of the Universe that needs to be converted into redshift.

    Return
    ======

      Redshift.

    '''

    # Return the redshift
    temp_var = 1.5340669e-12 * lambda_0**0.5 * H_0 * t_gtzft
    return (lambda_0 / omega_0)**0.3333333 / \
            math.sinh(temp_var)**0.66666667 - 1


##############################################
#               Get Branches                 #
##############################################
def get_branches(nodes, redshifts, times):

    # Declare the arrays
    br_halo_ID  = []  # List of connected halo IDs (in redshift order)
    br_age      = []  # Age of the branch
    br_z        = []  # Redshift of the branch
    br_t_merge  = []  # Duration of the branches (delay between formation and merger)
    br_ID_merge = []  # Last halo ID of the branch (once it has merged)
    br_m_halo   = []  # Array of dark matter halo masses
    br_r_vir   = []  # Array of dark matter halo masses
    br_is_prim  = []  # True or False depending whether the branch is primordial

    # Create an entry for each redshift
    for i_z in range(0,len(redshifts)):
        br_halo_ID.append([])
        br_age.append([])
        br_z.append([])
        br_t_merge.append([])
        br_ID_merge.append([])
        br_m_halo.append([])
        br_r_vir.append([])
        br_is_prim.append([])

    # For each node in the tree ..
    for i_n in range(len(nodes)):

        # Get the index for the redshift array
        i_z = redshifts.index(nodes[i_n]['redshift'])

        # If the halo is the creation point of a new branch ..
        if nodes[i_n].ancestors == None:
            creation_point = True
        elif not len(nodes[i_n].ancestors) == 1:
            creation_point = True
        else:
            creation_point = False
        if creation_point:

            # Create a new branch for the considered redshift
            br_halo_ID[i_z].append([])
            br_age[i_z].append([])
            br_z[i_z].append([])
            br_t_merge[i_z].append(0.0)
            br_ID_merge[i_z].append(0.0)
            br_m_halo[i_z].append([])
            br_r_vir[i_z].append([])

            # Assign whether or not this is a primordial branch
            if nodes[i_n].ancestors == None:
                br_is_prim[i_z].append(True)
            else:
                br_is_prim[i_z].append(False)

            # Fill the halo ID, age, mass, and radius
            fill_branch_info(br_halo_ID, br_age, br_z, br_m_halo, \
                br_r_vir, nodes[i_n].uid, i_z, i_z, times, redshifts, nodes[i_n])

            # If the trunk is not recovered ..
            if not nodes[i_n].is_root:

                # Go to the next connected halo
                node_search = nodes[i_n].descendent

                # While the descendent is not the product of a merger ..
                while len(node_search.ancestors) < 2:

                    # Get the redshift index of the current children
                    i_z_cur = redshifts.index(node_search['redshift'])

                    # Fill the halo ID, age, and mass
                    fill_branch_info(br_halo_ID, br_age, br_z, br_m_halo, \
                        br_r_vir, node_search.uid, i_z, i_z_cur, times, redshifts, node_search)

                    # Stop if the trunk is recovered ..
                    if node_search.is_root:
                        break

                    # Go to the next children/descendent
                    node_search = node_search.descendent

                # Calculate the time before merger
                i_z_last = redshifts.index(node_search['redshift'])
                br_t_merge[i_z][-1] = times[i_z_last] - times[i_z]

                # Copy the last halo ID (when the branch has merged)
                br_ID_merge[i_z][-1] = node_search.uid

    # Return the branches
    return br_halo_ID, br_age, br_z, br_t_merge, \
           br_ID_merge, br_m_halo, br_r_vir, br_is_prim


##############################################
#              Fill Branch Info              #
##############################################
def fill_branch_info(br_halo_ID, br_age, br_z, br_m_halo, \
            br_r_vir, hid, iz, iz_cur, times, redshifts, node_cur):

    '''
    Fill the halo ID, age, and mass of the current halo in a branch.

    '''

    # Copy the halo ID
    br_halo_ID[iz][-1].append(hid)

    # Calculate the age of the branch
    br_age[iz][-1].append(times[iz_cur] - times[iz])

    # Calculate the redshift of the branch
    br_z[iz][-1].append(redshifts[iz_cur])

    # Copy the halo mass of the current halo
    br_m_halo[iz][-1].append(float(node_cur['Mvir'].in_units('Msun')))

    # Copy the rivial radius
    br_r_vir[iz][-1].append(float(node_cur['Rvir'].in_units('kpc') /\
        (1.0 + redshifts[iz_cur])))

