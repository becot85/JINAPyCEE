"""
Author: Alex Ji
31 Jul 2018
"""
import numpy as np
import haloutils
import time, sys, os

OUTPUTDIR="/blender/data/alexji/gamma_trees"
assert os.path.exists(OUTPUTDIR)

"""
Merger Tree Columns
tree.data.dtype.names
['scale', 'id', 'desc_id', 'num_prog', 'pid', 'upid', 'phantom',
'sam_mvir', 'mvir', 'rvir', 'rs', 'vrms', 'mmp', 'scale_of_last_MM',
'vmax', 'posX', 'posY', 'posZ', 'pecVX', 'pecVY', 'pecVZ', 'Jx',
'Jy', 'Jz', 'spin', 'bfid', 'dfid', 'origid', 'lastprog_dfid',
'm200c_all', 'm200b', 'xoff', 'voff', 'spin_bullock',
'b_to_a(500c)', 'c_to_a(500c)', 'A[x](500c)', 'A[y](500c)',
'A[z](500c)', 'T/|U|', 'snap']
"""

h0 = 0.6711

def fill_branch_info(br_halo_ID, br_age, br_z, br_m_halo,
                     br_r_vir, hid, iz, iz_cur, times, redshifts, node_cur,
                     br_is_sub):

    """
    Fill the halo ID, age, and mass of the current halo in a branch.
    """

    # Copy the halo ID
    br_halo_ID[iz][-1].append(hid)

    # Calculate the age of the branch
    br_age[iz][-1].append(times[iz_cur] - times[iz])

    # Calculate the redshift of the branch
    br_z[iz][-1].append(redshifts[iz_cur])

    # Copy the halo mass of the current halo
    #br_m_halo[iz][-1].append(float(node_cur['Mvir'].in_units('Msun')))
    br_m_halo[iz][-1].append(float(node_cur['mvir']/h0))

    # Copy the virial radius
    br_r_vir[iz][-1].append(float(node_cur['rvir']/h0 /\
        (1.0 + redshifts[iz_cur])))

    # Copy subhalo status
    br_is_sub[iz][-1].append(node_cur["upid"]!=-1)

def convert_tree(hpath, tree, treeoutputpath, verbose=True):
    # Maps to go up and down trees
    # desc_map is analogous to merger_tree_ytree.add_descendants()
    # mmp = "most massive progenitor" flag
    start = time.time()
    desc_map = tree.get_desc_map()
    if verbose: print "  Time to precompute tree maps: {:.1f}".format(time.time()-start); sys.stdout.flush()
    num_nodes_in_tree = len(tree.data)
    num_nodes_processed = 0

    ## This gets the scale factors etc for Caterpillar
    snaps = np.unique(tree["snap"])
    times = haloutils.get_t_snap(hpath,snaps)*1e9 #Gyr
    redshifts = haloutils.get_z_snap(hpath,snaps)
    ## Convert some things to lists for list.index() method
    snaps = list(snaps)

    # NOTE: a branch is a multi-snapshot segment of the tree that experiences no mergers
    # Declare the arrays
    br_halo_ID  = []  # List of connected halo IDs (in redshift order)
    br_age      = []  # Age of the branch
    br_z        = []  # Redshift of the branch
    br_t_merge  = []  # Duration of the branches (delay between formation and merger)
    br_ID_merge = []  # Last halo ID of the branch (once it has merged)
    br_m_halo   = []  # Array of dark matter halo masses
    br_r_vir    = []  # Array of dark matter halo radii
    br_is_sub   = []  # Array of True if the halo is a subhalo, False if is a host halo
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
        br_is_sub.append([])
        br_is_prim.append([])

    start = time.time()
    ## Loop through all nodes of the tree that are branch points
    for irow,row in enumerate(tree.data):
        i_z = snaps.index(row["snap"])
        ## If exactly one progenitor, this is not a branching point, so skip it
        if row["num_prog"] == 1: continue
        # Create a new branch for the considered redshift
        br_halo_ID[i_z].append([])
        br_age[i_z].append([])
        br_z[i_z].append([])
        br_t_merge[i_z].append(0.0)
        br_ID_merge[i_z].append(0.0)
        br_m_halo[i_z].append([])
        br_r_vir[i_z].append([])
        br_is_sub[i_z].append([])

        ## Assign whether or not this is a primordial branch
        br_is_prim[i_z].append(row["num_prog"] == 0)

        ## Fill the start of the branch
        ## Fill the halo ID, age, mass, and radius
        # Note: the ID is the mtid!!! not sure if this is what we actually want
        # To access that object in the halo catalogs, we need origid and snapshot
        fill_branch_info(br_halo_ID, br_age, br_z, br_m_halo, br_r_vir,
                         row["id"], i_z, i_z, times, redshifts, row, br_is_sub)
        num_nodes_processed += 1
        ## Step down the tree
        desc_irow = tree.getDesc(irow, desc_map)
        if desc_irow is None: continue # reached the root
        desc_row = tree[desc_irow]
        ## Loop through subsequent parts of the branch, defined as single-progenitor
        while desc_row["num_prog"] == 1:
            ## Fill next part of the branch
            i_z_cur = snaps.index(desc_row["snap"])
            fill_branch_info(br_halo_ID, br_age, br_z, br_m_halo, br_r_vir,
                             desc_row["id"], i_z, i_z_cur, times, redshifts, desc_row, br_is_sub)
            num_nodes_processed += 1
            ## Step down the tree
            desc_irow = tree.getDesc(desc_irow, desc_map)
            if desc_irow is None: break # reached the root
            desc_row = tree[desc_irow]

        # Calculate the time before merger
        i_z_last = snaps.index(desc_row["snap"])
        br_t_merge[i_z][-1] = times[i_z_last] - times[i_z]
        # Copy the last halo ID (when the branch has merged)
        br_ID_merge[i_z][-1] = desc_row["id"]
    if verbose: print "  Time to convert: {:.1f}".format(time.time()-start); sys.stdout.flush()
    if num_nodes_processed != num_nodes_in_tree:
        raise ValueError("ERROR! num nodes processed != num nodes in tree ({} != {})".format(num_nodes_processed, num_nodes_in_tree))

    start = time.time()
    np.save(treeoutputpath,[br_halo_ID, br_age, br_z, br_t_merge, br_ID_merge, \
                            br_m_halo, br_r_vir, br_is_prim, redshifts, times, tree[0]['id'], br_is_sub])
    if verbose:
        print "  Time to save: {:.1f}".format(time.time()-start); sys.stdout.flush()
        print

def convert_all_trees(hid, lx, Mpeakmin):
    print "Converting all halos for H{} LX{}, Mpeakmin={:.1e}".format(hid,lx,Mpeakmin)
    halodir = os.path.join(OUTPUTDIR,"{}_LX{}".format(haloutils.hidstr(hid),lx))
    print "Outputting to",halodir
    if not os.path.exists(halodir):
        print halodir,"does not exist, creating"
        os.makedirs(halodir)

    start = time.time()
    hpath = haloutils.get_hpath_lx(hid, lx)
    zoomid = haloutils.load_zoomid(hpath)
    # This contains a dictionary of Trees, keyed by RSID at z=0
    arbor = haloutils.load_zoom_mtc(hpath, indexbyrsid=True)
    print "Time to load catalog: {:.1f}".format(time.time()-start); sys.stdout.flush()
    print

    num_trees_tried = 0
    num_trees_written = 0
    startall = time.time()
    for itree, tree in arbor.Trees.iteritems():
        treeoutputpath = os.path.join(halodir,"rsid{}.npy".format(itree))
        num_trees_tried += 1
        ## TODO use Kaley's better Mpeaks
        mb = tree.getMainBranch()
        Mpeak = np.max(mb["mvir"]/h0)
        if Mpeak < Mpeakmin: continue
        print "  Converting tree {} Mpeak={:.1e} to {}".format(itree, Mpeak, treeoutputpath)
        try:
            convert_tree(hpath, tree, treeoutputpath)
            num_trees_written += 1
        except Exception as e:
            print "ERROR!!!"
            print e
            print "Skipping this tree"
            print
    print "DONE! Processed {}/{} trees, Total time = {:.1f}".format(num_trees_written, num_trees_tried, time.time()-startall)

if __name__=="__main__":
    hids = [1725272, 1387186, 5320]
    lxs = [11]#, 12, 13, 14]
    #lxs = [11, 12]
    #hids = [1725272]
    #hids = [1387186,5320]
    #lxs = [13,14]
    Mpeakmin = 1e8
    for lx in lxs:
        for hid in hids:
            convert_all_trees(hid, lx, Mpeakmin)

