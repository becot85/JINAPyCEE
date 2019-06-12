# Parameters to be sampled
# var_range['dictionary_key'][0] -- minimum parameter value
# var_range['dictionary_key'][1] -- range (max - min)
var_range = {}
var_range["sfe_m_index"]       = [-1.0, 2.0]
var_range["mass_loading"]      = [0.0, 2.0]
var_range["f_halo_to_gal_out"] = [0.0, 1.0]
# etc ..
# Number of sampled imput parameters
dimensions = len(var_range)

print('List of parameters (',dimensions,'in total )')
for key in var_range.keys():
    print('   ',key)

# Create dummy latin hypercube
# This is just to test my script and to show the dictionary part
# Create also the "em_sample_points" array
sampled_points = 10
lhd = []
em_sample_points = []
for i in range(0, sampled_points):
    lhd.append({})
    em_sample_points.append({})
    for key in var_range.keys():
        lhd[i][key] = i / float(sampled_points)

i_test = 7
print()
print('lhd for index',i_test)
print('   ',lhd[i_test])

# Fill the em_sample_points array
for i in range(0, sampled_points):
    for key in var_range.keys():
        em_sample_points[i][key] = (lhd[i][key]*(var_range[key][1]))+var_range[key][0]

print()
print('sampled parameters for index',i_test)
print('   ',em_sample_points[i_test])

# Parameters that should not be changed
C17_eta_z_dep = False
DM_outflow_C17 = True
sfe_m_dep = True
t_star = -1
t_inflow = -1

# For each set of sampled parameters ..
for i in range(0, sampled_points):
    
    # Get the default "empty" list of parameters
    gamma_kwargs = {"print_off":True, "C17_eta_z_dep":C17_eta_z_dep, \
        "DM_outflow_C17":DM_outflow_C17, "t_star":t_star, \
            "t_inflow":t_inflow, "sfe_m_dep":sfe_m_dep}

# Add the sampled parameters
for key in var_range.keys():
    gamma_kwargs[key] = copy.deepcopy(em_sample_points[i][key])
    
    # Run GAMMA here
    
    # This is just a test for printing (see below)
    if i == i_test:
        gamma_kwargs_i_test = gamma_kwargs

print()
print('sampled parameters for index',i_test)
print('   ',gamma_kwargs_i_test)
