import numpy as np
import dielectric_tensor as dt
import ldlb_plotting as ldlb_plt
import python_util
import matplotlib.pyplot as plt

#frequently run post-processing routines for scripts--highly specific, not generalizable
#was copying the same stuff over an over though, so this ended up being necessary
#this is purely electronic
def lin_spec_from_tddft(filename,spec,damping_spec,tddft_results):
    lin_abs_spec = tddft_results.linear_lorentzian_spec(spec,damping_spec)
    plt.plot(dt.nm_to_eV(spec),lin_abs_spec)
    plt.savefig(filename+"lin_abs")
    plt.show()

def two_film_from_tddft(filename,prefactor,damping_tuner,tddft_results,second_film_rotation = np.array([0,0,np.pi/4]),spec = dt.nm_to_eV(np.linspace(250,600,2000))):
    dipole_mat_init = tddft_results.dip_mat
    dipole_mat_init = python_util.remove_unnecessary_indices(dipole_mat_init)
    unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297))
    dielectric_inf = 1 #placeholder
    volume = 1.25e-7 #in eV vol units
    cd_per_mdeg_to_cd_factor = 3.491e-5
    dipole_mat = prefactor*dipole_mat_init[:,:]
    e_array =tddft_results.energies
    damping_factor = damping_tuner
    dp = dt.DIELECTRIC_PARAMS(dielectric_inf, volume, damping_factor,length = 1,gamma_type = "linear")
    ldlb_plt.plot_ldlb_orientations(filename+"orientations",dipole_mat,e_array, spec, second_film_rotation, dp,
                                    cd_conversion_factor= 1/cd_per_mdeg_to_cd_factor)

    ldlb_plt.plot_dipoles(dipole_mat[:, :2], filename=filename + "reprod_dipoles.png")

def two_film_from_tddft_vibronic(filename,prefactor,damping_tuner,tddft_results,vib_dressing,second_film_rotation = np.array([0,0,np.pi/4]),spec = dt.nm_to_eV(np.linspace(250,600,2000))):
    vibronic_results = tddft_results.vibronic_dressing(vib_dressing.vib_index,vib_dressing.vib_dist,vib_dressing.huang_rhys)
    dipole_mat_init = vibronic_results.dip_mat
    dipole_mat_init = python_util.remove_unnecessary_indices(dipole_mat_init)
    dielectric_inf = 1 #placeholder
    volume = 1.25e-7 #in eV vol units
    cd_per_mdeg_to_cd_factor = 3.491e-5
    dipole_mat = prefactor*dipole_mat_init[:,:]
    e_array =vibronic_results.energies
    damping_factor = damping_tuner*e_array
    dp = dt.DIELECTRIC_PARAMS(dielectric_inf, volume, damping_factor,length = 1)
    ldlb_plt.plot_ldlb_orientations(filename+"orientations",dipole_mat,e_array, spec, second_film_rotation, dp,
                                    cd_conversion_factor= 1/cd_per_mdeg_to_cd_factor)

    ldlb_plt.plot_dipoles(dipole_mat[:, :2], filename=filename + "reprod_dipoles.png")

def get_lin_abs_anal(spec,dp,e_array,dip_mags,dip_angles,unit_defs = dt.unit_defs_base):
    '''treats dielectric function fully '''
    dipole_mat = dt.create_dipole_matrix_polar_3D(dip_mags, dip_angles)
    dielectric_tensor = dt.create_dielectric_tensor(dp, dipole_mat, e_array, spec, unit_defs, **{"dimension": 3})
    n_tensor = np.sqrt(dielectric_tensor)
    print("anal"+str(n_tensor[:,:,0]))
    prefactor = dp.length/(unit_defs.hbar*unit_defs.c)
    #note that abs_length = 2*(k_x+k_y)/2
    lin_abs = prefactor*spec*np.imag(n_tensor[0,0,:]+n_tensor[1,1,:])
    return lin_abs

def get_lin_abs_anal_pert(spec,dp,e_array,dip_mags,unit_defs = dt.unit_defs_base):
    '''analytic expression in JACS paper (23) '''
    prefactor = dt.ldlb_prefactor_2(dp.epsilon_inf,dp.v,dp.length,unit_defs=unit_defs)
    s_n_array = dip_mags**2*e_array
    s_n_matrix = np.tile(s_n_array, (np.size(spec), 1)).T
    v_n_matrix = dt.f_dielectric_im(e_array,spec,dp.damping_array(energy_array=e_array))
    lin_abs = prefactor*spec*np.sum(s_n_matrix*v_n_matrix,axis = 0)
    return lin_abs




def get_triple_spec_from_rotation_indices_optimization(indices_net,dip_mags,e_array,spec,dp, unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297)),dip_angle_style = "v1",flip_angles = False):
    min_cost_d = np.argmin(indices_net[3,:])
    params = indices_net[:3,min_cost_d]
    if (dip_angle_style == "v1"):
        dip_angles = np.array([params[0],params[1],0,0,0,0])
    if (dip_angle_style == "v2"):
        dip_angles = np.hstack((np.zeros(8),np.ones(8)*params[0],np.ones(8)*params[1]))
    else:
        num_vib = np.size(e_array)-2
        dip_angles = np.hstack((np.zeros(num_vib),np.array([params[0], params[1]])))
    second_film_rotation = np.array([0, 0, params[2]])
    if (flip_angles):
        second_film_rotation = -1*second_film_rotation
        dip_angles = -1*dip_angles
    return get_ldlb_two_film(dip_mags,dip_angles,e_array,spec,dp,second_film_rotation)

def get_ldlb_two_film(dip_mags,dip_angles,e_array,spec,dielectric_params, second_film_rotation, unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297))):
    dipole_mat = dt.create_dipole_matrix_polar_3D(dip_mags, dip_angles)
    dielectric_tensor = dt.create_dielectric_tensor(dielectric_params, dipole_mat, e_array, spec, unit_defs, **{"dimension": 3})
    lin_opt_params = dt.linear_optics_from_dielectric_tensor(dielectric_tensor, spec, to_print=False,
                                                             **{"length_over_c": dielectric_params.length / unit_defs.c})
    rotate_dip_mat_second_film = dt.rotate_vector(second_film_rotation, dipole_mat, transpose=True)
    rotated_dielectric_second_film = dt.create_dielectric_tensor(dielectric_params, rotate_dip_mat_second_film, e_array, spec,
                                                                 unit_defs, **{"dimension": 3})
    lin_opt_params_2 = dt.linear_optics_from_dielectric_tensor(rotated_dielectric_second_film, spec,
                                                               **{"length_over_c": dielectric_params.length / unit_defs.c})
    ldlb_two_film_response = dt.ldlb_two_film_from_params(lin_opt_params, lin_opt_params_2)
    r_flip = np.array([np.pi, 0, 0])
    net_rotation_flip = second_film_rotation + np.array([np.pi, 0, 0]) #this is fine bc z axis is just being inverted,but probably not the best way of generally doing rotations
    dip_mat_flip = dt.rotate_vector(r_flip, dipole_mat, transpose=True)
    dielectric_flip = dt.create_dielectric_tensor(dielectric_params, dip_mat_flip, e_array, spec, unit_defs, **{"dimension": 3})
    rotate_dip_mat_second_film_flip = dt.rotate_vector(net_rotation_flip, dipole_mat, transpose=True)
    rotated_dielectric_second_film_flip = dt.create_dielectric_tensor(dielectric_params, rotate_dip_mat_second_film_flip, e_array,
                                                                      spec, unit_defs, **{"dimension": 3})
    lin_opt_params_flip = dt.linear_optics_from_dielectric_tensor(dielectric_flip, spec,
                                                                  **{"length_over_c": dielectric_params.length / unit_defs.c})
    lin_opt_params_2_flip = dt.linear_optics_from_dielectric_tensor(rotated_dielectric_second_film_flip, spec,
                                                                    **{"length_over_c": dielectric_params.length / unit_defs.c})
    flipped_response = dt.ldlb_two_film_from_params(lin_opt_params_2_flip, lin_opt_params_flip)
    ldlb_two_film_response = ldlb_two_film_response
    flipped_response = flipped_response

    ldlb_semi_sum = (ldlb_two_film_response + flipped_response) / 2
    absorbance = lin_opt_params.absorbance
    return ldlb_two_film_response,flipped_response,ldlb_semi_sum,absorbance

def get_triple_spec_helix_params(spectrum,dielectric_params,e_array,dip_mags,dip_angles,gamma_array,total_rotation):
    ldlb = dt.ldlb_helical_perturbative(spectrum,dielectric_params,e_array,dip_mags,dip_angles,gamma_array,total_rotation)
    ldlb_flip = dt.ldlb_helical_perturbative(spectrum,dielectric_params,e_array,dip_mags,-1*dip_angles,gamma_array,total_rotation)
    ldlb_ss = (ldlb_flip+ldlb)/2
    return ldlb,ldlb_flip,ldlb_ss

#needs first set of combos to bee all dipole helical self-interactions
def get_subpeak_contributions(peak_combos,spectral_set):
    num_combos = np.size(peak_combos,axis= 1)
    sub_peak_contributions = np.zeros(np.shape(spectral_set))
    sole_peak_contrib_indices = []
    for i in np.arange(0,num_combos):
        first_peak = peak_combos[0,i]
        combo = peak_combos[:,i]
        if (np.all(combo==first_peak)):
            sole_peak_contrib_indices.append(i)
    num_peaks = np.size(sole_peak_contrib_indices)
    sole_peaks = np.zeros((num_peaks,np.size(spectral_set,axis=1)))
    for i in np.arange(0,num_peaks):
        sole_peaks[i,:] = spectral_set[sole_peak_contrib_indices[i],:]
    for i in np.arange(0,num_combos):
        first_peak = peak_combos[0,i]
        combo = peak_combos[:,i]
        if (np.all(combo==first_peak)):
            #note: this assumes that sole peaks are ordered 1,2,3... by index 0,1,2
            #a more formal code would work for arbitrary indexing
            sub_peak_contributions[i,:] = sole_peaks[int(first_peak-1),:]
        else:
            mask = np.full(num_peaks,False)
            mask[np.array([combo])-1] = True
            mask_full = np.tile(mask,(np.size(spectral_set,axis=1),1)).T
            sole_peak_contributions = np.sum(sole_peaks,axis=0,where = mask_full)
            corrected_spec=  spectral_set[i,:]-sole_peak_contributions
            sub_peak_contributions[i,:] = corrected_spec
    return sub_peak_contributions

def extract_dip_mags_eV(dip_vecs):
    eV_to_atomic = 1 / 0.00026811961479979727
    atomic_to_eV = 1 / eV_to_atomic
    dip_vecs_norm = np.sqrt(np.sum(dip_vecs**2,axis =1))
    return dip_vecs_norm*atomic_to_eV
