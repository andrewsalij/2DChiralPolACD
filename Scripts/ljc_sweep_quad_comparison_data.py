import numpy as np
import dielectric_tensor as dt
import ldlb_jaynes_cummings as ljc
import combinatorics as combo
import ldlb_plotting as lp

gam_set = np.array([.5,.10,.15])
vp_set = np.array([35,70])/np.sqrt(2)
del_energy_set = np.array([1,.5])
dip_rat_set = np.array([.5,1,2])

spectrum = np.linspace(0.5, 3.5, 2000) #needs to be high to prevent aggregate beating
cav_freq_array = np.linspace(1.5,3.5,201)

helical_pol_lp_total = np.zeros((8,np.size(cav_freq_array)))
#DEFAULT Parameters
damp_init = .1
dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=8,volume_cell= 5.8075e-7,damping_factor=damp_init,
                     length=1,gamma_type="linear")

energy_array = np.array([2,2+del_energy_set[0]])
dip_mags = 1.05474e-3 * np.array([1, 1])  # 10 Debye
dip_angles = np.array([0,-np.pi/4])


vec_pot = vp_set[0]
polarization = np.array([1,1]) #nonpolarized
num_excited_states = np.size(energy_array)
full_basis = combo.construct_organic_basis(num_excited_states,1+1)

#note that unchanging params are constants such as spectrum, cav freq array
#function does not generalize outside this script--do not use elsewhere
def get_helical_pol_lower_polariton(dielectric_params,energy_array,dip_mags,dip_angles,vec_pot,loss = 0,visualize = True):
    length_array = ljc.mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal",loss=loss)
    dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
    e_x_full, a_x_full,chiral_int = ljc.jaynes_cummings_organic_ldlb_sweep(1,energy_array,cav_freq_array,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length_array,mueller_correction = False,brown_style="pert")

    helical_pol = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_full,full_basis,[0,1],[2])
    helical_pol = np.nan_to_num(helical_pol)
    helical_pol_lower_polariton = helical_pol[1, :]  # recall that lowest state is the vacuum state
    if (visualize):
        lp.plot_set_colored("",cav_freq_array,e_x_full,helical_pol,norm_min = -.5,norm_max = .5)
    return helical_pol_lower_polariton

#Default
helical_pol_lp_total[0,:] = get_helical_pol_lower_polariton(dielectric_params,energy_array,dip_mags,dip_angles,vec_pot)


#Dipole changes
dip_mags_2 = np.array([dip_mags[0],dip_mags[1]*.5])
helical_pol_lp_total[1,:] = get_helical_pol_lower_polariton(dielectric_params,energy_array,dip_mags_2,dip_angles,vec_pot)

dip_mags_3 = np.array([dip_mags[0],dip_mags[1]*2])
helical_pol_lp_total[2,:] = get_helical_pol_lower_polariton(dielectric_params,energy_array,dip_mags_3,dip_angles,vec_pot)

#Damping changes
dielectric_params.gamma = gam_set[0]
helical_pol_lp_total[3,:] = get_helical_pol_lower_polariton(dielectric_params,energy_array,dip_mags,dip_angles,vec_pot)

dielectric_params.gamma = gam_set[1]
helical_pol_lp_total[4,:] = get_helical_pol_lower_polariton(dielectric_params,energy_array,dip_mags,dip_angles,vec_pot)

dielectric_params.gamma = gam_set[2]
helical_pol_lp_total[5,:] = get_helical_pol_lower_polariton(dielectric_params,energy_array,dip_mags,dip_angles,vec_pot)

#resetting
dielectric_params.gamma = damp_init
vec_pot_high = vp_set[1]
helical_pol_lp_total[6,:] = get_helical_pol_lower_polariton(dielectric_params,energy_array,dip_mags,dip_angles,vec_pot_high)

energy_array_close = np.array([2,2+del_energy_set[1]])
helical_pol_lp_total[7,:] = get_helical_pol_lower_polariton(dielectric_params,energy_array_close,dip_mags,dip_angles,vec_pot)

filename = "lower_polariton_variable_comparison_.npy"
np.save(filename,helical_pol_lp_total)
