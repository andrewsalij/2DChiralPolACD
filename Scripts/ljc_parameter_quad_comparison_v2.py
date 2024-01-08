import numpy as np
import dielectric_tensor as dt
import ldlb_jaynes_cummings as ljc
import combinatorics as combo
import brioullinzonebasefinal as base
set_size  =100
gam_set = np.linspace(.03,.15,set_size)
del_energy_set = np.linspace(0,2,set_size)
mu_ratio_set = np.linspace(.3,3,100)

spectrum = np.linspace(0.5, 3.5, 2000) #needs to be high to prevent aggregate beating
cav_freq = 2.1

helical_pol_lp_total = np.zeros((3,set_size))
#DEFAULT Parameters
damp_init = .1
dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=8,volume_cell= 5.8075e-7,damping_factor=damp_init,
                     length=1,gamma_type="linear")

energy_array = np.array([2,3])
dip_mags = 1.05474e-3 * np.array([1, 1])  # 10 Debye
dip_angles = np.array([0,np.pi/4])


vec_pot = 35
polarization = np.array([1,1]) #nonpolarized
num_excited_states = np.size(energy_array)
full_basis = combo.construct_organic_basis(num_excited_states,1+1)

#note that unchanging params are constants such as spectrum, cav freq array
#function does not generalize outside this script--do not use elsewhere
def get_helical_pol_lower_polariton(dielectric_params,energy_array,cav_freq,dip_mags,dip_angles,vec_pot,loss = 0):
    length = ljc.mean_interaction_length(cav_freq,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal",loss=loss)[0]

    dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
    new_jc_h,new_chiral_int = ljc.organic_hamiltonian_ldlb_no_vib(1,energy_array,cav_freq,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length,mueller_correction = False,brown_style="pert")
    cur_e_x, cur_a_x = base.solve_hamiltonian(new_jc_h)
    helical_pol = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(cur_a_x,full_basis,[0,1],[2])
    helical_pol = np.nan_to_num(helical_pol)
    helical_pol_lower_polariton = helical_pol[1]  # recall that lowest state is the vacuum state
    return helical_pol_lower_polariton

def get_helical_pol_mu_sweep(mu_ratio_set,dielectric_params,energy_array,cav_freq,dip_mags,dip_angles,vec_pot):
    helical_pol_set = np.zeros(np.size(mu_ratio_set))
    for i in range(0, np.size(mu_ratio_set)):
        dip_mags_to_use = np.array([dip_mags[0],dip_mags[0]*mu_ratio_set[i]])
        helical_pol = get_helical_pol_lower_polariton(dielectric_params, energy_array, cav_freq, dip_mags_to_use, dip_angles,
                                                      vec_pot)
        helical_pol_set[i] = helical_pol
    return helical_pol_set

def get_helical_pol_gam_sweep(gam_set,dielectric_params,energy_array,cav_freq,dip_mags,dip_angles,vec_pot):
    helical_pol_set = np.zeros(np.size(gam_set))
    gam_init = dielectric_params.gamma
    for i in range(0,np.size(gam_set)):
        dielectric_params.gamma = gam_set[i]
        helical_pol = get_helical_pol_lower_polariton(dielectric_params,energy_array,cav_freq,dip_mags,dip_angles,vec_pot)
        helical_pol_set[i] = helical_pol
    dielectric_params.gamma = gam_init #resetting so that dielectric stays unchanged--this is bad practice but somewhat necc.
    return helical_pol_set

def get_helical_pol_energy_sweep(delta_e_set,dielectric_params,energy_array,cav_freq,dip_mags,dip_angles,vec_pot):
    helical_pol_set = np.zeros(np.size(delta_e_set))
    base_e = energy_array[0]
    for i in range(0,np.size(gam_set)):
        energy_array_to_use = np.array([base_e,base_e+delta_e_set[i]])
        helical_pol = get_helical_pol_lower_polariton(dielectric_params,energy_array_to_use,cav_freq,dip_mags,dip_angles,vec_pot)
        helical_pol_set[i] = helical_pol
    return helical_pol_set

helical_pol_lp_total[0,:] = get_helical_pol_mu_sweep(mu_ratio_set,dielectric_params,energy_array,cav_freq,dip_mags,dip_angles,vec_pot)
helical_pol_lp_total[1,:] = get_helical_pol_gam_sweep(gam_set,dielectric_params,energy_array,cav_freq,dip_mags,dip_angles,vec_pot)
helical_pol_lp_total[2,:] = get_helical_pol_energy_sweep(del_energy_set,dielectric_params,energy_array,cav_freq,dip_mags,dip_angles,vec_pot)
filename = "lower_polariton_2pt1comparison_bcd.npy"
np.save(filename,helical_pol_lp_total)