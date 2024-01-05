import numpy as np
import dielectric_tensor as dt
import ldlb_jaynes_cummings as ljc
import ldlb_plotting as lp
import numpy.ma as ma
import albano_params as ap
import matplotlib.pyplot as plt
import brioullinzonebasefinal as base
from matplotlib.collections import LineCollection

gamma= .1
gamma_type = "linear"
non_vib_energies =np.array([2.6,3.99])
elec_results = my_mock_results = ap.TDDFT_RESULTS(non_vib_energies, None, np.array([.455,.10]), "my_brain", "my_brain")

beta = .175*np.pi
dip_angles = np.array([0,beta])
#second_film_rotation = -.021*np.pi
dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=8,volume_cell= 5.8075e-7,damping_factor=gamma ,
                     length=1,gamma_type="linear")
init_dip_mags = np.sqrt(elec_results.osc_array)
first_transition_strength = 0.0014407702432843387# in e hc/eV units, 13.66 Debye , same as all-cis TDDFT fit
dip_mags = first_transition_strength / np.sqrt(elec_results.osc_array[0]) * init_dip_mags
dip_mags = dip_mags
energy_array = non_vib_energies
energies = energy_array


spectrum = np.linspace(1, 5, 1000)
cav_freq_array = np.linspace(1,5,1000)
energies_to_save =5
energy_save = np.zeros((energies_to_save,np.size(cav_freq_array)))
a_x_save =  np.zeros((energies_to_save,energies_to_save,np.size(cav_freq_array)),dtype= np.csingle)
vec_pot = 50
#polarization = np.array([1,1]) # linearly-polarized
polarization = np.array([1,1]) # NP
#polarization = np.array([0,1]) #RHP
#polarization = np.array([np.cos(.65),np.sin(.65)]) #theta .6
length = dielectric_params.length
num_excited_states = np.size(energies)
import combinatorics as combo
full_basis = combo.construct_organic_basis(num_excited_states,1+1)
loss = 0 #loss percent per twice film thickness

length_array = ljc.mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal",loss=loss)



dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
e_x_full, a_x_full,chiral_int = ljc.jaynes_cummings_organic_ldlb_sweep(1,energies,cav_freq_array,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length_array,mueller_correction = False)
energy_save[:,:] = e_x_full[:energies_to_save,:]
a_x_save[:,:,:] = a_x_full[:energies_to_save,:energies_to_save,:]


filename_to_save = "di_bari_reprod_subset_vec50"
filename_to_save = filename_to_save.replace(".","pt")
lp.plot_set(filename_to_save+"energies.png",cav_freq_array,energy_save[:,:])
helical_pol = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_save[:,:,:],full_basis,[0,1],[2])
helical_pol = np.nan_to_num(helical_pol,0)
chi_pol = ljc.polaritonic_characteristic_v2_from_eigenvecs(a_x_save[:,:,:],full_basis,[0,1],[2])
helicity = ljc.helicity_from_eigenvecs(a_x_save[:,:,:],full_basis,0,1)
lp.plot_set_colored(filename_to_save + "hel_pol.png", cav_freq_array, energy_save[:, :],
                    helical_pol, x_label="Cavity Energy (eV)", y_label="Eigenstate Energy (eV)",
                    opacity=.7, norm_max=1, norm_min=-1, colorbar_label=r"$\eta_{pol}$",
                    **{"show_min_max": False})
leader = "di_bari_subset"
np.save(leader+"_cav_freq.npy",cav_freq_array)
np.save(leader+"_energy.npy",energy_save)
np.save(leader+"_helical_pol.npy",helical_pol)