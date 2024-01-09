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
dielectric_inf = 8
damping_factor = .1
volume = 5.8075e-7#in eV vol units
dielectric_params= dt.DIELECTRIC_PARAMS(dielectric_inf, volume, damping_factor, length=1, gamma_type="linear")
first_transition_strength = 0.0014407702432843387# in e hc/eV units, 13.66 Debye , same as all-cis TDDFT fit

non_vib_energies = np.array([2.6, 3.11, 3.99])
my_mock_results = ap.TDDFT_RESULTS(non_vib_energies, None,  np.array([.455, .11, .10]), "manual", "manual")
vib_results = my_mock_results.vibronic_dressing(np.array([0, 1, 2]), np.array([.185, .185, .185]),
                                                np.array([1.04, 0, 0]), vib_modes=np.arange(8))

#note this is reverse sign from JACS paper
alpha =-.0583*np.pi
beta = .175*np.pi
init_dip_mags = np.sqrt(vib_results.osc_array)
dip_mags = first_transition_strength/np.sqrt(my_mock_results.osc_array[0])* init_dip_mags
dip_angles = np.hstack((np.ones(8)*0,np.ones(8)*alpha,np.ones(8)*beta))
energy_array = vib_results.energies
energies = energy_array


spectrum = np.linspace(1, 5, 1000)
cav_freq_array = np.linspace(1,5,1000)
energies_to_save =27
energy_save = np.zeros((energies_to_save,np.size(cav_freq_array)))
a_x_save =  np.zeros((energies_to_save,energies_to_save,np.size(cav_freq_array)),dtype= np.csingle)
vec_pot = 50
#polarization = np.array([1,1]) # linearly-polarized
polarization = np.array([1,1]) # NP

length = dielectric_params.length
num_excited_states = np.size(energies)
import combinatorics as combo
full_basis = combo.construct_organic_basis(num_excited_states,1+1)
dip_angles = np.hstack((np.ones(8)*0,np.ones(8)*alpha,np.ones(8)*beta))
loss = 0 #loss percent per twice film thickness

length_array = ljc.mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal",loss=loss)



dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
e_x_full, a_x_full,chiral_int = ljc.jaynes_cummings_organic_ldlb_sweep(1,energies,cav_freq_array,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length_array,mueller_correction = False,brown_style ="pert")
energy_save[:,:] = e_x_full[:energies_to_save,:]
a_x_save[:,:,:] = a_x_full[:energies_to_save,:energies_to_save,:]


filename_to_save = "di_bari_reprod_vec50_v2_brown_style_pert"
filename_to_save = filename_to_save.replace(".","pt")
lp.plot_set(filename_to_save+"energies.png",cav_freq_array,energy_save[:,:])
helical_pol = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_save[:,:,:],full_basis,[0,1],[2])
chi_pol = ljc.polaritonic_characteristic_v2_from_eigenvecs(a_x_save[:,:,:],full_basis,[0,1],[2])
helicity = ljc.helicity_from_eigenvecs(a_x_save[:,:,:],full_basis,0,1)
lp.plot_set_colored(filename_to_save + "hel_pol.png", cav_freq_array, energy_save[:, :],
                    helical_pol, x_label="Cavity Energy (eV)", y_label="Eigenstate Energy (eV)",
                    opacity=1, norm_max=1, norm_min=-1, colorbar_label=r"$\eta_{pol}$",
                    **{"show_min_max": False})
np.save("di_bari_cav_freq.npy",cav_freq_array)
np.save("di_bari_energy.npy",energy_save)
np.save("di_bari_helical_pol.npy",helical_pol)