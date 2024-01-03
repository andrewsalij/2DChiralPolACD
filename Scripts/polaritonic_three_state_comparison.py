import ldlb_jaynes_cummings as ljc
import numpy as np
import dielectric_tensor as dt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mueller
import ldlb_plotting as lp
import numpy.ma as ma


dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=2,volume_cell= 5e-8,damping_factor=.08,
                     length=1,gamma_type="linear")
energy_array = np.array([2,3])
energies = energy_array
dip_mags = 1e-4*np.array([1,1])

dip_angles = np.array([0,np.pi/4])
dipole_matrix = dt.create_dipole_matrix_polar_2D(dip_mags,dip_angles)
spectrum = np.linspace(1, 5, 1000)
#cavity frequency must be subset of spectrum for code to not have finite numeric errors
cav_freq_array = spectrum
size = np.size(cav_freq_array)

energies_to_save = 5
mean_int_len = ljc.mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal")
length_by_energy = mean_int_len

vec_pot = 20/np.sqrt(2)
polarization = np.array([1,1]) # linearly-polarized
dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)

e_x_full, a_x_full,chiral_int_set = ljc.jaynes_cummings_organic_ldlb_sweep(1,energy_array,cav_freq_array,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length_by_energy)
energy_save, a_x_save = e_x_full,a_x_full

cav_freq_subset_values = [200,300]
cav_freq_subset = cav_freq_array[cav_freq_subset_values[0]:cav_freq_subset_values[1]]
subsize = np.size(cav_freq_subset)
chiral_factor = chiral_int_set[0,cav_freq_subset_values[0]:cav_freq_subset_values[1]]
detuning = cav_freq_subset-energy_array[0]
achiral_int_set = ljc.achiral_interaction(energy_array,dip_mags,vec_pot)

achiral_int = np.ones(subsize)*achiral_int_set[0]

three_state_polaritons = ljc.THREE_STATE_ANAL_POLARITONS(achiral_int,chiral_factor,detuning)

import combinatorics as combo
full_basis  = combo.construct_organic_basis(2,2)
phot,lop,up = three_state_polaritons.all_states()

basis,phot_indices,elec_indices = ljc.three_state_basis()

gamma_plus = three_state_polaritons.gamma_factor()

lp_polaritonic_char = ljc.polaritonic_characteristic_from_eigenvecs(lop,basis,phot_indices,elec_indices)
up_polaritonic_char = ljc.polaritonic_characteristic_from_eigenvecs(up,basis,phot_indices,elec_indices)

lp_helical = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(lop,basis,phot_indices,elec_indices)

polaritons_extracted = ljc.extract_polaritons(a_x_save,[3])
polaritons_numeric = polaritons_extracted[0]
upper_polariton_numeric = polaritons_numeric[:,1,:]
up_polaritonic_char_numeric = ljc.polaritonic_characteristic_from_eigenvecs(upper_polariton_numeric,full_basis,phot_indices,elec_indices)

up_helical_metric = ljc.helicity_from_eigenvecs(up,basis,0,1)
up_helical_anal = three_state_polaritons.sigma
up_helical_numeric = ljc.helicity_from_eigenvecs(upper_polariton_numeric,full_basis,0,1)


filename_to_save = "show_polariton_states_testing"
filename_to_save = filename_to_save.replace(".","pt")
r = np.abs(a_x_save[2, :, :]) ** 2
l = np.abs(a_x_save[1, :, :]) ** 2
polarized_states=  (l-r)/(l+r+1e-8)
helical_pol = ljc.helical_polaritonic_characteristic_from_eigenvecs(a_x_save,full_basis,[0,1],[2])
chi_phot, chi_elec = ljc.polaritonic_characteristic_from_eigenvecs(a_x_save,full_basis,[0,1],[2])
chi_pol = chi_phot+chi_elec
mixed_state_metric = ljc.mixed_electronic_state_characteristic_from_eigenvecs(a_x_save,full_basis,[2])
mixed_state_masked = ma.masked_less(mixed_state_metric,.0002)
mask_value = 1.001
r_masked = ma.masked_greater(r,mask_value)
polarized_states_masked = ma.masked_greater(polarized_states,mask_value)
helicity = ljc.helicity_from_eigenvecs(a_x_save[:,:,:],full_basis,0,1)
# lp.plot_set_colored(filename_to_save+"a_x_r.png",cav_freq_array,energy_save,
#                 r_masked,x_label = "Cavity Energy (eV)",y_label= "Eigenstate Energy (eV)",opacity = .7,norm_max = mask_value,colorbar_label=r"$A_r$")
#
lp.plot_set_colored(filename_to_save + "helicity_pol.png", cav_freq_array, energy_save,
                    helical_pol,x_label = "Cavity Energy (eV)",y_label= "Eigenstate Energy (eV)",
                    opacity = .7,norm_max = 0.5,norm_min = -.5,colorbar_label = r"$\eta_{pol}$",**{"show_min_max":False,"x_bounds":[1.9,2.1],"y_bounds":[1.9,2.1]})
# lp.plot_set_colored(filename_to_save + "chi_pol.png", cav_freq_array, energy_save,
#                     chi_pol, x_label="Cavity Energy (eV)", y_label="Eigenstate Energy (eV)",
#                     opacity=.7, norm_max=.1, norm_min=-.1, colorbar_label=r"$\chi_{pol}$",
#                     **{"show_min_max": False})
# lp.plot_set_colored(filename_to_save + "hel_pol.png", cav_freq_array, energy_save,
#                     helical_pol, x_label="Cavity Energy (eV)", y_label="Eigenstate Energy (eV)",
#                     opacity=.7, norm_max=.1, norm_min=-.1, colorbar_label=r"$\eta_{pol}$",
#                     **{"show_min_max": False})





chi_numeric = up_polaritonic_char_numeric[0]+up_polaritonic_char_numeric[1]
chi_numeric_subset=  chi_numeric[cav_freq_subset_values[0]:cav_freq_subset_values[1]]
plt.plot(cav_freq_subset,chi_numeric_subset,label = "Numeric")
plt.plot(cav_freq_subset,up_polaritonic_char[0]+up_polaritonic_char[1],label = "Three State Analytic")
plt.plot(cav_freq_subset,-np.abs(detuning/(2*np.sqrt(2)*achiral_int))+1,label = "Linear Approximation")
plt.legend()
plt.xlabel("Cavity Energy (eV)")
plt.ylabel(r"$\chi_{pol}$")
plt.savefig("chi_pol_comparison.pdf")
plt.show()


plt.plot(cav_freq_subset,up_helical_anal,label = "Analytic")
plt.plot(cav_freq_subset,up_helical_metric,label = "Three State Analytic")
plt.plot(cav_freq_subset,up_helical_numeric[cav_freq_subset_values[0]:cav_freq_subset_values[1]],label = "Numeric")
plt.legend()
plt.xlabel("Cavity Energy (eV)")
plt.ylabel(r"$\eta$")
plt.savefig("eta_comparison.pdf")
plt.show()

chiral_simple = ljc.chiral_factor_ultra_approximate(energy_array[0],energy_array[1],dielectric_params.gamma*energy_array[1],cav_freq_array)
chiral_semisimple = ljc.chiral_factor_semi_approximate(cav_freq_array,energy_array,dip_mags,dielectric_params,length_by_energy)
chiral_anal = ljc.chiral_factor_two_dipole_anal(cav_freq_array,energy_array,dip_mags,dip_angles,dielectric_params,length_by_energy)
#plt.plot(cav_freq_array,chiral_simple)
plt.plot(cav_freq_array,chiral_int_set[0,:])
plt.plot(cav_freq_array,chiral_anal)
#plt.plot(cav_freq_array,chiral_semisimple)
plt.show()