import ldlb_jaynes_cummings as ljc
import numpy as np
import dielectric_tensor as dt
import matplotlib.pyplot as plt
import ldlb_plotting as lp
import numpy.ma as ma



dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=2,volume_cell= 5e-8,damping_factor=.15,
                     length=1,gamma_type="linear")
energy_array = np.array([2,3])
energies = energy_array
dip_mags = 4e-3*np.array([1,1])
dip_angles = np.array([0,np.pi/100])
spectrum = np.linspace(1, 5, 1000)
cav_freq_array = np.linspace(1,4,100)

energies_to_save = 5
mean_int_len = ljc.mean_interaction_length(cav_freq_array,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal")
#mean_int_len_by_energy_array = ljc.select_from_energies(mean_int_len,spectrum,energy_array)
length_by_energy = 1/mean_int_len

plt.plot(cav_freq_array,length_by_energy)
plt.ylim(0,10)
plt.xlabel(r"$\omega_{cav}$",fontsize= 16)
plt.ylabel(r'$z$ (eV/$(\hbar c)$)',fontsize = 16)
plt.tight_layout()
plt.savefig("mean_int_len.png")
plt.show()

#needs to be by spectrum for later code to work
mean_int_len = ljc.mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal")
#mean_int_len_by_energy_array = ljc.select_from_energies(mean_int_len,spectrum,energy_array)
length_by_energy = 1/mean_int_len


#length_by_energy = 10

vec_pot = 20/np.sqrt(2)
polarization = np.array([1,1]) # linearly-polarized
#polarization = np.array([1,0]) # LHP
#polarization = np.array([0,1]) #RHP
#polarization = np.array([np.cos(.65),np.sin(.65)]) #theta .6

num_excited_states = np.size(energy_array)
import combinatorics as combo
full_basis = combo.construct_organic_basis(num_excited_states,1+1)

dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
e_x_full, a_x_full,chiral_int_set = ljc.jaynes_cummings_organic_ldlb_sweep(1,energy_array,cav_freq_array,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length_by_energy)
energy_save = e_x_full[:energies_to_save,:]
a_x_save = a_x_full[:energies_to_save,:energies_to_save,:]


plt.plot(cav_freq_array,chiral_int_set[0,:],label = r"$\mu_1$",color = "red")
plt.plot(cav_freq_array,chiral_int_set[1,:],label = r"$\mu_2$",color = "blue")
plt.xlabel(r"$\omega_{cav}$",fontsize= 16)
plt.ylabel(r'$\sigma$ (dim.)',fontsize = 16)
plt.legend()
plt.tight_layout()
plt.savefig("chiral_int.png")
plt.show()

filename_to_save = "strong_dip_length_dep_dipoles_NP_vec_20"
filename_to_save = filename_to_save.replace(".","pt")
lp.plot_set(filename_to_save+"energies.png",cav_freq_array,energy_save)
r = np.abs(a_x_save[ 2, :, :]) ** 2
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
lp.plot_set_colored(filename_to_save+"a_x_r.png",cav_freq_array,energy_save,
                r_masked,x_label = "Cavity Energy (eV)",y_label= "Eigenstate Energy (eV)",opacity = .7,norm_max = mask_value,colorbar_label=r"$A_r$")

lp.plot_set_colored(filename_to_save + "helicity.png", cav_freq_array, energy_save,
                    helicity,x_label = "Cavity Energy (eV)",y_label= "Eigenstate Energy (eV)",
                    opacity = .7,norm_max = mask_value,norm_min = -mask_value,colorbar_label = r"$\eta$",**{"show_min_max":False})
lp.plot_set_colored(filename_to_save + "chi_pol.png", cav_freq_array, energy_save,
                    chi_pol, x_label="Cavity Energy (eV)", y_label="Eigenstate Energy (eV)",
                    opacity=.7, norm_max=.1, norm_min=-.1, colorbar_label=r"$\chi_{pol}$",
                    **{"show_min_max": False})
lp.plot_set_colored(filename_to_save + "hel_pol.png", cav_freq_array, energy_save,
                    helical_pol, x_label="Cavity Energy (eV)", y_label="Eigenstate Energy (eV)",
                    opacity=.7, norm_max=.1, norm_min=-.1, colorbar_label=r"$\eta_{pol}$",
                    **{"show_min_max": False})