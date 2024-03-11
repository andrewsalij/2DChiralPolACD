
import numpy as np
import dielectric_tensor as dt
import ldlb_jaynes_cummings as ljc
import ldlb_plotting as lp
dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=8,volume_cell= 5.8075e-7,damping_factor=.1,
                     length=1,gamma_type="linear")

energy_array = np.array([2,3])
energies = energy_array
dip_mags = 1.05474e-3*np.array([1,1]) #10 Debye

dip_angles = np.array([0,np.pi/4])
spectrum = np.linspace(1, 5, 1000)
cav_freq_array = np.linspace(1,4,100)
energies_to_save =12
energy_save = np.zeros((energies_to_save,np.size(cav_freq_array),2))
a_x_save =  np.zeros((energies_to_save,energies_to_save,np.size(cav_freq_array),2),dtype= np.csingle)
vec_pot_1 = 70/np.sqrt(2)
polarization = np.array([1,1])

length = dielectric_params.length
num_excited_states = np.size(energies)
import combinatorics as combo
full_basis = combo.construct_organic_basis(num_excited_states,1+1)

full_basis_extended = combo.construct_organic_basis(num_excited_states,3)
energy_save = np.zeros((energies_to_save,np.size(cav_freq_array),2))
a_x_save =  np.zeros((energies_to_save,energies_to_save,np.size(cav_freq_array),2),dtype= np.csingle)

loss = 0 #loss percent per twice film thickness

length_array = ljc.mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal",loss=loss)

dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
e_x_full, a_x_full,chiral_int = ljc.jaynes_cummings_organic_ldlb_sweep(2,energies,cav_freq_array,vec_pot_1,polarization,dipole_matrix,dielectric_params,spectrum,length_array,
                                                                       mueller_correction = False,brown_style="pert",interaction_mask = None,to_counter_rotate=True)

e_x_masked, a_x_masked,chiral_int = ljc.jaynes_cummings_organic_ldlb_sweep(2,energies,cav_freq_array,vec_pot_1,polarization,dipole_matrix,dielectric_params,spectrum,length_array,mueller_correction = False,brown_style="pert",interaction_mask = np.array([1,0]))

energy_save[:,:,0] = e_x_full
energy_save[:,:,1] = e_x_masked
a_x_save[:,:,:,0] = a_x_full
a_x_save[:,:,:,1] = a_x_masked


helical_pol_1 = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_full,full_basis_extended,[0,1],[2])
helical_pol_2 = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_masked,full_basis_extended,[0,1],[2])

helical_pol_stack = np.stack((helical_pol_1,helical_pol_2),axis= 2)
filename = "fig_one_v_two_v4"
lp.plot_double_set_colored_shared_y_axis(filename,cav_freq_array,energy_save,helical_pol_stack,
                                         x_label=r"$\Omega$ (eV/$\hbar$)",y_label= r"$E^\alpha$ (eV)",norm_min = -.25,norm_max= .25,figsize = (3.3,3),cbar_style = "horizontal",
                                         **{"x_bounds" : [1.7,2.3],"y_bounds" :[1.8,2.2]})

np.save("extended_one_v_two_energy.npy",energy_save)
np.save("extended_one_v_two_helical_pol.npy",helical_pol_stack)
plot_chirality = False
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
#for chirality presentation
subplot_adjustments = {"left":.15,"right":.85,"bottom":.15}
if (plot_chirality):
    lp.plot_set_colored("four_state_hel_pol.png", cav_freq_array, e_x_full,
                        helical_pol_1,  x_label=r"Cavity Energy (eV)", y_label=r"Eigenstate Energy (eV)",
                        opacity=1, norm_max=.4, norm_min=-.4, cmap="bwr",
                        colorbar_label=r"Polaritonic Helicity ($\tilde{g}$)",
                        **{"show_min_max": False,"y_bounds":[1.88,2.0],"x_bounds":[1.85,2.0],"label_fontsize": 20,
                           "linewidth":10,"subplot_adjustments":subplot_adjustments})