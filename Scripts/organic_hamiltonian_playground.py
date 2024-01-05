import numpy as np
import dielectric_tensor as dt
import ldlb_jaynes_cummings as ljc
import ldlb_plotting as lp
import numpy.ma as ma


'''
Script for testing arbitrary systems'''


dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=2,volume_cell= 2.323e-7,damping_factor=.09,
                     length=1,gamma_type="linear")
energy_array = np.array([2,3])
energies = energy_array
dip_mags = 1e-3*np.array([1,1])
dip_second_angle_set = np.array([np.pi/8])
dip_angles = np.array([-np.pi/8,dip_second_angle_set[0]])
spectrum = np.linspace(1, 5, 1000)
cav_freq_array = np.linspace(1,4,100)
energies_to_save =5
energy_save = np.zeros((np.size(dip_second_angle_set),energies_to_save,np.size(cav_freq_array)))
a_x_save =  np.zeros((np.size(dip_second_angle_set),energies_to_save,energies_to_save,np.size(cav_freq_array)),dtype= np.csingle)
vec_pot = 50/np.sqrt(2)

#polarization = np.array([1,1]) # linearly-polarized
polarization = np.array([1,1]) # LHP
#polarization = np.array([0,1]) #RHP
#polarization = np.array([np.cos(.65),np.sin(.65)]) #theta .6
length = dielectric_params.length
num_excited_states = np.size(energies)
import combinatorics as combo
full_basis = combo.construct_organic_basis(num_excited_states,1+1)

loss = 0 #loss percent per twice film thickness

for i in range(0,np.size(dip_second_angle_set)):
    length_array = ljc.mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal",loss=loss)


    dip_angles = np.array([0,dip_second_angle_set[i]])
    dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
    e_x_full, a_x_full,chiral_int = ljc.jaynes_cummings_organic_ldlb_sweep(1,energies,cav_freq_array,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length_array,mueller_correction = False)
    energy_save[i,:,:] = e_x_full[:energies_to_save,:]
    a_x_save[i,:,:,:] = a_x_full[:energies_to_save,:energies_to_save,:]



for i in range(0,np.size(dip_second_angle_set)):
    angle_index = i
    filename_to_save = "di_bari_mock_no_low_chiral_int"+str(i)
    filename_to_save = filename_to_save.replace(".","pt")
    lp.plot_set(filename_to_save+"energies.png",cav_freq_array,energy_save[angle_index,:,:])
    r = np.abs(a_x_save[angle_index, 2, :, :]) ** 2
    l = np.abs(a_x_save[angle_index, 1, :, :]) ** 2
    polarized_states=  (l-r)/(l+r+1e-8)
    helical_pol = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_save[angle_index,:,:,:],full_basis,[0,1],[2])
    chi_pol = ljc.polaritonic_characteristic_v2_from_eigenvecs(a_x_save[angle_index,:,:,:],full_basis,[0,1],[2])
    helical_pol = lp.mask_array_by_other_array_condition(helical_pol,chi_pol,np.greater,.01)
    mixed_state_metric = ljc.mixed_electronic_state_characteristic_from_eigenvecs(a_x_save[angle_index,:,:,:],full_basis,[2])
    mixed_state_masked = ma.masked_less(mixed_state_metric,.0002)
    mask_value = 1.001
    r_masked = ma.masked_greater(r,mask_value)
    polarized_states_masked = ma.masked_greater(polarized_states,mask_value)
    helicity = ljc.helicity_from_eigenvecs(a_x_save[angle_index,:,:,:],full_basis,0,1)
    lp.plot_set_colored(filename_to_save+"a_x_r.png",cav_freq_array,energy_save[angle_index,:,:],
                    r_masked,x_label = "Cavity Energy (eV)",y_label= "Eigenstate Energy (eV)",opacity = .7,norm_max = mask_value,colorbar_label=r"$A_r$")

    lp.plot_set_colored(filename_to_save + "helicity.png", cav_freq_array, energy_save[angle_index, :, :],
                        helicity,x_label = "Cavity Energy (eV)",y_label= "Eigenstate Energy (eV)",
                        opacity = .7,norm_max = mask_value,norm_min = -mask_value,colorbar_label = r"$\eta$",**{"show_min_max":True})
    lp.plot_set_colored(filename_to_save + "hel_pol.png", cav_freq_array, energy_save[angle_index, :, :],
                        helical_pol, x_label="Cavity Energy (eV)", y_label="Eigenstate Energy (eV)",
                        opacity=.7, norm_max=.5, norm_min=-.5, colorbar_label=r"$\eta_{pol}$",
                        **{"show_min_max": False})
    import matplotlib.cm as cm
    mixed_state_cmap = cm.get_cmap("Reds")
    mixed_state_cmap.set_bad(color= "white")
    lp.plot_set_colored(filename_to_save + "mixing.png", cav_freq_array, energy_save[angle_index, :, :],
                        mixed_state_masked, x_label="Cavity Energy (eV)", y_label="Eigenstate Energy (eV)",
                        opacity=.7, norm_max=.2, norm_min=0, cmap = mixed_state_cmap,colorbar_label=r"$\mathcal{M}_{el}$",
                        **{"show_min_max": False})


