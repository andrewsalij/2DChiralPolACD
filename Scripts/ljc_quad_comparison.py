import numpy as np
import dielectric_tensor as dt
import ldlb_jaynes_cummings as ljc
import ldlb_plotting as lp
import numpy.ma as ma
import matplotlib.pyplot as plt
import brioullinzonebasefinal as base
from matplotlib.collections import LineCollection

dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=8,volume_cell= 5.8075e-7,damping_factor=.1,
                     length=1,gamma_type="linear")

energy_array = np.array([2,3])
energies = energy_array
dip_mags = 1.05474e-3*np.array([1,1]) #10 Debye

dip_angles = np.array([0,-np.pi/4])
spectrum = np.linspace(1, 5, 1000)
cav_freq_array = np.linspace(1,4,100)
energies_to_save =5
energy_save = np.zeros((energies_to_save,np.size(cav_freq_array),2))
a_x_save =  np.zeros((energies_to_save,energies_to_save,np.size(cav_freq_array),2),dtype= np.csingle)
vec_pot_1 = 35/np.sqrt(2)
vec_pot_2 = 70/np.sqrt(2)
polarization = np.array([1,1])

length = dielectric_params.length
num_excited_states = np.size(energies)
import combinatorics as combo
full_basis = combo.construct_organic_basis(num_excited_states,1+1)

loss = 0 #loss percent per twice film thickness

length_array = ljc.mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal",loss=loss)

dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
e_x_full, a_x_full,chiral_int = ljc.jaynes_cummings_organic_ldlb_sweep(1,energies,cav_freq_array,vec_pot_1,polarization,dipole_matrix,dielectric_params,spectrum,length_array,mueller_correction = False)
energy_save[:,:,0] = e_x_full[:energies_to_save,:]
a_x_save[:,:,:,0] = a_x_full[:energies_to_save,:energies_to_save,:]

e_x_full, a_x_full,chiral_int = ljc.jaynes_cummings_organic_ldlb_sweep(1,energies,cav_freq_array,vec_pot_2,polarization,dipole_matrix,dielectric_params,spectrum,length_array,mueller_correction = False,brown_style="pert")
energy_save[:,:,1] = e_x_full[:energies_to_save,:]
a_x_save[:,:,:,1] = a_x_full[:energies_to_save,:energies_to_save,:]

filename_to_save = "vector_potential_comparison_v2_resized"
helical_pol_1 = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_save[:,:,:,0],full_basis,[0,1],[2])
chi_pol_1 = ljc.polaritonic_characteristic_v2_from_eigenvecs(a_x_save[:,:,:,0],full_basis,[0,1],[2])
elec_occ = ljc.state_occupancy_from_eigenvecs(a_x_save[:,:,:,0],np.array([3,4]))
#helical_pol_1 = lp.mask_array_by_other_array_condition(helical_pol_1,chi_pol_1*elec_occ,np.greater,.001)
#helical_pol_1 = ma.filled(helical_pol_1,0)

helical_pol_2 = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_save[:,:,:,1],full_basis,[0,1],[2])
chi_pol_2 = ljc.polaritonic_characteristic_v2_from_eigenvecs(a_x_save[:,:,:,1],full_basis,[0,1],[2])
elec_occ = ljc.state_occupancy_from_eigenvecs(a_x_save[:,:,:,1],np.array([3,4]))
#helical_pol_2 = lp.mask_array_by_other_array_condition(helical_pol_2,chi_pol_2*elec_occ,np.greater,.001)
#helical_pol_2 = ma.filled(helical_pol_2,0)

#old way of separation
polariton_assignments_1 = ljc.eigenvector_state_assignment(a_x_save[:,:,:,0],np.array([3,4]))
separated_helical_pol_1 = ljc.separate_eigenvector_characteristics_by_states(a_x_save[:,:,:,0],helical_pol_1,np.array([3,4]))
separated_helical_pol_1 = ma.filled(separated_helical_pol_1,None)

polariton_assignments_2 = ljc.eigenvector_state_assignment(a_x_save[:,:,:,1],np.array([3,4]))
separated_helical_pol_2 = ljc.separate_eigenvector_characteristics_by_states(a_x_save[:,:,:,1],helical_pol_2,np.array([3,4]))
separated_helical_pol_2 = ma.filled(separated_helical_pol_2,None)

#manual separation
bp_mask = ljc.mask_for_rows(np.shape(e_x_full),np.array([0,2,4]))
up_mask = ljc.mask_for_rows(np.shape(e_x_full),np.array([0,1,3]))
bottom_pol_1 = np.ma.masked_array(np.copy(helical_pol_1),mask = np.logical_or(ma.getmask(helical_pol_1),bp_mask))
upper_pol_1 = np.ma.masked_array(np.copy(helical_pol_1),mask = np.logical_or(ma.getmask(helical_pol_1),up_mask))

bottom_pol_2 = np.ma.masked_array(np.copy(helical_pol_2),mask = np.logical_or(ma.getmask(helical_pol_1),bp_mask))
upper_pol_2 = np.ma.masked_array(np.copy(helical_pol_2),mask = np.logical_or(ma.getmask(helical_pol_1),up_mask))

helical_pol_stack = np.ma.stack((bottom_pol_1,upper_pol_1,bottom_pol_2,upper_pol_2),axis = 2)
helical_pol_stack_filled = np.ma.filled(helical_pol_stack,None)
#helical_pol_stack = np.concatenate((separated_helical_pol_1,separated_helical_pol_2),axis = 2)

np.save("rev_separated_helical_pol_vp_35_70.npy",helical_pol_stack_filled)

np.save("rev_energy_vp_35_70.npy",energy_save)
