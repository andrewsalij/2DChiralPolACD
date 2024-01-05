import ldlb_jaynes_cummings as ljc
import numpy as np
import dielectric_tensor as dt
import matplotlib.pyplot as plt


suffix = "2pt1eV_num"
dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=8,volume_cell= 5.8075e-7,damping_factor=.1,
                     length=1,gamma_type="linear")

energy_array = np.array([2,3])
energies = energy_array
dip_mags = 1.05474e-3*np.array([1,1]) #10 Debye

dip_angles = np.array([0,np.pi/4])
dipole_matrix = dt.create_dipole_matrix_polar_2D(dip_mags,dip_angles)

cav_freq_array = np.linspace(1,5,100)
spectrum = np.linspace(1, 5, 100)
spectrum = cav_freq_array
size = np.size(cav_freq_array)

loss = 0

energies_to_save = 5
mean_int_len = ljc.mean_interaction_length(cav_freq_array,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_numeric",loss = loss)


mean_int_len_pert = ljc.mean_interaction_length(cav_freq_array,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal")

pert_len = False
if (pert_len):
    mean_int_len = mean_int_len_pert

chiral_int_pert, m_plus,m_minus = ljc.coupling_element_cavity_sweep(energy_array,cav_freq_array,dipole_matrix,dielectric_params,spectrum,mean_int_len,style = "pert")

#energy sweep
chiral_int_full, m_plus,m_minus = ljc.coupling_element_cavity_sweep(energy_array,cav_freq_array,dipole_matrix,dielectric_params,spectrum,mean_int_len,style = "full",brown_style = "pert")

chiral_semisimple = ljc.chiral_factor_semi_approximate(cav_freq_array,energy_array,dip_mags,dielectric_params,dip_angles)
chiral_semisimple_2 = ljc.chiral_factor_second_semi_approximate(cav_freq_array,energy_array,dip_mags,dielectric_params,dip_angles)

energy_to_save = np.vstack((cav_freq_array,chiral_int_full[0,:],chiral_int_full[1,:],chiral_semisimple,chiral_semisimple_2))
np.save("chiral_int_e_scan"+suffix+".npy",energy_to_save)

#gamma sweep \omega = 2
e_cav = 2.1
damping_array = np.linspace(.03,.15,100)
chiral_int_g_full, m_plus,m_minus = ljc.coupling_element_gamma_sweep(energy_array,e_cav,dipole_matrix,dielectric_params,damping_array,spectrum,style = "full",brown_style = "pert")

chiral_gamma_approx = ljc.chiral_factor_approximate_gamma_sweep(e_cav,energy_array,dip_mags,dip_angles,damping_array,dielectric_params)

gamma_to_save = np.vstack((damping_array,chiral_int_g_full[0,:],chiral_int_g_full[1,:],chiral_gamma_approx[0,:],chiral_gamma_approx[1,:]))

np.save("chiral_int_gamma_scan"+suffix+".npy",gamma_to_save)

plt.plot(damping_array,chiral_int_g_full[0,:])
plt.plot(damping_array,chiral_gamma_approx[0,:])
plt.show()

ratio_array = np.linspace(.3,3,100)
dielectric_params.gamma = 0.10
chiral_int_r_full, m_plus,m_minus = ljc.coupling_element_dip_ratio_sweep(energy_array,e_cav,dip_mags[1],dip_angles,dielectric_params,ratio_array,spectrum,style = "full",brown_style = "pert",order = "21")

chiral_r_approx = ljc.chiral_factor_approximate_dipole_sweep(e_cav,energy_array,dip_mags[1],ratio_array,dip_angles,dielectric_params,order = "21")

ratio_to_save = np.vstack((ratio_array,chiral_int_r_full[0,:],chiral_int_r_full[1,:],chiral_r_approx[0,:],chiral_r_approx[1,:]))
np.save("chiral_int_ratio_scan"+suffix+".npy",ratio_to_save)

w2_array = np.linspace(2,4,100)
chiral_int_w_full, m_plus,m_minus = ljc.coupling_element_w2_sweep(energy_array,e_cav,dip_mags,dip_angles,dielectric_params,w2_array,spectrum,style = "full",brown_style = "pert")

chiral_w_approx = ljc.chiral_factor_approximate_w2_sweep(e_cav,energy_array,dip_mags,w2_array,dip_angles,dielectric_params)

ratio_to_save = np.vstack((w2_array,chiral_int_w_full[0,:],chiral_int_w_full[1,:],chiral_w_approx[0,:],chiral_w_approx[1,:]))
np.save("chiral_int_w2_scan"+suffix+".npy",ratio_to_save)



