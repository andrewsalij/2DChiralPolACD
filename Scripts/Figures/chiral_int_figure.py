import ldlb_jaynes_cummings as ljc
import numpy as np
import dielectric_tensor as dt
import matplotlib.pyplot as plt

dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=8,volume_cell= 2e-7,damping_factor=.09,
                     length=1,gamma_type="linear")
energy_array = np.array([2,3])
energies = energy_array
dip_mags = 1e-3*np.array([1,1])

dip_angles = np.array([0,-np.pi/4])
dipole_matrix = dt.create_dipole_matrix_polar_2D(dip_mags,dip_angles)

cav_freq_array = np.linspace(1,5,100)
spectrum = np.linspace(1, 5, 100)
spectrum = cav_freq_array
size = np.size(cav_freq_array)

energies_to_save = 5
mean_int_len = ljc.mean_interaction_length(cav_freq_array,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_numeric")
length_by_energy = mean_int_len

mean_int_len_pert = ljc.mean_interaction_length(cav_freq_array,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal")

chiral_int_pert, m_plus,m_minus = ljc.coupling_element_cavity_sweep(energy_array,cav_freq_array,dipole_matrix,dielectric_params,spectrum,length_by_energy,style = "pert")
chiral_int_full, m_plus,m_minus = ljc.coupling_element_cavity_sweep(energy_array,cav_freq_array,dipole_matrix,dielectric_params,spectrum,mean_int_len,style = "full")


chiral_semisimple = ljc.chiral_factor_semi_approximate(cav_freq_array,energy_array,dip_mags,dielectric_params,dip_angles)
chiral_semisimple_2 = ljc.chiral_factor_second_semi_approximate(cav_freq_array,energy_array,dip_mags,dielectric_params,dip_angles)

fig, ax = plt.subplots(figsize = (3.3,2.5))

ax.plot(cav_freq_array,chiral_int_full[0,:],label = r"$\mu_1$",color ="red")
ax.plot(cav_freq_array,chiral_semisimple,color = "red",linestyle = "dotted")
ax.plot(cav_freq_array,chiral_int_full[1,:],label = r"$\mu_2$ ",color = 'blue')
ax.plot(cav_freq_array,chiral_semisimple_2,color= "blue",linestyle = "dotted" )

#plt.plot(cav_freq_array,chiral_simple)
ax.set_xlim(1.5,4)
ax.set_ylim(-1,1)
ax.set_xlabel(r" $\Omega$ (eV)",fontsize= 14)
ax.set_ylabel(r"$\sigma_n$",fontsize = 14)
fig.legend(loc = 'upper center',bbox_to_anchor = (.52,.87),fontsize = 10)
fig.tight_layout()
fig.savefig("chiral_int_comparisonv4.pdf")
fig.show()