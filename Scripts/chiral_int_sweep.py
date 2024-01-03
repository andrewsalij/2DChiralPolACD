import ldlb_jaynes_cummings as ljc
import numpy as np
import dielectric_tensor as dt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mueller
import ldlb_plotting as lp
import numpy.ma as ma


dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=8,volume_cell= 5.8075e-7,damping_factor=.1,
                     length=1,gamma_type="linear")
energy_array = np.array([2,3])
energies = energy_array
dip_mags = 1.05474e-3*np.array([1,1]) #10 Debye

dip_angles = np.array([0,np.pi/4])
dipole_matrix = dt.create_dipole_matrix_polar_2D(dip_mags,dip_angles)

cav_freq_array = np.linspace(1.5,2.5,100)
spectrum = np.linspace(1, 5, 100)
spectrum = cav_freq_array
size = np.size(cav_freq_array)

energies_to_save = 5
mean_int_len = ljc.mean_interaction_length(cav_freq_array,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_numeric")
length_by_energy = mean_int_len

chiral_int_pert, m_plus,m_minus = ljc.coupling_element_cavity_sweep(energy_array,cav_freq_array,dipole_matrix,dielectric_params,spectrum,length_by_energy,style = "pert")
chiral_int_full, m_plus,m_minus = ljc.coupling_element_cavity_sweep(energy_array,cav_freq_array,dipole_matrix,dielectric_params,spectrum,length_by_energy,style = "full")


chiral_semisimple = ljc.chiral_factor_semi_approximate(cav_freq_array,energy_array,dip_mags,dielectric_params,length_by_energy)

plt.plot(cav_freq_array,chiral_int_full[0,:],label = "Numeric")
plt.plot(cav_freq_array,chiral_int_pert[0,:],label = "First Analytic")
plt.plot(cav_freq_array,chiral_semisimple,label = "Ultra-Analytic")

#plt.plot(cav_freq_array,chiral_simple)
#plt.ylim(0,1)
plt.xlabel("Cavity Energy (eV)")
plt.ylabel(r"$\sigma$")
plt.title("Equal")
plt.legend()
plt.show()