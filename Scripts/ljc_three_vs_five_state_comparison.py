
import numpy as np
import dielectric_tensor as dt
import ldlb_jaynes_cummings as ljc
import ldlb_plotting as lp
import brioullinzonebasefinal as base
def hamiltonian_three_state_sweep(cav_freq_array,chiral_int_array,coupling,trans_freq):
    e_x_array = np.zeros((3,np.size(cav_freq_array)))
    a_x_array = np.zeros((3,3,np.size(cav_freq_array)))
    for i in range(np.size(cav_freq_array)):
        cav_freq = cav_freq_array[i]
        v_plus = coupling*np.sqrt(1/2+1/2*chiral_int_array[i])
        v_minus = coupling*np.sqrt(1/2-1/2*chiral_int_array[i])
        jc_h = np.array([[cav_freq,0,v_plus],
                         [0,cav_freq,v_minus],
                         [v_plus,v_minus,trans_freq]])
        e_x, a_x = base.solve_hamiltonian(jc_h)
        e_x_array[:,i] = e_x
        a_x_array[:,:,i] = a_x
    return e_x_array,a_x_array

def hamiltonian_five_state_sweep(cav_freq_array,chiral_int_array,valley_coupling,chiral_coupling,trans_freq,valley_freq):
    e_x_array = np.zeros((5,np.size(cav_freq_array)))
    a_x_array = np.zeros((5,5,np.size(cav_freq_array)))
    for i in range(np.size(cav_freq_array)):
        cav_freq = cav_freq_array[i]
        v_plus = chiral_coupling*np.sqrt(1/2+1/2*chiral_int_array[i])
        v_minus = chiral_coupling*np.sqrt(1/2-1/2*chiral_int_array[i])
        w = valley_coupling
        jc_h = np.array([[cav_freq,0,w,0,v_plus],
                         [0,cav_freq,0,w,v_minus],
                         [w,0,valley_freq,0,0],
                         [0,w,0,valley_freq,0],
                         [v_plus,v_minus,0,0,trans_freq]])
        e_x, a_x = base.solve_hamiltonian(jc_h)
        e_x_array[:,i] = e_x
        a_x_array[:,:,i] = a_x
    return e_x_array,a_x_array

dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=8,volume_cell= 5.8075e-7,damping_factor=.1,
                     length=1,gamma_type="linear")
energy_array = np.array([2,3])
energies = energy_array
dip_mags = 1.05474e-3*np.array([.5,1]) #10 Debye

dip_angles = np.array([0,np.pi/4])
dipole_matrix = dt.create_dipole_matrix_polar_2D(dip_mags,dip_angles)

cav_freq_array = np.linspace(1.8,2.2,100)
spectrum = np.linspace(1.8, 2.2, 100)
spectrum = cav_freq_array
size = np.size(cav_freq_array)

energies_to_save = 5
mean_int_len = ljc.mean_interaction_length(cav_freq_array,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_numeric")
length_by_energy = mean_int_len

chiral_semisimple = ljc.chiral_factor_semi_approximate(cav_freq_array,energy_array,dip_mags,dielectric_params,length_by_energy)

vec_pot = 80/np.sqrt(2)
acd_coupling = vec_pot*dip_mags[0]*energy_array[0]
valley_coupling = .5*acd_coupling
trans_freq = energy_array[0]
valley_freq = energy_array[0]
e_x_three_state, a_x_three_state= hamiltonian_three_state_sweep(cav_freq_array,chiral_semisimple,acd_coupling,trans_freq)

e_x_five_state,a_x_five_state= hamiltonian_five_state_sweep(cav_freq_array,chiral_semisimple,valley_coupling,acd_coupling,trans_freq,valley_freq)

three_state_basis =np.array([[1,0,-1],
                    [0,1,-1],
                    [0,0,0]])

five_state_basis = np.array([[1,0,-1,-1,-1],
                    [0,1,-1,-1,-1],
                    [0,0,0,-1,-1],
                    [0,0,-1,0,-1],
                    [0,0,-1,-1,0]])



helicity_three_state =ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_three_state,three_state_basis,[0,1],[2],sign_convention="reversed")

helicity_five_state = ljc.helical_polaritonic_characteristic_v2_from_eigenvecs(a_x_five_state,five_state_basis,[0,1],[2,3,4],sign_convention="reversed")

import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
#removing middle (photonic) states from plotting
fontsize = 24
subplot_adjustments = {"left":.15,"right":.85,"bottom":.15}
e_x_three_state_masked = np.vstack((e_x_three_state[0,:],e_x_three_state[2,:]))
helicity_three_state_masked = np.vstack((helicity_three_state[0,:],helicity_three_state[2,:]))
lp.plot_set_colored("three_state_hel.png", cav_freq_array,e_x_three_state_masked,
                    helicity_three_state_masked, x_label=r"Cavity Energy (eV)", y_label=r"Eigenstate Energy (eV)",
                    opacity=1, norm_max=.35, norm_min=-.35, cmap="bwr",
                    colorbar_label=r"Polaritonic Helicity ($\tilde{g}$)",
                    **{"show_colorbar":True,"show_min_max": False,"y_bounds":[1.9,2.1],"x_bounds":[1.8,2.2],"linewidth":10,
                       "label_fontsize": fontsize,"subplot_adjustments":subplot_adjustments,
                       "x_ticks":[1.80,1.90,2.0,2.10,2.20],"y_ticks":[1.9,1.95,2.0,2.05,2.1]})

e_x_five_state_masked = np.concatenate((e_x_five_state[:2,:],e_x_five_state[3:]),axis=0)
helicity_five_state_masked = np.concatenate((helicity_five_state[:2,:],helicity_five_state[3:]),axis=0)
lp.plot_set_colored("five_state_hel.png", cav_freq_array,e_x_five_state_masked,
                    helicity_five_state_masked,x_label=r"Cavity Energy (eV)", y_label=r"Eigenstate Energy (eV)",
                    opacity=1, norm_max=.35, norm_min=-.35, cmap="bwr",
                    colorbar_label=r"Polaritonic Helicity ($\tilde{g}$)",
                    **{"show_min_max": False,"y_bounds":[1.9,2.1],"x_bounds":[1.8,2.2],"linewidth":10,
                       "label_fontsize":fontsize,"subplot_adjustments":subplot_adjustments,
                       "x_ticks":[1.80,1.90,2.0,2.10,2.20],"y_ticks":[1.9,1.95,2.0,2.05,2.1]})