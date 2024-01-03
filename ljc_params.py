import dielectric_tensor as dt
import numpy as np
import ldlb_jaynes_cummings as ljc

def two_dipole_init(e_1,e_2,vec_pot):
    dielectric_params = dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const=1, volume_cell=1e-8, damping_factor=.09,
                                             length=1, gamma_type="linear")
    energies = np.array([e_1, e_2])
    dip_mags = 1e-3 * np.array([1, 1])
    spectrum = np.linspace(1, 4, 1000)
    cav_freq_array = np.linspace(1, 4, 1000)
    vec_pot = vec_pot / np.sqrt(2)
    dip_angles = np.array([0, np.pi / 4])
    dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags, dip_angles)
    return dielectric_params,energies, dip_mags, spectrum, cav_freq_array, vec_pot, dip_angles, dipole_matrix

def two_dipole_simple(e_1,e_2,vec_pot,polarization = np.array([1,1]),omit_zero_state = False):
    dielectric_params, energies, dip_mags, spectrum, cav_freq_array, vec_pot, dip_angles, dipole_matrix = two_dipole_init(e_1,e_2,vec_pot)
    e_x_full, a_x_full,_ = ljc.jaynes_cummings_organic_ldlb_sweep(1, energies, cav_freq_array, vec_pot, polarization,
                                                                dipole_matrix, dielectric_params, spectrum,dielectric_params.length,omit_zero_state)
    return e_x_full ,a_x_full,cav_freq_array,spectrum

def chiral_couplings_two_dipoles(e_1,e_2,vec_pot,polarization = np.array([1,1])):
    dielectric_params, energies, dip_mags, spectrum, cav_freq_array, vec_pot, dip_angles, dipole_matrix = two_dipole_init(e_1,e_2,vec_pot)
    couplings = ljc.get_chiral_couplings_two_dipoles(cav_freq_array,dipole_matrix,dielectric_params,spectrum,energies,dielectric_params.length,vec_pot,
                                                     polarization)
    return couplings
