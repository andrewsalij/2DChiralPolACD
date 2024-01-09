import numpy as np
import dielectric_tensor as dt
import jaynes_cummings as jc
import combinatorics as combo
import brioullinzonebasefinal as base
import copy
import ldlb_post_processing as lpp
import python_util as pu
import time
import mueller

def sum_cross_interaction(energy_array,dipole_mags_array,dipole_angles_array,base_angle,spectrum,gamma_array):
    '''this is the \sum_m s_m w_m sin(2\beta_nm) term in writing'''
    s_m = energy_array*dipole_mags_array**2
    s_m = np.tile(s_m,(np.size(spectrum),1)).T
    w_m = dt.f_dielectric_real(energy_array,spectrum,gamma_array)
    sin_beta = np.sin(2*(base_angle-dipole_angles_array))
    sin_beta = np.tile(sin_beta,(np.size(spectrum),1)).T
    return np.sum(s_m*w_m*sin_beta,axis= 0)
def cd_analytic(dielectric_params, energy_array,dipole_mags_array,dipole_angles_array,spectrum,gamma_array,length =1,set_energy = None,unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297))):
    '''cd analytic expression in the perturbative limit'''
    dipole_matrix = dt.create_dipole_matrix_polar_2D(dipole_mags_array,dipole_angles_array)
    total_cross_int = np.zeros((np.size(dipole_mags_array),np.size(spectrum)))
    for i in range(0,np.size(dipole_mags_array)):
        cross_int = sum_cross_interaction(energy_array,dipole_mags_array,dipole_angles_array,dipole_angles_array[i],spectrum,gamma_array)
        total_cross_int[i,:] = cross_int
    xi = 1/(unit_defs.hbar*unit_defs.c*unit_defs.e0*np.sqrt(dielectric_params.epsilon_inf)*dielectric_params.v)
    s_n = energy_array * dipole_mags_array ** 2
    if (set_energy is not None):
        brown_params = brown_params_from_raw_set_energy(dielectric_params,dipole_matrix,energy_array,spectrum,set_energy,length)
        a_1 = brown_params[0][1]#here, a_1 is a function of length, if anything
        v_n = dt.f_dielectric_im(energy_array,spectrum,gamma_array)
        energy_index = np.argmin(np.abs(spectrum-set_energy))
        v_n = v_n[:,energy_index]
        total_cross_int = total_cross_int[:,energy_index]
        second_dipole_contrib = v_n * s_n
        spec_to_use = set_energy
    elif (np.isscalar(length)):
        brown_params= brown_params_from_raw(dielectric_params,dipole_matrix,energy_array,spectrum,length)
        a_1 = brown_params[0][1] #here, a_1 is a function of spectral frequency
        v_n = dt.f_dielectric_im(energy_array,spectrum,gamma_array)
        s_n = energy_array * dipole_mags_array ** 2
        s_n =  np.tile(s_n,(np.size(spectrum),1)).T
        second_dipole_contrib = v_n*s_n
        spec_to_use = spectrum
    else:
        ValueError("Either length must be a scalar or a single energy is probed")
    length_prefactor = a_1 / length
    total_prefactor = length_prefactor * xi ** 2 * spec_to_use** 2
    abs_cd_anal = total_prefactor*np.sum(total_cross_int*second_dipole_contrib,axis = 0)
    return abs_cd_anal

def sum_cross_interaction_cos(energy_array,dipole_mags_array,dipole_angles_array,base_angle,spectrum,gamma_array):
    '''Returns dipole-dipole (cosine) interaction tensor of set fo dipoles'''
    s_m = energy_array*dipole_mags_array**2
    s_m = np.tile(s_m,(np.size(spectrum),1)).T
    v_m = dt.f_dielectric_im(energy_array,spectrum,gamma_array)
    cos_beta = np.cos(2*(dipole_angles_array-base_angle))
    cos_beta = np.tile(cos_beta,(np.size(spectrum),1)).T
    return np.sum(s_m*v_m*cos_beta,axis= 0)
class POLARIZANCE(mueller.POLARIZANCE):
    '''
    Class for creating and manipulating polarizance parameters (dichroism/diattenuation and birefringence)
    assumes that 0 axis is the linear, linear_prime, circular axes
     1 axis is some collection, generally over some spectrum.
     See SI to https://arxiv.org/abs/2208.14461 or https://doi.org/10.1117/12.366361 (Brown 1999)
     '''
    def __init__(self,birefringence,diattenuation,polarizance,isotropic_dichroic_loss):
        mueller.POLARIZANCE.__init__(self,birefringence,diattenuation,polarizance,isotropic_dichroic_loss)
def brown_params(r_p,i_p,n_p,length = 1):
    return mueller.brown_params(r_p,i_p,n_p,length = length)

def create_mueller_elems(polarizance_object,length = 1):
    p_m_array, r_p_array, i_p_array, n_p_array = polarizance_object.decompose_polarizance()
    brown_params_list = brown_params(r_p_array,i_p_array,n_p_array,length= length)
    b = polarizance_object.b_matrix
    d = polarizance_object.d_matrix
    a0,a1,a2,a3= brown_params_list[0],brown_params_list[1],brown_params_list[2],brown_params_list[3]
    m00 = a0+a1*(np.sum(b**2,axis = 0))
    m03  = a1*(b[1,:]*d[0,:]-b[0,:]*d[1,:])-a2*d[2,:]+a3*b[2,:]
    return m00, m03



def get_polarizance_params_dielectric(dielectric_params,dipole_matrix,energy_array,spectrum,style = "default"):
    '''
    Provides the first 2 parameters in Brown and Bak, 1999 (i.e. A_0 and A_1) 
    Only calculates dielectric fucntion, with assumptions that include that microscopic
    circular dichroism and circular birefringence are negligible
    That is , beta_3 = d_3 = 0, to isolate linear contributions
    :param dielectric_params:
    :param dipole_matrix:
    :param energy_array:
    :param spectrum:
    :param style:
    Order to expand refractive index in solving for Brown polarizance parameters.
    For most cases studied, these should give roughly equivalent results
    "pert": second-order analytic expansion--use this for direct invocation of equations in paper
    "default": numerical creation of dielectric tensor--use this for greater accuracy
    :return:
    '''
    dp_copy = copy.deepcopy(dielectric_params)
    dp_copy.length = 1 #ensuring no length dependence for differential elements
    dielectric_tensor = dt.create_dielectric_tensor(dp_copy,dipole_matrix,energy_array,spectrum)
    if (style == "pert"):
        linear_optics_params = dt.get_linear_optics_pert(dp_copy,dipole_matrix,energy_array,spectrum)
    else:
        linear_optics_params = dt.linear_optics_from_dielectric_tensor(dielectric_tensor,spectrum)
    return polarizance_from_linear_optics(linear_optics_params)

def get_polarizance_params_set_energy(dielectric_params,dipole_matrix,energy_array,spectrum,set_energy,style = "default"):
    ''' get_polarizance_params_dielectric() but for a single energy'''
    dp_copy = copy.deepcopy(dielectric_params)
    dp_copy.length = 1  # ensuring no length dependence for differential elements
    dielectric_tensor = dt.create_dielectric_tensor(dp_copy, dipole_matrix, energy_array, spectrum)
    if (style == "pert"):
        linear_optics_params = dt.get_linear_optics_pert(dp_copy, dipole_matrix, energy_array, spectrum)
    else:
        linear_optics_params = dt.linear_optics_from_dielectric_tensor(dielectric_tensor, spectrum)
    energy_index = np.argmin(np.abs(spectrum-set_energy))
    linear_optics_params_selected = linear_optics_params.select_by_index(energy_index)
    return polarizance_from_linear_optics(linear_optics_params_selected)


def polarizance_from_linear_optics(linear_optics_params):
    '''Returns POLARIZANCE for a system without CB or CD using dt.LINEAR_OPTICS'''
    return mueller.polarizance_from_linear_optics(linear_optics_params)

def brown_params_from_raw(dielectric_params,dipole_matrix,energy_array,spectrum,length = 1,style = "default"):
    '''Returns Brown params from dielectric paramaterization'''
    polarizance_object = get_polarizance_params_dielectric(dielectric_params,dipole_matrix,energy_array,spectrum,style = style)
    return brown_params_from_polarizance(polarizance_object,length=length),polarizance_object

def brown_params_from_polarizance(polarizance_object,length =1):
    '''Returns Brown params from a POLARIZANCE object of some length (default 1).'''
    return mueller.brown_params_from_polarizance(polarizance_object,length = length)

def brown_params_from_raw_set_energy(dielectric_params,dipole_matrix,energy_array,spectrum,set_energy,length = 1,style = "default"):
    '''Returns Brown params from dielectric paramaterization for a specific energy'''
    polarizance_object = get_polarizance_params_set_energy(dielectric_params, dipole_matrix, energy_array, spectrum,set_energy,style = style)
    return brown_params_from_polarizance(polarizance_object, length=length), polarizance_object

def brown_two_fast(polarizance_selected,length):
    '''Calculates the first two Brown parameters (A_0, A_1) a bit faster than the more general method'''
    p_m, r_p, i_p, n_p = polarizance_selected.decompose_polarizance()
    a_0 = (r_p / n_p) ** 2 * np.cosh(i_p * length) + (i_p / n_p) ** 2 * np.cos(r_p * length)
    a_1 = (np.cosh(i_p * length) - np.cos(r_p * length)) / (n_p ** 2)
    return a_0,a_1


def transition_dipole_ldlb_operators(dielectric_params,energy_array,dipole_mags_array, dipole_angle_array,spectrum,cd_characteristic,style = "heuristic",film_thickness = 1):
    '''Deprecated. Used to create opeators representing "chiral" transition dipoles due to LDLB'''
    Warning("The creation of LDLB operators in this manner is deprecated.")
    zeta_factor = dt.ldlb_prefactor_2(dielectric_params.epsilon_inf,dielectric_params.v,film_thickness)
    operator_factors = np.zeros((np.size(energy_array),np.size(spectrum)))
    dipole_array = dt.create_dipole_matrix_polar_2D(dipole_mags_array,dipole_angle_array)
    dipole_operators = np.zeros((np.size(dipole_array,0),np.size(spectrum)))
    gamma_array = dielectric_params.damping_array(energy_array)
    for n in range(0,np.size(energy_array)):
        angle_n = dipole_angle_array[n]
        cross_interaction = sum_cross_interaction(energy_array,dipole_mags_array,dipole_angle_array,angle_n,spectrum,
                                                  gamma_array=gamma_array)
        cos_cross_interaction = sum_cross_interaction_cos(energy_array,dipole_mags_array,dipole_angle_array,angle_n,spectrum,
                                                  gamma_array=gamma_array)
        brown_params_list, polarizance_data = brown_params_from_raw(dielectric_params, dipole_array, energy_array,
                                                                    spectrum, length=film_thickness)
        dichroic_loss = polarizance_data.absorbance*film_thickness
        edz = np.exp(-dichroic_loss)
        #full treatment of m_00 and m_03
        #treats total absorbance matrix element perturbatively (i.e, as unity after e^-A has
        #been accounted for
        if(style == "heuristic"):
            operator_factors[n,:] = np.sqrt(1-1/2*cd_characteristic*zeta_factor*spectrum*cross_interaction)
        if (style == "ldlb_full"):
            operator_factors[n, :] =np.sqrt(1-1/2*zeta_factor*spectrum*(cos_cross_interaction+cd_characteristic*cross_interaction))
        #defaults to double perturbative approximation in absence of direction or
        else:
            operator_factors[n,:] =  np.sqrt(1-1/2*zeta_factor*spectrum*(cd_characteristic*cross_interaction))
        dipole_operators[n,:] = dipole_mags_array[n]*operator_factors[n,:]
    return dipole_operators

def abs_from_operator(dielectric_params,energy_array,spectrum,dipole_operators,length=1,separate= False):
    '''Gives absorption from a set of dipole operators'''
    dip_op_dim = dipole_operators.ndim
    if (np.size(dipole_operators,axis = dip_op_dim-1) != np.size(spectrum)):
        dipole_operators = np.tile(dipole_operators, (np.size(spectrum), 1)).T
    v_n = dt.f_dielectric_im(energy_array,spectrum,damping_factor=dielectric_params.damping_array(energy_array))
    zeta_factor = dt.ldlb_prefactor_2(dielectric_params.epsilon_inf,dielectric_params.v,length)
    if (separate):
        abs = spectrum * zeta_factor * v_n * dipole_operators ** 2 * np.tile(energy_array, (np.size(spectrum), 1)).T
    else:
        abs = spectrum*zeta_factor*np.sum(v_n*dipole_operators**2*np.tile(energy_array,(np.size(spectrum),1)).T,axis = 0)
    return abs

def dirac_selector(spectrum,energy_array,weighting = None):
    '''Selects a idices in spectrum corresponding to a set of energies. If weighting is provided, multiplies by given weight.'''
    selected = np.zeros(np.size(spectrum))
    for i in range(0,np.size(energy_array)):
        idx = np.argmin(np.abs(spectrum-energy_array[i]))
        if (weighting is None):
            selected[idx] = spectrum[idx]
        else:
            selected[idx] = spectrum[idx]*weighting[i]
    return selected

def dirac_selector_separated(spectrum,energy_array,weighting = None):
    '''dirac_selector() with output over 2 dimensions'''
    selected = np.zeros((np.size(energy_array),np.size(spectrum)))
    for i in range(0, np.size(energy_array)):
        idx = np.argmin(np.abs(spectrum - energy_array[i]))
        if (weighting is None):
            selected[i,idx] = spectrum[idx]
        else:
            selected[i,idx] = spectrum[idx] * weighting[i]
    return selected

def vec_pot_dipole_selected(vec_pot_array,dipole_matrix,e_array,spectrum):
    a0_mu = np.einsum("ij,j->i",dipole_matrix,vec_pot_array)
    return dirac_selector(spectrum,a0_mu*e_array)

def interaction_hamiltonian(dipole_operators,energy_array,spectrum,vector_potential,unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297))):
    c = unit_defs.c
    to_sum = np.tile(energy_array,(np.size(spectrum),1)).T*dipole_operators
    return -np.sum(to_sum*1j/c,axis= 0)*vector_potential

def interaction_hamiltonian_element(dipole_operator,energy,vector_potential,unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297))):
    c = unit_defs.c
    return (dipole_operator*vector_potential*energy*1j/c)


#perturbative treatment
#vec_pot is scalar value for nonpolarized light
def interaction_hamiltonian_v2(dipole_matrix,energy_array,spectrum,vector_potential,dielectric_params,length,unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297))):
    dip_mags, dip_angles = dt.dipole_matrix_to_params(dipole_matrix)
    v_n = dt.f_dielectric_im(resonant_energy=energy_array,spectrum=spectrum,damping_factor=dielectric_params.damping_array(energy_array))
    w_m = dt.f_dielectric_real(resonant_energy=energy_array,spectrum=spectrum,damping_factor=dielectric_params.damping_array(energy_array))
    #normal_interaction = np.einsum("ij,j->i",dipole_matrix,vector_potential)
    normal_interaction = -1j*dip_mags*vector_potential
    #normal_interaction_spec = np.outer(normal_interaction,spectrum)
    #normal_interaction_spec = np.tile(normal_interaction,(np.size(spectrum),1)).T*v_n
    normal_interaction_spec = dirac_selector_separated(spectrum,energy_array,normal_interaction)
    brown_params = brown_params_from_raw(dielectric_params,dipole_matrix,energy_array,spectrum,length)
    a_1 = brown_params[0][1]
    xi = 1/(unit_defs.hbar*unit_defs.c*unit_defs.e0*np.sqrt(dielectric_params.epsilon_inf)*dielectric_params.v)
    chiral_pref = a_1*spectrum*xi
    s_m = dip_mags**2*energy_array
    chiral_contribution = np.outer(s_m,chiral_pref)*w_m
    angles_tiled = np.tile(dip_angles,(np.size(dip_angles),1))
    #index of tiled array is m,n with transpose changing with m and normal changing with n
    beta_nm = np.nan_to_num(np.sin(2*(angles_tiled-angles_tiled.T)))
    sum_chiral_contribution = np.einsum("ij,il->lj",chiral_contribution,beta_nm)
    return normal_interaction_spec,sum_chiral_contribution

def h_int_v2_show(normal_interaction_spec,sum_chiral_contribution):
    h_n = np.sum(normal_interaction_spec,axis = 0)
    h_l = np.sum(normal_interaction_spec*np.sqrt(1+sum_chiral_contribution),axis =0)
    h_r =  np.sum(normal_interaction_spec*np.sqrt(1-sum_chiral_contribution),axis =0)
    return h_n, h_l, h_r


def get_chiral_interaction_constants(dipole_matrix,dielectric_params,spectrum,energy_array,length, cavity_freq,unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297)),style= "default"):
    w_m = dt.f_dielectric_real(resonant_energy=energy_array, spectrum=spectrum,
                               damping_factor=dielectric_params.damping_array(energy_array))
    v_n = dt.f_dielectric_im(resonant_energy=energy_array, spectrum=spectrum,
                               damping_factor=dielectric_params.damping_array(energy_array))
    #for i in range(0,np.size(energy_array)):
    #    plt.plot(spectrum,w_m[i,:])
    #plt.show()
    cavity_index = np.argmin(np.abs(spectrum - cavity_freq))
    w_m_at_omega_cav = w_m[:, cavity_index]
    v_n_at_omega_cav = v_n[:,cavity_index]
    if (type(length) is np.ndarray):
        if (np.size(length) == 1):
            length = length.item()
        else:
            brown_params = brown_params_from_raw(dielectric_params, dipole_matrix, energy_array, spectrum, length,style = style)
            a_1 = brown_params[0][1]
            a_1_at_omega_cav = a_1[cavity_index]
    if (np.isscalar(length)):
        brown_params = brown_params_from_raw(dielectric_params, dipole_matrix, energy_array, spectrum, length,style = style)
        a_1 = brown_params[0][1]
        a_1_at_omega_cav = a_1[cavity_index]
    else:
        ValueError("Length type error (not scalar or array)")

    xi = 1 / (unit_defs.hbar * unit_defs.c * unit_defs.e0 * np.sqrt(
        dielectric_params.epsilon_inf) * dielectric_params.v)
    return a_1_at_omega_cav,xi,w_m_at_omega_cav,v_n_at_omega_cav


def get_chiral_interaction_constants_set_energy(dipole_matrix,dielectric_params,spectrum,energy_array,length, cavity_freq,unit_defs = dt.UNIT_DEFINITIONS(1,1,1/(4*np.pi*0.007297)),style = "default"):
    '''
    cavity_freq is the set energy
    :param dipole_matrix:
    :param dielectric_params:
    :param spectrum:
    :param energy_array:
    :param length:
    :param cavity_freq:
    :param unit_defs:
    :param style:
    :return:
    '''
    w_m = dt.f_dielectric_real(resonant_energy=energy_array, spectrum=spectrum,
                               damping_factor=dielectric_params.damping_array(energy_array))
    v_n = dt.f_dielectric_im(resonant_energy=energy_array, spectrum=spectrum,
                               damping_factor=dielectric_params.damping_array(energy_array))
    #for i in range(0,np.size(energy_array)):
    #    plt.plot(spectrum,w_m[i,:])
    #plt.show()
    cavity_index = np.argmin(np.abs(spectrum - cavity_freq))
    w_m_at_omega_cav = w_m[:, cavity_index]
    v_n_at_omega_cav = v_n[:,cavity_index]
    if (type(length) is np.ndarray):
        if (np.size(length) == 1):
            length = length.item()
        else:
            brown_params = brown_params_from_raw_set_energy(dielectric_params, dipole_matrix, energy_array, spectrum, cavity_freq,length,style = style)
            a_1_at_omega_cav = brown_params[0][1]
    if (np.isscalar(length)):
        brown_params = brown_params_from_raw_set_energy(dielectric_params, dipole_matrix, energy_array, spectrum, cavity_freq,length,style = style )
        a_1_at_omega_cav = brown_params[0][1]
    else:
        ValueError("Length type error (not scalar or array)")

    xi = 1 / (unit_defs.hbar * unit_defs.c * unit_defs.e0 * np.sqrt(
        dielectric_params.epsilon_inf) * dielectric_params.v)
    return a_1_at_omega_cav,xi,w_m_at_omega_cav,v_n_at_omega_cav

def create_index_mask(array_shape,index):
    mask = np.zeros(array_shape)
    mask[index] = 1
    return mask

def sum_different_indices(array):
    shape = np.shape(array)
    different_index_sum_array = np.zeros(np.shape(array))
    for i in range(0,np.size(array)):
        index_mask = create_index_mask(shape,i)
        masked_array = np.ma.masked_array(array,index_mask)
        different_index_sum_array[i] = np.sum(masked_array)
    return different_index_sum_array

#this is referred to as sigma in the notes--it is the \frac{A_1}{z}\omega\xi |\mu_m|^2\omega_m W_m \sin(2\beta_{nm}) term
#provides chiral factor over n index where n is the electronic eigenstates
def chiral_factor(a_1,length,cavity_freq,energy_array,dip_mags,dip_angles,w_m_at_cavity_freq,xi,style = "full"):
    if (style == "full"):
        prefactor = cavity_freq*xi*a_1/length
    else:
        prefactor = cavity_freq*xi*1/2*length
    dipole_contrib = dip_mags**2*energy_array*w_m_at_cavity_freq
    dipole_contrib = np.tile(dipole_contrib,(np.size(dipole_contrib),1))
    angles_tiled = np.tile(dip_angles, (np.size(dip_angles), 1))
    beta_nm = np.nan_to_num(np.sin(2 * (angles_tiled.T - angles_tiled)))
    chiral_factor_nm = prefactor*dipole_contrib*beta_nm
    np.fill_diagonal(chiral_factor_nm,0)
    return -np.sum(chiral_factor_nm,axis=  1)

#all perturbations included in full anal
#a_1 numerically determined in z_anal
def chiral_factor_pert(a_1,xi,cavity_freq,energy_array,dip_mags,dip_angles,w_m_at_cavity_freq,v_n_at_cavity_freq,length,style = "full_anal"):
    dipole_contrib_m = dip_mags ** 2 * energy_array * w_m_at_cavity_freq
    dipole_contrib_n = dip_mags ** 2 * energy_array * v_n_at_cavity_freq
    dipole_contrib_m = np.tile(dipole_contrib_m, (np.size(dipole_contrib_m), 1))
    dipole_contrib_n = np.tile(dipole_contrib_n, (np.size(dipole_contrib_n), 1)).T
    angles_tiled = np.tile(dip_angles, (np.size(dip_angles), 1))
    beta_nm = np.nan_to_num(np.sin(2 * (angles_tiled.T - angles_tiled)))
    if (style == "full_anal"):
        prefactor = 1/2
        chiral_factor_nm = prefactor * dipole_contrib_m / dipole_contrib_n * beta_nm
    elif (style == "z_anal"):
        prefactor = a_1/length*cavity_freq**2*xi*1
        chiral_factor_nm = prefactor*dipole_contrib_m/dipole_contrib_n*beta_nm
    np.fill_diagonal(chiral_factor_nm, 0)
    return -np.sum(chiral_factor_nm, axis=0)

#case for two dipoles
#this approximation is basically useless, use semi-approximate instead
def chiral_factor_ultra_approximate(energy_1,energy_2,damping_2,spectrum):
    return -1/(2)*((energy_2**2-spectrum**2))/(damping_2*spectrum)

def get_two_dipole_params(spectrum,energy_array,dielectric_params):
    v_n = dt.f_dielectric_im(energy_array, spectrum, dielectric_params.damping_array(energy_array))
    w_m = dt.f_dielectric_real(energy_array, spectrum, dielectric_params.damping_array(energy_array))
    w1, w2 = energy_array[0], energy_array[1]
    return v_n,w_m,w1,w2

def chiral_factor_semi_approximate(spectrum,energy_array,dip_mags,dielectric_params,dip_angles):
    v_n, w_m, w1, w2 = get_two_dipole_params(spectrum,energy_array,dielectric_params)
    return -np.sin(2*(dip_angles[0]-dip_angles[1]))/(2)*w2*dip_mags[1]**2*w_m[1,:]/(w1*v_n[0,:]*dip_mags[0]**2+w2*v_n[1,:]*dip_mags[1]**2)

def chiral_factor_second_semi_approximate(spectrum,energy_array,dip_mags,dielectric_params,dip_angles):
    v_n, w_m, w1, w2 = get_two_dipole_params(spectrum,energy_array,dielectric_params)
    return -np.sin(2*(dip_angles[1]-dip_angles[0]))/(2)*w1*dip_mags[0]**2*w_m[0,:]/(w1*v_n[0,:]*dip_mags[0]**2+w2*v_n[1,:]*dip_mags[1]**2)

def get_gamma_sweep(dielectric_params,energy_array,gamma_array):
    n_energy = np.size(energy_array)
    n_gam = np.size(gamma_array)
    gamma_set = np.zeros((n_energy,n_gam))
    for i in range(0,n_gam):
        dielectric_params.gamma = gamma_array[i]
        new_gam = dielectric_params.damping_array(energy_array)
        gamma_set[:,i] = new_gam
    return gamma_set

#all inputs tiled equally or scalar
def w_v_tiled(energy_set,cav_energy,gamma_tiled):
    w = (energy_set**2-cav_energy**2)/((energy_set**2-cav_energy**2)**2+gamma_tiled**2*cav_energy**2)
    v = (gamma_tiled*cav_energy)/((energy_set**2-cav_energy**2)**2+gamma_tiled**2*cav_energy**2)
    return w, v

def chiral_factor_approximate_gamma_sweep(cav_freq,energy_array,dip_mags,dip_angles,gamma_array,dielectric_params):
    sigma_set = np.zeros((np.size(energy_array),np.size(gamma_array)))
    spec_size = np.size(gamma_array)
    gamma_tiled = get_gamma_sweep(dielectric_params,energy_array,gamma_array)
    dips_tiled = np.tile(dip_mags,(spec_size,1)).T
    angles_tiled = np.tile(dip_angles,(spec_size,1)).T
    energies_tiled = np.tile(energy_array,(spec_size,1)).T
    w_n, v_n = w_v_tiled(energies_tiled,cav_freq,gamma_tiled)
    abs = np.sum(dips_tiled**2*energies_tiled*v_n,axis = 0)
    for i in range (0,np.size(energy_array)):
        numerator = np.sum(dips_tiled**2*energies_tiled*w_n*np.sin(2*(dip_angles[i]-angles_tiled)),axis = 0)
        sigma_set[i,:] = 1/2*numerator/abs
    return sigma_set


def chiral_factor_approximate_dipole_sweep(cav_freq,energy_array,dip_base_mag,dip_ratio_array,dip_angles,dielectric_params,order = "default"):
    sigma_set = np.zeros((np.size(energy_array),np.size(dip_ratio_array)))
    spec_size = np.size(dip_ratio_array)
    gamma_tiled = np.tile(dielectric_params.damping_array(energy_array),(spec_size,1)).T
    if (order == "21"):
        dip_mag_1 = dip_base_mag
        dips_tiled = np.vstack((np.ones(spec_size) * dip_mag_1,dip_mag_1*dip_ratio_array))
    else:
        dip_mag_2 = dip_base_mag
        dips_tiled = np.vstack((dip_ratio_array*dip_mag_2,np.ones(spec_size)*dip_mag_2))
    angles_tiled = np.tile(dip_angles,(spec_size,1)).T
    energies_tiled = np.tile(energy_array,(spec_size,1)).T
    w_n, v_n = w_v_tiled(energies_tiled,cav_freq,gamma_tiled)
    abs = np.sum(dips_tiled**2*energies_tiled*v_n,axis = 0)
    for i in range (0,np.size(energy_array)):
        numerator = np.sum(dips_tiled**2*energies_tiled*w_n*np.sin(2*(dip_angles[i]-angles_tiled)),axis = 0)
        sigma_set[i,:] = 1/2*numerator/abs
    return sigma_set

def chiral_factor_approximate_w2_sweep(cav_freq,energy_array,dip_mags,w2_array,dip_angles,dielectric_params):
    sigma_set = np.zeros((np.size(energy_array),np.size(w2_array)))
    spec_size = np.size(w2_array)
    gamma_tiled = np.tile(dielectric_params.damping_array(energy_array),(spec_size,1)).T
    dips_tiled = np.tile(dip_mags,(spec_size,1)).T
    angles_tiled = np.tile(dip_angles,(spec_size,1)).T
    energies_tiled =  np.vstack((np.ones(spec_size)*energy_array[0],w2_array))
    w_n, v_n = w_v_tiled(energies_tiled,cav_freq,gamma_tiled)
    abs = np.sum(dips_tiled**2*energies_tiled*v_n,axis = 0)
    for i in range (0,np.size(energy_array)):
        numerator = np.sum(dips_tiled**2*energies_tiled*w_n*np.sin(2*(dip_angles[i]-angles_tiled)),axis = 0)
        sigma_set[i,:] = 1/2*numerator/abs
    return sigma_set



#acting somewhat bizzare:
def chiral_factor_two_dipole_anal(spectrum,energy_array,dip_mags,dip_angles,dielectric_params,length):
    dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
    polarizance = get_polarizance_params_dielectric(dielectric_params,dipole_matrix,energy_array,spectrum)
    brown_params = brown_params_from_polarizance(polarizance,length)
    v_n, w_m, w1, w2 = get_two_dipole_params(spectrum,energy_array,dielectric_params)
    length_factor = brown_params[1]/length
    return length_factor*w2*dip_mags[1]**2*w_m[1,:]

def isotropic_avg_absorption_correction(polarizance,length):
    brown_params = brown_params_from_polarizance(polarizance, length=length)
    d1, d2, d3, b1, b2, b3 = polarizance.provide_tuples()
    a0, a1 = brown_params[0], brown_params[1]
    m00 = a0 + a1 * (d1 ** 2 + d2 ** 2+d3**2)
    correction = -np.log(m00)/(length)
    return correction

def apparent_cd_correction(polarizance,length):
    cd_full = mueller.cd_from_polarizance(polarizance,length,"full")
    cd_m03 = mueller.cd_from_polarizance(polarizance,length,"partial")
    correction = cd_full-cd_m03
    return correction

# gives 1 where the approximation a_1 = 1/2 z^2 holds, diverges otherwise
def get_a_1_pert_metric(a_1,length):
    return a_1*2/length**2

def check_if_dirac_spectral_hamiltonian(spectral_hamiltonian,energy_array):
    if (np.size(np.nonzero(spectral_hamiltonian))<=np.size(energy_array)):
        return True
    else:
        return False

def extract_discrete_hamiltonian_from_spectral_hamiltonian(spectral_hamiltonian,energy_array,spectrum):
    is_dirac = check_if_dirac_spectral_hamiltonian(spectral_hamiltonian,energy_array)
    if (not is_dirac):
        assert ValueError("Spectral Hamiltonian Must Be in Dirac Delta Form ")
    else:
        dim = np.ndim(spectral_hamiltonian)
        if (dim >= 3):
            assert ValueError("Too many dimensions in spectral hamiltonian--only 1D and 2D arrays supported ")
        if (dim == 2):
            spectral_hamiltonian = np.sum(spectral_hamiltonian,axis= 0)
        nonzero_idx = np.nonzero(spectral_hamiltonian)
        saved_hamiltonian = spectral_hamiltonian[nonzero_idx]
        saved_energies = spectrum[nonzero_idx]
        if (not np.isclose(energy_array,saved_energies,rtel = .01).any()):
            assert ValueError("Saved Spectral Energies Have Shifted to Far from Reference")
        return saved_hamiltonian, saved_energies

def integrate_region(function,domain,center,distance):
    low_index = int(np.argmin(np.abs(domain-center+distance)))
    high_index = int(np.argmin(np.abs(domain-center-distance)))
    return np.sum(function[low_index:high_index])

#polarization index is 1 for LHP, -1 for RHP
#gives quantized interaction for a given polarization vector potential
def organic_excitonic_array(energy_array,dipole_mags,chiral_int,correction = 0,interaction_mask = None):
    net_term = chiral_int+correction
    net_term = np.clip(net_term,-1,1)
    if interaction_mask is None:
        interaction_mask = np.ones(np.size(energy_array))
    interaction_array = 1j*energy_array*dipole_mags*np.sqrt(1/2+1/2*net_term)*interaction_mask
    return interaction_array

def achiral_interaction(energy_array,dipole_mags,vector_potential):
    m_elem = organic_excitonic_array(energy_array,dipole_mags,chiral_int= 0)
    return m_elem*vector_potential

def extract_polaritons(eigenvectors,electronic_indices):
    polariton_set = []
    for i in range(0,np.size(electronic_indices)):
        e_idx = electronic_indices[i]
        selected_electronic_states = np.abs(eigenvectors[e_idx,:,:])
        sorted_idx = np.argsort(selected_electronic_states,axis = 0)
        polariton_indices = sorted_idx[3:,:]
        polariton_indices_sorted = np.sort(polariton_indices,axis = 0)
        extracted_polaritons = np.zeros((np.size(eigenvectors,axis = 0),2,np.size(eigenvectors,axis=2)),dtype = np.csingle)
        for i in range(0,np.size(polariton_indices,axis = 1)):
            new_polaritons = eigenvectors[:,polariton_indices_sorted[:,i],i]
            extracted_polaritons[:,:,i] = new_polaritons
        polariton_set.append(extracted_polaritons)
    return polariton_set

def diag_organic_jc_term(basis_state,cavity_freq,energy_array):
    n_plus = basis_state[0]
    n_minus= basis_state[1]
    h_el = 0
    electronic_index =int(basis_state[2])
    if (electronic_index>=0):
        h_el = h_el+energy_array[electronic_index]
    term = h_el+(cavity_freq)*(n_plus+n_minus)
    return term

def diag_organic_tmd_jc_term(basis_state,cavity_freq,organic_oscillator_energy_array,e_x_up_array,e_x_down_array):
    h_el = 0
    organic_basis = basis_state[:3]
    tmd_basis = np.hstack((basis_state[:2],basis_state[3:]))
    h_el = h_el+diag_organic_jc_term(organic_basis,cavity_freq,organic_oscillator_energy_array)
    h_el = h_el+jc.jc_diagonal_term(tmd_basis,e_x_down_array,e_x_up_array,0) #cavity freq 0 so as to not double count it
    return h_el


def interaction_array_to_matrices(full_basis,interaction_array,energy_index = 2):
    interaction_array_to_project =np.hstack((interaction_array,np.zeros(1)))
    electronic_states= full_basis[:,energy_index]
    projected_states_m = np.outer(electronic_states,np.ones(np.size(electronic_states))).astype(int)
    return interaction_array_to_project[projected_states_m]

def init_int_m_elems(energy_array,array_to_iterate_over):
    chiral_int_to_save = np.zeros((np.size(energy_array), np.size(array_to_iterate_over)), dtype=np.csingle)
    m_elem_plus_to_save = np.zeros(np.shape(chiral_int_to_save), dtype=np.csingle)
    m_elem_minus_to_save = np.zeros(np.shape(chiral_int_to_save), dtype=np.csingle)
    return chiral_int_to_save,m_elem_plus_to_save,m_elem_minus_to_save
def get_and_store_int_m_elems(i,chiral_int_to_save,m_elem_plus_to_save,m_elem_minus_to_save,a_1, length_to_use, cavity_freq, energy_array, dip_mags, dip_angles, w_m_at_cavity_freq,
                               xi, style):
    chiral_int = chiral_factor(a_1, length_to_use, cavity_freq, energy_array, dip_mags, dip_angles, w_m_at_cavity_freq,
                               xi, style=style)
    m_elem_array_plus = organic_excitonic_array(energy_array, dip_mags, chiral_int)  # LHP
    m_elem_array_minus = organic_excitonic_array(energy_array, dip_mags, -1 * chiral_int)  # RHP
    chiral_int_to_save[:, i] = chiral_int
    m_elem_plus_to_save[:, i] = m_elem_array_plus
    m_elem_minus_to_save[:, i] = m_elem_array_minus

def coupling_element_cavity_sweep(energy_array,cavity_freq_array,dipole_matrix,dielectric_params,spectrum,length = 1,style = "full",brown_style = "default"):
    dip_mags,dip_angles = dt.dipole_matrix_to_params(dipole_matrix)
    chiral_int_to_save, m_elem_plus_to_save, m_elem_minus_to_save = init_int_m_elems(energy_array,cavity_freq_array)
    for i in range(0,np.size(cavity_freq_array)):
        cavity_freq = cavity_freq_array[i]
        if type(length) is np.ndarray:
            length_to_use = select_from_energies(length, spectrum, cavity_freq)
        a_1, xi, w_m_at_cavity_freq,v_n_at_cavity_freq= get_chiral_interaction_constants(dipole_matrix, dielectric_params, spectrum,
                                                                        energy_array, length_to_use, cavity_freq,style = brown_style)
        get_and_store_int_m_elems(i, chiral_int_to_save, m_elem_plus_to_save, m_elem_minus_to_save, a_1, length_to_use,
                                  cavity_freq, energy_array, dip_mags, dip_angles, w_m_at_cavity_freq,
                                  xi, style)
    return chiral_int_to_save,m_elem_plus_to_save,m_elem_minus_to_save

def coupling_element_gamma_sweep(energy_array,cavity_freq,dipole_matrix,dielectric_params,damping_array,spectrum,style = "full",brown_style = "default"):
    dip_mags,dip_angles = dt.dipole_matrix_to_params(dipole_matrix)
    chiral_int_to_save, m_elem_plus_to_save, m_elem_minus_to_save = init_int_m_elems(energy_array, damping_array)
    for i in range(0,np.size(damping_array)):
        dielectric_params.gamma = damping_array[i]
        length_to_use = mean_interaction_length(cavity_freq,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal")
        a_1, xi, w_m_at_cavity_freq,v_n_at_cavity_freq= get_chiral_interaction_constants(dipole_matrix, dielectric_params, spectrum,
                                                                       energy_array, length_to_use, cavity_freq,style = brown_style)
        get_and_store_int_m_elems(i, chiral_int_to_save, m_elem_plus_to_save, m_elem_minus_to_save, a_1, length_to_use,
                                  cavity_freq, energy_array, dip_mags, dip_angles, w_m_at_cavity_freq,
                                  xi, style)
    return chiral_int_to_save,m_elem_plus_to_save,m_elem_minus_to_save

def coupling_element_dip_ratio_sweep(energy_array,cavity_freq,dip_base_mag,dip_angles,dielectric_params,dip_ratio_array,spectrum,style = "full",brown_style = "default",order = "default"):
    chiral_int_to_save, m_elem_plus_to_save, m_elem_minus_to_save = init_int_m_elems(energy_array, dip_ratio_array)
    for i in range(0,np.size(dip_ratio_array)):
        #dip ratio is mu1:mu2
        if (order == "21"):
            dip_1_mag = dip_base_mag
            dip_mags = np.array([dip_1_mag, dip_1_mag*dip_ratio_array[i]])
        else:
            dip_2_mag = dip_base_mag
            dip_mags = np.array([dip_ratio_array[i]*dip_2_mag,dip_2_mag])
        dipole_matrix = dt.create_dipole_matrix_polar_2D(dip_mags,dip_angles)
        length_to_use = mean_interaction_length(cavity_freq,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal")
        a_1, xi, w_m_at_cavity_freq,v_n_at_cavity_freq= get_chiral_interaction_constants(dipole_matrix, dielectric_params, spectrum,
                                                                       energy_array, length_to_use, cavity_freq,style= brown_style)
        get_and_store_int_m_elems(i, chiral_int_to_save, m_elem_plus_to_save, m_elem_minus_to_save, a_1, length_to_use,
                                  cavity_freq, energy_array, dip_mags, dip_angles, w_m_at_cavity_freq,
                                  xi, style)
    return chiral_int_to_save,m_elem_plus_to_save,m_elem_minus_to_save

def coupling_element_w2_sweep(energy_array,cavity_freq,dip_mags,dip_angles,dielectric_params,w2_array,spectrum,style = "full",brown_style = "default"):
    chiral_int_to_save, m_elem_plus_to_save, m_elem_minus_to_save = init_int_m_elems(energy_array, w2_array)
    for i in range(0,np.size(w2_array)):
        dipole_matrix = dt.create_dipole_matrix_polar_2D(dip_mags,dip_angles)
        energy_array = np.array([energy_array[0],w2_array[i]])
        length_to_use = mean_interaction_length(cavity_freq,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal")
        a_1, xi, w_m_at_cavity_freq,v_n_at_cavity_freq = get_chiral_interaction_constants(dipole_matrix, dielectric_params, spectrum,
                                                                       energy_array, length_to_use, cavity_freq,style = brown_style)
        get_and_store_int_m_elems(i, chiral_int_to_save, m_elem_plus_to_save, m_elem_minus_to_save, a_1, length_to_use,
                                  cavity_freq, energy_array, dip_mags, dip_angles, w_m_at_cavity_freq,
                                  xi, style)
    return chiral_int_to_save,m_elem_plus_to_save,m_elem_minus_to_save

def get_chiral_couplings_two_dipoles(cav_array,dipole_matrix,dielectric_params,spectrum,energy_array,length,vec_pot,polarization):
    a0_left, a0_right = jc.chiral_a0(vec_pot, polarization)
    dip_mags, dip_angles = dt.dipole_matrix_to_params(dipole_matrix)
    gl1_array = np.zeros(np.shape(cav_array))
    gl2_array = np.zeros(np.shape(cav_array))
    gr1_array = np.zeros(np.shape(cav_array))
    gr2_array = np.zeros(np.shape(cav_array))
    for i in range(0,np.size(cav_array)):
        cavity_freq = cav_array[i]
        a_1, xi, w_m_at_cavity_freq, v_n_at_cavity_freq = get_chiral_interaction_constants(dipole_matrix, dielectric_params, spectrum,
                                                                       energy_array, length, cavity_freq)
        chiral_int = chiral_factor(a_1, length, cavity_freq, energy_array, dip_mags, dip_angles, w_m_at_cavity_freq, xi)
        m_elem_array_left = organic_excitonic_array(energy_array, dip_mags, chiral_int)  # LHP
        m_elem_array_right = organic_excitonic_array(energy_array, dip_mags, -1 * chiral_int)  # RHP
        #I know this isn't super efficient--this doesn't really matter for runtime though and doesn't require rewriting half the
        #code base
        gl1_array[i] = m_elem_array_left[0]*a0_left
        gl2_array[i] = m_elem_array_left[1]*a0_left
        gr1_array[i] = m_elem_array_right[0]*a0_right
        gr2_array[i] = m_elem_array_right[1]*a0_right
    return np.vstack((gl1_array,gl2_array,gr1_array,gr2_array))

def get_mueller_correction(spectrum,cavity_freq,dielectric_params,dipole_matrix,energy_array,length):
    spectrum_selected = select_from_energies(spectrum, spectrum, cavity_freq)
    abs_correction = np.zeros(np.size(energy_array))
    cd_correction = np.zeros(np.size(energy_array))
    for i in range(0,np.size(energy_array)):
        polarizance_selected = get_polarizance_params_dielectric(dielectric_params, dipole_matrix[i,:], energy_array[i], spectrum_selected)
        abs_correction[i] = isotropic_avg_absorption_correction(polarizance_selected,length)
        cd_correction[i] = apparent_cd_correction(polarizance_selected,length)
    return abs_correction,cd_correction

def organic_hamiltonian_ldlb_no_vib(num_quanta,energy_array,cavity_freq,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length = 1,omit_zero_state = False,mueller_correction =False,brown_style = "default",interaction_mask = None,to_counter_rotate=False):
    '''
    In dipole gauge 
    :param num_quanta:
    :param energy_array:
    :param cavity_freq:
    :param vec_pot:
    :param polarization:
    :param dipole_matrix:
    :param dielectric_params:
    :param spectrum:
    :param length:
    :param omit_zero_state:
    :param mueller_correction:
    Whether to account for numerics to further update m_00 and m_03--not generally necessary
    :param brown_style:
    Level of expansion for the construction of the dielectric tensor for the creation of the brown parameters
    Either "pert" (i.e., second order), or "default" (numeric)
    :param interaction_mask:
    Mask to control which interactions are permitted
    :param to_counter_rotate:
    :return:
    '''
    elem_charge = 1
    m_e = 5.10999e5 # eV/c**2
    if type(length) is np.ndarray:
        length = select_from_energies(length,spectrum,cavity_freq)
    dip_mags,dip_angles = dt.dipole_matrix_to_params(dipole_matrix)
    a0 = vec_pot
    a0_plus,a0_minus = jc.chiral_a0(a0,polarization) #LHP, RHP
    num_excited_states = np.size(energy_array)
    full_basis = combo.construct_organic_basis(num_excited_states,num_quanta+1)
    if (omit_zero_state):
        full_basis = np.delete(full_basis,0,axis= 0)# removing 0 basis state--for testing purposes
    num_basis_states = np.size(full_basis,axis = 0)
    diagonal_terms = np.zeros(num_basis_states, dtype=np.csingle)
    for i in range(0, num_basis_states):
        cur_state = full_basis[i, :]
        diagonal_terms[i] = diag_organic_jc_term(cur_state, cavity_freq, energy_array)
    jc_h = np.diag(diagonal_terms) + 0 * 1j  # adding 0 imag unit to insure complex data type
    state_m, state_n = jc.create_state_tensor(full_basis, 0), jc.create_state_tensor(full_basis, 1)
    #elec_m, elec_n = jc.create_state_tensor(full_basis[:,2],0),jc.create_state_tensor(full_basis[:,2],1)
    phot_m, phot_n = state_m[:,:,:2],state_n[:,:,:2]
    acd_basis_type =["ladder","ladder","bool"]
    a_plus_dag_b = jc.state_matrix_relation(state_n, state_m, np.array([1, 0,-1]),acd_basis_type)
    a_plus_b_dag = jc.state_matrix_relation(state_n, state_m, np.array([-1, 0,1]), acd_basis_type)
    a_minus_dag_b = jc.state_matrix_relation(state_n, state_m,np.array([0, 1,-1]), acd_basis_type)
    a_minus_b_dag = jc.state_matrix_relation(state_n, state_m, np.array([0, -1,1]), acd_basis_type)



    a_1, xi, w_m_at_cavity_freq,v_n_at_cavity_freq = get_chiral_interaction_constants(dipole_matrix,dielectric_params,spectrum,energy_array,length,cavity_freq,style = brown_style)
    chiral_int = chiral_factor(a_1,length,cavity_freq,energy_array,dip_mags,dip_angles,w_m_at_cavity_freq,xi,style = "full")
    chiral_int_pert = chiral_factor_pert(a_1,xi,cavity_freq,energy_array,dip_mags,dip_angles,w_m_at_cavity_freq,v_n_at_cavity_freq,length,style = "full_anal")
    chiral_int_z_anal = chiral_factor_pert(a_1, xi, cavity_freq, energy_array, dip_mags, dip_angles, w_m_at_cavity_freq,
                                         v_n_at_cavity_freq,length, style="z_anal")
    a_1_metric = get_a_1_pert_metric(a_1,length)
    print_variables = True
    if (print_variables):
        print("length pref:" + str(a_1 / length))
        print("length:"+str(length))
        print("A_1:"+str(a_1))
        print("a_1_metric:" + str(a_1_metric))
        print("W_m:" + str(w_m_at_cavity_freq))
        print("sigma:"+str(chiral_int))
        print("sigma_pert:" + str(chiral_int_pert))
        print("sigma_z_anal:"+str(chiral_int_z_anal))

    if (mueller_correction):
        abs_correct, cd_correct = get_mueller_correction(spectrum,cavity_freq,dielectric_params,dipole_matrix,energy_array,length)
        iso_abs = abs_from_operator(dielectric_params,energy_array,cavity_freq,dip_mags,length = 1,separate = True)
        iso_abs = np.array(iso_abs[:,0])
        abs_correct_normed = abs_correct/iso_abs
        cd_correct_normed = cd_correct/iso_abs
    else:
        abs_correct_normed, cd_correct_normed = 0,0
    if (interaction_mask is None):
        int_mask = np.ones(np.size(energy_array))
    else: int_mask = interaction_mask
    #minus is lhp, plus is rhp
    m_elem_array_plus = organic_excitonic_array(energy_array,dip_mags,chiral_int,correction= abs_correct_normed+cd_correct_normed,interaction_mask=int_mask) #LHP
    m_elem_array_minus = organic_excitonic_array(energy_array,dip_mags,-chiral_int,correction = abs_correct_normed-cd_correct_normed,interaction_mask=int_mask) #RHP
    m_elem_matrix_plus = interaction_array_to_matrices(full_basis,m_elem_array_plus)
    m_elem_matrix_minus = interaction_array_to_matrices(full_basis,m_elem_array_minus)
    m_elem_plus = (a_plus_dag_b*m_elem_matrix_plus.T + a_plus_b_dag * m_elem_matrix_plus.conj())
    m_elem_minus = (a_minus_dag_b* m_elem_matrix_minus.T + a_minus_b_dag * m_elem_matrix_minus.conj())
    if (to_counter_rotate):
        a_plus_dag_b_dag = jc.state_matrix_relation(state_n, state_m, np.array([1, 0, 1]), acd_basis_type)
        a_plus_b = jc.state_matrix_relation(state_n, state_m, np.array([-1, 0, -1]), acd_basis_type)
        a_minus_dag_b_dag = jc.state_matrix_relation(state_n, state_m, np.array([0, 1, 1]), acd_basis_type)
        a_minus_b = jc.state_matrix_relation(state_n, state_m, np.array([0, -1, -1]), acd_basis_type)
        m_elem_plus = m_elem_plus+(a_plus_b * m_elem_matrix_plus.T + a_plus_dag_b_dag * m_elem_matrix_plus.conj())
        m_elem_minus = m_elem_minus+(a_minus_b * m_elem_matrix_minus.T + a_minus_dag_b_dag * m_elem_matrix_minus.conj())
    jc_h = jc_h + (a0_plus* m_elem_plus + a0_minus * m_elem_minus)
    net_int_perturbation = chiral_int+cd_correct_normed+abs_correct_normed
    return jc_h, net_int_perturbation

def jaynes_cummings_organic_alter_fast(jc_h,full_basis,cav_freq,energy_array):
    num_basis_states = np.size(full_basis,0)
    for i in range(0,num_basis_states):
        cur_state = full_basis[i,:]
        jc_h[i,i] = diag_organic_jc_term(cur_state,cav_freq,energy_array)
    return jc_h

def jaynes_cummings_organic_ldlb_sweep(num_quanta,energy_array,cavity_freq_array,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length = 1,omit_zero_state = False,mueller_correction = False,brown_style = "default",interaction_mask = None,to_counter_rotate=False,to_self_interact=False):
    cavity_its = np.size(cavity_freq_array)
    init_jc_h,chiral_int_init = organic_hamiltonian_ldlb_no_vib(num_quanta,energy_array,cavity_freq_array[0],vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length,\
                                                                omit_zero_state= omit_zero_state,mueller_correction=mueller_correction,brown_style = brown_style,interaction_mask = interaction_mask)
    e_x_init,a_x_init = base.solve_hamiltonian(init_jc_h)
    e_x_to_save = np.zeros((np.size(e_x_init),cavity_its))
    a_x_to_save = np.zeros((np.size(a_x_init,0),np.size(a_x_init,1),cavity_its),dtype=np.csingle)
    chiral_int_set = np.zeros((np.size(energy_array),np.size(cavity_freq_array)))
    e_x_to_save[:,0] = e_x_init
    a_x_to_save[:,:,0] = a_x_init
    for i in range(1,cavity_its):
        print(i)
        new_jc_h,new_chiral_int = organic_hamiltonian_ldlb_no_vib(num_quanta,energy_array,cavity_freq_array[i],vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length,omit_zero_state,mueller_correction,brown_style,interaction_mask=interaction_mask,to_counter_rotate=to_counter_rotate)
        chiral_int_set[:,i] = new_chiral_int
        cur_e_x, cur_a_x = base.solve_hamiltonian(new_jc_h.astype(np.csingle)) #type fixing to ensure that there isn't a crash due to typing issues
        e_x_to_save[:,i] = cur_e_x
        a_x_to_save[:,:,i] = cur_a_x
    return e_x_to_save, a_x_to_save, chiral_int_set

def create_organic_tmd_hamiltonian(acd_length,num_quanta,tmd_ex_basis_size,organic_energy_array,cavity_freq,vec_pot,polarization,tmd_excitonic_vars_up,tmd_excitonic_vars_down,organic_dipole_matrix,dielectric_params,spectrum,
                            omit_zero_state = False,mueller_correction =False,brown_style = "default",interaction_mask = None,**kwargs):
    '''
    :param num_quanta:
    :param energy_array:
    :param cavity_freq:
    :param vec_pot:
    :param polarization:
    :param excitonic_vars_up:
    :param excitonic_vars_down:
    :param angle_mom_space:
    :param dipole_matrix:
    :param dielectric_params:
    :param spectrum:
    :param acd_length:
    :param omit_zero_state:
    :param mueller_correction:
    :param brown_style:
    :param interaction_mask:
    :param kwargs:
    :return:
    '''
    resolution_factor = 1 #defaults to not accounting for resolution
    n_el = 1
    acd_tmd_vec_pot_ratio=1
    start_time = time.time()
    print("Constructing ACD-TMD hamiltonian")
    for key,value in kwargs.items():
        if key == "resolution":
            resolution_factor = value
        if key == "n_el":
            n_el = value
        if key == "acd_tmd_interaction_ratio":
            acd_tmd_vec_pot_ratio = value
    if type(acd_length) is np.ndarray:
        length = select_from_energies(acd_length, spectrum, cavity_freq)
    else: length = acd_length
    dip_mags, dip_angles = dt.dipole_matrix_to_params(organic_dipole_matrix)
    a0 = vec_pot
    a0_plus, a0_minus = jc.chiral_a0(a0, polarization)  # LHP, RHP
    num_excited_states = np.size(organic_energy_array)
    num_spin_excitons = tmd_ex_basis_size
    evd,evu = tmd_excitonic_vars_down, tmd_excitonic_vars_up
    e_x_up_array = tmd_excitonic_vars_up.e_x[:tmd_ex_basis_size]
    a_x_up_matrix = tmd_excitonic_vars_up.a_x[:,:tmd_ex_basis_size]
    e_x_down_array = tmd_excitonic_vars_down.e_x[:tmd_ex_basis_size]
    a_x_down_matrix = tmd_excitonic_vars_down.a_x[:, :tmd_ex_basis_size]
    p_minus_down, p_minus_vv_down, p_minus_cc_down,p_minus_cv_down = evd.p_minus,evd.p_minus_vv,evd.p_minus_cc,evd.p_minus_cv
    p_plus_down, p_plus_vv_down, p_plus_cc_down,p_plus_cv_down = evd.p_plus, evd.p_plus_vv, evd.p_plus_cc,evd.p_plus_cv
    p_minus_up, p_minus_vv_up, p_minus_cc_up,p_minus_cv_up = evu.p_minus,evu.p_minus_vv,evu.p_minus_cc ,evu.p_minus_cv
    p_plus_up, p_plus_vv_up, p_plus_cc_up ,p_plus_cv_up= evu.p_plus, evu.p_plus_vv, evu.p_plus_cc,evu.p_plus_cv
    p_x_up,p_x_vv_up,p_x_cc_up,p_x_cv_up = evu.p_x, evu.p_x_vv, evu.p_x_cc  ,evu.p_x_cv
    p_x_down,p_x_vv_down,p_x_cc_down,p_x_cv_down = evd.p_x, evd.p_x_vv, evd.p_x_cc    ,evd.p_x_cv

    full_basis = combo.construct_organic_spin_exciton_basis(num_excited_states,num_spin_excitons, num_quanta + 1)
    num_basis_states = np.size(full_basis, axis=0)
    diagonal_terms = np.zeros(num_basis_states, dtype=np.csingle)
    for i in range(0, num_basis_states):
        cur_state = full_basis[i, :]
        diagonal_terms[i] = diag_organic_tmd_jc_term(cur_state,cavity_freq,organic_energy_array,e_x_up_array,e_x_down_array)
    jc_h = np.diag(diagonal_terms) + 0 * 1j  # adding 0 imag unit to insure complex data type
    phot_m, phot_n = jc.create_state_tensor(full_basis[:, :2], 0), jc.create_state_tensor(full_basis[:, :2], 1)
    phot_equal = jc.photons_matrix_relation(phot_n, phot_m, np.array([0, 0]), relation_type="bool")
    a_plus_dag = jc.photons_matrix_relation(phot_n, phot_m, np.array([1, 0]), "a")
    a_plus = jc.photons_matrix_relation(phot_n, phot_m, np.array([-1, 0]), "a")
    a_minus_dag = jc.photons_matrix_relation(phot_n, phot_m, np.array([0, 1]), "a")
    a_minus = jc.photons_matrix_relation(phot_n, phot_m, np.array([0, -1]), "a")
    a_1, xi, w_m_at_cavity_freq, v_n_at_cavity_freq = get_chiral_interaction_constants(organic_dipole_matrix, dielectric_params,
                                                                                       spectrum, organic_energy_array, length,cavity_freq,style=brown_style)
    chiral_int = chiral_factor(a_1, length, cavity_freq, organic_energy_array, dip_mags, dip_angles, w_m_at_cavity_freq, xi,
                               style="full")
    if (mueller_correction):
        abs_correct, cd_correct = get_mueller_correction(spectrum, cavity_freq, dielectric_params, organic_dipole_matrix,
                                                         organic_energy_array, length)
        iso_abs = abs_from_operator(dielectric_params, organic_energy_array, cavity_freq, dip_mags, length=1, separate=True)
        iso_abs = np.array(iso_abs[:, 0])
        abs_correct_normed = abs_correct / iso_abs
        cd_correct_normed = cd_correct / iso_abs
    else:
        abs_correct_normed, cd_correct_normed = 0, 0
    if (interaction_mask is None):
        int_mask = np.ones(np.size(organic_energy_array))
    else:
        int_mask = interaction_mask
    # minus is lhp, plus is rhp
    m_nm_fit_param = n_el #n_el as in (4) in Latini et al 2019 https://pubs.acs.org/doi/pdf/10.1021/acs.nanolett.9b00183
    acd_m_elem_array_plus = organic_excitonic_array(organic_energy_array, dip_mags, chiral_int,
                                                correction=abs_correct_normed + cd_correct_normed,
                                                interaction_mask=int_mask)  # LHP
    acd_m_elem_array_minus = organic_excitonic_array(organic_energy_array, dip_mags, -1*chiral_int,
                                                 correction=abs_correct_normed - cd_correct_normed,
                                                 interaction_mask=int_mask)  # RHP
    m_0n_plus_up, m_0n_minus_up, m_0n_x_up, m_0n_plus_down, m_0n_minus_down, m_0n_x_down,\
     m_nm_plus_up, m_nm_minus_up, m_nm_x_up, m_nm_plus_down, m_nm_minus_down, m_nm_x_down= \
        jc.create_tmd_m_elem_matrices(a_x_up_matrix, a_x_down_matrix, p_plus_up, p_minus_up, p_x_up, p_plus_down, p_minus_down, p_x_down,
                                p_plus_vv_up, p_plus_cc_up, p_minus_vv_up,
                               p_minus_cc_up, p_plus_vv_down, p_plus_cc_down, p_minus_vv_down, p_minus_cc_down, p_x_vv_up, p_x_cc_up, p_x_vv_down, p_x_cc_down, m_nm_fit_param,
                               to_mask_coupling_lb=False, mask_factor=1e-5)
    tmd_basis = np.concatenate((full_basis[:,:2],full_basis[:,3:]),axis= 1)
    acd_basis = full_basis[:,:3]
    tmd_m_elem_matrix_plus,tmd_m_elem_matrix_minus,tmd_m_elem_matrix_x= jc.get_tmd_m_elem_matrices(tmd_basis,m_0n_plus_up,m_0n_minus_up,m_0n_x_up,m_0n_plus_down,m_0n_minus_down,m_0n_x_down,\
           m_nm_plus_up,m_nm_minus_up,m_nm_x_up,m_nm_plus_down ,m_nm_minus_down,m_nm_x_down)
    acd_m_elem_matrix_plus = interaction_array_to_matrices(acd_basis, acd_m_elem_array_plus)
    acd_m_elem_matrix_minus = interaction_array_to_matrices(acd_basis, acd_m_elem_array_minus)
    m_e = 5.10999e5  # ev/c**2
    m_elem_matrix_plus = acd_m_elem_matrix_plus.T*acd_tmd_vec_pot_ratio+ tmd_m_elem_matrix_plus/m_e
    m_elem_matrix_minus = acd_m_elem_matrix_minus.T*acd_tmd_vec_pot_ratio+ tmd_m_elem_matrix_minus/m_e
    m_elem_matrix_x = acd_tmd_vec_pot_ratio/np.sqrt(2)*(acd_m_elem_matrix_plus+acd_m_elem_matrix_minus).T+tmd_m_elem_matrix_x/m_e
    if (np.array_equiv(full_basis[0,:],np.array([0,0,-1,-1,-1]))):
        m_0n_arrays = jc.M_0N_ARRAYS(m_elem_matrix_plus[0,:],m_elem_matrix_minus[0,:],m_elem_matrix_x[0,:]) #assumes that first state is the null state
    else: ValueError("First basis element must be null state")
    #selecting out by specific states
    m_elem_matrix_plus = a_plus_dag*m_elem_matrix_plus+a_plus*m_elem_matrix_plus.T.conj()
    m_elem_matrix_minus = a_minus_dag * m_elem_matrix_minus + a_minus * m_elem_matrix_minus.T.conj()
    a0_plus,a0_minus = a0_plus/resolution_factor,a0_minus/resolution_factor
    jc_h = jc_h + a0_plus * m_elem_matrix_plus + a0_minus * m_elem_matrix_minus
    net_int_perturbation = chiral_int + cd_correct_normed + abs_correct_normed
    end_time = time.time()
    print("took " + str(end_time - start_time) + " seconds")
    return jc_h, net_int_perturbation,m_0n_arrays


def calculate_cavity_n_eff(angle_array,cavity_length,dielectric_length,n_diel,n_back = 1):
    '''
    Determines the effective refractive index for the entire cavity in terms of how much it increases the wavelength
    Dielectrics assumed to be non-angle dependent
    This may be incorrect. Needs further testing
    :param angle_array:
    :param cavity_length:
    :param dielectric_length:
    :param n_diel:
    :param n_back:
    :return:
    '''
    if (np.isscalar(dielectric_length)):
        theta_diel = np.arcsin(n_back*np.sin(angle_array)/n_diel)
        ratio = dielectric_length/cavity_length
        return np.cos(angle_array)*(n_back/np.cos(angle_array)*(1-ratio)+n_diel/np.cos(theta_diel)*ratio)
    else:
        n_eff = np.cos(angle_array)*(n_back/np.cos(angle_array)*(1-np.sum(dielectric_length)/cavity_length))
        for i in np.arange(np.size(dielectric_length)):
            theta_diel = np.arcsin(n_back * np.sin(angle_array) / n_diel[i])
            ratio = dielectric_length[i]/cavity_length
            n_eff = n_eff+ np.cos(angle_array)*n_diel/np.cos(theta_diel)*ratio
        return n_eff

def jaynes_cummings_organic_tmd_alter_fast(jc_h,full_basis,cavity_freq,tmd_ex_up_energy_array,tmd_ex_down_energy_array,organic_oscillator_energy_array):
    num_basis_states = np.size(full_basis,0)
    for i in range(0,num_basis_states):
        cur_state = full_basis[i,:]
        jc_h[i,i] = diag_organic_tmd_jc_term(cur_state,cavity_freq,organic_oscillator_energy_array,tmd_ex_up_energy_array,tmd_ex_down_energy_array)
    return jc_h

def get_peak_types_and_widths_from_tmd_acd_basis(full_basis,dielectric_damping_array):
    '''
    Assumes basis in form (gamma_plus,gamma_minus,mu,tmd_a,tmd_b)
    Only works for states of single quanta such that either tmd or acd is occupied,
    not both
    :param full_basis:
    :param dielectric_damping_array
    :return:
    '''
    tmd_basis = full_basis[:,3:5]+1
    acd_basis = full_basis[:,2]+1
    tmd_occupied = np.nonzero(np.sum(tmd_basis,axis = 1))[0]
    acd_occupied = np.nonzero(acd_basis)[0]
    peak_types = np.zeros(np.size(full_basis,axis=0),dtype = object)
    peak_types[tmd_occupied] = "dirac"
    peak_types[acd_occupied] = "lorentzian"
    peak_widths = np.zeros(np.size(full_basis,axis=0),dtype = float)
    peak_widths[tmd_occupied]= .01 #10 meV broadening
    peak_widths[acd_occupied] = dielectric_damping_array
    return peak_types,peak_widths


def tmd_acd_hamiltonian_sweep(acd_length,spectrum,organic_dipole_matrix,dielectric_params,organic_energy_array,raw_system_up,raw_system_down,tmd_basis_size,cavity_freq_array, vec_pot_strength, polarization,mueller_correction = False,
                              brown_style = "default",**kwargs):
    num_quanta = 1
    brown_style = "default"
    mueller_correction = False
    energy_renorm = 0
    for key,value in kwargs.items():
        if key == "num_quanta":
            num_quanta = value
        if key == "renorm_energy":
            energy_renorm = value
    coupling = True
    start_time = time.time()
    tmd_ex_vars_up = jc.get_excitonic_vars(raw_system_up, **{"coupled": coupling,"basis_size":tmd_basis_size})
    tmd_ex_vars_down = jc.get_excitonic_vars(raw_system_down, **{"coupled": coupling,"basis_size":tmd_basis_size})
    end_time = time.time()
    print("BSE_took:"+str(end_time-start_time)+"sec")
    iterations = np.size(cavity_freq_array)
    spectral_res=  np.size(spectrum)
    spectral_matrix = np.zeros((np.size(cavity_freq_array), spectral_res,3))

    full_basis = combo.construct_organic_spin_exciton_basis(np.size(organic_energy_array),tmd_basis_size,num_quanta+1)
    full_basis_size = np.size(full_basis,axis =0)
    peak_types,peak_widths = get_peak_types_and_widths_from_tmd_acd_basis(full_basis,dielectric_params.damping_array(organic_energy_array))
    to_save_energies = tmd_basis_size*2+np.size(organic_energy_array)
    a_x_matrix = np.zeros((full_basis_size,to_save_energies,iterations),dtype= np.csingle)
    calculated_energies_matrix = np.zeros((to_save_energies, iterations))
    for i in range(0, iterations):
        cavity_freq = cavity_freq_array[i]
        h_jc,net_acd_int_perturbation, m_elem_tensor = create_organic_tmd_hamiltonian(acd_length,num_quanta,tmd_basis_size,organic_energy_array,cavity_freq,vec_pot_strength,polarization,tmd_ex_vars_up,
                                                                  tmd_ex_vars_down,organic_dipole_matrix,dielectric_params,spectrum,mueller_correction=mueller_correction,
                                                                  brown_style = brown_style,**kwargs)
        e_x_jc, a_x_jc = base.solve_hamiltonian(h_jc)
        #e_x_jc = (e_x_jc-e_x_jc[0]) #rebasing
        m_pol = jc.polariton_matrix_set(a_x_jc,m_elem_tensor)
        e_x_sub = e_x_jc[:to_save_energies]
        a_x_sub = a_x_jc[:,:to_save_energies]
        print("Iteration:"+str(i))
        m_elem_tensors = [m_elem_tensor.m_plus,m_elem_tensor.m_minus,m_elem_tensor.m_x]
        for j in range(2,3):
            m_elem_tensor = m_elem_tensors[j]
            lin_abs_spectrum = linear_absorption_multiple_peak_shapes(spectrum,full_basis,a_x_sub,e_x_sub,m_elem_tensor,dielectric_params)
            spectral_matrix[i, :, j] = lin_abs_spectrum
        calculated_energies_matrix[:, i] = e_x_sub
        a_x_matrix[:, :, i] = a_x_sub

    return calculated_energies_matrix, a_x_matrix, spectral_matrix,tmd_ex_vars_up.a_x,tmd_ex_vars_down.a_x

def linear_absorption_multiple_peak_shapes(energy_spectrum,full_basis,a_x_matrix,e_x_array,m_0n_array,dielectric_params):
    '''
    :param full_basis:
    :param a_x_matrix:
    :param e_x_array:
    :param m_elem_matrix:
    :return:
    '''
    vec_pot = 1 # in eV/(ec)
    num_states = np.size(e_x_array)
    num_basis_states = np.size(full_basis,axis= 0)
    intensity_array=  np.zeros(np.size(energy_spectrum))
    oscillator_strengths = (vec_pot) ** 2 * 2 * np.pi * np.abs(np.einsum("ij,i->ij", a_x_matrix, m_0n_array)) ** 2
    for i in range(num_states):
        cur_e_x = e_x_array[i]
        damping_array = dielectric_params.damping_array(cur_e_x)
        peak_types,peak_widths = get_peak_types_and_widths_from_tmd_acd_basis(full_basis,damping_array)
        for j in range(num_basis_states):
            peak_shape = 0
            if (peak_types[j] =="dirac"):
                peak_shape = base.dirac_delta(energy_spectrum, e_x_array[i],**{"width":peak_widths[j]})
            elif (peak_types[j] =="lorentzian"):
                peak_shape = np.imag(dt.lorentzian(energy_spectrum,e_x_array[i],peak_widths[j]))*0
            intensity_array = intensity_array + oscillator_strengths[j, i] * peak_shape
            #plt.plot(energy_spectrum,intensity_array);plt.show()
    return np.abs(intensity_array)

def cavity_wavevector_dispersion(cavity_length,q_parallel,n_eff =1,units = dt.unit_defs_base):
    return units.hbar*units.c/n_eff*np.sqrt((np.pi/cavity_length)**2+q_parallel**2)
class CAVITY():
    def __init__(self,length,n_eff_array,q_in_plane_array,finesse, modes = 1):
        '''
        Object for storing cavity paramters
        :param length: length of cavity from mirror to mirror
        :param n_eff_array: effective dielectric frequency of the cavity (consider substrates, e.g.)
        array dimensions same as for q
        :param q_in_plane_array: wave vector for q_parallel, q_x, k_x, k_parallel
        :param finesse: 2*\pi/loss where loss is round trip loss
        :param modes: cavity mode number
        '''
        self.length = length
        self.n_eff_array = n_eff_array
        self.q_parallel = q_in_plane_array
        self.finesse = finesse
        self.modes = modes
        self.energy_dispersion = self.cavity_wavevector_dispersion() #for q_parallel
    def cavity_wavevector_dispersion(self):
        return cavity_wavevector_dispersion(self.length,self.q_parallel,self.n_eff_array)
    def get_q_factor(self,wavelength):
        return self.finesse*self.length*2/wavelength


def jaynes_cummings_organic_q_vector_sweep_simple(num_quanta,energy_array,cavity_object,vec_pot,polarization,dipole_matrix,dielectric_params,spectrum,length = 1,omit_zero_state = False,mueller_correction = False,brown_style = "default",interaction_mask = None):
    '''
    Simplest way to account for angle dispersion--
    DOES NOT REPRODUCE EXPERIMENTAL RESULTS
    :param num_quanta:
    :param energy_array:
    :param cavity_object:
    :param vec_pot:
    :param polarization:
    :param dipole_matrix:
    :param dielectric_params:
    :param spectrum:
    :param length:
    :param omit_zero_state:
    :param mueller_correction:
    :param brown_style:
    :param interaction_mask:
    :return:
    '''
    cavity_freq_array = cavity_object.energy_dispersion
    return jaynes_cummings_organic_ldlb_sweep(num_quanta, energy_array, cavity_freq_array, vec_pot, polarization,
                                       dipole_matrix, dielectric_params, spectrum, length=length, omit_zero_state=omit_zero_state,
                                       mueller_correction=mueller_correction, brown_style=brown_style, interaction_mask=interaction_mask)

class VIBRATIONAL_PARAMS():
    def __init__(self,electronic_array,huang_rhys_array,vibrational_dist_array):
        self.el = electronic_array
        self.huang_rhys = huang_rhys_array
        self.vib = vibrational_dist_array

def organic_hamiltonian_vib(num_quanta,energy_array,cavity_freq,vec_pot,dipole_operators_r,dipole_operators_l,spectrum,vib_states):
    a0 = vec_pot
    num_excited_states = np.size(energy_array)
    full_basis = combo.construct_organic_vib_basis(num_excited_states, num_quanta + 1,4)
    num_basis_states = np.size(full_basis, axis=0)
    diagonal_terms = np.zeros(num_basis_states, dtype=np.csingle)
    for i in range(0, num_basis_states):
        cur_state = full_basis[i, :]
        diagonal_terms[i] = diag_organic_jc_term(cur_state, cavity_freq, energy_array)
    jc_h = np.diag(diagonal_terms) + 0 * 1j  # adding 0 imag unit to insure complex data type
    phot_m, phot_n = jc.create_state_tensor(full_basis[:, :2], 0), jc.create_state_tensor(full_basis[:, :2], 1)
    elec_m, elec_n = jc.create_state_tensor(full_basis[:, 2:3], 0), jc.create_state_tensor(full_basis[:, 2:], 1)
    vib_m,vib_n = jc.create_state_tensor(full_basis[:, 3:], 0), jc.create_state_tensor(full_basis[:, 2:], 1)
    a_plus_dag = jc.photons_matrix_relation(phot_n, phot_m, np.array([1, 0]), "a")
    a_plus = jc.photons_matrix_relation(phot_n, phot_m, np.array([-1, 0]), "a")
    a_minus_dag = jc.photons_matrix_relation(phot_n, phot_m, np.array([0, 1]), "a")
    a_minus = jc.photons_matrix_relation(phot_n, phot_m, np.array([0, -1]), "a")
    m_elem_array_plus = organic_excitonic_array(energy_array, dipole_operators_r, spectrum)
    m_elem_array_minus = organic_excitonic_array(energy_array, dipole_operators_l, spectrum)
    to_rotate = False
    if (to_rotate):  # rotatino of the interaction elements doesn't (and shouldn't matter)
        phi = .1
        rotation = np.exp(1j * phi)
        m_elem_array_minus * rotation
        m_elem_array_plus * rotation
    m_elem_matrix_plus = interaction_array_to_matrices(full_basis, m_elem_array_plus)
    m_elem_matrix_minus = interaction_array_to_matrices(full_basis, m_elem_array_minus)
    m_elem_plus = (a_plus_dag * m_elem_matrix_plus + a_plus * m_elem_matrix_plus.T.conj())
    m_elem_minus = (a_minus_dag * m_elem_matrix_minus + a_minus * m_elem_matrix_minus.T.conj())
    jc_h = jc_h + (a0* m_elem_plus + a0 * m_elem_minus)
    return jc_h

def show_eigenvalue_content(energy_spectrum,energy_array,chosen_eigenvalue_array):
    energy_array = np.array(energy_array)
    a_x_array = np.array(chosen_eigenvalue_array)
    num_states = np.size(a_x_array,axis = 0)
    size = energy_spectrum.size
    intensity_array = np.zeros((size,np.size(energy_array,axis=1)))
    for i in range(0,num_states):
        oscillator_strength = np.abs(a_x_array[i])**2
        peak_shape = base.dirac_delta_set(energy_spectrum,energy_array[i],width = .04)
        intensity_array = intensity_array+np.tile(oscillator_strength,(np.size(energy_spectrum),1))*peak_shape
    return intensity_array

def basis_weighting(full_basis,index,boolean_weighting = False,elec_style = False):
    selected_basis = np.copy(full_basis[:,index])
    if (elec_style==False):
        if (boolean_weighting):
            selected_basis[np.abs(selected_basis) > 0] = 1
            return selected_basis
        else:
            return selected_basis
    else:
        if (boolean_weighting):
            selected_basis[selected_basis > -0.5] = 1 #recall that -1 is the null index for electonic occupancy
            selected_basis[selected_basis <=0.5] = 0 #makes -1 (the previous null index), 0
            return selected_basis
        else:
            return selected_basis+1



#plus index, minus index in terms of basis
#need to check whether this really works for num_quanta > 1
#minus index is lhp, plus is rhp --note that this opposite the quantum optics convention
#that is, for optics, in the alternate convention, (1 i) is RHP for the Jones vector
#we return LHP-RHP
def helicity_from_eigenvecs(eigenvecs,full_basis,plus_index,minus_index,sign_convention = "default"):
    '''
    Note that this form of the helicity varies from -1 to 1, unlike the -2 to 2 convention often encountered
    in the literature (g = 2*(L-R)/(L+R)), instead of g = (L-R)/(L+R). In many plotting routines, this is
    accounted for by multiplying by 2 if "old_factor=True"
    :param eigenvecs:
    :param full_basis:
    :param plus_index:
    :param minus_index:
    :param sign_convention:
    :return:
    '''
    eigenvec_dimension = np.ndim(eigenvecs)
    plus_values = np.array(basis_weighting(full_basis,plus_index),dtype=np.float64)
    minus_values = np.array(basis_weighting(full_basis,minus_index),dtype= np.float64)
    if (eigenvec_dimension == 2):
        plus_eigenvecs = np.einsum("i,ij->ij",plus_values,eigenvecs)
        minus_eigenvecs = np.einsum("i,ij->ij",minus_values,eigenvecs)
    elif (eigenvec_dimension == 3):
        plus_eigenvecs = np.einsum("i,ijk->ijk",plus_values,eigenvecs)
        minus_eigenvecs = np.einsum("i,ijk->ijk",minus_values,eigenvecs)
    else:
        assert ValueError("Eigenvector tensor dimension must be 2 or 3")
    p_plus = np.sum(np.abs(plus_eigenvecs)**2,axis =0)
    p_minus = np.sum(np.abs(minus_eigenvecs)**2,axis = 0)
    helicity = (p_minus-p_plus)/(p_plus+p_minus)
    if (sign_convention != "default"):
        helicity = -1*helicity
    return  helicity
#note that this sign convention is opposite of paper

def polaritonic_characteristic_from_eigenvecs(eigenvecs,full_basis,photonic_indices,electronic_indices):
    eigenvec_dimension = np.ndim(eigenvecs)
    photon_weights, elec_weights = get_weights_from_indices(full_basis,photonic_indices,electronic_indices)
    if (eigenvec_dimension == 2):
        elec_eigenvecs = np.einsum("i,ij->ij", elec_weights, eigenvecs)
        photon_eigenvecs = np.einsum("i,ij->ij",photon_weights, eigenvecs)
    elif (eigenvec_dimension == 3):
        elec_eigenvecs = np.einsum("i,ijk->ijk", elec_weights, eigenvecs)
        photon_eigenvecs = np.einsum("i,ijk->ijk",photon_weights, eigenvecs)
    else:
        assert ValueError("Eigenvector tensor dimension must be 2 or 3")
    chi_elec = -np.abs(np.sum(np.abs(elec_eigenvecs)**2,axis= 0)-.5)+.5
    chi_phot = -np.abs(np.sum(np.abs(photon_eigenvecs)**2,axis= 0)-.5)+.5
    return chi_phot,chi_elec
def get_weights_from_indices(full_basis,photonic_indices,electronic_indices):
    photon_types = np.size(photonic_indices)
    elec_types = np.size(electronic_indices)
    num_states = np.size(full_basis,axis = 0)
    photon_weights =np.zeros(num_states)
    elec_weights = np.zeros(num_states)
    for i in range(photon_types):
        photon_weights = photon_weights+basis_weighting(full_basis,photonic_indices[i])
    for i in range(elec_types):
        elec_weights = elec_weights+basis_weighting(full_basis,electronic_indices[i],boolean_weighting=True,elec_style=True)
    photon_weights = np.array(photon_weights,dtype = np.float64)
    elec_weights = np.array(elec_weights,dtype = np.float64)
    return photon_weights, elec_weights
#V2: Chi_pol = 2|C_ex C_phot|, not the summed absolute values as before
def polaritonic_characteristic_v2_from_eigenvecs(eigenvecs,full_basis, photonic_indices,electronic_indices):
    eigenvec_dimension = np.ndim(eigenvecs)
    photon_weights, elec_weights = get_weights_from_indices(full_basis,photonic_indices,electronic_indices)
    if (eigenvec_dimension == 2):
        elec_eigenvecs = np.einsum("i,ij->ij", elec_weights, eigenvecs)
        photon_eigenvecs = np.einsum("i,ij->ij",photon_weights, eigenvecs)
    elif (eigenvec_dimension == 3):
        elec_eigenvecs = np.einsum("i,ijk->ijk", elec_weights, eigenvecs)
        photon_eigenvecs = np.einsum("i,ijk->ijk",photon_weights, eigenvecs)
    else:
        assert ValueError("Eigenvector tensor dimension must be 2 or 3")
    chi_elec = np.sqrt(np.sum(elec_eigenvecs ** 2, axis=0))
    chi_phot = np.sqrt(np.sum(photon_eigenvecs** 2, axis=0))
    return np.abs(2*chi_elec*chi_phot)

def helical_polaritonic_characteristic_v2_from_eigenvecs(eigenvecs,full_basis,photonic_indices,electronic_indices,sign_convention = "default"):
    chi_pol = polaritonic_characteristic_v2_from_eigenvecs(eigenvecs,full_basis,photonic_indices,electronic_indices)
    plus_index, minus_index = photonic_indices[0], photonic_indices[1]
    helicity = helicity_from_eigenvecs(eigenvecs,full_basis,plus_index,minus_index,sign_convention = sign_convention)
    return helicity*chi_pol

#assumes eigenvecs in 3D--will update at some point
def helical_polaritonic_characteristic_from_eigenvecs(eigenvecs,full_basis,photonic_indices,electronic_indices):
    chi_phot, chi_elec=    polaritonic_characteristic_from_eigenvecs(eigenvecs, full_basis, photonic_indices, electronic_indices)
    plus_index, minus_index = photonic_indices[0],photonic_indices[1]
    helicity = helicity_from_eigenvecs(eigenvecs,full_basis,plus_index,minus_index)
    return helicity*(chi_phot+chi_elec)

def state_occupancy_from_eigenvecs(eigenvecs,state_index):
    '''
    Provides occupancy percentage for some state
    :param eigenvecs:
    :param state_index:
    :return:
    '''
    if (np.isscalar(state_index)):
        return np.abs(eigenvecs[state_index,:,:])**2
    else:
        for i in range(0,np.size(state_index)):
            if i==0:
                occupancy = np.abs(eigenvecs[state_index[i],:,:])**2
            else:
                occupancy = occupancy+np.abs(eigenvecs[state_index[i],:,:])**2
        return occupancy


def get_basis_indices_from_labels(basis_labels,target_label):
    selected_indices = np.char.count(basis_labels,target_label)
    return np.atleast_1d(np.squeeze(np.argwhere(selected_indices==1)))

def acd_tmd_eigenvec_characterization(eigenvecs,full_basis,
                                      basis_labels = np.array(["photon_plus","photon_minus","elec_dipole","tmd_K","tmd_prime_K"])):
    '''

    :param eigenvecs:
    :param full_basis:
    :param basis_labels:
    :return:
    '''
    photon_indices = get_basis_indices_from_labels(basis_labels,"photon")
    tmd_indices = get_basis_indices_from_labels(basis_labels,"tmd")
    k_indices = get_basis_indices_from_labels(basis_labels, "tmd_K")
    k_prime_indices = get_basis_indices_from_labels(basis_labels, "tmd_prime_K")
    elec_dipole_indices=  get_basis_indices_from_labels(basis_labels,"elec")
    electron_indices = np.hstack((elec_dipole_indices,tmd_indices))
    photon_weights,k_weights = get_weights_from_indices(full_basis,photon_indices,k_indices)
    photon_weights, k_prime_weights = get_weights_from_indices(full_basis, photon_indices, k_prime_indices)
    photon_weights,elec_dipole_weights = get_weights_from_indices(full_basis,photon_indices,elec_dipole_indices)
    photonic_occupancy = state_occupancy_from_eigenvecs(eigenvecs,np.nonzero(photon_weights)[0])
    k_occupancy = state_occupancy_from_eigenvecs(eigenvecs,np.nonzero(k_weights)[0])
    k_prime_occupancy = state_occupancy_from_eigenvecs(eigenvecs,np.nonzero(k_prime_weights)[0])
    elec_dipole_occupancy = state_occupancy_from_eigenvecs(eigenvecs,np.nonzero(elec_dipole_weights)[0])
    hel_pol_characteristic = helical_polaritonic_characteristic_v2_from_eigenvecs(eigenvecs,full_basis,photon_indices,electron_indices)
    return photonic_occupancy,k_occupancy,k_prime_occupancy,elec_dipole_occupancy,hel_pol_characteristic

def mask_for_rows(shape,indices):
    truth_array = np.zeros(shape,dtype = bool)
    if (np.isscalar(indices)):
        indices = np.array(indices)
    for i in range(0,np.size(indices)):
        truth_array[indices[i],:] = np.ones(shape[1],dtype = bool)
    return truth_array

#reads eigenvectors, gets occupancies, then assigns states to each eigenvector
#among the chosen states
#for instance, say you have a basis of [a,b,c] and you want to distinguish between
#b and c but don't care about a (perhaps b and c are different electronic states
#while a is the photonic component); this tells you which vectors are b or c
def eigenvector_state_assignment(eigenvecs,state_indices):
    n_states = np.size(state_indices)
    states_to_compare = np.zeros((n_states,np.size(eigenvecs,axis = 1),np.size(eigenvecs,axis=2)))
    for i in range(n_states):
        states_to_compare[i,:,:] = state_occupancy_from_eigenvecs(eigenvecs,state_indices[i])
    return state_indices[np.argmax(states_to_compare,axis =0)]

# eigenvecs.ndim == 3, characterstics.ndim ==2 as it is a truncation that assigns
# a single value (typically -1 to 1 or -.5 to .5) to each vector
def separate_eigenvector_characteristics_by_states(eigenvecs,characteristics,state_indices):
    states_assigments = eigenvector_state_assignment(eigenvecs,state_indices)
    masked_separated_chars = np.zeros((np.size(characteristics,axis = 0),np.size(characteristics,axis = 1),np.size(state_indices)))
    mask_for_separated_chars = np.zeros(np.shape(masked_separated_chars),dtype = bool)
    for i in range(0,np.size(state_indices)):
        cur_state = state_indices[i]
        #masks all states except current state
        masked_separated_chars[:,:,i] = np.copy(characteristics)
        #ensures that polariton masking cascades into state assignment masking
        if (np.ma.is_masked(characteristics)==True):
            mask_for_separated_chars[:, :, i] = np.logical_or(states_assigments != cur_state,np.ma.getmask(characteristics))
        else:
            mask_for_separated_chars[:,:,i] = states_assigments!=cur_state
    return np.ma.masked_array(masked_separated_chars,mask_for_separated_chars)

def mask_eigenvector_characteristic_by_state_threshold(eigenvecs,characteristics,state_indices,threshold = .95):
    if (np.isscalar(state_indices)):
        state_indices = np.array([state_indices])
    for i in range(0,np.size(state_indices)):
        if (i == 0):
            occupancy = state_occupancy_from_eigenvecs(eigenvecs,state_indices[i])
        else:
            occupancy = occupancy+state_occupancy_from_eigenvecs(eigenvecs,state_indices[i])
    mask = occupancy<threshold #so that values above threshold will be masked
    return np.ma.masked_array(characteristics,mask)

#assumes eigenvecs in 3D--will update at some point
def mixed_state_metric(fixed_eigenvecs):
    probabilities=  np.abs(fixed_eigenvecs)**2
    num_states = np.size(probabilities,axis = 0)
    mixed_state_metric_value = np.zeros((np.size(probabilities,axis = 1),np.size(probabilities,axis =2)))
    for i in range(0,num_states):
        cur_probability = probabilities[i,:,:]
        all_other_probabilities= np.sum(probabilities,axis = 0)-cur_probability
        mixed_state_metric_value = mixed_state_metric_value+cur_probability*all_other_probabilities
    return mixed_state_metric_value

def mixed_electronic_state_characteristic_from_eigenvecs(eigenvecs,full_basis,electronic_indices):
    eigenvec_dimension = np.ndim(eigenvecs)
    elec_types = np.size(electronic_indices)
    num_states = np.size(full_basis, axis=0)
    elec_weights = np.zeros(num_states)
    for i in range(elec_types):
        elec_weights = elec_weights + basis_weighting(full_basis, electronic_indices[i], boolean_weighting=True,elec_style=True)
    elec_weights = np.array(elec_weights, dtype=np.float64)
    if (eigenvec_dimension == 2):
        elec_eigenvecs = np.einsum("i,ij->ij", elec_weights, eigenvecs)
    elif (eigenvec_dimension == 3):
        elec_eigenvecs = np.einsum("i,ijk->ijk", elec_weights, eigenvecs)
    return mixed_state_metric(elec_eigenvecs)

#energy_matrix gives energy states for some parameters of the matrix (or tensor) to mask
def mask_over_spectral_energies(matrix_to_mask,energy_matrix,energy_bounds,indices_to_ignore = None):
    energy_truth_lb  = energy_matrix >energy_bounds[0]
    energy_truth_ub = energy_matrix<energy_bounds[1]
    energy_truth_bounded = energy_truth_ub*energy_truth_lb
    #this extends energy mask over the indices of te matrix that you want to mask
    #e.g. matrix is of size (2,10,10) and energy is (10,10), indices to ignore will be 0
    #reshapes energy to (2,10,10) with proper truth values
    if (indices_to_ignore is not None):
        indices_to_ignore_arr = np.sort(pu.arb_to_array(indices_to_ignore))
        for i in range(0,np.size(indices_to_ignore_arr)):
            repeats = np.size(matrix_to_mask,axis = indices_to_ignore_arr[i])
            energy_truth_bounded = pu.repeat_array_along_new_axis(energy_truth_bounded,repeats,indices_to_ignore_arr[i])
    #Note that where bounds are true, matrix should be UNMASKED, hence the inversion
    return np.ma.masked_array(matrix_to_mask,mask = ~energy_truth_bounded)

def second_dipole_mixing_function(gl1,gl2,gr1,gr2):
    return (gl2*gl1+gr2*gr1)/(1-gl2**2-gr2**2)

def second_dipole_mixing_function_from_matrix(coupling_matrix):
    return second_dipole_mixing_function(coupling_matrix[0,:],coupling_matrix[1,:],coupling_matrix[2,:],coupling_matrix[3,:])


#Analytic handling of eigenvector, eigenvalues
class FOUR_EIGENVECS_ANAL():
    #parameter array is the array which characterizes the parameter sweep of the coupling values
    #in general, this is cavity frequency (eV), but it is written to be more general
    #final index of couplings is the parameter array
    # copulings are l1,l2,r1,r2
    def __init__(self,parameter_array,couplings,eigenvalues_region_array):
        self.x_array = parameter_array
        self.y_array = eigenvalues_region_array
        self.gl1 = couplings[0,:]
        self.gl2 = couplings[1,:]
        self.gr1 = couplings[2,:]
        self.gr2 = couplings[3,:]
    def create_eigenvectors(self,energy_cav,energy_2):
        eigenvectors = np.zeros((4,np.size(self.x_array),np.size(self.y_array)))
        mixing_factor = second_dipole_mixing_function(self.gl1,self.gl2,self.gr1,self.gr2)
        mixing_tiled = np.tile(mixing_factor,(np.size(self.y_array),1)).T
        gl1_tiled = np.tile(self.gl1,(np.size(self.y_array),1)).T
        gl2_tiled = np.tile(self.gl2, (np.size(self.y_array), 1)).T
        gr1_tiled = np.tile(self.gr1, (np.size(self.y_array), 1)).T
        gr2_tiled = np.tile(self.gr2, (np.size(self.y_array), 1)).T
        e_cav_tiled = np.tile(energy_cav,(np.size(self.y_array),1)).T
        energy_2_tiled=  np.tile(np.ones(np.size(self.x_array))*energy_2,(np.size(self.y_array),1)).T
        eigenvals_tiled = np.tile(self.y_array,(np.size(self.x_array),1))
        eigenvectors[2, :, :] = np.ones((np.size(self.x_array), np.size(self.y_array)))
        eigenvectors[3,:,:] =mixing_tiled/((eigenvals_tiled-e_cav_tiled)*(eigenvals_tiled-energy_2_tiled))
        eigenvectors[0,:,:] = (gl1_tiled+gl2_tiled*eigenvectors[3,:,:])/(eigenvals_tiled-e_cav_tiled)
        eigenvectors[1, :, :] = (gr1_tiled+gr2_tiled*eigenvectors[3,:,:])/(eigenvals_tiled-e_cav_tiled)
        norm = np.sqrt(np.sum(eigenvectors*eigenvectors.conjugate(),axis=0))
        norm_tiled = np.tile(norm,(4,1,1))
        return eigenvectors/norm_tiled
    # symbolic handling
import sympy as sym

def create_hamiltonian_two_dipoles():
    e_cav, e_1, e_2, g_1l, g_1r, g_2l, g_2r = sym.symbols("E_{cav} E_1 E_2 g_1^l g_1^r g_2^l g_2^r")
    return sym.Matrix([[e_cav,0,g_1l,g_2l],[0,e_cav,g_1r, g_2r],[g_1l.conjugate(),g_1r.conjugate(),e_1,0],[g_2l.conjugate(),g_2r.conjugate(),0,e_2]])

def create_hamiltonian_one_dipole():
    e_cav, e_1, g_l, g_r = sym.symbols("E_{cav} E_1  g_l g_r")
    return sym.Matrix([[e_cav,0,g_l],[0,e_cav,g_r],[g_l.conjugate(),g_r.conjugate(),e_1]])

def create_hamiltonian_one_dipole_real():
    e_cav, e_1, g_l, g_r = sym.symbols("E_{cav} E_1  g_l g_r")
    return sym.Matrix([[e_cav,0,g_l],[0,e_cav,g_r],[g_l,g_r,e_1]])

def solve_sym_matrix(matrix):
    vals = matrix.eigenvals()
    vecs = matrix.eigenvects()
    return vals, vecs

def extract_piecewise_otherwise_solutions(eigen_solutions):
    eigen_energies = list(eigen_solutions.keys())
    solutions = []
    for i in range(0,len(eigen_energies)):
        piecewise = eigen_energies[i]
        solutions.append(piecewise.args[1][0])
    return solutions

def produce_two_dipole_eigenvalues():
    e_avg, alpha,beta, gamma = sym.symbols(r"E_{avg} \alpha \beta \gamma")
    lambda_1 = e_avg-alpha+sym.sqrt(beta+gamma)/2
    lambda_2 =  e_avg-alpha-sym.sqrt(beta+gamma)/2
    lambda_3 =  e_avg+alpha+sym.sqrt(beta-gamma)/2
    lambda_4 =  e_avg+alpha-sym.sqrt(beta-gamma)/2
    return [lambda_1,lambda_2,lambda_3,lambda_4]

# perturbative
#note that dielectric_params must be of length = 1
#also, mean interaction length is an INVERSE LENGTH
import scipy.optimize as sp
#args is the polarizance object
def numeric_mean_int_length(z,*polarizance_sel_tuple):
    polarizance_sel = polarizance_sel_tuple[0]
    loss = polarizance_sel_tuple[1]
    a0,a1 = brown_two_fast(polarizance_sel,z)
    d1,d2,d3,b1,b2,b3= polarizance_sel.provide_tuples()
    m00 = a0+a1*(d1**2+d2**2+d3**2)
    function_to_zero = (1+np.log(m00))/(polarizance_sel.absorbance+loss)-z
    return function_to_zero

def intensity_fraction(z,polarizance_sel,loss =0):
    m00 = m00_quick(z,polarizance_sel)
    intensity_frac =  np.exp(-(polarizance_sel.absorbance*z))*m00*np.exp(-loss*z)
    return intensity_frac

def m00_quick(z,polarizance_sel):
    a0, a1 = brown_two_fast(polarizance_sel, z)
    d1, d2, d3, b1, b2, b3 = polarizance_sel.provide_tuples()
    m00 = a0 + a1 * (d1 ** 2 + d2 ** 2 + d3 ** 2)
    return m00

def solve_mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,loss=0):
    initial_solutions = 1/(lpp.get_lin_abs_anal(spectrum,dielectric_params,energy_array,dip_mags,dip_angles)+loss)
    dipole_matrix = dt.create_dipole_matrix_polar_3D(dip_mags,dip_angles)
    polarizance_total = get_polarizance_params_dielectric(dielectric_params,dipole_matrix,energy_array,spectrum)
    final_solutions = np.zeros(np.size(initial_solutions))
    for i in range(0,np.size(initial_solutions)):
        polarizance_selected = polarizance_total.select_by_index(i)
        polarizance_tuple = (polarizance_selected,loss)
        roots = sp.fsolve(numeric_mean_int_length,initial_solutions[i],args=polarizance_tuple)
        #print(str(roots))
        intensity_frac = intensity_fraction(roots[0],polarizance_selected,loss)
        print(str(i)+":"+str(intensity_frac))
        if (.49 <intensity_frac <.51):
            final_solutions[i] = roots[0]
        else:
            final_solutions[i] = initial_solutions[i]
    return final_solutions

#loss is in terms of loss_percent_per_cycle/2*film_thickness (.01 is a fine test paramter, corresponding to 1% loss per round trip) and a film ~500 nm
def mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,style = "full_anal",loss = 0):
    '''
    :param spectrum:
    :param dielectric_params:
    :param energy_array:
    :param dip_mags:
    :param dip_angles:
    :param style:
    :param loss: l_%/2l where l is film thickness
    :return:
    '''
    if (dielectric_params.length>= 1.01 or dielectric_params.length<= 0.99):
        ValueError("Dielectric parameters must be length independent, indicated by length = 1")
    #fully perturbative--the expression in the manuscript
    if (style == "full_anal"):
        dipole_operators = dip_mags #you don't need to account for isotropic averaging as int cos^2(x) = 1/2 for a full cycle as 2k = abs, which cancels out that factor
        mean_int_length = 1/(abs_from_operator(dielectric_params,energy_array,spectrum,dipole_operators,length = 1)+loss)
    #refractive index tensor is calculated numerically, but the rest is analytic
    elif (style == "anal"):
        mean_int_length = 1/(lpp.get_lin_abs_anal(spectrum,dielectric_params,energy_array,dip_mags,dip_angles)+loss)
    elif (style == "full_numeric"):
        mean_int_length = solve_mean_interaction_length(spectrum,dielectric_params,energy_array,dip_mags,dip_angles,loss)
    else:
        ValueError("Style not found")
    return mean_int_length
# general, accounts for m00 shift
def mean_interaction_length_mueller(mueller_matrix):
    mat_dim = mueller_matrix.ndim()
    if (mat_dim ==2):
        iso_abs = mueller_matrix[0,0]
    elif(mat_dim ==3 ):
        iso_abs = mueller_matrix[0,0,:]
    return iso_abs

def select_from_energies(value_array,spectrum,energy_array):
    if (np.size(value_array) != np.size(spectrum)):
        ValueError("Spectrum and array must have equal size")
    selected_array = np.zeros(np.size(energy_array),dtype = value_array.dtype)
    if (np.isscalar(energy_array)):
        energy_array = np.array([energy_array])
    for i in range(0,np.size(energy_array)):
        selected_array[i] = value_array[np.argmin(np.abs(spectrum-energy_array[i]))]
    return selected_array

#in terms of physical, not phase space parameters
class THREE_STATE_ANAL_POLARITONS():
    def __init__(self,achiral_coupling,chiral_factor,detuning):
        self.delta = detuning
        self.phi = achiral_coupling
        self.sigma = chiral_factor
    def sqrt_chiral_ratio(self):
        return np.sqrt((1-self.sigma)/(1+self.sigma))
    def gamma_factor(self,sign = 1):
        return 1/(self.delta/2*(1+sign*np.sqrt(1+8*self.phi**2/self.delta**2)))
    def photonic_state(self):
        norm = 1/np.sqrt(1+self.sqrt_chiral_ratio())
        return norm*np.asarray((self.sqrt_chiral_ratio(),np.ones(np.size(self.sqrt_chiral_ratio())),np.zeros(np.size(self.sqrt_chiral_ratio()))))
    def upper_polariton_norm(self):
        return np.cos(1/2*np.arctan(2*np.sqrt(2)*self.phi/self.delta))
    def lower_polariton_norm(self):
        return np.sin(1/2*np.arctan(2*np.sqrt(2)*self.phi/self.delta))
    def upper_polariton(self):
        norm = self.upper_polariton_norm()
        gamma_plus = self.gamma_factor(sign = 1)
        return norm*np.asarray((gamma_plus*self.phi*np.sqrt(1+self.sigma),gamma_plus*self.phi*np.sqrt(1-self.sigma),np.ones(np.size(gamma_plus))))
    def lower_polariton(self):
        norm = self.lower_polariton_norm()
        gamma_minus = self.gamma_factor(sign=-1)
        return norm * np.asarray(
            (gamma_minus * self.phi * np.sqrt(1 + self.sigma), gamma_minus* self.phi * np.sqrt(1 - self.sigma), np.ones(np.size(gamma_minus))))
    def all_states(self):
        return self.photonic_state(),self.lower_polariton(),self.upper_polariton()
    def detuning_ratio(self):
        return self.delta/(2*np.sqrt(2)*self.phi)
    def mixed_state_metric_anal(self):
        chi_ratio = self.detuning_ratio()
        return np.abs(1/(2*chi_ratio*np.sqrt(1+1/(chi_ratio**2))))

def three_state_basis():
    basis = combo.construct_organic_basis(1,2)
    basis_no_zero_state = basis[1:,:]
    photonic_indices = [0,1]
    electronic_indices = [2]
    return basis_no_zero_state,photonic_indices,electronic_indices

def q_factor(round_trip_loss,mode_number=1):
    '''
    Simple manner to calculate resonant q factor from loss
    :param round_trip_loss:
    :param mode_number:
    :return:
    '''
    return 2*np.pi*mode_number/round_trip_loss

def round_trip_loss(q_factor,mode_number = 1):
    '''
    Simple manner to
    :param q_factor:
    :param mode_number:
    :return:
    '''
    return 2 * np.pi * mode_number / q_factor