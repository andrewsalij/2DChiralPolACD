import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import brioullinzonebasefinal as base
import combinatorics as combo
import time
import numpy.ma as ma
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class EXCITONIC_VARS:
    '''
    Data transfer class for excitonic variables
    '''
    def __init__(self,e_x_bare,a_x_bare,p_minus,p_plus,p_minus_vv,p_minus_cc,p_plus_vv,p_plus_cc,p_x,p_x_vv,p_x_cc,p_minus_cv,p_plus_cv,p_x_cv):
        '''
        Inputs are set of np.ndarrays over a desired k-space and number of states
        See base.create_polarized_momentum_matrices() and create_momentum_matrix_set() for details on momentum matrix construction
        :param e_x_bare np.ndarray
        Energies of excitonic states
        :param a_x_bare: np.ndarray
        Eigenvectors before cavity coupling
        :param p_minus: np.ndarray
        :param p_plus: np.ndarray
        :param p_minus_vv: np.ndarray
        :param p_minus_cc: np.narray
        :param p_plus_vv: np.narray
        :param p_plus_cc: np.ndarray
        :param p_x: np.ndarray
        :param p_x_vv: np.ndarray
        :param p_x_cc: np.ndarray
        :param p_minus_cv: np.ndarray
        :param p_plus_cv: np.ndarray
        :param p_x_cv: np.ndarray
        '''
        self.e_x = e_x_bare
        self.a_x = a_x_bare
        self.p_minus = p_minus
        self.p_plus = p_plus
        self.p_minus_vv = p_minus_vv
        self.p_minus_cc = p_minus_cc
        self.p_plus_vv = p_plus_vv
        self.p_plus_cc = p_plus_cc
        self.p_plus_cv = p_plus_cv
        self.p_minus_cv = p_minus_cv
        self.p_x_cv = p_x_cv
        self.p_x = p_x
        self.p_x_vv = p_x_vv
        self.p_x_cc= p_x_cc

class MATRIX_ELEMENTS:
    '''Data transfer class for coupling matrix elements'''
    def __init__(self,m_nm_plus,m_nm_minus,m_nm_x):
        '''
        :param m_nm_plus:
        :param m_nm_minus:
        :param m_nm_x:
        '''
        self.m_plus = m_nm_plus
        self.m_minus = m_nm_minus
        self.m_x = m_nm_x

class M_0N_ARRAYS():
    '''Data transfer class for photonic-electronic (i.e., not the mixing) elements'''
    def __init__(self,m_0n_plus,m_0n_minus,m_0n_x):
        '''
        :param m_0n_plus: np.ndarray
        :param m_0n_minus: np.ndarray
        :param m_0n_x: np.ndarray
        '''
        self.m_plus = m_0n_plus
        self.m_minus = m_0n_minus
        self.m_x = m_0n_x

class ORGANIC_PARAMS:
    '''Data transfer class for'''
    def __init__(self,rabi_freq_org,org_elec_freq,transition_matrix):
        '''
        :param rabi_freq_org: float
        :param org_elec_freq: float
        :param transition_matrix: np.ndarray
        '''
        self.rabi = rabi_freq_org
        self.elec = org_elec_freq
        self.m_elem_matrix = transition_matrix

def init_momentum_int_arrays(p_minus,p_plus,angle,**kwargs):
    '''
    Initializes momentum integral arrays
    :param p_minus: np.ndarray 
    :param p_plus: np.ndarray
    :param angle: float
    :param kwargs: dict
    :return:
    '''
    to_truncate = False
    to_toggle = False
    init_zeros = 2 #to correspond to two blank photon modes
    for key, value in kwargs.items():
        if key == "trunc":
            to_truncate = True
            init_width = value
        if key == "blank_initial":
            init_zeros = value
        if key == "toggle":
            to_toggle = True
    if (not to_truncate):
        init_width = np.size(p_minus)
    minus_factor = np.cos(angle)
    # since np.cos(np.pi/2) is not exactly zero
    if (minus_factor < 1e-12):
        minus_factor = 0
    plus_factor = np.sin(angle)
    if (plus_factor < 1e-12):
        plus_factor = 0
    if (to_toggle):
        if (minus_factor >0):
            p_minus=  p_minus
        if (plus_factor > 0):
            p_plus = p_plus
    else:
        p_minus = minus_factor* p_minus
        p_plus = plus_factor * p_plus  # in momentum space, p_minus is x-axis and p_plus is y-axis
    p_minus_int_array = np.zeros(init_width + init_zeros, dtype=np.csingle)
    p_plus_int_array = np.zeros(init_width + init_zeros, dtype=np.csingle)
    return p_minus, p_plus, p_minus_int_array, p_plus_int_array


def create_momentum_matrix_set(valence_vec_array,conduction_vec_array,tau_array,a,t):
    '''
    Creates momentum matrix set
    :param valence_vec_array: 
    :param conduction_vec_array: 
    :param tau_array: 
    :param a: 
    :param t: 
    :return: 
    '''
    p_vc_x, p_vc_y = base.create_momentum_matrix(valence_vec_array,conduction_vec_array,tau_array,a,t)
    p_cv_x, p_cv_y = base.create_momentum_matrix(conduction_vec_array,valence_vec_array,tau_array,a,t)
    p_cc_x, p_cc_y = base.create_momentum_matrix(conduction_vec_array,conduction_vec_array,tau_array,a,t)
    p_vv_x, p_vv_y = base.create_momentum_matrix(valence_vec_array,valence_vec_array,tau_array,a,t)
    p_vc = np.row_stack((p_vc_x,p_vc_y))
    p_cv =  np.row_stack((p_cv_x,p_cv_y))
    p_cc = np.row_stack((p_cc_x,p_cc_y))
    p_vv = np.row_stack((p_vv_x,p_vv_y))
    return p_vc,p_cv, p_vv, p_cc

def excitonic_matrix_element_zero(a_n_array,p_array):
    '''Creates M_0n element'''
    return np.sum(p_array*a_n_array)


def excitonic_matrix_array_zero(a_n_matrix,p_array):
    '''Creates M_0n array'''
    m_array = np.zeros(np.size(a_n_matrix,1),dtype = np.csingle)
    for i in range(0,np.size(m_array)):
        m_array[i] = excitonic_matrix_element_zero(a_n_matrix[:,i],p_array)
    return m_array

def excitonic_matrix_element_nonzero(a_n_array,a_m_array,p_vv_array,p_cc_array,fit_param):
    '''Creates M_nm element'''
    m_elem = fit_param*np.sum(a_m_array.conj()*a_n_array*(p_cc_array-p_vv_array))
    return m_elem

def excitonic_matrix_array_nonzero(a_n_matrix,p_vv_array,p_cc_array,fit_param):
    '''Creates M_nm array'''
    m_matrix = np.zeros((np.size(a_n_matrix,1),np.size(a_n_matrix,1)),dtype = np.csingle)
    for i in range(0,np.size(m_matrix,0)):
        for j in range(0,np.size(m_matrix,1)):
            m_matrix[i,j] = excitonic_matrix_element_nonzero(a_n_matrix[:,i],a_n_matrix[:,j],p_vv_array,p_cc_array,fit_param)
    return m_matrix

def coupling_matrix_mask(basis_set,spin,photonic_state,coupling_matrix):
    '''
    Provides a boolean masking array for the coupling matrix
    :param basis_set: np.narray
    :param spin: int (-1 or 1)
    :param photonic_state: np.array
    :param coupling_matrix: np.narray
    :return:
    '''
    coupling_matrix_mask = np.zeros((np.size(coupling_matrix,0),np.size(coupling_matrix,1)))
    for a in range(0,np.size(basis_set,0)):
        for b in range(0,np.size(basis_set,0)):
            state_m =  basis_set[a,:]
            state_n = basis_set[b,:]
            spin_m = find_spin(basis_set[a,:],2,3)
            spin_n = find_spin(basis_set[b,:],2,3)
            if (np.array_equal(state_m[:2],photonic_state) and np.array_equal(state_n[:2],photonic_state)):
                if ((spin_m == spin or spin_m ==0) and (spin_n == spin or spin_n ==0)):
                    coupling_matrix_mask[b,a] = coupling_matrix[b,a]
    return coupling_matrix_mask

def polaritonic_matrix_array(c_x_matrix,m_0n_array):
    '''
    Construction of \Psi(0) after excitation
    :param c_x_matrix:
    :param m_0n_array:
    :return:
    '''
    m_array = np.einsum("i,ij->j",m_0n_array,c_x_matrix)
    return m_array

def linear_absorption_polaritonic(energy_spectrum,e_i_array,m_pol_array):
    '''
    Calculates and returns linear absorption spectrum
    :param energy_spectrum:
    :param e_i_array:
    :param m_pol_array:
    :return:
    '''
    num_states = np.size(m_pol_array)
    intensity_array = np.zeros(np.size(energy_spectrum))
    vec_pot = 1 # in eV/(ec)
    for i in range(0, num_states):
        oscillator_strength = (vec_pot)**2*2*np.pi*np.abs(m_pol_array[i])**2
        peak_shape = base.dirac_delta(energy_spectrum, e_i_array[i],**{"width":.01})
        intensity_array = intensity_array + oscillator_strength*peak_shape
    return np.abs(intensity_array)



def jc_diagonal_term(basis_state,e_x_up,e_x_down,cav_freq):
    '''
    Diagonal element for Jaynes-Cummings Hamiltonian
    :param basis_state:
    :param e_x_up:
    :param e_x_down:
    :param cav_freq:
    :return:
    '''
    n_plus ,n_minus= basis_state[0],basis_state[1]
    up_index = int(basis_state[2])
    down_index = int(basis_state[3])
    h_el = 0
    if (up_index>=0):
        h_el = h_el+e_x_up[up_index]
    if (down_index>= 0):
        h_el = h_el+e_x_down[down_index]
    term = h_el+(cav_freq)*(n_plus+n_minus)
    #final term is a self-energy term--generally omitted as photonic states are [0,0] in the number of quanta used
    return term

def init_all_momenta_arrays(p_minus,p_minus_vv,p_minus_cc,p_plus,p_plus_vv,p_plus_cc,angle,**kwargs):
    '''Initializes all momenta arrays'''
    p_minus, p_plus, a, b = init_momentum_int_arrays(p_minus,p_plus,angle,**kwargs)
    p_minus_vv, p_plus_vv, a, b = init_momentum_int_arrays(p_minus_vv, p_plus_vv, angle, **kwargs)
    p_minus_cc, p_plus_cc, a, b = init_momentum_int_arrays(p_minus_cc, p_plus_cc, angle, **kwargs)
    return p_minus,p_minus_vv,p_minus_cc, p_plus,p_plus_vv,p_plus_cc

def find_spin(state,up_index,down_index):
    '''Gives total spin index of a state'''
    if (state.ndim == 2):
        n_up = (state[:,up_index] >= 0).astype(int)*1
        n_down = (state[:,down_index] >= 0).astype(int)*-1
    else:
        n_up = (state[up_index] >= 0).astype(int) * 1
        n_down = (state[down_index] >= 0).astype(int) * -1
    return n_up+n_down

def ladder_up(a,b,index):
    '''Creation operator between two (a and b) states'''
    if (np.array_equal(np.delete(a,index),np.delete(b,index))):
        if (a[index] == b[index]+1):
            return np.sqrt(a[index])
    return 0

def double_ladder_up(a,b,index1,index2):
    '''Creation operator over two indices betwen two (a and b) states '''
    if (np.array_equal(np.delete(a,[index1,index2]),np.delete(b,[index1,index2]))):
        if (a[index1] == b[index1]+1 and a[index2] == b[index2]+1):
            return np.sqrt(a[index1])*np.sqrt(a[index2])
    return 0
def ladder_down(a,b,index):
    '''Annihilation operator between two (a and b) states'''
    if (np.array_equal(np.delete(a,index),np.delete(b,index))):
        if (a[index] == b[index]-1):
            return np.sqrt(b[index])
    return 0
def double_ladder_down(a,b,index1,index2):
    '''Annihilation operator over two indices betwen two (a and b) states '''
    if (np.array_equal(np.delete(a,[index1,index2]),np.delete(b,[index1,index2]))):
        if (a[index1] == b[index1]-1 and a[index2] == b[index2]-1):
            return np.sqrt(b[index1])*np.sqrt(b[index2])
    return 0

def number_operator(a,b,index):
    '''Number operator over two (a and b) states'''
    if (np.array_equal(np.delete(a,index),np.delete(b,index))):
        if (a[index] == b[index]):
            return a[index]
    return 0
def create_state_tensor(photon_basis,axis):
    '''
    Creates photonic state tensor
    axis inserts along chosen axis
    :param photon_basis:
    :param axis:
    :return:
    '''
    states = np.size(photon_basis,0)
    ones = np.ones(states)
    if axis == 1:
        photon_tensor = np.einsum('ij,k->ikj',np.asarray(photon_basis,dtype= int),
                                  np.asarray(ones,dtype=int))
    if axis == 0:
        photon_tensor = np.einsum('ij,k->kij',np.asarray(photon_basis,dtype= int),
                                  np.asarray(ones,dtype=int))
    return photon_tensor


def photons_matrix_relation(photon_tensor_n,photon_tensor_m,desired_difference,relation_type = "ladder"):
    '''
    desired difference incorporates some sense of ladder operators or equality
    that is the ladder operator for a state |1> is a^\dag|elec> = > |\gamma>
    generalized and made more robust in state_matrix_relation()
    :param photon_tensor_n: np.ndarray
    :param photon_tensor_m:
    :param desired_difference:
    :param relation_type: str
    "bool" returns boolean matrix
    Any other value returns ladder operation
    :return:
    '''
    # m is initial, n is final
    diff_tensor = photon_tensor_n-photon_tensor_m
    diff_matrix = ~np.sum(np.abs(diff_tensor-desired_difference),axis= 2).astype(bool)
    if relation_type == "bool":
        return diff_matrix
    else:
        return_tensor = np.ones(np.shape(photon_tensor_m))
        if (desired_difference[0] == 1):
            return_tensor[:,:,0] = np.sqrt(photon_tensor_n[:,:,0])
        if (desired_difference[1] == 1):
            return_tensor[:, :, 1] = np.sqrt(photon_tensor_n[:, :, 1])
        if (desired_difference[0] == -1):
            return_tensor[:,:,0] = np.sqrt(photon_tensor_m[:,:,0])
        if (desired_difference[1] == -1):
            return_tensor[:, :, 1] = np.sqrt(photon_tensor_m[:, :, 1])
        return_tensor = return_tensor[:,:,0]*return_tensor[:,:,1]*diff_matrix
        return return_tensor
def state_matrix_relation(state_tensor_n,state_tensor_m,desired_difference,basis_type):
    '''

    :param state_tensor_n:
    :param state_tensor_m:
    :param desired_difference:
    :param basis_type: np.ndarray
    array of ladder relation,
    "ladder", meaning standard creation/annihilation, e.g., sqrt(n)|n>
    "bool", meaning to return "1" for any valid index (used for transition indices of many potential states)
    :return:
    '''
    # m is initial, n is final, final axis is basis
    #converting states into appropriate index
    cop_state_tensor_n = np.copy(state_tensor_n)
    cop_state_tensor_m = np.copy(state_tensor_m)
    for i in np.arange(np.size(basis_type)):
        if basis_type[i] == "bool": #-1 to 0, values 0 onwards -> 1
            cop_state_tensor_n[:,:,i] = (cop_state_tensor_n[:,:,i]+1).astype(bool).astype(float)
            cop_state_tensor_m[:,:,i] = (cop_state_tensor_m[:,:,i]+1).astype(bool).astype(float)
    diff_tensor = cop_state_tensor_n-cop_state_tensor_m
    diff_matrix = ~np.sum(np.abs(diff_tensor-desired_difference),axis= 2).astype(bool)
    sel_matrix = diff_matrix.astype(np.csingle)
    sel_state_tensor_m = np.einsum("ij,ijk->ijk", sel_matrix, cop_state_tensor_m)
    return_tensor = np.ones(np.shape(diff_matrix))*sel_matrix
    for i in np.arange(np.size(desired_difference)):
        cur_basis_diff = desired_difference[i]
        cur_sel_state_m = np.copy(sel_state_tensor_m[:,:,i]) #init selected state
        if (cur_basis_diff>0):
            counter = 0
            while (counter < cur_basis_diff):
                cur_sel_state_m = cur_sel_state_m+1*sel_matrix
                return_tensor = return_tensor*np.sqrt(cur_sel_state_m)
                counter = counter+1
        elif (cur_basis_diff<0):
            counter = 0
            while (counter < np.abs(cur_basis_diff)):
                return_tensor = return_tensor * np.sqrt(cur_sel_state_m)
                cur_sel_state_m = cur_sel_state_m-1*sel_matrix
                counter = counter+1
    return return_tensor


def chiral_a0(a0,polarization):
    '''Gives tuple of vector potentail multiplied by some polarization state of size 2'''
    return a0*polarization[0],a0*polarization[1]

def matrix_lower_bound_mask(matrix,bound_ratio):
    '''Returns matrix with values below some bound set to 0.
    To see if very weak couplings have an appreciable effect (they don't).'''
    matrix_max = np.max(matrix)
    mask_indices = np.abs(matrix) < np.abs(matrix_max*bound_ratio)
    matrix[mask_indices] = 0
    return matrix

def create_tmd_m_elem_matrices(a_x_up,a_x_down,p_plus_up,p_minus_up,p_x_up,p_plus_down,p_minus_down,p_x_down,p_plus_vv_up,p_plus_cc_up,p_minus_vv_up,p_minus_cc_up,\
                               p_plus_vv_down,p_plus_cc_down,p_minus_vv_down,p_minus_cc_down,p_x_vv_up,p_x_cc_up,p_x_vv_down,p_x_cc_down,m_nm_fit_param,
                               to_mask_coupling_lb = False,mask_factor = 1e-5):
    '''
    Creates all TMD coupling matrix element arrays
    :param a_x_up:
    :param a_x_down:
    :param p_plus_up:
    :param p_minus_up:
    :param p_x_up:
    :param p_plus_down:
    :param p_minus_down:
    :param p_x_down:
    :param p_plus_vv_up:
    :param p_plus_cc_up:
    :param p_minus_vv_up:
    :param p_minus_cc_up:
    :param p_plus_vv_down:
    :param p_plus_cc_down:
    :param p_minus_vv_down:
    :param p_minus_cc_down:
    :param p_x_vv_up:
    :param p_x_cc_up:
    :param p_x_vv_down:
    :param p_x_cc_down:
    :param m_nm_fit_param:
    :param to_mask_coupling_lb:
    :param mask_factor:
    :return:
    '''
    m_0n_plus_up = excitonic_matrix_array_zero(a_x_up, p_plus_up)
    m_0n_minus_up = excitonic_matrix_array_zero(a_x_up, p_minus_up)
    m_0n_x_up = excitonic_matrix_array_zero(a_x_up, p_x_up)
    m_0n_plus_down = excitonic_matrix_array_zero(a_x_down, p_plus_down)
    m_0n_minus_down = excitonic_matrix_array_zero(a_x_down, p_minus_down)
    m_0n_x_down = excitonic_matrix_array_zero(a_x_up, p_x_down)
    m_nm_plus_up = excitonic_matrix_array_nonzero(a_x_up, p_plus_vv_up, p_plus_cc_up, m_nm_fit_param).T
    m_nm_minus_up = excitonic_matrix_array_nonzero(a_x_up, p_minus_vv_up, p_minus_cc_up, m_nm_fit_param).T
    m_nm_x_up = excitonic_matrix_array_nonzero(a_x_up, p_x_vv_up, p_x_cc_up, m_nm_fit_param).T
    m_nm_plus_down = excitonic_matrix_array_nonzero(a_x_down, p_plus_vv_down, p_plus_cc_down, m_nm_fit_param).T
    m_nm_minus_down = excitonic_matrix_array_nonzero(a_x_down, p_minus_vv_down, p_minus_cc_down, m_nm_fit_param).T
    m_nm_x_down = excitonic_matrix_array_nonzero(a_x_down, p_x_vv_down, p_x_cc_down, m_nm_fit_param).T
    if (to_mask_coupling_lb):
        m_0n_plus_up = matrix_lower_bound_mask(m_0n_plus_up, mask_factor)
        m_0n_minus_up = matrix_lower_bound_mask(m_0n_minus_up, mask_factor)
        m_0n_x_up = matrix_lower_bound_mask(m_0n_x_up, mask_factor)
        m_0n_plus_down = matrix_lower_bound_mask(m_0n_plus_down, mask_factor)
        m_0n_minus_down = matrix_lower_bound_mask(m_0n_minus_down, mask_factor)
        m_0n_x_down = matrix_lower_bound_mask(m_0n_x_down, mask_factor)
        m_nm_plus_up = matrix_lower_bound_mask(m_nm_plus_up, mask_factor)
        m_nm_minus_up = matrix_lower_bound_mask(m_nm_minus_up, mask_factor)
        m_nm_x_up = matrix_lower_bound_mask(m_nm_x_up, mask_factor)
        m_nm_plus_down = matrix_lower_bound_mask(m_nm_plus_down, mask_factor)
        m_nm_minus_down = matrix_lower_bound_mask(m_nm_minus_down, mask_factor)
        m_nm_x_down = matrix_lower_bound_mask(m_nm_x_down, mask_factor)
    return m_0n_plus_up,m_0n_minus_up,m_0n_x_up,m_0n_plus_down,m_0n_minus_down,m_0n_x_down , \
           m_nm_plus_up,m_nm_minus_up,m_nm_x_up,m_nm_plus_down ,m_nm_minus_down,m_nm_x_down

def get_tmd_m_elem_matrices(full_basis,m_0n_plus_up,m_0n_minus_up,m_0n_x_up,m_0n_plus_down,m_0n_minus_down,m_0n_x_down, \
           m_nm_plus_up,m_nm_minus_up,m_nm_x_up,m_nm_plus_down ,m_nm_minus_down,m_nm_x_down):
    '''

    :param full_basis:
    Must be the TMD basis gamma_plus,gamma_minus,spin up,spin_down
    :param m_0n_plus_up:
    :param m_0n_minus_up:
    :param m_0n_x_up:
    :param m_0n_plus_down:
    :param m_0n_minus_down:
    :param m_0n_x_down:
    :param m_nm_plus_up:
    :param m_nm_minus_up:
    :param m_nm_x_up:
    :param m_nm_plus_down:
    :param m_nm_minus_down:
    :param m_nm_x_down:
    :return:
    '''
    ones = np.ones(np.size(full_basis, 0))
    spin_states = find_spin(full_basis, 2, 3)
    null_states = spin_states == 0
    spin_up_states = spin_states == 1
    spin_down_states = spin_states == -1
    spin_up_mat = np.outer(spin_up_states,spin_up_states)
    null_spin_up_mat = np.outer(null_states,spin_up_states)
    null_spin_down_mat = np.outer(null_states,spin_down_states)
    spin_down_mat = np.outer(spin_down_states,spin_down_states)
    up_m = np.outer(full_basis[:,2],ones).astype(int)
    up_n =np.outer( ones,full_basis[:, 2]).astype(int)
    down_m = np.outer(full_basis[:,3],ones).astype(int)
    down_n = np.outer(ones,full_basis[:,3]).astype(int)
    #ordering is final, init state
    up_up_plus_int = ma.filled(ma.array(m_nm_plus_up[up_m,up_n],mask = ~spin_up_mat),0)
    null_up_plus_int = ma.filled(ma.array(m_0n_plus_up[up_n], mask=~null_spin_up_mat), 0)
    down_down_plus_int = ma.filled(ma.array(m_nm_plus_down[down_m,down_n],mask = ~spin_down_mat),0)
    null_down_plus_int = ma.filled(ma.array(m_0n_plus_down[down_n], mask=~null_spin_down_mat), 0)
    m_elem_matrix_plus = (up_up_plus_int+null_up_plus_int+down_down_plus_int+null_down_plus_int)
    up_up_minus_int = ma.filled(ma.array(m_nm_minus_up[up_m, up_n], mask=~spin_up_mat), 0)
    null_up_minus_int = ma.filled(ma.array(m_0n_minus_up[up_n], mask=~null_spin_up_mat), 0)
    down_down_minus_int = ma.filled(ma.array(m_nm_minus_down[down_n, down_m], mask=~spin_down_mat), 0)
    null_down_minus_int = ma.filled(ma.array(m_0n_minus_down[down_n], mask=~null_spin_down_mat), 0)
    m_elem_matrix_minus = (up_up_minus_int + null_up_minus_int + down_down_minus_int + null_down_minus_int)
    up_up_x_int = ma.filled(ma.array(m_nm_x_up[up_m, up_n], mask=~spin_up_mat), 0)
    null_up_x_int = ma.filled(ma.array(m_0n_x_up[up_n], mask=~null_spin_up_mat), 0)
    down_down_x_int = ma.filled(ma.array(m_nm_x_down[down_m, down_n], mask=~spin_down_mat), 0)
    null_down_x_int = ma.filled(ma.array(m_0n_x_down[down_n], mask=~null_spin_down_mat), 0)
    m_elem_matrix_x = (up_up_x_int  + null_up_x_int + down_down_x_int  + null_down_x_int)
    return m_elem_matrix_plus,m_elem_matrix_minus,m_elem_matrix_x

def create_jc_h_extended_basis_fast(basis_size,excitonic_vars_up,excitonic_vars_down, angle_mom_space,cavity_freq,vec_pot_strength,**kwargs):
    '''
    # creates the QED hamiltonian from BSE solutions in the up and down spin bases
    #considerably faster than an iterated approach--that algorithm has been omitted but can be found by looking at commits from June 2020
    For reproducing paper 1, go back to January 2021 commit
    Indices have been flipped to get into agreement with ACD hamiltonian.
    Not really designed for properly accounting for self-interaction term (on purpose, this is intentional)
    :param basis_size:
    :param excitonic_vars_up:
    :param excitonic_vars_down:
    :param angle_mom_space:
    :param cavity_freq:
    :param vec_pot_strength:
    :param kwargs:
    :return:
    '''
    resolution_factor = 1 #defaults to not accounting for resolution
    start_time = time.time()
    print("constructing extended hamiltonian 2")
    max_quanta = 1
    n_el = 1
    for key,value in kwargs.items():
        if key == "max_quanta":
            max_quanta = value
        if key == "resolution":
            resolution_factor = value
        if key == "n_el":
            n_el = value
    evd = excitonic_vars_down
    evu = excitonic_vars_up
    a0 = vec_pot_strength #units, hbar/bohr
    e_x_up = excitonic_vars_up.e_x[:basis_size]
    a_x_up = excitonic_vars_up.a_x[:,:basis_size]
    e_x_down = excitonic_vars_down.e_x[:basis_size]
    a_x_down = excitonic_vars_down.a_x[:, :basis_size]
    p_minus_down, p_minus_vv_down, p_minus_cc_down,p_minus_cv_down = evd.p_minus,evd.p_minus_vv,evd.p_minus_cc,evd.p_minus_cv
    p_plus_down, p_plus_vv_down, p_plus_cc_down,p_plus_cv_down = evd.p_plus, evd.p_plus_vv, evd.p_plus_cc,evd.p_plus_cv
    p_minus_up, p_minus_vv_up, p_minus_cc_up,p_minus_cv_up = evu.p_minus,evu.p_minus_vv,evu.p_minus_cc ,evu.p_minus_cv
    p_plus_up, p_plus_vv_up, p_plus_cc_up ,p_plus_cv_up= evu.p_plus, evu.p_plus_vv, evu.p_plus_cc,evu.p_plus_cv
    p_x_up,p_x_vv_up,p_x_cc_up,p_x_cv_up = evu.p_x, evu.p_x_vv, evu.p_x_cc  ,evu.p_x_cv
    p_x_down,p_x_vv_down,p_x_cc_down,p_x_cv_down = evd.p_x, evd.p_x_vv, evd.p_x_cc    ,evd.p_x_cv
    full_basis = combo.construct_spin_exciton_basis(basis_size,max_quanta+1)
    num_basis_states = np.size(full_basis,0)
    diagonal_terms = np.zeros(num_basis_states,dtype = np.csingle)
    a0 = a0
    a0_minus, a0_plus = chiral_a0(a0,angle_mom_space)
    for i in range(0,num_basis_states):
        cur_state = full_basis[i,:]
        diagonal_terms[i] = jc_diagonal_term(cur_state,e_x_up,e_x_down,cavity_freq)

    jc_h = np.diag(diagonal_terms) + 0 * 1j  # adding 0 imag unit to insure complex data type

    m_nm_fit_param = n_el #n_el as in (4) in Latini et al 2019 https://pubs.acs.org/doi/pdf/10.1021/acs.nanolett.9b00183
    print("diagonals_created")
    m_0n_plus_up, m_0n_minus_up, m_0n_x_up, m_0n_plus_down, m_0n_minus_down, m_0n_x_down,  \
     m_nm_plus_up, m_nm_minus_up, m_nm_x_up, m_nm_plus_down, m_nm_minus_down, m_nm_x_down= \
        create_tmd_m_elem_matrices(a_x_up, a_x_down, p_plus_up, p_minus_up, p_x_up, p_plus_down, p_minus_down, p_x_down,
                               p_plus_vv_up, p_plus_cc_up, p_minus_vv_up,
                               p_minus_cc_up, p_plus_vv_down, p_plus_cc_down, p_minus_vv_down, p_minus_cc_down, p_x_vv_up, p_x_cc_up, p_x_vv_down, p_x_cc_down, m_nm_fit_param,
                               to_mask_coupling_lb=True, mask_factor=1e-5)


    phot_m, phot_n = create_state_tensor(full_basis[:, :2], 0), create_state_tensor(full_basis[:, :2], 1)
    phot_equal = photons_matrix_relation(phot_n,phot_m,np.array([0,0]),relation_type= "bool")
    a_plus_dag = photons_matrix_relation(phot_n,phot_m,np.array([1,0]),"a")
    a_plus = photons_matrix_relation(phot_n,phot_m,np.array([-1,0]),"a")
    a_minus_dag = photons_matrix_relation(phot_n,phot_m,np.array([0,1]),"a")
    a_minus = photons_matrix_relation(phot_n,phot_m,np.array([0,-1]),"a")
    m_elem_matrix_plus,m_elem_matrix_minus,m_elem_matrix_x= get_tmd_m_elem_matrices(full_basis,m_0n_plus_up,m_0n_minus_up,m_0n_x_up,m_0n_plus_down,m_0n_minus_down,m_0n_x_down,  \
          m_nm_plus_up,m_nm_minus_up,m_nm_x_up,m_nm_plus_down ,m_nm_minus_down,m_nm_x_down)
    m_e = 5.10999e5 # ev/c2
    m_elem_plus = (a_plus_dag * m_elem_matrix_plus+ a_plus*m_elem_matrix_plus.T.conj())/m_e
    m_elem_minus = (a_minus_dag * m_elem_matrix_minus+ a_minus * m_elem_matrix_minus.T.conj())/m_e
    m_0n_arrays = M_0N_ARRAYS(m_elem_matrix_plus[0,:]/m_e,m_elem_matrix_minus[0,:]/m_e,m_elem_matrix_x[0,:]/m_e) #in terms of full basis
    jc_h = jc_h+((a0_plus * m_elem_plus + a0_minus * m_elem_minus)/resolution_factor)
    end_time = time.time()
    print("took " + str(end_time - start_time) + " seconds")
    return jc_h, m_0n_arrays



def jaynes_cummings_alter_fast(jc_h,basis_size,cav_freq,max_quanta,evu,evd):
    '''takes initial hamiltonian and alters the relevant indices for different cavity frequencies
     Considerably faster than creating a new hamiltonian from scratch'''
    new_jc_h = np.copy(jc_h)
    e_x_up = evu.e_x[:basis_size]
    e_x_down = evd.e_x[:basis_size]
    full_basis = combo.construct_spin_exciton_basis(basis_size,max_quanta+1)
    num_basis_states = np.size(full_basis,0)
    for i in range(0,num_basis_states):
        cur_state = full_basis[i,:]
        new_jc_h[i,i] = jc_diagonal_term(cur_state,e_x_up,e_x_down,cav_freq)
    return new_jc_h
def smooth_signal(intensity_array,time_array,window):
    '''
    Takes a signal and provides a moving average and moving standard deviation
    :param intensity_array: np.ndarray
    :param time_array: np.ndarray
    :param window: int
    :return: tuple (np.ndarray, np.ndarray)
    Smoothed dated, smoothed standard deviation data
    '''
    if np.ndim(intensity_array ==1):
        intensity_array = np.atleast_2d(intensity_array)
    smoothed_data = np.zeros(np.shape(intensity_array))
    smoothed_std_data = np.zeros(np.shape(intensity_array))
    for i in range(0,np.size(intensity_array,0)):
        data_series = pd.Series(intensity_array[i,:],time_array)
        smoothed_series = data_series.rolling(window = window).mean()
        smoothed_std = data_series.rolling(window= window).std()
        smoothed_data[i,:] = np.array(smoothed_series)
        smoothed_std_data[i,:] =  np.array(smoothed_std)
    return smoothed_data, smoothed_std_data


def plot_dispersion(filename,cavity_freq_array,response_freq_array,spectral_matrix,**kwargs):
    '''
    Plots the spectral response (see Figs 2a-b and 4a-b in main text)
    :param filename: str
    :param cavity_freq_array: np.ndarray
    :param response_freq_array: np.ndarray
    :param spectral_matrix: np.ndarray
    :param kwargs: dict
        "tight_layout": bool (default False)
            To use set tight layout or not
        "color_map": str (default "Purples")
            What color map to use
        "show_axes": bool
            To show axes or not
        "colorbar": str
            Colorbar label given by value here
        "axis": plt.Axes
        "figure": plt.Figure
        "x_limits": arraylike (2)
        "y_limits": arraylike (2)
        "x_label": str
        "y_label": str
        "axis_fontsize":int
        "axis_labelsize": int
        "colorbar_fontsize": int
        "to_colorbar_norm": bool
        "power_law": float
            Value gives what factor to apply a power law for plotting
        "bounds": arraylike (2)
            Bounds for the colormap (vmin, vmax)
    :return:
    '''
    #defaults
    color_map = plt.get_cmap("Purples")
    to_show_axes = True
    set_bounds = False
    to_show_colorbar = False
    x_limits = False
    y_limits = False
    x_label  = "$\Omega$ (eV)"
    y_label = "$\omega$ (eV)"
    axis_fs = 11
    colorbar_fs = 10
    make_fig = True
    to_colorbar_norm  = False
    tight_layout = False
    power_law_n = .33333
    axis_labelsize= 9
    for key, value in kwargs.items():
        if key == "tight_layout":
            tight_layout = value
        if key == "color_map":
            color_map = plt.get_cmap(value)
        if key == "show_axes":
            to_show_axes = value
        if key == "colorbar":
            to_show_colorbar = True
            colorbar_label = value
        if key == "axis":
            ax = value
            make_fig = False
        if key == "figure":
            fig = value
            make_fig = False
        if key == "x_limits":
            x_limits = True
            xlim = value
        if key == "y_limits":
            y_limits = True
            ylim = value
        if key == "x_label":
            x_label = value
        if key == "y_label":
            y_label = value
        if key == "axis_fontsize":
            axis_fs = value
        if key == "axis_labelsize":
            axis_labelsize = value
        if key == "colorbar_fontsize":
            colorbar_fs = value
        if key == "to_colorbar_norm":
            to_colorbar_norm = value
        if key == "power_law":
            power_law_n = value
        if key == "bounds":
            set_bounds = True
            bounds = value
    if (make_fig):
        fig, ax = plt.subplots()
    omeg = np.array(cavity_freq_array)
    freq = np.array(response_freq_array)
    y_min = np.min(response_freq_array)
    y_max = np.max(response_freq_array)
    ax.set_ylim(y_min, y_max)
    omeg2 = np.transpose(np.tile(omeg, (np.size(freq), 1)))
    freq2 = np.tile(freq, (np.size(omeg), 1))
    to_plot = spectral_matrix
    num_levels = 200
    if (not to_colorbar_norm):
        if (set_bounds):
            cs = plt.contourf(omeg2, freq2, (to_plot) ** (1), levels=num_levels,
                              vmin=bounds[0], vmax=bounds[1],
                              norm=colors.PowerNorm(gamma=power_law_n), cmap=color_map)
        else:
            cs = plt.contourf(omeg2, freq2, (to_plot)**(1), levels=num_levels,vmin = np.min(to_plot)+1/num_levels*np.max(to_plot), vmax = np.max(to_plot), norm = colors.PowerNorm(gamma=power_law_n),cmap = color_map)
    else:
        levels_normed = np.linspace(0,1,num_levels)
        cs = plt.contourf(omeg2, freq2, (to_plot) ** (1), levels=levels_normed,
                          vmin=1e-2, vmax=1.001,
                          norm=colors.Normalize(vmin = 1e-2,vmax = 1.001), cmap=color_map)
    if (to_show_axes):
        ax.tick_params(axis = 'both',labelsize = axis_labelsize)
        plt.xlabel(x_label,fontsize = axis_fs, labelpad = 0)
        plt.ylabel(y_label, fontsize =axis_fs, labelpad = -1 )
    if (to_show_colorbar):
        tickmin = 0
        tickmax = np.max(spectral_matrix)
        #https://matplotlib.org/3.4.2/gallery/axes_grid1/demo_colorbar_with_inset_locator.html
        cbaxes = inset_axes(ax,width="100%",height="5%",
                            loc='lower left',bbox_to_anchor=(0, 1.1, 1, 1),
                            bbox_transform=ax.transAxes,borderpad=0,)
        if (to_colorbar_norm):
            plt.gcf().subplots_adjust(top=0.83)
            cbar = fig.colorbar(cs, cax=cbaxes, format='%.1f', orientation="horizontal",ticks = [0,1.00])
            cbar.ax.set_xlabel(colorbar_label, labelpad=-15, fontsize=colorbar_fs)
        else:
            cbar = fig.colorbar(cs, cax = cbaxes, format='%.2f',orientation = "horizontal",ticks = [tickmin,tickmax])
            cbar.ax.set_xlabel(colorbar_label,labelpad = -13,fontsize = colorbar_fs)
        cbar.ax.tick_params(labelsize=12)
    if (tight_layout):
        plt.tight_layout()
    if (x_limits):
        ax.set_xlim(xlim[0],xlim[1])
    if (y_limits):
        ax.set_ylim(ylim[0],ylim[1])
    if (filename != ""):
        plt.savefig(filename)
        plt.show()

def polariton_matrix_set(a_x_matrix, matrix_element_object):
    '''Gives set coupled matrix elements (for polarized absorption)'''
    m_pol_minus = polaritonic_matrix_array(a_x_matrix, matrix_element_object.m_minus)
    m_pol_plus = polaritonic_matrix_array(a_x_matrix, matrix_element_object.m_plus)
    m_pol_x = polaritonic_matrix_array(a_x_matrix, matrix_element_object.m_x)
    return m_pol_plus,m_pol_minus,m_pol_x

def create_cavity_dispersion_excitonic_spin(raw_system_up,raw_system_down,basis_size,cavity_freq_array, spectral_res,vec_pot_strength, a0_polarization,**kwargs):
    '''
    Determines relation between cavity frequency, eigenvectors, energies , and spectral response
    :param raw_system_up:
    :param raw_system_down:
    :param basis_size:
    :param cavity_freq_array:
    :param spectral_res:
    :param vec_pot_strength:
    :param a0_polarization:
    :param kwargs:
    :return:
    '''
    coupling = True
    start_time = time.time()
    ex_vars_up = get_excitonic_vars(raw_system_up, **{"coupled": coupling,"basis_size":basis_size})
    ex_vars_down = get_excitonic_vars(raw_system_down, **{"coupled": coupling,"basis_size":basis_size})
    end_time = time.time()
    print("BSE_took:"+str(end_time-start_time)+"sec")
    iterations = np.size(cavity_freq_array)
    energy_spectrum = np.linspace(1, 3, spectral_res) #1 to 3 eV
    spectral_matrix = np.zeros((np.size(cavity_freq_array), spectral_res,3))
    to_save_energies = basis_size*2
    calculated_energies_matrix = np.zeros((to_save_energies,iterations))
    full_basis_size = np.size(combo.construct_spin_exciton_basis(basis_size,2),0)
    a_x_matrix = np.zeros((full_basis_size,to_save_energies,iterations),dtype= np.csingle)
    for i in range(0, iterations):
        cavity_freq = cavity_freq_array[i]
        if (i == 0):
            h_jc_init, m_elem_tensor = create_jc_h_extended_basis_fast(basis_size,ex_vars_up,ex_vars_down,a0_polarization,cavity_freq,vec_pot_strength,**kwargs)
            h_jc= h_jc_init
        else:
            h_jc = jaynes_cummings_alter_fast(h_jc_init,basis_size,cavity_freq,1,ex_vars_up,ex_vars_down)
        e_x_jc, a_x_jc = base.solve_hamiltonian(h_jc)
        e_x_jc = (e_x_jc-e_x_jc[0])
        m_pol = polariton_matrix_set(a_x_jc,m_elem_tensor)
        e_x_sub = e_x_jc[:to_save_energies]
        a_x_sub = a_x_jc[:,:to_save_energies]

        print("Iteration:"+str(i))
        for j in range(0,3):
            lin_abs_spectrum = linear_absorption_polaritonic(energy_spectrum,e_x_jc,m_pol[j])
            spectral_matrix[i, :, j] = lin_abs_spectrum

        calculated_energies_matrix[:, i] = e_x_sub
        a_x_matrix[:, :, i] = a_x_sub

    return calculated_energies_matrix, a_x_matrix, spectral_matrix,ex_vars_up.a_x,ex_vars_down.a_x

def momentum_rep_from_excitonic_spin(a_x_excitonic,a_x_bare_up,a_x_bare_down,full_basis):
    '''
    Gives momentum representation from the calculated eigenvectors
    :param a_x_excitonic: np.ndarray
    :param a_x_bare_up: np.ndarray
    :param a_x_bare_down: np.ndarray
    :param full_basis:
    :return:
    '''
    num_k = np.size(a_x_bare_up,0)
    if (a_x_excitonic.ndim == 2):
        num_excitons = np.size(a_x_excitonic,1)
    else:
        num_excitons = 1
    a_x_rep_up = np.zeros((num_k,num_excitons),dtype = np.csingle)
    a_x_rep_down = np.zeros((num_k, num_excitons), dtype=np.csingle)
    for i in range(0,num_excitons):
        mom_rep_array_up = np.zeros(num_k, dtype=np.csingle)
        mom_rep_array_down = np.zeros(num_k, dtype=np.csingle)
        for j in range(0,np.size(a_x_excitonic,0)):
            cur_basis = full_basis[j,:]
            state_up = cur_basis[2]
            state_down = cur_basis[3]
            excitonic_elem = a_x_excitonic[j, i]
            if (state_up>= 0):
                mom_array_elem = a_x_bare_up[:,state_up]
                mom_rep_array_up = mom_rep_array_up + excitonic_elem * mom_array_elem
            if (state_down>= 0):
                mom_array_elem = a_x_bare_down[:,state_down]
                mom_rep_array_down = mom_rep_array_down+excitonic_elem*mom_array_elem
        a_x_rep_up[:,i] = mom_rep_array_up
        a_x_rep_down[:, i] = mom_rep_array_down
    return a_x_rep_up, a_x_rep_down

def envelope(a_x_k_up,a_x_k_down,p_vc_x_up,p_vc_x_down):
    net_sig = np.einsum("i,ij->ij",p_vc_x_up,a_x_k_up)+np.einsum("i,ij->ij",p_vc_x_down,a_x_k_down)
    return net_sig

def integrate_bz_region_prob_density(wavefunction,mask):
    k_valley_psi = wavefunction[mask]
    total_prob = np.sum(k_valley_psi.conj()*k_valley_psi)
    return total_prob


def time_dependent_momentum(e_x_array,a_x_matrix, time_array,polaritonic_matrix):
    '''
    Performs TDSE time evolution on set of eigenvectors provided some polariton matrix
    :param e_x_array: np.ndarray (1D) (eV)
    :param a_x_matrix: np.ndarray (2D)
    :param time_array:  np.ndarray (1D) (seconds)
    :param polaritonic_matrix:
    :return: np.ndarray (2D)
    '''
    hbar = 6.582e-16 #eV*s
    num_t = np.size(time_array)
    e_x_array = e_x_array/hbar
    a_x_time_matrix = np.zeros((np.size(a_x_matrix,0),num_t),dtype = np.csingle)
    for i in range(0,num_t):
        time = time_array[i]
        mom_vals = np.zeros((np.size(a_x_matrix,0)),dtype = np.csingle)
        for j in range(0,np.size(a_x_matrix,0)):
            mom_vals[j] = np.exp(-1j*time*e_x_array[j])*polaritonic_matrix[j]
        a_x_time_matrix[:,i] = mom_vals

    return a_x_time_matrix

def time_dependent_probe(probe_momentum,momentum_time_matrix):
    '''(13) in the published paper, doi 10.1103/PhysRevB.103.035431'''
    probe_signal = np.abs(np.einsum("i,it->t",probe_momentum.conj(),momentum_time_matrix))**2
    return probe_signal


def time_dependent_jc_dynamics(e_x_array,a_x_matrix, time_array,m_pol_plus_array,m_pol_minus_array,optical_excitation):
    '''
    Taking solutions to QED hamiltonian, gives measurements for time dependent intensities
    :param e_x_array: np.ndarray (1D)
    :param a_x_matrix: np.ndarray (2D)
    :param time_array: np.ndarray (1D)
    :param m_pol_plus_array: np.ndarray(1D)
    :param m_pol_minus_array: np.ndarray(1D)
    :param optical_excitation: str
    :return:
    '''
    p_plus_t_matrix = time_dependent_momentum(e_x_array,a_x_matrix, time_array,m_pol_plus_array)
    p_minus_t_matrix = time_dependent_momentum(e_x_array, a_x_matrix, time_array, m_pol_minus_array)
    if (optical_excitation == "plus"):
        i_plus = time_dependent_probe(p_plus_t_matrix[:,0],p_plus_t_matrix)
        i_minus = time_dependent_probe(p_minus_t_matrix[:,0],p_plus_t_matrix)
    elif (optical_excitation == "minus"):
         i_plus = time_dependent_probe(p_plus_t_matrix[:,0],p_minus_t_matrix)
         i_minus = time_dependent_probe(p_minus_t_matrix[:,0],p_minus_t_matrix)
    else: raise ValueError("Unsupported optical excitation string ('plus' or 'minus' required)")
    helicity = get_helicity_electronic(i_plus,i_minus)

    return i_plus, i_minus ,helicity

def time_dependent_set(raw_system_up,raw_system_down,basis_size,vec_pot_array,cavity_p_array_angle,time_array,**kwargs):
    '''
    Gives tuple of intensity arrays for plus, minus polarization and a net helicity from RAW_SYSTEM objects in the spin basis
    :param raw_system_up:
    :param raw_system_down:
    :param basis_size:
    :param vec_pot_array:
    :param cavity_p_array_angle:
    :param time_array:
    :param kwargs:
    :return:
    '''
    ex_vars_up = get_excitonic_vars(raw_system_up)
    ex_vars_down = get_excitonic_vars(raw_system_down)
    res_freq = raw_system_up.tmdc.baseline
    helicity_matrix = np.zeros((np.size(time_array),np.size(vec_pot_array)))
    i_plus = np.zeros((np.size(time_array), np.size(vec_pot_array)))
    i_minus =  np.zeros((np.size(time_array), np.size(vec_pot_array)))
    for vp in range(0,np.size(vec_pot_array)):
        hamiltonian, matrix_elements_tensor = create_jc_h_extended_basis_fast(basis_size,ex_vars_up,ex_vars_down,cavity_p_array_angle,res_freq,vec_pot_array[vp],**kwargs)
        e_x, a_x = base.solve_hamiltonian(hamiltonian)
        m_pol_minus = polaritonic_matrix_array(a_x, matrix_elements_tensor.m_minus)
        m_pol_plus = polaritonic_matrix_array(a_x, matrix_elements_tensor.m_plus)
        i_plus[:,vp], i_minus[:,vp], helicity_matrix[:,vp] = time_dependent_jc_dynamics(e_x,a_x,time_array,m_pol_plus,m_pol_minus,"plus")
    return i_plus, i_minus, helicity_matrix
def get_excitonic_vars(raw_system,**kwargs):
    '''
    for some raw TMDC system , creates and solves the BSE
    :param raw_system: base.RAW_SYSTEM
    :param kwargs:
    :return:
    '''
    to_couple = True
    to_trunc = False
    energy_shift = raw_system.tmdc.baseline #centering lowest energy at baseline
    for key,value in kwargs.items():
        if key == "coupled":
            to_couple = value
        if key == "basis_size":
            to_trunc= True
            truncation_size = value

    if (to_couple):
        h_bare = base.create_bse_coupled(raw_system)
    else:
        h_bare = base.create_bse(raw_system)

    start_time = time.time()
    if (np.size(h_bare,0)<10):
        print(h_bare)
    e_x_bare, a_x_bare = base.solve_hamiltonian(h_bare)
    end_time = time.time()
    print("BSE took:"+str(end_time-start_time)+"sec")
    if (to_trunc):
        a_x_bare = a_x_bare[:,:truncation_size]
    p_minus = raw_system.p_minus
    p_plus = raw_system.p_plus
    p_vc, p_cv, p_vv, p_cc = create_momentum_matrix_set(raw_system.c_v, raw_system.c_c, raw_system.tau, raw_system.tmdc.a, raw_system.tmdc.t)
    p_minus_vv, p_plus_vv = base.create_polarized_momentum_matrices(p_vv[0], p_vv[1])
    p_minus_cv, p_plus_cv = base.create_polarized_momentum_matrices(p_cv[0], p_cv[1])
    p_minus_cc, p_plus_cc = base.create_polarized_momentum_matrices(p_cc[0], p_cc[1])
    e_x_bare = e_x_bare+energy_shift-e_x_bare[0]
    excitonic_vars = EXCITONIC_VARS(e_x_bare, a_x_bare, p_minus, p_plus,p_minus_vv,p_minus_cc,p_plus_vv,p_plus_cc,p_vc[0],p_vv[0],p_cc[0],p_minus_cv,p_plus_cv,p_cv[0])
    return excitonic_vars


def get_split(branch_a,branch_b,cavity_freq_array,res_freq):
    '''Returns energy split between two branches at some resonance frequency'''
    index = np.argmin(np.abs(np.array(cavity_freq_array)-res_freq))
    return np.abs(branch_a[index]-branch_b[index])

def get_helicity_electronic(i_plus_array,i_minus_array):
    '''Returns electronic state helicity metric. (12) in doi 10.1103/PhysRevB.103.035431'''
    return (i_plus_array-i_minus_array)/(i_plus_array+i_minus_array)



