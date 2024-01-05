import matplotlib.pyplot as plt
import numpy as np
import time
import numpy.ma as ma
import scipy as sp
'''Core file for handling reciprocal space systems and solving for Hamiltonians in light-matter coupled systems.
Focus on transition metal dichalcogenides (TMDs, or TMDCs). Andrew Salij and Roel Tempelaar'''

class RAW_SYSTEM:
    '''
    Data transfer class for (raw) TMDC information
    '''
    def __init__(self,resolution,tmdc, energy_v, energy_c, valence_vec, conduction_vec, lattice_vec_1,
                             lattice_vec_2, kx, ky,p_minus,p_plus,tau):
        '''
        :param resolution: int
        :param tmdc: str
        :param energy_v: float
        :param energy_c: float
        :param valence_vec: np.ndarray
        :param conduction_vec: np.ndarray
        :param lattice_vec_1: np.narray  (2)
        :param lattice_vec_2:np.ndarray (2)
        :param kx: float
        :param ky: float
        :param p_minus: np.ndarray
        :param p_plus: np.ndarray
        :param tau: int (-1 or 1)
        '''
        self.res = resolution
        self.tmdc = tmdc
        self.energy_v = energy_v
        self.energy_c = energy_c
        self.c_v = valence_vec
        self.c_c = conduction_vec
        self.vec_1 = lattice_vec_1
        self.vec_2 = lattice_vec_2
        self.kx = kx
        self.ky = ky
        self.p_minus = p_minus
        self.p_plus = p_plus
        self.tau = tau
def monkhorst_packing_centered(resolution, lattice_vec_1, lattice_vec_2):
    '''
    Unlike monkhorst_packing(), this results in a pack centered at the origin--for symmetry considerations,
    This is what is used in published manuscript.
    :param resolution:int
    :param lattice_vec_1: np.array
    :param lattice_vec_2: np.array
    :return:
    '''
    grid = []
    res = resolution
    for res_id_1 in range(res):
        p = res_id_1 + 1
        up = (2 * p - res - 1) / (2.0 * res)
        for res_id_2 in range(res):
            r = res_id_2 + 1
            ur = (2 * r - res - 1) / (2.0 * res)
            grid.append(up * lattice_vec_1 + ur * lattice_vec_2)
    return np.array(grid)
def fill_bz(kx_array,ky_array,lattice_vec_1,lattice_vec_2,monkhorst_array):
    '''
    kx_array, ky_array, monkhorst_array all same size
    :param kx_array:
    :param ky_array:
    :param lattice_vec_1:
    :param lattice_vec_2:
    :param monkhorst_array:
    :return:
    '''
    new_size = np.size(kx_array)*9
    kx_new = np.zeros(new_size)
    ky_new = np.zeros(new_size)
    new_monkhorst_array = np.zeros(new_size,dtype= np.csingle)
    for i in range(-1,2):
        for j in range(-1,2):
            kx_add = kx_array+lattice_vec_1[0]*(i)+lattice_vec_2[0]*j
            ky_add = ky_array+lattice_vec_1[1]*(i)+lattice_vec_2[1]*j
            index = 3*(i+1)+(j+1)
            kx_new[index*np.size(kx_array):(index+1)*np.size(kx_array)] = kx_add
            ky_new[index*np.size(kx_array):(index+1)*np.size(kx_array)] = ky_add
            new_monkhorst_array[index*np.size(kx_array):(index+1)*np.size(kx_array)]  = monkhorst_array
    return kx_new,ky_new, new_monkhorst_array

def point_to_cartesian_point(point,vec_1,vec_2):
    v1 = np.array(vec_1)
    v2 = np.array(vec_2)
    point = np.array(point)
    trans_mat = np.array([v1,v2]).T
    inv_trans_mat = np.linalg.inv(trans_mat)
    cartesian_point = np.dot(inv_trans_mat, point)
    return cartesian_point

def point_to_monkhorst_point(point,vec_1,vec_2):
    v1 = np.array(vec_1)
    v2 = np.array(vec_2)
    point = np.array(point)
    trans_mat = np.array([v1,v2]).T
    cartesian_point = point_to_cartesian_point(point,v1,v2)
    x_val = cartesian_point[0]
    y_val = cartesian_point[1]
    x_val_rem = (x_val)-np.floor(x_val+0.5)
    y_val_rem = (y_val)-np.floor(y_val+0.5)
    cartesian_point_shifted = np.array([x_val_rem,y_val_rem])
    monkhorst_point = np.dot(trans_mat,cartesian_point_shifted)
    return monkhorst_point


#takes an array of length n^2 and reforms it into a square matrix of length n
def reshape_1D_to_2D(array_1D):
    dim = int(np.sqrt(array_1D.size))
    array_2D = array_1D.reshape(dim, dim)
    return array_2D

# takes a square matrix of length n and reforms it into an array of length n^2
def reshape_2D_to_1D(array_2D):
    dim = int(array_2D.size)
    array_1D = array_2D.reshape(1,dim)
    return array_1D


def k_to_q(k, k_points, k_points_tau):
    mask_shape = k.shape[0]
    mask = np.zeros(mask_shape, dtype=np.int64)
    for i in np.arange(mask_shape): mask[i] = np.argmin(np.round(np.apply_along_axis(np.linalg.norm, axis=1,arr= k[i] - k_points), decimals=5))
    q = k-k_points[mask]
    tau = k_points_tau[mask]
    return q, tau

def fix_phase(vecs,n_vecs):
    '''
    Fix phase of orbitals such that the first element of all eigenvectors is purely real
    :param vecs: np.ndarray (2D)
    :param n_vecs: float
    :return:
    '''
    for n in range(n_vecs):
        phase = np.angle(vecs[0,n])
        vecs[:,n] = vecs[:,n]*np.exp(-1j*phase)
    return vecs


def two_band_init(tmdc,qx_array):
    energy_gap = tmdc.energy_gap
    t = tmdc.t
    c_split = tmdc.c_so
    v_split = tmdc.v_so
    size = np.array(qx_array).size
    energy_v_array = np.zeros(size, dtype=np.csingle)
    energy_c_array = np.zeros(size, dtype=np.csingle)
    vec_c_array = np.zeros([size, 2], dtype=np.csingle)
    vec_v_array = np.zeros([size, 2], dtype=np.csingle)
    return energy_gap, t, c_split, v_split, energy_v_array, energy_c_array, vec_v_array, vec_c_array

def calc_two_band_spin(tmdc,spin,qx_array,qy_array, tau_array):
    '''
    :param tmdc: TMDC object
    :param spin:
    :param qx_array:
    :param qy_array:
    :param tau_array:
    :return:
    '''
    size = np.size(qx_array)
    energy_gap, t, c_split, v_split, energy_v_array, energy_c_array, vec_v_array, vec_c_array = two_band_init(tmdc, qx_array)
    a = 1  # lattice constant set to 1 for plotting purposes--accounted for in area elsewhere
    alpha = a * t
    beta = energy_gap / 2
    for i in range(0, size):
        x = qx_array[i]
        y = qy_array[i]
        tau = tau_array[i]
        spin_product = spin*tau
        h00 = beta + spin_product*c_split / 2
        h11 = -beta + spin_product*v_split / 2
        h01 = alpha * (tau * x - 1j * y)
        hamiltonian = np.array([[h00, h01], [h01.conj(), h11]])
        if sp.linalg.ishermitian(hamiltonian):
            bands, vecs = np.linalg.eigh(hamiltonian)
        else:
            bands, vecs = np.linalg.eig(hamiltonian)
        idx = bands.argsort()
        bands = bands[idx]
        vecs_init = vecs[:, idx]
        vecs = fix_phase(vecs_init, 2)
        energy_v_array[i] = bands[0]
        energy_c_array[i] = bands[1]
        vec_v_array[i, :] = vecs[:, 0]
        vec_c_array[i, :] = vecs[:, 1]
    return energy_v_array, energy_c_array, vec_v_array, vec_c_array

def obtain_two_band(q, tau, energy_gap, lambda_c, lambda_v, t, spin):
    '''
    Modified solution to (2) in (10.1103/PhysRevB.92.085413)
    :param q:
    :param tau: np.ndarray (1D) (int) (-1 or 1)
    :param energy_gap: float (eV)
    :param lambda_c: float (eV)
    :param lambda_v: float (eV)
    :param t: float (eV)
    :param spin: int (-1 or 1)
    :return:
    '''
    energy_v = -energy_gap/2 + tau/2 * lambda_v * spin
    energy_c = energy_gap/2 + tau/2 * lambda_c * spin
    energy_gap = energy_c - energy_v
    energy_avg = (energy_c + energy_v)/2
    epsilon = 1/2* np.sqrt(np.square(energy_gap) + 4 * np.square(t) * np.sum(np.square(q), 1))
    spin_qx = q[:,0]*t*tau
    spin_qy = q[:,1]*t
    #1e-10 added to prevent singularities
    coef_c = np.column_stack((np.ones(epsilon.size), -(energy_gap/ 2 - epsilon) / (spin_qx - 1j * spin_qy - 1j * 1e-10)))
    coef_v = np.column_stack((np.ones(epsilon.size), -(energy_gap/ 2 + epsilon) / (spin_qx - 1j * spin_qy - 1j * 1e-10)))
    coef_c_norms = np.apply_along_axis(np.linalg.norm, 1, coef_c)
    coef_c /= np.column_stack((coef_c_norms, coef_c_norms))
    coef_v_norms = np.apply_along_axis(np.linalg.norm, 1, coef_v)
    coef_v /= np.column_stack((coef_v_norms, coef_v_norms))
    return -epsilon + energy_avg, epsilon + energy_avg,  coef_v, coef_c

def create_momentum_matrix(vec1_array,vec2_array,tau_array,lattice_constant,transfer_integral):
    '''
    See (6) in doi: 10.1103/PhysRevB.92.085413, by Berkelbach, T.C. Hybertsen, M.S. and Reichman, D.R. (2015)., Phys. Rev. B. 92, 085415.
    Note that the k gradient Hamiltonian here, the factor of m/\hbar is included
    :param vec1_array: np.ndarray (1D) (dim)
    :param vec2_array: np.ndarray (1D) (dim)
    :param tau_array: np.ndarray (1D) (dim)
    :param lattice_constant: float (angstrom)
    :param transfer_integral: float (eV)
    :return: np.ndarray (1D) (eV/c)
    '''
    #a, t in units of Angstrom, eV to start
    angstrom_to_eV_dist = 1973.27
    a = lattice_constant/angstrom_to_eV_dist
    t = transfer_integral
    m_e = 5.10999e5 # electron mass ev/c2
    hbar = 1 #au
    alpha = a*t*m_e/hbar
    cur_size = int(vec1_array.size/2)
    momentum_matrix_x = np.zeros(cur_size, dtype = np.csingle)
    momentum_matrix_y= np.zeros(cur_size, dtype = np.csingle)
    for i in range(0,cur_size):
           psi_1 = vec1_array[i,:]
           psi_2 = vec2_array[i,:]
           tau = tau_array[i]
           h01x = alpha*tau
           h01y = -alpha*1j
           delH_x = np.array([[0, h01x],[h01x,0]])
           delH_y = np.array([[0, h01y],[-h01y,0]])
           p_x = np.dot(np.dot(psi_1.conj().T,delH_x),psi_2)
           p_y = np.dot(np.dot(psi_1.conj().T,delH_y),psi_2)
           momentum_matrix_x[i] = p_x
           momentum_matrix_y[i] = p_y
    #returns in units of eV/c
    return momentum_matrix_x, momentum_matrix_y
def create_linear_polarized_momentum_matrix(p_x,p_y,angle):
    '''angle dehigh_resd from x-axis'''
    p_modified = p_x*np.cos(angle)+p_y*np.sin(angle)
    return p_modified


def create_polarized_momentum_matrices(p_x,p_y):
    '''For circular polarization. Sign convention is that P_\pm is c*(P_x\pm 1j*P_y)'''
    p_minus = 1/np.sqrt(2)*(p_x-1j*p_y)
    p_plus = 1/np.sqrt(2)*(p_x+1j*p_y)
    return p_minus, p_plus

def brillouin_zone_plot(filename, k_x, k_y, plot_this, k_points, vec_1, vec_2,**kwargs):
    '''
    For 2D matrix inputs (same size) of k_x,k_y, and plot_this, saves a contour plot of plot_this
    normalized by (a^-1) where a is the lattice constant
    :param filename:
    :param k_x:
    :param k_y:
    :param plot_this:
    :param k_points:
    :param vec_1:
    :param vec_2:
    :param kwargs:
    :return:
    '''
    #defaults
    to_show_axes = True
    to_show_ticks = True
    to_show_bz_lines = True
    to_show_lv_lines = True
    to_show_whole_bz = False
    color_map = plt.get_cmap("viridis")
    axis_exists = False
    normalized = False
    show_colorbar = False
    z_bounds = np.array([0,0])
    show_label = False
    for key, value in kwargs.items():
        if key == "show_axes":
            to_show_axes = value
        if key == "ticks":
            to_show_ticks = value
        if key == "bz_lines":
            to_show_bz_lines = value
        if key == "lv_lines":
            to_show_lv_lines = value
        if key == "color_map":
            color_map = plt.get_cmap(value)
        if key == "colorbar":
            show_colorbar = True
            colorbar_label = value
        if key == "whole_bz":
            to_show_whole_bz = value
        if key == "axis":
            ax = value
            axis_exists = True
        if key == "normalized":
            normalized = value
        if key == "z_bounds":
            normalized = False
            z_bounds = value
        if key == "bz_label":
            bz_label = value
            plt.rcParams.update({'font.size': 4})
            show_label = True
        if key == "inset_region":
            inset_region = value
    if (not axis_exists):
        fig, ax = plt.subplots()
        axis_exists = True

    base = [0,0]
    k_points = k_points[k_points[:,0].argsort()]
    kp_x = k_points[:,0]
    kp_y = k_points[:,1]

    zero_mask = plot_this == 0
    plot_this = ma.masked_array(plot_this,zero_mask)
    if (normalized):
        cs = plt.contourf(k_x, k_y, plot_this, levels=200, cmap=color_map)
    elif(z_bounds.any()):
        cs = plt.contourf(k_x, k_y, plot_this, levels=200, cmap=color_map, vmin=z_bounds[0], vmax=z_bounds[1])
    else:
        max = np.max(np.abs(plot_this))
        cs = plt.contourf(k_x, k_y, plot_this, levels=200, cmap=color_map,vmin = -max,vmax = max)

    if (show_label):
        plt.text(0.5, 0.5, bz_label, fontsize = 7, horizontalalignment='center',verticalalignment = 'center',transform=ax.transAxes)
    if (not to_show_ticks):
        plt.axis("off")
    if (show_colorbar):
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel(colorbar_label)
    if (to_show_whole_bz):
        buffer = 1
        x_min = np.min(k_points[:,0])
        x_max = np.max(k_points[:,0])
        y_min = np.min(k_points[:,1])
        y_max = np.max(k_points[:,1])
        ax.set_xlim(x_min-buffer,x_max+buffer)
        ax.set_ylim(y_min-buffer,y_max+buffer)
    if (to_show_lv_lines):
        #vectors for monkhorst pack
        plt.arrow(base[0],base[1],vec_1[0],vec_1[1])
        plt.arrow(base[0],base[1],vec_2[0],vec_2[1])

    if (to_show_bz_lines):
        # hexagon of brioullin zone
        linewidth = 0.3
        #probably should come up with a more elegant plotting solution
        ax.plot((kp_x[0], kp_x[1]),(kp_y[0], kp_y[1]),color = "black",linewidth = linewidth)
        ax.plot((kp_x[0], kp_x[2]),(kp_y[0], kp_y[2]),color = "black",linewidth = linewidth)
        ax.plot((kp_x[2], kp_x[3]), (kp_y[2], kp_y[3]),color = "black",linewidth = linewidth)
        ax.plot((kp_x[4], kp_x[1]), (kp_y[4], kp_y[1]),color = "black",linewidth = linewidth)
        ax.plot((kp_x[4], kp_x[5]), (kp_y[4], kp_y[5]),color = "black",linewidth = linewidth)
        ax.plot((kp_x[3], kp_x[5]), (kp_y[3], kp_y[5]),color = "black",linewidth = linewidth)
    if (to_show_axes):
        plt.xlabel("$k_x (a^{-1})$")
        plt.ylabel("$k_y (a^{-1})$")
    if (filename):
        plt.savefig(filename)
        plt.show()

def complex_triple_brillouin_zone_plot(filename, k_x, k_y, raw_plot_this, k_points, vec_1, vec_2,**kwargs):
    '''Plots set of three Brioullin zones for some variable (raw_plot_this) --the real, the imaginary, and absolute values'''
    brillouin_zone_plot(filename+"real", k_x, k_y, raw_plot_this.real, k_points, vec_1, vec_2,**kwargs)
    brillouin_zone_plot(filename+"imag", k_x, k_y, raw_plot_this.real, k_points, vec_1, vec_2,**kwargs)
    brillouin_zone_plot(filename+"abs", k_x, k_y, np.abs(raw_plot_this), k_points, vec_1, vec_2,**kwargs)

def distance(vec_1,vec_2):
    '''Returns Cartesian distance between two vectors of size 2'''
    distance = np.sqrt((vec_1[0]-vec_2[0])**2+(vec_1[1]-vec_2[1])**2)
    return distance
def filter_near_points(grid_matrix,selected_points_matrix,radius):
    '''
    Takes initial kx/ky matrix and returns only the points some radial distance to symmetry points
    def filter_near_points(grid_matrix,selected_points_mat
    :param grid_matrix: 
    :param selected_points_matrix: 
    :param radius: 
    :return: 
    '''
    grid = np.array(grid_matrix)
    sel_points = np.array(selected_points_matrix)
    r = radius
    num_grid_points = np.size(grid, 1)
    num_sel_points = np.size(sel_points,0)
    filtered_grid = np.zeros([2,1])
    counter =  0
    for i in range(0,num_grid_points):
        cur_point = grid[:,i]
        for j in range(0,num_sel_points):
            cur_sel_point = sel_points[j,:]
            d = distance(cur_point,cur_sel_point)
            if d <= r :
                filtered_grid = np.column_stack((filtered_grid,cur_point))
                counter = counter+1
                break
    filtered_grid = np.delete(filtered_grid,0,1)
    return filtered_grid
def fill_filter(filtered_matrix,mask,resolution,num_of_states):
    '''Fills a filtered (see filter_near_points()) matrix with zeros. Needed for getting shapes of arrays to be correct for 
    plotting after truncation'''
    res = resolution
    if num_of_states == 1:
        fill_matrix = np.zeros(res ** 2, dtype=np.csingle)
        fill_matrix[mask] = filtered_matrix[:]
    else:
        fill_matrix = np.zeros((res ** 2, num_of_states), dtype=np.csingle)
        for i in range(0,num_of_states):
            fill_matrix[mask, i] = filtered_matrix[:, i]
    return fill_matrix
def screened_interaction(k,chi_2D):
    '''Gives 2D screened interaction. (33) in (10.1103/PhysRevB.92.085413). Fine structure constant needed for units'''
    fsc = 0.007297
    w_k = 2*np.pi*fsc/(k*(1+2*np.pi*chi_2D*k))
    return w_k
def create_pack_params(tmdc,vec_1,vec_2,res):
    '''
    Creates parameters for Monkhorst-Pack grid construction for a given TMDC
    :param tmdc: str
    :param vec_1: np.ndarray (2)
    :param vec_2: np.ndarray (2)
    :param res: int
    :return: tuple
    '''
    angstrom_to_eV_dist = 1973.27
    lattice_const = tmdc.a / angstrom_to_eV_dist
    chi_2D = tmdc.chi / angstrom_to_eV_dist
    total_area = np.cross(vec_1, vec_2) / lattice_const ** 2 #cross product evals to scalar when vectors are 2D
    d_k = np.linalg.norm(vec_1) / (res)/ lattice_const
    int_fac = total_area / (res ** 2 * (2 * np.pi) ** 2)
    return lattice_const, chi_2D, total_area, d_k, int_fac
def create_bse(raw_system):
    '''K points decoupled from one another--default case'''
    cv = np.array(raw_system.c_v)
    cc = np.array(raw_system.c_c)
    res = raw_system.res
    vec_1 = np.array(raw_system.vec_1)
    vec_2 = np.array(raw_system.vec_2)
    k_x_array = np.array(raw_system.kx)
    k_y_array = np.array(raw_system.ky)
    tau = np.array(raw_system.tau)
    energy_c = np.array(raw_system.energy_c)
    energy_v = np.array(raw_system.energy_v)
    tmdc = raw_system.tmdc
    num_k = np.size(k_x_array)
    lattice_const, chi_2D, total_area, d_k, int_fac = create_pack_params(tmdc,vec_1,vec_2,res)
    start_time = time.time()
    print("Building BSE matrix")
    diagonal_values = (energy_c-energy_v) #treating screened interaction at 0 distance as negligible
    bse_hamiltonian = np.diag(diagonal_values) + 0 * 1j #adding 0 imag unit to insure complex data type
    tau1_mask = tau == 1
    k_x_mask, k_y_mask = ma.masked_array(k_x_array, tau1_mask-1), ma.masked_array(k_y_array, tau1_mask-1)
    k_x_mask2, k_y_mask2 = ma.masked_array(k_x_array, tau1_mask), ma.masked_array(k_y_array, tau1_mask)
    for i in range(1, num_k):
        if (tau1_mask[i]):
            k_val = np.sqrt((k_x_array[i] - k_x_mask[range(i)]) ** 2 + (k_y_array[i] - k_y_mask[range(i)]) ** 2)/lattice_const
            overlap_factor = (cc[i, 0].conj() * cc[range(i), 0] + cc[i, 1].conj() * cc[range(i), 1]) *\
                                     (cv[range(i), 0].conj() * cv[i, 0] + cv[range(i), 1].conj() * cv[i, 1])
            overlap_integral = int_fac * overlap_factor
            off_diagonal_value = -screened_interaction(k_val,chi_2D)*overlap_integral
            bse_hamiltonian[i, range(i)] = ma.filled(off_diagonal_value,0)
        else:
            k_val = np.sqrt(
                (k_x_array[i] - k_x_mask2[range(i)]) ** 2 + (k_y_array[i] - k_y_mask2[range(i)]) ** 2) / lattice_const
            overlap_factor = (cc[i, 0].conj() * cc[range(i), 0] + cc[i, 1].conj() * cc[range(i), 1]) * \
                             (cv[range(i), 0].conj() * cv[i, 0] + cv[range(i), 1].conj() * cv[i, 1])
            overlap_integral = int_fac * overlap_factor
            off_diagonal_value = -screened_interaction(k_val, chi_2D) * overlap_integral
            bse_hamiltonian[i, range(i)] = ma.filled(off_diagonal_value,0)
    end_time = time.time()
    print(str(start_time-end_time)+"sec")
    return bse_hamiltonian
def create_bse_coupled(raw_system):
    '''K points coupled to one another'''
    cv = np.array(raw_system.c_v)
    cc = np.array(raw_system.c_c)
    res = raw_system.res
    vec_1 = np.array(raw_system.vec_1)
    vec_2 = np.array(raw_system.vec_2)
    k_x_array = np.array(raw_system.kx)
    k_y_array = np.array(raw_system.ky)
    energy_c = np.array(raw_system.energy_c)
    energy_v = np.array(raw_system.energy_v)
    tmdc = raw_system.tmdc
    num_k = np.size(k_x_array)
    lattice_const, chi_2D, total_area, d_k, int_fac = create_pack_params(tmdc,vec_1,vec_2,res)
    start_time = time.time()
    print("Building BSE matrix")
    diagonal_values = (energy_c-energy_v)
    bse_hamiltonian = np.diag(diagonal_values) + 0 * 1j #adding 0 imag unit to insure complex data type

    for i in range(1, num_k):
            k_val = np.sqrt((k_x_array[i] - k_x_array[range(i)]) ** 2 + (k_y_array[i] - k_y_array[range(i)]) ** 2)/lattice_const
            overlap_factor = (cc[i, 0].conj() * cc[range(i), 0] + cc[i, 1].conj() * cc[range(i), 1]) *\
                                     (cv[range(i), 0].conj() * cv[i, 0] + cv[range(i), 1].conj() * cv[i, 1])
            overlap_integral = int_fac * overlap_factor
            off_diagonal_value = -screened_interaction(k_val,chi_2D)*overlap_integral
            bse_hamiltonian[i, range(i)] = off_diagonal_value
    end_time = time.time()
    print(str(start_time-end_time)+"sec")
    return bse_hamiltonian

def solve_hamiltonian(hamiltonian_matrix):
    '''
    Solves some square Hamiltonian matrix, returns sorted energies and eigenvectors
    :param hamiltonian_matrix:
    :return:
    '''
    if (sp.linalg.ishermitian(hamiltonian_matrix)):
        e_X, vec_X = np.linalg.eigh(hamiltonian_matrix)
    else:
        e_X, vec_X = np.linalg.eig(hamiltonian_matrix)
    id_X = e_X.argsort()
    e_X = e_X[id_X]
    vec_X = vec_X[:, id_X]
    return e_X, vec_X

def solve_bse(raw_system):
    '''For a given RAW_SYSTEM, solves Bethe-Salpeter equation'''
    bse_hamiltonian = create_bse_coupled(raw_system)
    print("Solving Hamiltonian")
    e_X, vec_X = solve_hamiltonian(bse_hamiltonian)
    return e_X, vec_X

def dirac_delta(a,b,**kwargs):
    '''Simple implementation of peak that approaches the Dirac delta function (in the limit of width->0, it would be so)'''
    c = .01
    for key,value in kwargs.items():
        if key == "width":
            c = value/4.291
    delta = np.exp(-.5 *((a-b)/c)**2)
    return delta

def dirac_delta_set(spectrum,energies,width = .01):
    '''For a set of energies, gives a spectrum that models the Dirac delta function for all'''
    c = width/4.291
    spec_set = np.tile(spectrum,(np.size(energies),1)).T
    energy_set = np.tile(energies,(np.size(spectrum),1))
    delta = np.exp(-.5*(energy_set-spec_set)**2/c**2)
    return delta
def create_linear_absorption(energy_spectrum,energy_array,polarized_momentum_array,bse_solutions_matrix):
    '''energy_spectrum is the input of energies into system, whereas energy_array is the eigenvalues to bse'''
    energy_array = np.array(energy_array)
    p_array = np.array(polarized_momentum_array)
    elem_charge = 1
    mass = 5.10999e5 # MeV/c^2
    fsc =  0.007297
    a_x_matrix = np.array(bse_solutions_matrix)
    num_states = np.size(a_x_matrix,1)
    size = energy_spectrum.size
    intensity_array = np.zeros(size)
    for i in range(0,num_states):
        oscillator_strength = np.abs(np.sum((2*np.pi*elem_charge*p_array*a_x_matrix[:,i]/mass)))**2
        peak_shape = fsc/(energy_spectrum**2)*dirac_delta(energy_spectrum,energy_array[i])
        intensity_array = intensity_array+oscillator_strength*peak_shape
    return intensity_array

def element_anisotropy(a_x_array,p_array_1,p_array_2):
    '''p arrays are basis-- p_minus and p_plus for my analysis'''
    osc_strength1 = np.abs(np.sum(a_x_array*p_array_1))
    osc_strength2 = np.abs(np.sum(a_x_array*p_array_2))
    return (osc_strength1-osc_strength2)/(osc_strength1+osc_strength2)