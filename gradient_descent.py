import pandas
import scipy.optimize as op
import numpy as np
import dielectric_tensor as dt
import matplotlib.pyplot as plt
import python_util as pu

'''
Core functionality for optimization of spectra calculations 

Functions w/o documentation are largely deprecated and should be thought more of as
scripts that are no longer in use. 
Written by Andrew Salij
'''
def flatten_cost(rot_array,vector_matrix):
    '''Cost function for projection of vectors (xyz) into xy plane'''
    rot_array = np.array(rot_array).flatten()
    vector_matrix = dt.rotate_vector(rot_array,vector_matrix.T).T
    return np.sum(vector_matrix[:,2]**2)

def flatten_dipoles(init_dipoles):
    '''
    Flattens matrix of vectors (xyz) into xy-plane such that most length of all is in xy plane
    Finds different solutions upon rerun as z rotation axis of solution is a free parameters
    which must be rebased
    :param init_dipoles: np.ndarray
    :return:np.ndarray (shape = np.shape(init_dipoles)
    '''
    rot_array = tuple((0,0,0))
    full_arguments = {"args":(init_dipoles,)}
    sols = op.basinhopping(flatten_cost,rot_array,minimizer_kwargs=full_arguments,niter = 50)
    return dt.rotate_vector(np.array(sols.x),init_dipoles.T).T, sols.x

def angles_from_2d_matrix(matrix):
    '''
    Converts 2D matrix in xy coordinates to composite angles
    :param matrix: np.ndarray
    :return: np.ndarray
    '''
    mags = np.sqrt(np.sum(matrix**2,axis = 1))
    mags_mat = np.array([mags,mags]).T
    normalized_matrix = matrix/mags_mat
    x_array = normalized_matrix[:,0]
    y_array= normalized_matrix[:,1]
    return np.arctan2(y_array,x_array)


#data must be same shape
def cost_function(training_data,test_data,lower_bound = 0,normed = True):
    '''Mean square cost function for gradient descent'''
    if (np.max(np.abs(test_data))<=lower_bound):
        return 1e5
    else:
        if (normed):
            return np.sum((pu.norm_array(training_data)-pu.norm_array(test_data))**2)/(np.sum(pu.norm_array(training_data)**2))
        else:
            return np.sum(((training_data)-(test_data))**2)/(np.sum((training_data)**2))
#array sizes must be same
def create_training_peaks(spectrum_linspace,peak_energy_array,peak_height_array, gamma_array):
    '''Adds peaks to training data'''
    num_vals = np.size(peak_energy_array)
    spec = np.zeros(np.size(spectrum_linspace))
    for i in range(0,num_vals):
        peak_to_add = np.abs(np.imag(dt.lorenzian(spectrum_linspace,peak_energy_array[i],gamma_array[i])))
        scale_factor = peak_height_array[i]/np.max(np.abs(peak_to_add))
        spec = scale_factor*peak_to_add+spec
    return spec

#input needs to be a 1D array, so order as all dipole magnitudes, then all x-axis declination angles
#polar coords inputs
def test_dipole_params(array_dipole_params,dielectric_params=  dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const = 1, volume_cell = 1e-7, damping_factor = .03,length = 1),
                       spectrum = np.linspace(0,1,100), ldlb_mimic = np.linspace(0,1,100),transition_energies =  np.ones(3), space_dim = 2):
    '''Calculates cost for a given set of dipole parameters'''
    unit_defs = dt.UNIT_DEFINITIONS(1, 1, np.pi / 0.007297)
    if space_dim == 2:
        num_dips = np.int(np.size(array_dipole_params)/space_dim)
        dip_mags = array_dipole_params[:num_dips]
        dip_angles = array_dipole_params[num_dips:]
        dipole_matrix = dt.create_dipole_matrix_polar_2D(dip_mags,dip_angles)
        a,b,c, d, ldlb_sig = dt.get_ldlb_2D(dielectric_params, dipole_matrix, transition_energies, spectrum, unit_defs)
        cost = cost_function(ldlb_mimic,ldlb_sig[:,10])
        cost = cost+cost_function(ldlb_mimic,ldlb_sig[:,20])
        return cost
    else:
    #may add 3D handling at some point-we'll see
        return 0


def optimize_dipole_params(array_dipole_params,dielectric_params=  dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const = 1, volume_cell = 1e-7, damping_factor = .03,length = 1),
                       spectrum = np.linspace(0,1,100), ldlb_mimic = np.linspace(0,1,100),transition_energies =  np.ones(3), space_dim = 2):
    '''Optimizes dipole parameters using a basin hopping algorithm'''
    full_arguments = {"args":(dielectric_params,spectrum, ldlb_mimic,transition_energies, space_dim,)}
    sols = op.basinhopping(test_dipole_params,array_dipole_params,minimizer_kwargs=full_arguments,niter = 30)
    print("Cost minimized to:"+np.str(sols.fun))
    return sols.x

#b = optimize_dipole_params(np.array([1,1,1,1,1,1]),ldlb_mimic = np.linspace(0,2,100))
class DIPOLE_PARAMS():
    '''
    Container class for parameters relating to a set of Lorentzian oscillator dipoles
    '''
    def __init__(self, dielectric_params,spectrum,target_signal,transition_energies,dipole_params):
        self.dielectric = dielectric_params
        self.spec = spectrum
        self.target = target_signal
        self.energy_array = transition_energies
        self.dipole_array = dipole_params

#minimizes perterbative 5 param function
def three_peak_func_minimize(dip_params,linear_sols = np.array([1,1,1]),e_array = np.array([1,1,1])):
    i1 =dip_params[0]**2*e_array[0]
    i2 = dip_params[1]**2*e_array[1]
    i3 = dip_params[2]**2*e_array[2]
    alpha = dip_params[3]
    beta = dip_params[4]
    eq_1 =  np.abs(i1*i2*np.sin(2*alpha)-linear_sols[0])
    eq_2 =  np.abs(i2*i3*np.sin(2*beta)-linear_sols[1])
    eq_3 =  np.abs(i1*i3*np.sin(2*(alpha+beta))-linear_sols[2])
    return eq_1+eq_2+eq_3

def n_peak_func_minimize(dip_params,linear_sols= np.zeros(4),e_array = np.zeros(4)):
    num_peaks = np.size(e_array)
    i_array = dip_params[:num_peaks]**2*e_array
    angle_array = dip_params[num_peaks:]
    eq_array = np.zeros(num_peaks)
    cost = np.sum(eq_array)
    return cost



def gradient_descent_sweep(dip_param_sets_to_store,dip_param_measure_delta,
                           dielectric_params=  dt.DIELECTRIC_PARAMS(iso_hf_dielectric_const = 1, volume_cell = 1e-7, damping_factor = .03,length = 1),
                       spectrum = np.linspace(100,500,100), ldlb_mimic = np.linspace(100,500,100),transition_energies =  np.ones(3), space_dim = 2):
    unit_defs = dt.unit_defs_base
    test_style = "random"
    num_dip_params = np.int(np.size(transition_energies))
    sols_data_array = []
    magnitude_bounds = np.array([1e-6,1e-3])
    limit_param = 3 #1 for testing
    limit = np.int(limit_param*dip_param_sets_to_store)#keeps this from running forever
    counter = 0
    while (len(sols_data_array) < dip_param_sets_to_store and counter <= limit):
        counter = counter +1
        if (test_style == "random"):
            dipole_magnitude_array = (magnitude_bounds[1]-magnitude_bounds[0])*np.random.random_sample(num_dip_params)+magnitude_bounds[0]
            dipole_angle_array = 2*np.pi*np.random.random_sample(num_dip_params)
        if (test_style == "perturbative_3_peak"):
            gamma = dielectric_params.gamma
            w1 = transition_energies[0]
            w2 = transition_energies[1]
            w3 = transition_energies[2]
            ldlb_1 = ldlb_mimic[np.argmin(np.abs(spectrum-w1))]
            ldlb_2 = ldlb_mimic[np.argmin(np.abs(spectrum - w2))]
            ldlb_3 = ldlb_mimic[np.argmin(np.abs(spectrum - w3))]
            lin_sys = np.array([[dt.pert_lineshape(w1,w2,w1,gamma),0,dt.pert_lineshape(w1,w3,w1,gamma)],
                                [dt.pert_lineshape(w1,w2,w2,gamma),dt.pert_lineshape(w2,w3,w2,gamma),0],
                                [0,dt.pert_lineshape(w2,w3,w3,gamma),dt.pert_lineshape(w1,w3,w3,gamma)]])
            n_factor = dt.ldlb_pert_factor(dielectric_params.epsilon_inf,unit_defs,dielectric_params.v)
            lin_peaks = np.array([ldlb_1,ldlb_2,ldlb_3])
            lin_sys = lin_sys*n_factor
            lin_sols = np.linalg.solve(lin_sys,lin_peaks)
            dipole_magnitude_array = (magnitude_bounds[1] - magnitude_bounds[0]) * np.random.random_sample(
                num_dip_params) + magnitude_bounds[0]
            dipole_angle_array = 2 * np.pi * np.random.random_sample(num_dip_params-1)
            dipole_params_array_start = np.hstack((dipole_magnitude_array, dipole_angle_array))
            magnitude_angle_bounds = ((magnitude_bounds[0], magnitude_bounds[1]), (magnitude_bounds[0], magnitude_bounds[1]),(magnitude_bounds[0], magnitude_bounds[1]),
                                        (0,2*np.pi),(0,2*np.pi))
            full_args = {"args": (lin_sols,),"bounds":magnitude_angle_bounds}
            optimized_params = op.basinhopping(three_peak_func_minimize,dipole_params_array_start,minimizer_kwargs=full_args,niter = 100)
            print("init_cost:"+str(optimized_params.fun))
            dipole_magnitude_array = optimized_params.x[:3]
            dipole_angle_array = np.array([0,optimized_params.x[3],optimized_params.x[4]])

        dipole_params_array = np.hstack((dipole_magnitude_array,dipole_angle_array))
        init_all_params = DIPOLE_PARAMS(dielectric_params,spectrum,ldlb_mimic,transition_energies,dipole_params_array)
        init_peaks = dipole_params_to_ldlb(init_all_params)[4][:, 10]
        plt.plot(spectrum,init_peaks)
        plt.show()
        print("testing:"+str(dipole_params_array))
        #for testing
        #dipole_params_array = np.array([1.5e-3*np.sqrt(.1),1.5e-3*np.sqrt(.15),1.5e-3*np.sqrt(.25),np.pi,np.pi/3,np.pi/3-np.pi/4])
        optimized_sols = optimize_dipole_params(dipole_params_array,dielectric_params,spectrum,ldlb_mimic,transition_energies,space_dim)
        if (len(sols_data_array) ==  0):
            dipole_to_save_object = DIPOLE_PARAMS(dielectric_params, spectrum, ldlb_mimic, transition_energies,
                                                  optimized_sols)
            sols_data_array.append(dipole_to_save_object)
        else:
            for i in range(0,len(sols_data_array)):
                measure = np.sqrt(np.sum((sols_data_array[i].dipole_array-optimized_sols)**2))
                #print(measure)
                if (np.sum(np.abs(sols_data_array[i].dipole_array)) > dip_param_measure_delta*measure):
                    dipole_to_save_object = DIPOLE_PARAMS(dielectric_params,spectrum,ldlb_mimic,transition_energies,optimized_sols)
                    sols_data_array.append(dipole_to_save_object)

    return sols_data_array

def dipole_params_to_ldlb(dipole_params,to_flip = False):
    params = dipole_params.dipole_array
    num_dips = np.int(np.size(params) / 2)
    dip_mags = params[:num_dips]
    dip_angles = params[num_dips:]
    dipole_matrix = dt.create_dipole_matrix_polar_2D(dip_mags, dip_angles)
    if (to_flip):
        dipole_matrix[:,0] = -1*dipole_matrix[:,0]
    signal = dt.get_ldlb_2D(dipole_params.dielectric,dipole_matrix,dipole_params.energy_array,dipole_params.spec)
    return signal

def gaussian(x_array, height,center,sigma):
    return height*(np.exp((-0.5/(sigma**2))*((x_array-center)**2)))

def lorentzian_parameterized(x_array,height,center,width):
    loren_param = (x_array-center)/(width*0.5)
    return height*1/(1+loren_param**2)

def lorentzian_dielectric(x_array,gamma_array,height,center):
    return x_array*height*gamma_array*x_array/((x_array**2-center**2)**2+gamma_array**2*x_array**2)

def lorenztian_dielectric_multi(x_array,gamma_array,*params):
    params = np.array(params).flatten()
    num_peaks = np.int(np.size(params) / 2)
    y = np.zeros(np.size(x_array))
    for i in range(0,num_peaks):
        y = y + lorentzian_dielectric(x_array,gamma_array,params[2*i],params[2*i+1])
    return y

def params_fit(func,x_array,y_data,params_array,bounds = []):
    if (bounds):popt, pcov = op.curve_fit(func, x_array, y_data, p0=params_array,bounds=bounds)
    else:popt, pcov = op.curve_fit(func, x_array, y_data, p0=params_array,maxfev = 10000)
    return popt


def get_lin_abs_from_params(spectrum,ldlb_prefactor,e_array,dip_mags,gamma_array):
    '''Converts set of dipole parameters into a linear absorption spectrum
    Somewhat deperecated--use LINEAR_OPTICS() in dielectric_tensor.py instead'''
    height_array = dip_mags ** 2 * e_array * ldlb_prefactor
    params = tuple(pu.interweave_arrays(height_array, e_array))
    lin_abs = lorenztian_dielectric_multi(spectrum, gamma_array, params)
    return lin_abs


def get_ldlb_double_spec_helix_params(spectrum,dielectric_params,e_array,dip_mags,dip_angles,gamma_array,total_rotation):
    ldlb = dt.ldlb_helical_perturbative(spectrum,dielectric_params,e_array,dip_mags,dip_angles,gamma_array,total_rotation)
    ldlb_flip = dt.ldlb_helical_perturbative(spectrum,dielectric_params,e_array,dip_mags,-1*dip_angles,gamma_array,total_rotation)
    ldlb_ss = (ldlb_flip+ldlb)/2
    return ldlb,ldlb_ss

#note that bounds are equidistant in each direction from initial
# it may proof worthwhile to make this allow for assymetric bounds, but this is
#fine for now
def get_bounds_set(init_arg_array,percent_bounds = 1,offset_bounds = 0):
    '''
    Provides set of boundaries with arbitrary relative and absolute offsets
    :param init_arg_array: np.ndarray
    :param percent_bounds: np.float (default 1.0)
    :param offset_bounds: np.float (default 0.0)
    :return: np.ndarray
    '''
    bounds = []
    if (np.isscalar(percent_bounds)):
        percent_bounds = percent_bounds*np.ones(np.size(init_arg_array))
    if (np.isscalar(offset_bounds)):
        offset_bounds = offset_bounds*np.ones(np.size(init_arg_array))
    percent_bounds, offset_bounds = np.abs(percent_bounds), np.abs(offset_bounds)
    for i in range(0,np.size(init_arg_array)):
        new_bounds = (init_arg_array[i]-offset_bounds[i]-percent_bounds[i]*init_arg_array[i],
                      init_arg_array[i]+offset_bounds[i]+percent_bounds[i]*init_arg_array[i])
        bounds.append(new_bounds)
    return bounds


def simple_ldlb(spectrum,energies,dip_mags,dip_angles,damping_array,prefactor):
    '''this is a model that minimizes total parameters for easy solving
    params are prefactor = xi, e_1,e_2,mu_1,mu_2,gamma_1,gamma_2'''
    ldlb,abs = np.zeros(np.size(spectrum)), np.zeros(np.size(spectrum))
    for n in range(0,np.size(energies)):
        v_n = dt.f_dielectric_im(energies[n],spectrum,damping_array[n])
        for m in range(0,np.size(energies)):
            w_m = dt.f_dielectric_real(energies[m],spectrum,damping_array[m])
            dipole_contributions = dip_mags[n]**2*dip_mags[m]**2*energies[m]*energies[n]
            total_contribution_ldlb = dipole_contributions*v_n*w_m*np.sin(2*(dip_angles[n]-dip_angles[m]))
            ldlb = ldlb+total_contribution_ldlb
    ldlb_total = prefactor**2*spectrum**2*ldlb
    return ldlb_total

def simple_ldlb_from_params(params_tuple,spectrum,dip_angles,prefactor):
    '''Provides LDLB from a tuple of dipole parameters for fitting'''
    e_1,e_2 ,mu_1,mu_2 ,gamma_1,gamma_2 = params_tuple
    energies = np.array([e_1,e_2])
    dip_mags = np.array([mu_1,mu_2])
    damping_array = np.array([gamma_1,gamma_2])
    return simple_ldlb(spectrum,energies,dip_mags,dip_angles,damping_array,prefactor)

def simple_two_dipole_ldlb_to_optimize(params,spectrum = np.linspace(1,2,100),experimental_data = np.linspace(1,2,100),dip_angles= np.array([0,np.pi/4]),prefactor = 1):
    predicted_data = simple_ldlb_from_params(params,spectrum,dip_angles,prefactor)
    return cost_function(experimental_data,predicted_data,normed = False)

def optimize_two_dipole_ldlb_to_data(energies_0,dip_mags_0,damping_array_0,spectrum,experimental_data,dip_angles,prefactor,per_offset = 1,raw_offset = 0):
    init_params = np.hstack((energies_0,dip_mags_0,damping_array_0))
    bounds = get_bounds_set(init_params,per_offset,raw_offset) #energies don't get to shift much, other params do
    method = "L-BFGS-B"
    full_arguments = {"args": (spectrum, experimental_data,dip_angles,prefactor,),
                      "bounds":bounds,"method":method}
    sols = op.basinhopping(simple_two_dipole_ldlb_to_optimize, init_params, minimizer_kwargs=full_arguments, niter=100)
    return sols.x

def two_dipole_ldlb_to_spectrum(energies_0,dip_mags_0,damping_array_0,spectrum,experimental_data,dip_angles,prefactor):
    init_params = np.hstack((energies_0,dip_mags_0,damping_array_0))
    ldlb_cost = simple_two_dipole_ldlb_to_optimize(init_params,spectrum,experimental_data,dip_angles,prefactor)
    return ldlb_cost

def random_offset_to_array(array,offsets,type = "absolute"):
    '''Deprecated use random_offset_array_to_array() instead'''
    return random_offset_array_to_array(array,offsets,type= type,uniform_offset = True)

def random_offset_array_to_array(array,offsets,type ="absolute",uniform_offset = False):
    '''
    Takes an array and offsets in a variety of manners dictated by "type"
    :param array: np.ndarray
    :param offsets: np.ndarray
    :param type: str
    :param uniform_offset: bool
    :return: np.narray
    '''
    n = np.size(array)
    if np.isscalar(offsets): offsets = np.ones(n)*offsets
    if (uniform_offset): random_array = 2 * (np.random.rand(1) - .5)*np.ones(n)
    else: random_array = 2 * (np.random.rand(n) - .5)  # -1 to 1, single value
    if (type =="scale"):offset_array = array*(1+random_array*offsets)
    elif (type == "absolute"):offset_array = random_array*offsets + array
    else:ValueError("Invalid offset type")
    return offset_array

def random_offset_to_scalar(scalar,offset):
    random_scalar = (2*(np.random.rand(1).item())-.5)*offset
    return scalar+random_scalar

class Random2DDipoleOrientation():
    def __init__(self,dip_mags,dip_angles,dip_energies,mags_offsets,angles_offsets,energies_offsets):
        self.dip_mags = random_offset_to_array(dip_mags,mags_offsets)
        self.dip_angles=  random_offset_to_array(dip_angles,angles_offsets)
        self.dip_energies = random_offset_to_array(dip_energies,energies_offsets)


def add_row_array_to_dataframe(dataframe,row_array):
    '''Takes a pandas DataFrame and adds a numpy row array to it'''
    dataframe.loc[len(dataframe)] = row_array.tolist()

