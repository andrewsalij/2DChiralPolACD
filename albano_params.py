import numpy as np
import dielectric_tensor as dt
import python_util
import gradient_descent as gd
import python_util as pu
'''
Paramterization of Albano's 2017 oligothiophene 

Synthesis of molecule reported in 
Albano, G., Lissia, M., Pescitelli, G., Aronica, L. A., & Di Bari, L. (2017).
 Chiroptical response inversion upon sample flipping in thin films of a chiral benzo [1, 2-b: 4, 5-b′] 
 dithiophene-based oligothiophene. Materials Chemistry Frontiers, 1(10), 2047-2056.
doi 10.1039/C7QM00233E 
Also includes backend for handling TDDFT results, which is used in other places such as ptpo_params.py
'''
cd_per_mdeg_to_cd_factor = 3.491e-5

albano_2017_nm = np.array([300,354.2168675,421.686747])
albano_2017_cd = np.array([147.4123093,107.1517597,-70.02195976])*cd_per_mdeg_to_cd_factor

class TDDFT_RESULTS():
    def __init__(self,energies,dipole_matrix,osc_strength_array,optimization_str,tddft_str):
        '''

        :param energies:
        :param dipole_matrix:
        :param osc_strength_array: \propto \mu^2 \omega
        :param optimization_str:
        :param tddft_str:
        '''
        self.energies = energies.flatten()
        self.dip_mat = dipole_matrix
        self.osc_array = osc_strength_array.flatten()
        self.opt = optimization_str
        self.tddft = tddft_str

    def vibronic_dressing(self,vib_index_array,vib_dist_array,huang_rhys_array,vib_modes = np.arange(4),to_zero_unselected = False):
        peaks_per_vib_dressing = np.size(vib_modes)
        total_peaks = peaks_per_vib_dressing*np.size(vib_index_array)+np.size(self.energies)-np.size(vib_index_array)
        new_energies = np.zeros(total_peaks)
        new_osc_str = np.zeros(total_peaks)
        new_dip_mat = np.zeros((total_peaks,3))
        start_index = 0
        end_index = 0
        for i in range(0,np.size(self.energies)):
            start_index = end_index
            if (i in vib_index_array):
                end_index = end_index+np.size(vib_modes)
                #print("start:end"+str(start_index)+":"+str(end_index))
                cur_index = np.argwhere(vib_index_array==i)
                vib_osc = python_util.remove_unnecessary_indices(dt.vib_spec_osc_str(self.osc_array[i], vib_modes, huang_rhys_array[cur_index]))
                energies_to_add = vib_modes * vib_dist_array[cur_index] + self.energies[i]
                #vib_osc_ratio_array = vib_osc / self.osc_array[i]
                if (self.dip_mat is not None):
                    init_dipole = python_util.remove_unnecessary_indices(self.dip_mat[i,:])
                    new_dipoles = dt.vib_spec_dip_mat(init_dipole,vib_modes,huang_rhys_array[cur_index])
                    new_dip_mat[start_index:end_index,:] = new_dipoles
                new_energies[start_index:end_index] = energies_to_add
                new_osc_str[start_index:end_index] = vib_osc
            else:
                end_index = end_index+1
                if (to_zero_unselected):nonselected_factor = 0
                else: nonselected_factor = 1
                if (self.dip_mat is not None):
                    new_dip_mat[start_index:end_index, :] = self.dip_mat[i,:]*nonselected_factor
                new_energies[start_index:end_index] = self.energies[i]*nonselected_factor
                new_osc_str[start_index:end_index] = self.osc_array[i]*nonselected_factor
        return TDDFT_RESULTS(new_energies,new_dip_mat,new_osc_str,self.opt,self.tddft)
    def linear_lorentzian_spec(self,spectrum,gamma_array):
        params = tuple(python_util.interweave_arrays(self.osc_array/self.energies,self.energies))
        return gd.lorenztian_dielectric_multi(spectrum,gamma_array,params)
    def linear_lorentzian_spec_set_amp(self,spectrum,gamma_array,amp_array):
        params = tuple(python_util.interweave_arrays(amp_array, self.energies))
        return gd.lorenztian_dielectric_multi(spectrum, gamma_array, params)
    def truncate_selection(self,select_array):
        truncated_results = TDDFT_RESULTS(self.energies[select_array],self.dip_mat[select_array,:],self.osc_array[select_array],
                                          self.opt,self.tddft)
        return truncated_results

class VIB_DRESSING():
    def __init__(self,vib_index_array,vib_dist_array,huang_rhys_array):
        self.vib_index = vib_index_array
        self.vib_dist = vib_dist_array
        self.huang_rhys = huang_rhys_array


#CAM B3LYP Optimization, 6-31 g basis
energies = np.array([3.2945,3.5038,3.8503,4.2738,4.3353,4.4496,4.6163,4.6640,4.6824,4.7127])
osc = np.array([.24013,1.96077,.02808,0.1106,.20914,.00509,.05467,.18778,.04686,.22454])
td = np.array([[.26196,1.70473,-.01732],
            [-.0723,4.77697,-.13094],
               [.37666,-.38852,-.06997],
               [.20022,.19311,.16807],
               [.62570,-1.25456,-.06037],
               [-.09496,-.12305,-.15013],
               [0.05392,.68691,-.09307],
               [0.07104,1.27993,-.01015],
               [-.00861,.59724,.22747],
               [-.35659,1.34743,-.04547]
            ])

meijer_cam_b3lyp_results = TDDFT_RESULTS(energies,td,osc,"CAM-B3LYP","CAM-B3LYP")

#2017 albano TDDFT
#B3LYP TDDFT, no geometric optimization, 6-31 g basis
albano_2017_b3lyp_eV = np.array([2.6916,3.0051,3.1885,3.4128,3.7145])
albano_2017_b3lyp_osc_array = np.array([1.1541,.00918,.99384,.03802,.35345])
albano_2017_b3lyp_td_array = np.array([[.449646,4.1560,.16013],
                                    [.00471,.34546,.07394],
                                     [-.32020,3.55204,-.0613],
                                     [-.55946,.3684,.09930],
                                     [-.27563,1.947,.12842]  ])
albano_b3lyp_results = TDDFT_RESULTS(albano_2017_b3lyp_eV,albano_2017_b3lyp_td_array,albano_2017_b3lyp_osc_array,"none","B3LYP")


#CAM-B3LYP TDDFT (def2-TZVP), 6-31G optimization
albano_2017_cam_eV = np.array([3.3189,3.9060,4.4851])
albano_2017_cam_osc_array = np.array([2.575549,.130949,.028616])
albano_2017_cam_td_array = np.array([[5.6207,-.2844,-.0536],[-.9991,-.5770,-.1933],[-.3586,.3476,-.1050]])
albano_cam_results = TDDFT_RESULTS(albano_2017_cam_eV,albano_2017_cam_td_array,albano_2017_cam_osc_array,"wB76-D","CAM-B3LYP")

#same as above but with 6-31G(d,p) optimization
albano_2017_cam_eV_reprod = np.array([3.4709,3.9595,4.6705])
albano_2017_cam_osc_array_reprod = np.array([2.35609,.2551745580,.2489671694])
albano_2017_cam_td_array_reprod = np.array([[-5.2487,.3878,.0900],[-1.5039,-.5748,-.1959],[-1.4533,.2070,-.1443]])
albano_cam_reprod_results = TDDFT_RESULTS(albano_2017_cam_eV_reprod,albano_2017_cam_td_array_reprod,albano_2017_cam_osc_array_reprod,"wB76-D","CAM-B3LYP")


#from linear fitting to absorption spectra in wollfs et al 2007

vib_index_array = np.array([0])
huang_rhys_array = np.array([1.17746])
vib_dist_array = np.array([.18074])
opv_vibronic_dressing = VIB_DRESSING(vib_index_array,vib_dist_array,huang_rhys_array)

#lin abs_curve_fitting
#eV
albano_2017_lin_abs_energies = np.array([2.559,2.78,3.001,3.24,3.98])
albano_2017_lin_abs_intensities = np.array([1.04875,2.471,1.5000,1.92955,.9001])


#lin abs curve fitting vibronic progression poisson
#from lower res
# vib_dist = .16515
# vib_modes = np.arange(4)
# vib_osc = dt.vib_spec_osc_str(2.40440,vib_modes,0.85430)
# albano_2017_lin_abs_energies_possoin = np.hstack((np.array([3.03,4.0031]),vib_modes*vib_dist+2.59826))
# albano_2017_lin_abs_intensities_possoin = np.hstack((np.array([2.28283,.63938]),vib_osc))

#from higher res =320 points
vib_dist = .16493
vib_modes = np.arange(4)
vib_osc = dt.vib_spec_osc_str(.25198,vib_modes,0.86618)
albano_2017_lin_abs_energies_possoin = np.hstack((np.array([3.02339,3.98917]),vib_modes*vib_dist+2.59359))
albano_2017_lin_abs_intensities_possoin = np.hstack((np.array([.22553,.06388]),vib_osc))

vib_index_array = np.array([0])
huang_rhys_array = np.array([.86618])
vib_dist_array = np.array([.16493])

otp_vibronic_dressing = VIB_DRESSING(vib_index_array,vib_dist_array,huang_rhys_array)

#from lorentzian-oscillator model

#for thin film fit--use this for paper (!)
otp_energies = np.array([2.59847,3.25048,3.99654])
otp_osc = np.array([.08479,.07450,.06124])
otp_lorentz_film_results = TDDFT_RESULTS(otp_energies,None,otp_osc,"lorentz_fit","lorentz_fit")
otp_lorentz_film_vib_dressing = VIB_DRESSING(np.array([0]),np.array([.19833]),np.array([1.23691]))
otp_lorentz_film_results_vib = otp_lorentz_film_results.vibronic_dressing(otp_lorentz_film_vib_dressing.vib_index,otp_lorentz_film_vib_dressing.vib_dist,otp_lorentz_film_vib_dressing.huang_rhys,vib_modes= np.arange(8))


#MONOMER! DON'T YOU DARE THINK THAT THIS IS FROM A FIT TO THE FILM
#Andrew from the future.
#-sincerely, Andrew from the past
otp_energies = np.array([2.71874,3.3134,4.05212])
otp_osc = np.array([.96649,.11393,.10853])

otp_lorentz_results = TDDFT_RESULTS(otp_energies,None,otp_osc,"lorentz_fit","lorentz_fit")
otp_lorentz_vib_dressing = VIB_DRESSING(np.array([0]),np.array([.15959]),np.array([1.33175]))
otp_lorentz_results_vib = otp_lorentz_results.vibronic_dressing(otp_lorentz_vib_dressing.vib_index,
                                                                otp_lorentz_vib_dressing.vib_dist,
                                                                otp_lorentz_vib_dressing.huang_rhys,
                                                         vib_modes= np.arange(8))
#otp from monomer spec--old way, unfixed gamma parameter

vib_dist = .15433
vib_modes = np.arange(4)
vib_osc = dt.vib_spec_osc_str(1.51497,vib_modes,1.06724)
albano_2017_mon_lin_abs_energies_possoin = np.hstack((np.array([3.02339,3.98917]),vib_modes*vib_dist+2.71492))
albano_2017_mon_lin_abs_intensities_possoin = np.hstack((np.array([.41514,.013562]),vib_osc))

vib_index_array = np.array([0])
huang_rhys_array = np.array([1.06724])
vib_dist_array = np.array([.15433])
otp_mon_vibronic_dressing = VIB_DRESSING(vib_index_array,vib_dist_array,huang_rhys_array)

#sc-4 lin abs params--from single vib fitting to ~2.9 eV peak--to return to
vib_dist = .18619
vib_modes = np.arange(4)
vib_osc = dt.vib_spec_osc_str(.87154,vib_modes,.98905)
sc4_lin_abs_energies_possoin = np.hstack((np.array([2.68137,3.52306,3.68791]),vib_modes*vib_dist+2.87427))
sc4_lin_abs_intensities_possoin = np.hstack((np.array([.24005,.08140,.13962]),vib_osc))

#congo_red lin_abs params
congo_red_lin_abs_energies = np.hstack([2.49148,3.60883,5.0875,6.77050])
congo_red_lin_abs_intensities = np.hstack([.39324,.26651,.18528,.50553])

##Solved parameters
##DEPRECATED
def get_solved_params_v1():
    Warning("DO NOT RUN THIS FILE-----DEPRECATED- use most recent version of get_solved_params()")
    unit_defs = dt.unit_defs_base
    dielectric_inf = 2.1  # placeholder
    damping_factor = .098
    volume = 2.323e-7  # in eV vol units
    dp = dt.DIELECTRIC_PARAMS(dielectric_inf, volume, damping_factor, length=7.20e3 / 1973, gamma_type="linear")

    non_vib_energies = np.array([2.6, 3.11, 3.99])
    my_mock_results = TDDFT_RESULTS(non_vib_energies, None, np.array([.455, .11, .10]), "my_brain", "my_brain")
    vib_results = my_mock_results.vibronic_dressing(np.array([0, 1, 2]), np.array([.185, .185, .185]),
                                                    np.array([1.04, 0, 0]), vib_modes=np.arange(8))
    alpha = .06 * np.pi
    beta = -.175 * np.pi
    init_dip_mags = np.sqrt(vib_results.osc_array)
    first_transition_strength = 0.0014407702432843387  # in e hc/eV units
    dip_mags = first_transition_strength / np.sqrt(my_mock_results.osc_array[0]) * init_dip_mags
    dip_mags_solo = np.sqrt(my_mock_results.osc_array) * first_transition_strength / np.sqrt(
        my_mock_results.osc_array[0])
    e_array = vib_results.energies
    dip_angles = np.hstack((np.ones(8) * 0, np.ones(8) * alpha, np.ones(8) * beta))
    spec_its = 2000
    spec = dt.nm_to_eV(np.linspace(250, 500, spec_its))
    dipole_mat = dt.create_dipole_matrix_polar_3D(dip_mags, dip_angles)
    dielectric_tensor = dt.create_dielectric_tensor(dp, dipole_mat, e_array, spec, unit_defs, **{"dimension": 3})
    return spec,e_array, dp, dipole_mat,dielectric_tensor

#current usage 7.28.2022
def get_solved_params_v2(e_inf = 8):
    unit_defs = dt.unit_defs_base
    dielectric_inf = e_inf
    volume = 2.323e-7*2.5  # in eV vol units
    damping_factor = .098
    dp = dt.DIELECTRIC_PARAMS(dielectric_inf, volume, damping_factor, length=850 / 1973, gamma_type="linear")
    non_vib_energies = np.array([2.6, 3.11, 3.99])
    my_mock_results = TDDFT_RESULTS(non_vib_energies, None, np.array([.455, .11, .10]), "my_brain", "my_brain")
    vib_results = my_mock_results.vibronic_dressing(np.array([0, 1, 2]), np.array([.185, .185, .185]),
                                                    np.array([1.09, 0, 0]), vib_modes=np.arange(8))
    alpha = .0583 * np.pi #10.5 deg
    beta = -.175 * np.pi # -31.5 deg
    init_dip_mags = np.sqrt(vib_results.osc_array)
    first_transition_strength = 0.0014407702432843387  # in e hc/eV units
    dip_mags = first_transition_strength / np.sqrt(my_mock_results.osc_array[0]) * init_dip_mags
    e_array = vib_results.energies
    dip_angles = np.hstack((np.ones(8) * 0, np.ones(8) * alpha, np.ones(8) * beta))
    spec = dt.nm_to_eV(np.linspace(250, 600, 2000))
    dipole_mat = dt.create_dipole_matrix_polar_3D(dip_mags, dip_angles)
    dielectric_tensor = dt.create_dielectric_tensor(dp, dipole_mat, e_array, spec, unit_defs, **{"dimension": 3})
    return spec,e_array, dp, dipole_mat,dielectric_tensor