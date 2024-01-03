import matplotlib.pyplot as plt
import numpy as np
import ldlb_plotting
import numpy.ma as ma
import ldlb_jaynes_cummings as ljc
from matplotlib.cbook import get_sample_data
import os
#t indicates total, s indicates subset (only two dipoles, 1 and 3, chosen w/0 vib progression)
cav_freq_array_t = np.load("di_bari_cav_freq.npy")
energy_matrix_t = np.load("di_bari_energy.npy")
helical_pol_t = np.load("di_bari_helical_pol.npy")

g_factor_old_style = True
cav_freq_array_s = np.load("di_bari_subset_cav_freq.npy")
energy_matrix_s = np.load("di_bari_subset_energy.npy")
helical_pol_s = np.load("di_bari_subset_helical_pol.npy")

bp_mask = ljc.mask_for_rows(np.shape(energy_matrix_s),np.array([0,2,4]))
up_mask = ljc.mask_for_rows(np.shape(energy_matrix_s),np.array([0,1,3]))
bottom_pol = ma.filled(np.ma.masked_array(np.copy(helical_pol_s),mask = np.logical_or(ma.getmask(helical_pol_s),bp_mask)),np.nan)
upper_pol = ma.filled(np.ma.masked_array(np.copy(helical_pol_s),mask = np.logical_or(ma.getmask(helical_pol_s),up_mask)),np.nan)


helical_pol_up_container = np.zeros(np.shape(helical_pol_t))
energy_matrix_up_container= np.zeros(np.shape(energy_matrix_t))

helical_pol_bot_container = np.zeros(np.shape(helical_pol_t))
energy_matrix_bot_container= np.zeros(np.shape(energy_matrix_t))

helical_pol_up_container[:np.size(helical_pol_s,axis= 0),:] = upper_pol
energy_matrix_up_container[:np.size(energy_matrix_s,axis =0),:] = energy_matrix_s

helical_pol_bot_container[:np.size(helical_pol_s,axis= 0),:] = bottom_pol
energy_matrix_bot_container[:np.size(energy_matrix_s,axis =0),:] = energy_matrix_s

y_axis_stack = np.stack((energy_matrix_t,energy_matrix_bot_container,energy_matrix_up_container),axis = 2)
y_axis_set_color_values_stack = np.stack((helical_pol_t,helical_pol_bot_container,helical_pol_up_container),axis = 2)
if (g_factor_old_style):
    y_axis_set_color_values_stack = y_axis_set_color_values_stack*2

image_vectors_fn = get_sample_data(os.sep.join((os.getcwd(),"vectors_solo_cis.png")), asfileobj=False)
image_panel_b = plt.imread(image_vectors_fn)
filename = "figure_di_bari_comparison_v17.pdf"
ldlb_plotting.plot_triple_set_di_bari(filename,cav_freq_array_t,y_axis_stack,y_axis_set_color_values_stack,
                                      figure=  None,axis = None,x_label = r"$\Omega$ (eV $\hbar^{-1}$)",y_label = r"$E^\alpha$ (eV)",opacity = 1,norm_max =2,norm_min = -2,
                                      colorbar_label = r"$\tilde{g}^\alpha$",cmap = 'seismic',figsize = (3.46,4),cbar_style = "horizontal",
                                      **{"show_lines":True,"x_bounds":[2.25,4.75],"y_bounds":[2.25,4.75],"interp_factor":1,
                                         "x_labelpad":2,"label_fontsize":7,"ur_placeholder":True,"image":image_panel_b})


make_source_files = False
if (make_source_files):
    np.save("source_data_figure_6_cav_freq",cav_freq_array_t)
    np.save("source_data_figure_6_energy",y_axis_stack)
    np.save("source_data_figure_6_hel_pol",y_axis_set_color_values_stack)
