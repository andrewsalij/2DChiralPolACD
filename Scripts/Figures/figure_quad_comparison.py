import numpy as np
import ldlb_plotting as lp
import numpy.ma as ma

style = "default"

old_g_factor = True
cav_freq_array = np.linspace(1,4,100)
# helical_pol_stack = np.load("separated_helical_pol_vp_20_50.npy")
# energy_save=  np.load("energy_vp_20_50.npy")
if (style == "rev"): # \beta = -45 deg.
    helical_pol_stack = np.load("rev_separated_helical_pol_vp_35_70.npy")
    energy_save=  np.load("rev_energy_vp_35_70.npy")
else:
    helical_pol_stack = np.load("separated_helical_pol_vp_35_70.npy")
    energy_save=  np.load("energy_vp_35_70.npy")

if (old_g_factor):
    helical_pol_stack = helical_pol_stack*2
energy_ordered = np.stack((energy_save[:,:,0],energy_save[:,:,0],
                           energy_save[:,:,1],energy_save[:,:,1]),
                          axis = 2)
helical_pol_stack_ma = ma.masked_greater(helical_pol_stack,2)

hel_pol_stack_val = ma.filled(helical_pol_stack,0)


el_w = .3
el_h = .33
ellipse_matrix = np.array([[2.21,1.97,el_w,el_h],
                           [0,0,el_w,el_h],
                           [2.21,1.95,el_w,el_h],
                           [0,0,el_w,el_h]]) #truly elegant and clear coding my god

if (style== "rev"): prefix = "rev_"
else: prefix = ""

filename_to_save = prefix+"figure_vp_35_70_v6"
lp.plot_quad_set_shared_axes(filename_to_save + "quad_hel_pol.pdf", cav_freq_array, energy_ordered,
                    helical_pol_stack_ma, x_label=r"$\Omega$ (eV $\hbar^{-1}$)", y_label=r"$E^\alpha$ (eV)",
                    opacity=1, norm_max=2, norm_min=-2, colorbar_label=r"$\tilde{g}^\alpha$",figsize = (3.46,4),
                    **{"x_labelpad":2,"show_lines":True,"show_min_max": False,"x_bounds":[1.5,3.5],"y_bounds":[1.5,3.5],"label_fontsize":7,
                       "x_ticks":[2,2.5,3],
                       "y_ticks":[2,2.5,3],"interp_factor":1})

make_source_files = False
if (make_source_files):
    np.save("source_data_supp_figure_1_left_abcd_cav_freq",cav_freq_array)
    np.save("source_data_supp_figure_1_right_abcd_energies",energy_ordered)
    np.save("source_data_supp_figure_1_right_abcd_hel_pol",helical_pol_stack)

