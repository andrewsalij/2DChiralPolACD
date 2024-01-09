import ldlb_plotting as lp
import numpy as np
extended_basis = False
energy_save = np.load("one_v_two_energy.npy")
helical_pol_stack = np.load("one_v_two_helical_pol.npy")
if (extended_basis):
    energy_save = np.load("extended_one_v_two_energy.npy")
    helical_pol_stack = np.load("extended_one_v_two_helical_pol.npy")


old_g_factor = True
if (old_g_factor): #recall that my code goes from -1 to 1, whereas a convention is from -2 to 2
    helical_pol_stack = helical_pol_stack*2
helical_pol_stack = np.nan_to_num(helical_pol_stack)
cav_freq_array = np.linspace(1,4,100)

filename = "fig_one_v_two_11.pdf"
lp.plot_double_set_colored_shared_y_axis(filename,cav_freq_array,energy_save,helical_pol_stack,
                                         x_label=r"$\Omega$ (eV $\hbar^{-1}$)",y_label= r"$E^\alpha$ (eV)",
                                         colorbar_label = r"$\tilde{g}^{\alpha}$",norm_min = -.3,norm_max = .3,
                                         figsize = (3.46,2.3),cbar_style= "horizontal",**{"x_labelpad":2,
                                                                                         "y_labelpad":2,
                                                                                         "x_bounds":[1.8,2.2],
                                                                                       "y_bounds":[1.8,2.2],
                                                                                        "y_ticks": [1.8,1.9,2.0,2.1,2.2],
                                                                                         "cbar_ticks":[-.3,-.15,0,.15,.3],
                                                                                         "norm_style":"exp_tanh",
                                                                                         "interp_factor":20})
make_source_files = False
if (make_source_files):
    np.save("source_data_figure_5_energy",energy_save)
    np.save("source_data_figure_5_helical_pol",helical_pol_stack)
    np.save("source_data_figure_5_cav_freq",cav_freq_array)
