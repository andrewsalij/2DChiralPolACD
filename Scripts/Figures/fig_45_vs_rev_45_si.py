import ldlb_plotting as lp
import numpy as np
extended_basis = False
energy_save = np.load("one_v_two_energy.npy")
helical_pol_stack = np.load("one_v_two_helical_pol.npy")
energy_save_ext = np.load("rev_one_v_two_energy.npy")
helical_pol_stack_ext = np.load("rev_one_v_two_helical_pol.npy")

helical_pol_to_plot = np.zeros(np.shape(helical_pol_stack_ext))
energy_to_plot = np.zeros(np.shape(energy_save_ext))

helical_pol_to_plot[:5,:,0] = helical_pol_stack[:,:,0]
helical_pol_to_plot[:,:,1] = helical_pol_stack_ext[:,:,0]

energy_to_plot[:5,:,0] = energy_save[:,:,0]
energy_to_plot[:,:,1] = energy_save_ext[:,:,0]

old_g_factor = True
if (old_g_factor): #recall that my code goes from -1 to 1, whereas a convention is from -2 to 2
    helical_pol_to_plot = helical_pol_to_plot*2
helical_pol_stack = np.nan_to_num(helical_pol_stack)
cav_freq_array = np.linspace(1,4,100)

filename = "fig_45_vs_rev_45.pdf"
lp.plot_double_set_colored_shared_y_axis(filename,cav_freq_array,energy_to_plot,helical_pol_to_plot,
                                         x_label=r"$\Omega$ (eV/$\hbar$)",y_label= r"$E^\alpha$ (eV)",
                                         colorbar_label = r"$\tilde{g}^{\alpha}$",norm_min = -.3,norm_max = .3,
                                         figsize = (3.3,2.3),cbar_style= "horizontal",**{"x_labelpad":2,
                                                                                         "y_labelpad":2,
                                                                                         "x_bounds":[1.6,2.4],
                                                                                       "y_bounds":[1.8,2.2],
                                                                                        "y_ticks": [1.8,1.9,2.0,2.1,2.2],
                                                                                         "cbar_ticks":[-.3,-.15,0,.15,.3],
                                                                                         "norm_style":"exp_tanh"})