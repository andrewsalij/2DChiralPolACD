import ldlb_jaynes_cummings as ljc
import mueller
import numpy as np
import dielectric_tensor as dt
import ljc_params
import ldlb_plotting as lp
e_1 = 1.5
e_2 = 2.5
vec_pot = 180/np.sqrt(2)
polarization = np.array([1,0])

e_x_full ,a_x_full,cav_array, spec_array = ljc_params.two_dipole_simple(e_1,e_2,vec_pot,polarization,omit_zero_state=True)
e_x_full_zs ,a_x_full_zs,cav_array, spec_array = ljc_params.two_dipole_simple(e_1,e_2,vec_pot,polarization,omit_zero_state=False)

couplings = ljc_params.chiral_couplings_two_dipoles(e_1,e_2,vec_pot,polarization)
anal_eigenvecs_object = ljc.FOUR_EIGENVECS_ANAL(cav_array,couplings,spec_array)

eigenvecs_anal = anal_eigenvecs_object.create_eigenvectors(cav_array,e_2)
probabilites_anal = np.abs(eigenvecs_anal)**2
probability_matrix = np.abs(a_x_full)**2
probability_matrix_zs = np.abs(a_x_full_zs)**2
probability_matrix_zs_sub = probability_matrix_zs[1:,1:,:]
e_x_zs_sub = e_x_full_zs[1:,:]
filename = "lhp_a180"
for i in range(0,4):
    mueller.visualize_matrix(probabilites_anal[i,:,:].T,cav_array,spec_array,filename=filename+"c"+str(i)+".png",norm_bounds = [0,0.5])
    lp.plot_set_colored(filename+"nzs"+"P_"+str(i)+".png",cav_array,e_x_full[:,:],
                        probability_matrix[i,:,:],x_label = "Cavity Energy (eV)",y_label= "Eigenstate Energy (eV)",
                        opacity = 1,norm_max = 0.5,colorbar_label=r"$P_l$",
                        **{"y_bounds":[1,4]})
    lp.plot_set_colored(filename + "zs" + "P_" + str(i) + ".png", cav_array, e_x_zs_sub[:, :],
                        probability_matrix_zs_sub[i, :, :], x_label="Cavity Energy (eV)", y_label="Eigenstate Energy (eV)",
                        opacity=1, norm_max=0.5, colorbar_label=r"$P_l$",
                        **{"y_bounds": [1, 4]})
