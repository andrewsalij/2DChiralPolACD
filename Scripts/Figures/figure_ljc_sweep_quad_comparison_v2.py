import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

g_factor_old_style = True
extended_spectral_range = True

if (extended_spectral_range):
    filename = "lower_polariton_variable_comparison_exd_range.npy"
else:
    filename = "lower_polariton_variable_comparison.npy"

filename_2 = "lower_polariton_2pt1comparison_bcd.npy"

helical_pol_lp_a = np.load(filename)
helical_pol_lp_bcd = np.load(filename_2)

if (g_factor_old_style):
    helical_pol_lp_bcd = helical_pol_lp_bcd*2
    helical_pol_lp_a = helical_pol_lp_a*2
if (extended_spectral_range):
    cav_freq_array = np.linspace(1.5,3.5,201)
else:
    cav_freq_array = np.linspace(1.5,2.5,101)

set_size  =100
gam_array = np.linspace(.03,.15,set_size)
del_energy_array = np.linspace(0,2,set_size)
figure, axes = plt.subplots(2,2,figsize = (3.46,4)) #88 mm width


params = {'legend.fontsize':7,'legend.title_fontsize':7,
          'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7}
plt.rcParams.update(params)
label_fontsize = 7
default_color = "red"
second_color = "black"
third_color = "blue"

default_linestyle = "solid"
second_linestyle = "-."
third_linestyle = "dotted"


x_title_offset_1 = -.30
y_title_offset_1= .92

x_title_offset_2 = -.16
y_title_offset_2= .94

x_st = [x_title_offset_1,x_title_offset_2]
y_st = [y_title_offset_1,y_title_offset_2]

ylim_bounds = [-.73,.26]
xlim_bounds = [1.25,2.5]


def visible_x_axis(axis):
    axis.axline((0, 0), (1, 0), color="black", linestyle="solid", linewidth=1)


st_labels = ["a","b","c","d"]
k= 0
for i in range(0,2):
    for j in range(0,2):
        if (i == 0):
            axes[i,j].set_ylim(ylim_bounds[0],ylim_bounds[1])
            axes[i, j].set_xlim(xlim_bounds[0], xlim_bounds[1])
            #axes[i, j].set_yticks([-.6,-.4, -.2, 0, .2])
        axes[i,j].set_title(st_labels[k], fontsize=7, x=x_st[j], y=y_st[i],weight ="bold")
        axes[i,j].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        k = k+1
        visible_x_axis(axes[i,j])
if (extended_spectral_range):
    axes[0,0].set_xlim(np.min(cav_freq_array),np.max(cav_freq_array))
for i in range(0,2):
    #axes[0,1].set_yticklabels([])
    axes[i,0].tick_params(axis = 'y',labelsize = 7,pad = 0)
    axes[i, 1].tick_params(axis='y', labelsize=7, pad=0)
    #axes[0,i].set_xticklabels([])
    axes[1, i].tick_params(axis='x', labelsize=7)
    axes[0, i].tick_params(axis='x', labelsize=7)
    axes[i, 0].set_ylabel(r"$\tilde{g}^{\alpha = 1}$",labelpad=0,fontsize = label_fontsize)

shared_line_color = "orange"
ax = axes[0,1]
ratio_array= np.linspace(.3,3,100)
ax.set_xlabel(r"$\mu_2:\mu_1$",  fontsize=label_fontsize,labelpad=0)

ax.plot(ratio_array,helical_pol_lp_bcd[0,:],color= default_color)
ax.set_xlim(.5,2)
ax.set_ylim(-.8,.1)
ax.axline((1,-.1),(1,.1),color = shared_line_color)
ax.set_yticks([-.8,-.6,-.4, -.2, 0])
ax.set_xticks([.5,1,1.5,2])
ax.set_xticklabels(["0.5","1","1.5","2"])

ax = axes[1,0]
ax.plot(gam_array,helical_pol_lp_bcd[1,:],color = default_color,linestyle = default_linestyle)
ax.set_xlim(.02,.18)
ax.set_ylim(-.35,0)
ax.axline((.1,-.3),(.1,-.31),color = shared_line_color)
ax.set_xlabel(r"$\gamma$",fontsize = label_fontsize,labelpad= 0)
ax.set_yticks([-.3,-.2,-.1, 0])


ax = axes[0,0]
axes[0, 0].set_xlabel(r"$\Omega$ (eV $\hbar^{-1}$)",  fontsize=label_fontsize,labelpad=0)
ax.axline((2.1,-.3),(2.1,-.31),color = shared_line_color)

helical_pol_lp_a_to_use = np.vstack((helical_pol_lp_a[0,:],helical_pol_lp_a[6,:]))

ax.plot(cav_freq_array,helical_pol_lp_a_to_use[0,:],label = "35",color = default_color,linestyle = default_linestyle)
ax.plot(cav_freq_array,helical_pol_lp_a_to_use[1,:],label = "70",color = default_color,linestyle = second_linestyle)

ax.legend(title = r"$A_0$ (eV $ e^{-1}$ $c^{-1})$",bbox_to_anchor =(0.5,.5,.33,1.05),fontsize= 7)
ax.set_yticks([-.6,-.4, -.2, 0, .2])


ax = axes[1,1]
ax.plot(del_energy_array,helical_pol_lp_bcd[2,:],color= default_color,linestyle = default_linestyle)
ax.set_xlabel(r"$\omega_2-\omega_1$ (eV $\hbar^{-1}$)",fontsize = label_fontsize,labelpad= 0)
ax.set_xticks([0,1,2])
ax.axline((1,-.01),(1,.01),color = shared_line_color)
ax.set_yticks([-.5,-.4,-.3,-.2,-.1,0])

figure.subplots_adjust(wspace = .35,hspace= .42,top = .84,left = .12,right = .98,bottom = .10)
#figure.tight_layout()
figure.savefig("figure_lp_comparison_quadchart_v8.pdf")
figure.show()

x_arrays = np.vstack((ratio_array,gam_array,del_energy_array))

make_source_files = False
if (make_source_files):
    np.save("figure_4a_x",cav_freq_array)
    np.save("figure_4bcd_x",x_arrays)
    np.save("figure_4a_hel_pol_a",helical_pol_lp_a_to_use)
    np.save("figure_4bcd_hel_pol_bcd",helical_pol_lp_bcd)
