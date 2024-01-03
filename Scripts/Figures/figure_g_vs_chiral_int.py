import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

g_factor_old_style = True
filename = "lower_polariton_variable_comparison.npy"
helical_pol_lp_a = np.load(filename)

filename_2 = "lower_polariton_2pt1comparison_bcd.npy"
helical_pol_lp_bcd = np.load(filename_2)

if (g_factor_old_style):
    helical_pol_lp_bcd = helical_pol_lp_bcd*2
    helical_pol_lp_a = helical_pol_lp_a*2
cav_freq_array = np.linspace(1.5,2.5,101)
set_size  =100
gam_array = np.linspace(.03,.15,set_size)
del_energy_array = np.linspace(0,2,set_size)
figure, axes = plt.subplots(2,2,figsize = (3.3,3.4))


params = {'legend.fontsize':20,'legend.title_fontsize':20,
          'axes.labelsize':24,'xtick.labelsize':20,'ytick.labelsize':20,
          'axes.titlesize':28,'lines.linewidth':10}
plt.rcParams.update(params)
label_fontsize = 28
default_color = "red"
second_color = "black"
third_color = "blue"

default_linestyle = "solid"
second_linestyle = "-."
third_linestyle = "dotted"


x_title_offset_1 = -.38
y_title_offset_1= .92

x_title_offset_2 = -.16
y_title_offset_2= .94

x_st = [x_title_offset_1,x_title_offset_2]
y_st = [y_title_offset_1,y_title_offset_2]

ylim_bounds = [-.73,.26]
xlim_bounds = [1.25,2.5]


def visible_x_axis(axis):
    axis.axline((0, 0), (1, 0), color="black", linestyle="solid", linewidth=1)


fig,ax = plt.subplots()
shared_line_color = "orange"
fig,ax = plt.subplots()
ratio_array= np.linspace(.3,3,100)
ax.set_xlabel(r"$\mu_2:\mu_1$",  fontsize=label_fontsize,labelpad=0)

ax.plot(ratio_array,helical_pol_lp_bcd[0,:],color= default_color)
ax.set_xlim(.5,2)
ax.set_ylim(-.8,.1)
ax.axline((1,-.1),(1,.1),color = shared_line_color)
ax.set_yticks([-.8,-.6,-.4, -.2, 0])
ax.set_xticks([.5,1,1.5,2])
ax.set_xticklabels(["0.5","1","1.5","2"])
ax.set_ylabel(r"$\tilde{g}^{\alpha=1}$")
plt.tight_layout()
fig.savefig("g_mu.png")
fig.show()

fig,ax = plt.subplots()
ax.plot(gam_array,helical_pol_lp_bcd[1,:],color = default_color,linestyle = default_linestyle)
ax.set_xlim(.02,.18)
ax.set_ylim(-.35,0)
ax.axline((.1,-.3),(.1,-.31),color = shared_line_color)
ax.set_xlabel(r"$\gamma$",fontsize = label_fontsize,labelpad= 0)
ax.set_ylabel(r"$\tilde{g}^{\alpha=1}$")
ax.set_yticks([-.3,-.2,-.1, 0])
plt.tight_layout()
fig.savefig("g_gamma.png")
fig.show()

fig,ax = plt.subplots()
axes[0, 0].set_xlabel(r"$\Omega$ (eV/$\hbar$)",  fontsize=label_fontsize,labelpad=0)


ax.plot(cav_freq_array,helical_pol_lp_a[0,:],label = "35",color = default_color,linestyle = default_linestyle)
ax.axline((2.1,-.3),(2.1,-.31),color = shared_line_color)
ax.plot(cav_freq_array,helical_pol_lp_a[6,:],label = "70",color = second_color,linestyle = default_linestyle)

ax.legend(title = r"$A_0$ (eV$/ec)$")
ax.set_yticks([-.6,-.4, -.2, 0, .2])
ax.set_xlabel(r"$\Omega$(eV/$\hbar$)")
ax.set_ylabel(r"$\tilde{g}^{\alpha=1}$")
plt.tight_layout()
fig.savefig("g_omega.png")
fig.show()

fig,ax = plt.subplots()

ax.plot(del_energy_array,helical_pol_lp_bcd[2,:],color= default_color,linestyle = default_linestyle)
ax.axline((1,-.01),(1,.01),color = shared_line_color)
ax.set_xlabel(r"$\omega_2-\omega_1$(eV/$\hbar$)",fontsize = label_fontsize,labelpad= 0)
ax.set_xticks([0,1,2])

ax.set_yticks([-.5,-.4,-.3,-.2,-.1,0])
ax.set_ylabel(r"$\tilde{g}^{\alpha=1}$")
plt.tight_layout()
fig.savefig("g_omega_delta.png")
fig.show()