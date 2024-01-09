import os
import numpy as np
import matplotlib.pyplot as plt

path = os.getcwd()

dipole_ratio = np.linspace(-1,3,50)
ratio = dipole_ratio
gam_set = np.array([0.05,0.10,0.15])
vp_set = np.array([20,50])

data_min_total = np.zeros((np.size(gam_set),np.size(dipole_ratio)))
for a in range(0,np.size(gam_set)):
        vp_str = str(vp_set[0]).replace('.','_')
        gam_str = str(gam_set[a]).replace('.','_')
        filename_min = "dip_rat_swp_v3_lower_state_min_full_vp"+str(vp_str)+"res"+str(gam_str).replace('.','_')+".npy"
        data_min  =np.load(os.sep.join([path,filename_min]))
        data_min_total[a,:]  = data_min
filename_vp_50 ="dip_rat_swp_v3_lower_state_min_full_vp50res0_1.npy"
data_vp_50 = np.load(os.sep.join([path,filename_vp_50]))

filename_w2 = "dip_rat_swp_w2_25_v3_lower_state_min_full_vp20res0_1.npy"
data_w2 = np.load(os.sep.join([path,filename_w2]))

# filename_w1 = "dip_rat_swp_w1_1v2_lower_state_min_full_vp20res0_1.npy"
# data_w1 = np.load(os.sep.join([path,filename_w1]))


fig,axes = plt.subplots(nrows = 1,ncols = 3,figsize = (6.6,2.5))



ax = axes[0]
ax.set_title("a)", fontsize=11, x=.07, y=.8)
labels = ["0.05", "0.10", "0.15"]
color_set = ["red","blue","black"]
for a in range(0,np.size(gam_set)):
    ax.plot(ratio, np.abs(data_min_total[a,:]), color=color_set[a],linestyle=  "solid",label = labels[a])
ax.set_xlabel(r"$\log_2(\mu_2:\mu_1)$",fontsize = 14)
ax.set_ylim(0,.65)
#ax.set_xlabel(r"$\log_2(\mu_2:\mu_1)$",fontsize = 14)
ax.set_ylabel(r"max$(|\tilde{g}^{\alpha=1}|)$",fontsize= 14)
legend = ax.legend(title = r"$\gamma$",fontsize= 8,loc = "lower right",bbox_to_anchor = (.10,.5,.5,.5),frameon = False)
legend.get_title().set_fontsize(10)

to_plot = data_min_total
ax = axes[1]
ax.set_title("b)", fontsize=11, x=.07, y=.8)
labels = ["20","50"]
ax.tick_params(labelleft= False)
ax.plot(ratio,np.abs(data_min_total[1,:]),color = color_set[1],label = labels[0])
ax.plot(ratio,np.abs(data_vp_50),color = color_set[1],linestyle=  "-.",label = labels[1])
ax.set_ylim(0,.65)

legend = ax.legend(title = r"$A_{0,\pm}$ (eV$/ec$)",fontsize= 8,loc = "lower right",bbox_to_anchor = (.32,.58,.5,.5),frameon = False)
legend._legend_box.align = "left"
legend.get_title().set_fontsize(10)
ax.set_xlabel(r"$\log_2(\mu_2:\mu_1)$",fontsize = 14)

ax = axes[2]
ax.tick_params(labelleft= False)
ax.set_xlabel(r"$\log_2(\mu_2:\mu_1)$",fontsize = 14)
ax.set_title("c)", fontsize=11, x=.07, y=.8)
labels = ["3","4"]

ax.plot(ratio,np.abs(data_w2),color = color_set[1],linestyle = "dotted",label = "0.5")
ax.plot(ratio,np.abs(data_min_total[1,:]),color = color_set[1],label = "1.0")

# ax.plot(ratio,np.abs(data_w1))
ax.set_ylim(0,.65)

for i in range(3):
        axes[i].set_xticks(np.arange(-1,4,1))
        axes[i].set_xlim(-1.2,3.2)
legend = ax.legend(title = r"$\omega_2-\omega_1$ (eV/$\hbar$)",fontsize= 8,loc = "lower right",bbox_to_anchor = (.33,.58,.5,.5),frameon = False)
legend._legend_box.align = "left"
legend.get_title().set_fontsize(10)
#legend2 = fig.legend(title = r"$\gamma_n = x \omega_n$",fontsize= 8,loc = "lower right",bbox_to_anchor = (.45,.1,.5,.5))
#legend2.get_title().set_fontsize(10)
plt.subplots_adjust(wspace= 0.05,bottom = .24)
fig.savefig("gamma_comp_total_v8.pdf")
fig.show()