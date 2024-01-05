import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import matplotlib.pylab as pylab
suffix = "2pt1eV"
omega_data = np.real(np.load("chiral_int_e_scan"+suffix+".npy"))
gamma_data = np.real(np.load("chiral_int_gamma_scan"+suffix+".npy"))
r_data = np.real(np.load("chiral_int_ratio_scan"+suffix+".npy"))
w_data = np.real(np.load("chiral_int_w2_scan"+suffix+".npy"))


#designed to span two 3.3 columns
fig, ax = plt.subplots(2,2,figsize = (3.46,3.5))
ax = ax.flatten()
params = {'xtick.labelsize':7,'ytick.labelsize':7,'axes.labelsize':7}
pylab.rcParams.update(params)
#omega plot
shared_line_color = "orange"

x_title_offset_1 = -.32
y_title_offset_1= 1.05

x_title_offset_2 = -.19
y_title_offset_2= 1.05

gam_factor = .1
label_fontsize = 7
st_fs = 7 #subtitle fontsize

for i in range(0,4):
    ax[i].tick_params(axis="both",labelsize=7)
    #ax[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    if (i == 0):
        data = omega_data
    elif (i==1):
        data = r_data
    elif (i==2):
        data = gamma_data
    elif (i ==3):
        data = w_data
        data[0,:] = data[0,:]-2 #rebasing energy
    ax[i].plot(data[0,:],data[1,:],label = r"$n=1$",color ="red")
    ax[i].plot(data[0,:],data[3,:],color = "red",linestyle = "dotted")
    ax[i].plot(data[0,:],data[2,:],label = r"$n=2$ ",color = 'blue')
    ax[i].plot(data[0,:],data[4,:],color= "blue",linestyle = "dotted" )
    #ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #plt.plot(cav_freq_array,chiral_simple)
    def visible_x_axis(axis):
        axis.axline((0,0),(1,0),color = "black",linestyle = "solid",linewidth = 1)
    if (i == 0):
        visible_x_axis(ax[i])
        ax[i].set_title("a", fontsize=st_fs, x=x_title_offset_1, y=y_title_offset_1,weight= "bold")
        ax[i].tick_params(axis='y', which='major', pad=0)
        ax[i].set_xlim(1.5, 3.5)
        ax[i].set_ylim(-1,1)
        ax[i].set_ylabel(r"$\sigma_n$",fontsize = label_fontsize,labelpad= 0)
        ax[i].set_xlabel(r" $\Omega$ (eV $\hbar^{-1}$)",fontsize= label_fontsize, labelpad =0)
        ax[i].axline((2.1,-1),(2.1,1),color = shared_line_color)
    if (i == 1):
        ax[i].set_title("b", fontsize=st_fs, x=x_title_offset_2, y=y_title_offset_1,weight = "bold")
        ax[i].set_xlim(.5,2)
        ax[i].set_ylim(-.6,0)
        visible_x_axis(ax[i])
        yticks = ax[i].get_yticklabels()
        ax[i].tick_params(axis='y', which='major', pad=0)
        ax[i].set_xlabel(r" $\mu_2:\mu_1$",fontsize= label_fontsize,labelpad= 0)
        ax[i].axline((1,-.1),(1,0),color = shared_line_color)
        #ax[i].set_ylabel(r"$\sigma_n$",fontsize = 14)
    if (i == 2):
        visible_x_axis(ax[i])
        ax[i].set_ylabel(r"$\sigma_n$", fontsize=label_fontsize,labelpad =0)
        ax[i].set_xlim(.03,.15)
        ax[i].set_ylim(-1,0)
        ax[i].set_xlabel(r" $\gamma$",fontsize= 7,labelpad = 0)
        ax[i].axline((gam_factor,-.1),(gam_factor,0),color = shared_line_color)
        ax[i].tick_params(axis='y', which='major', pad=0)
        #yticks[4].set_visible(False)
        ax[i].set_title("c", fontsize=st_fs, x=x_title_offset_1, y=y_title_offset_2,weight= "bold")

    if (i == 3):
        visible_x_axis(ax[i])
        ax[i].set_xlim(0, 2)
        ax[i].tick_params(axis='y', which='major', pad=0)
        ax[i].set_title("d", fontsize=st_fs, x=x_title_offset_2, y=y_title_offset_2,weight = "bold")
        ax[i].set_xlabel(r" $\omega_2-\omega_1$ (eV $\hbar^{-1}$)", fontsize=label_fontsize,labelpad= 0)
        ax[i].axline((1, -.1), (1, 0), color=shared_line_color)
    if (i == 0):
        fig.legend(loc = 'upper center',bbox_to_anchor = (.34,1.01),fontsize = 7)
    if (i == 1):
        linestyles = ["solid","dotted"]
        lines = [matplotlib.lines.Line2D([0], [0], color="black", linewidth=1, linestyle=linestyle) for linestyle in linestyles]
        labels = ["Infinite-order","Second-order"]
        fig.legend(lines,labels,loc='upper center', bbox_to_anchor=(.80, 1.01), fontsize=7)

fig.tight_layout()
fig.subplots_adjust(wspace = .36,hspace= .42,top = .87,left = .13,right = .975,bottom = .11)
fig.savefig("chiral_int_triple_v14.pdf",dpi = 1000)
fig.show()


