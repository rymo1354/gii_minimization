#@author: rymo1354
# date - 1/20/2022

import matplotlib.pyplot as plt
import numpy as np

# Used to plot the dataset statistics distribution
def plot_giis_pearsons(giis, pearsons, orderings, name):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharey=False, sharex=False, dpi=300)
    fig.tight_layout()
    ax1.hist(giis, bins=np.arange(0, 2.21, 0.05), edgecolor='w', color='#1A1423')
    axins1 = ax1.inset_axes([0.25, 0.25, 0.67, 0.67])
    axins1.hist(giis, bins=np.arange(0, 2.21, 0.01), edgecolor='w', color='#1A1423')
    axins1.set_xlim(0, 0.6)
    axins1.set_ylim(0, 35)
    axins1.tick_params(axis='both', which='major', labelsize=10)
    #axins1.set_ylim(y1, y2)
    #axins1 = zoomed_inset_axes(ax1, 1.5, loc= 'lower left', bbox_to_anchor=(0,0), borderpad=3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 115])
    ax1.set_xlabel('$GII_{GS \: DFT} \\enspace (v.u.)$')
    #ax1.set_ylabel('$Count$')

    ax2.hist(pearsons, bins=np.arange(-1.0, 1.1, 0.1), edgecolor='w', color='#ED1C24')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([0, 200])
    ax2.set_xlabel('$Pearson \enspace coefficient, \enspace p$')
    ax2.set_ylabel('$Count$')
    axins2 = ax2.inset_axes([0.125, 0.25, 0.67, 0.67])
    axins2.hist(pearsons, bins=np.arange(-1.0, 1.1, 0.01), edgecolor='w', color='#ED1C24')
    axins2.set_xlim(0.5, 1)
    axins2.set_ylim(0, 60)
    axins2.tick_params(axis='both', which='major', labelsize=10)

    ys = [0 for i in range(1, 12, 1)]
    xs = [i for i in range(1, 12, 1)]
    for o in orderings:
        ys[o-1] += 1
    ax3.bar(xs, ys, edgecolor='w', color='#EE6C4D')
    ax3.set_xlim([0, 12])
    ax3.set_xticks([1, 6, 11])
    ax3.set_ylim([0, 70])
    ax3.set_xlabel('$ Structures \enspace correctly \enspace ordered, \enspace N^{correct}$')
    #ax3.set_ylabel('$Count$')
    #plt.tight_layout()
