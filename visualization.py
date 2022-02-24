#@author: rymo1354
# date - 1/20/2022

import matplotlib
from matplotlib import figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis import correct_ordering, get_parameters, get_coordination_envs, get_gs_ls_deviations, bar_plot
from scipy.stats import iqr, linregress
from gii_calculator import GIICalculator
from scipy.stats import pearsonr
from pymatgen.core.periodic_table import Element
from analysis import calculate_tbv_for_composition
from collections import OrderedDict
import sys
import re
import copy
import warnings

# Used to plot the dataset statistics distributions
def plot_giis_pearsons(giis, pearsons, orderings, name=None):
    matplotlib.rcParams.update({'font.size': 16})
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharey=False, sharex=False, dpi=300)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    ax1.hist(giis, bins=np.arange(0, 2.21, 0.05), edgecolor='w', color='#1A1423')
    axins1 = ax1.inset_axes([0.25, 0.25, 0.67, 0.67])
    axins1.hist(giis, bins=np.arange(0, 2.21, 0.01), edgecolor='w', color='#1A1423')
    axins1.set_xlim(0, 0.6)
    axins1.set_ylim(0, 35)
    axins1.tick_params(axis='both', which='major', labelsize=10)
    #axins1.set_ylim(y1, y2)
    #axins1 = zoomed_inset_axes(ax1, 1.5, loc= 'lower left', bbox_to_anchor=(0,0), borderpad=3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 150])
    ax1.set_xlabel('$DFT \enspace Ground \enspace State \enspace GII, \enspace GII_{GS - DFT} \enspace (v.u.)$')
    #ax1.set_ylabel('$Count$')

    ax2.hist(pearsons, bins=np.arange(-1.0, 1.1, 0.1), edgecolor='w', color='#ED1C24')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([0, 200])
    ax2.set_xlabel('$Pearson \enspace Coefficient, \enspace p$')
    ax2.set_ylabel('$Count$')
    axins2 = ax2.inset_axes([0.25, 0.25, 0.67, 0.67])
    axins2.hist(pearsons, bins=np.arange(-1.0, 1.1, 0.01), edgecolor='w', color='#ED1C24')
    axins2.set_xlim(0.5, 1)
    axins2.set_ylim(0, 60)
    #axins2.set_ylim(0, 80)
    axins2.tick_params(axis='both', which='major', labelsize=10)

    ys = [0 for i in range(1, 12, 1)]
    xs = [i for i in range(1, 12, 1)]
    for o in orderings:
        ys[o-1] += 1
    ax3.bar(xs, ys, edgecolor='w', color='#EE6C4D')
    ax3.set_xlim([0, 12])
    ax3.set_xticks([1, 6, 11])
    ax3.set_ylim([0, 70])
    #ax3.set_ylim([0, 90])
    ax3.set_xlabel('$ Structures \enspace Correctly \enspace Ordered, \enspace N^{correct}$')

    ax1.text(0.05, 142, 'a)', fontsize='large', weight='bold', fontfamily='serif', horizontalalignment='left', verticalalignment='top')
    ax2.text(-0.9, 189, 'b)', fontsize='large', weight='bold', fontfamily='serif', horizontalalignment='left', verticalalignment='top')
    ax3.text(0.6, 66, 'c)', fontsize='large', weight='bold', fontfamily='serif', horizontalalignment='left', verticalalignment='top')
    #ax3.text(0.6, 85, 'c)', fontsize='large', weight='bold', fontfamily='serif', horizontalalignment='left', verticalalignment='top')
    #ax3.set_ylabel('$Count$')
    #plt.tight_layout()
    if name != None:
        fig.savefig(name, bbox_inches = "tight")
    return

# Used to plot the GII vs dHd relationship for a given cation-anion pair
def plot_by_species(pair, structures_energies, param_dct, name=None):
    matplotlib.rcParams.update({'font.size': 18})
    colors = ['#01BAEF', '#8D5A97', '#ED1C24', '#1A1423', '#1EA896', '#214E34', '#EE6C4D', '#7B3E19']
    use_cmpds = []
    for cmpd in list(structures_energies.keys()):
        cmpd_structure = structures_energies[cmpd]['structures'][0]
        if pair[0] in cmpd_structure.species and pair[1] in cmpd_structure.species:
            use_cmpds.append(cmpd)

    fig, axs = plt.subplots(2, 4, figsize=(14, 7), sharey=True, sharex=True, dpi=500)
    #all_opt_energies, all_opt_giis, all_colors = [], [], []
    fig.tight_layout()
    count_1 = 0
    count_2 = 0
    total_count = 0
    for cmpd_ind, cmpd in enumerate(use_cmpds):
        name_cmpd = copy.deepcopy(cmpd)
        numbers = re.findall(r'\d+', cmpd)
        for n in numbers:
            name_cmpd = re.sub('\d', '_{%s}' % n, name_cmpd)
        name_cmpd = '$' + name_cmpd + '$'
        keys = list(param_dct.keys())
        if total_count >= 8:
            continue
        if 'Cation' in keys and 'Anion' in keys: # General parameters
            opt_params = param_dct
        else:
            opt_params = param_dct[cmpd]
        opt_structures = structures_energies[cmpd]['structures']
        opt_energies = structures_energies[cmpd]['energies']

        if 'Cation' in keys and 'Anion' in keys:
            other_cation = list(np.unique([s for s in structures_energies[cmpd]['structures'][0].species if s != pair[0] and np.sign(s.oxi_state) == 1]))[0]
            other_cation_inds = [i for i in range(len(opt_params['Cation'])) if opt_params['Cation'][i] == other_cation]
            anion_inds = [i for i in range(len(opt_params['Anion'])) if opt_params['Anion'][i] == pair[1]]
            opt_ind = list(set(other_cation_inds) & set(anion_inds))[0]
            comp_R0 = np.round(opt_params['R0'][opt_ind], 3)
        else:
            cation_inds = [i for i in range(len(opt_params['Cation'])) if opt_params['Cation'][i] == pair[0]]
            anion_inds = [i for i in range(len(opt_params['Anion'])) if opt_params['Anion'][i] == pair[1]]
            opt_ind = list(set(cation_inds) & set(anion_inds))[0]
            comp_R0 = np.round(opt_params['R0'][opt_ind], 3)

        gii_calc = GIICalculator(opt_params)
        opt_giis = [gii_calc.GII(s) for s in opt_structures]

        rounded_giis = np.round(opt_giis, 3)
        rounded_energies = np.round(opt_energies, 3)
        zipped = zip(rounded_energies, rounded_giis)
        sort = sorted(zipped, key = lambda t: t[1])
        s_energies, s_giis = [sort[i][0] for i in range(len(sort))], [sort[i][1] for i in range(len(sort))]
        co = correct_ordering(s_energies)

        p = np.round(pearsonr(opt_energies, opt_giis)[0], 3)
        print(cmpd, np.min(rounded_giis), p, co)
        color = colors[cmpd_ind]
        for i in range(len(rounded_energies)):
            if rounded_energies[i] == np.min(rounded_energies):
                axs[count_1][count_2].scatter(rounded_giis[i], rounded_energies[i], color=color, s=70,
                                              facecolors='none', edgecolors=color, label=cmpd, linewidth=2)
            else:
                axs[count_1][count_2].scatter(rounded_giis[i], rounded_energies[i], color=color, s=100,
                                              marker='o', label=cmpd, edgecolor='w')
        m, b = np.polyfit(rounded_giis, rounded_energies, 1)
        rg = np.linspace(np.min(rounded_giis), np.max(rounded_giis), 10)
        axs[count_1][count_2].plot(rg, np.add(np.multiply(m, rg), b), '--', c=color, alpha=1)
        axs[count_1][count_2].text(0.35, 0.85, '$R_{0} = %s \: \AA$' % format(comp_R0, '.3f'), ha='center', va='center',
                                   transform=axs[count_1][count_2].transAxes)
        axs[count_1][count_2].text(0.7, 0.1, '$p = %s$' % p, ha='center', va='center',
                                   transform=axs[count_1][count_2].transAxes)
        axs[count_1][count_2].set_title(name_cmpd)
        if count_1 != 0:
            axs[count_1][count_2].set_xlabel('$GII \\enspace (v.u.)$')
        if count_2 == 0:
            axs[count_1][count_2].set_ylabel('$\\Delta H^{DFT}_{d} \\enspace (eV/atom)$')
        #axs[count_1][count_2].legend(loc='upper left')
        count_2 += 1
        if count_2 == 4:
            count_1 += 1
            count_2 = 0
        total_count += 1
    if name != None:
        fig.savefig(name, bbox_inches = "tight")
    return

def periodic_table_heatmap(
    elemental_data,
    oxi_state_data,
    cbar_label="",
    cbar_label_size=18,
    show_plot=False,
    cmap="YlOrRd",
    cmap_range=None,
    blank_color="grey",
    value_format=None,
    max_row=9, name=None
):
    """
    A static method that generates a heat map overlayed on a periodic table.
    Args:
         elemental_data (dict): A dictionary with the element as a key and a
            value assigned to it, e.g. surface energy and frequency, etc.
            Elements missing in the elemental_data will be grey by default
            in the final table elemental_data={"Fe": 4.2, "O": 5.0}.
         oxi_state_data (dict): oxi_state_data={"Fe": +3}
         cbar_label (string): Label of the colorbar. Default is "".
         cbar_label_size (float): Font size for the colorbar label. Default is 14.
         cmap_range (tuple): Minimum and maximum value of the colormap scale.
            If None, the colormap will autotmatically scale to the range of the
            data.
         show_plot (bool): Whether to show the heatmap. Default is False.
         value_format (str): Formatting string to show values. If None, no value
            is shown. Example: "%.4f" shows float to four decimals.
         cmap (string): Color scheme of the heatmap. Default is 'YlOrRd'.
            Refer to the matplotlib documentation for other options.
         blank_color (string): Color assigned for the missing elements in
            elemental_data. Default is "grey".
         max_row (integer): Maximum number of rows of the periodic table to be
            shown. Default is 9, which means the periodic table heat map covers
            the first 9 rows of elements.
    """

    # Convert primitive_elemental data in the form of numpy array for plotting.
    if cmap_range is not None:
        max_val = cmap_range[1]
        min_val = cmap_range[0]
    else:
        max_val = max(elemental_data.values())
        min_val = min(elemental_data.values())

    max_row = min(max_row, 9)

    if max_row <= 0:
        raise ValueError("The input argument 'max_row' must be positive!")

    value_table = np.empty((max_row, 18)) * np.nan
    blank_value = min_val - 0.01

    for el in Element:
        if el.row > max_row:
            continue
        value = elemental_data.get(el.symbol, blank_value)
        value_table[el.row - 1, el.group - 1] = value

    # Initialize the plt object
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=500)
    plt.gcf().set_size_inches(12, 8)

    # We set nan type values to masked values (ie blank spaces)
    data_mask = np.ma.masked_invalid(value_table.tolist())
    heatmap = ax.pcolor(
        data_mask,
        cmap=cmap,
        edgecolors="w",
        linewidths=1,
        vmin=min_val - 0.001,
        vmax=max_val + 0.001,
    )
    cbar = fig.colorbar(heatmap)

    # Grey out missing elements in input data
    cbar.cmap.set_under(blank_color)

    # Set the colorbar label and tick marks
    cbar.set_label(cbar_label, rotation=270, labelpad=25, size=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_label_size)

    # Refine and make the table look nice
    ax.axis("off")
    ax.invert_yaxis()

    # Label each block with corresponding element and value
    for i, row in enumerate(value_table):
        for j, el in enumerate(row):
            if not np.isnan(el):
                symbol = Element.from_row_and_group(i + 1, j + 1).symbol
                try:
                    oxi_state = oxi_state_data[symbol]
                    if oxi_state == 1:
                        use_label = '$%s^{+}$' % (str(symbol))
                    else:
                        use_label = '$%s^{%s+}$' % (str(symbol), str(oxi_state))
                except:
                    use_label = '$%s$' % str(symbol)
                plt.text(
                    j + 0.5,
                    i + 0.25,
                    use_label,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=14,
                )
                if el != blank_value and value_format is not None:
                    plt.text(
                        j + 0.5,
                        i + 0.5,
                        value_format % el,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=10,
                    )

    plt.tight_layout()
    if name != None:
        plt.savefig(name, bbox_inches = "tight")

    if show_plot:
        plt.show()

    return plt

def periodic_table_heatmap_plot(species, counts, dct, threshold=2, show='IQR', name=None):
    use_species = []
    unique_elements = list(np.unique([s.element for s in species]))
    for el in unique_elements:
        use_specie = None
        count = 0
        element_species = [s for s in species if s.element == el]
        for element_specie in element_species:
            el_index = species.index(element_specie)
            el_count = counts[el_index]
            if el_count > count and el_count >= threshold:
                count = el_count
                use_specie = element_specie
        if use_specie != None:
            use_species.append(use_specie)

    params_list = [[] for i in range(len(use_species))]
    for cmpd_key in list(dct.keys()):
        entry = get_parameters(dct, cmpd=cmpd_key)
        cations = entry['Cation']
        R0s = entry['R0']
        use = True

        for cat_ind in range(len(cations)):
            if cations[cat_ind] in use_species and use == True:
                #print(cations[cat_ind])
                params_index = use_species.index(cations[cat_ind])
                params_list[params_index].append(R0s[cat_ind])
    print(len(use_species))
    plot_dict = {}
    oxi_dict = {}
    if show == 'IQR':
        vals = [np.round(iqr(s_R0), 3) for s_R0 in params_list]
        elements = [str(s.element) for s in use_species]
        oxi_states = [s.oxi_state for s in use_species]
        for element_ind in range(len(elements)):
            plot_dict[elements[element_ind]] = vals[element_ind]
            oxi_dict[elements[element_ind]] = oxi_states[element_ind]
        periodic_table_heatmap(plot_dict, oxi_dict, cmap='Wistia', cbar_label='$Interquartile \enspace Range, \enspace IQR \: (\\AA)$',
                               value_format ='%.3f', name=name)
        #plot_periodic_table_heatmap(plot_dict, cbar_label='IQR', value_format ='%.3f')
    elif show == 'std':
        vals = [np.std(s_R0) for s_R0 in params_list]
        elements = [str(s.element) for s in use_species]
        for element_ind in range(len(elements)):
            plot_dict[elements[element_ind]] = vals[element_ind]
        periodic_table_heatmap(plot_dict, cbar_label='std', value_format ='%.3f', name=name)
    elif show == 'median':
        vals = [np.round(np.median(s_R0), 3) for s_R0 in params_list]
        elements = [str(s.element) for s in use_species]
        oxi_states = [s.oxi_state for s in use_species]
        for element_ind in range(len(elements)):
            plot_dict[elements[element_ind]] = vals[element_ind]
            oxi_dict[elements[element_ind]] = oxi_states[element_ind]
        periodic_table_heatmap(plot_dict, oxi_dict, cmap='Wistia', cbar_label='$Median \enspace R_{0, \enspace GS - DFT}, \enspace \hatR_{0, \enspace GS - DFT} \: (\\AA)$',
                               value_format ='%.3f', name=name)
    elif show == 'mean':
        vals = [np.mean(s_R0) for s_R0 in params_list]
        elements = [str(s.element) for s in use_species]
        for element_ind in range(len(elements)):
            plot_dict[elements[element_ind]] = vals[element_ind]
        periodic_table_heatmap(plot_dict, cbar_label='mean', value_format ='%.3f', name=name)
    elif show == 'range':
        vals = [np.subtract(max(s_R0), min(s_R0)) for s_R0 in params_list]
        elements = [str(s.element) for s in use_species]
        for element_ind in range(len(elements)):
            plot_dict[elements[element_ind]] = vals[element_ind]
        periodic_table_heatmap(plot_dict, cbar_label='range', value_format ='%.3f', name=name)
    else:
        print('Not an option')
        sys.exit(1)
    return use_species, vals

# Use to compare R0, GS DFT to R0, RMSD Parameters
def compare_rmsd_to_gs_dft_R0(rmsd_dct, gs_dft_dct, name=None):
    alkali_metals = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']
    alkaline_earth_metals = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']
    transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                  'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
    lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    actinides = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']
    post_transition_metals = ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po']
    metalloids = ['B', 'Si', 'Ge', 'As', 'Sb', 'Te']

    use_colors = ['#1A1423', '#8D5A97', '#ED1C24', '#214E34', '#01BAEF', '#7B3E19']

    comp_rmsd_gii_gs_dct = {'cations': [], 'rmsds': [], 'gii_gss': [], 'labels': [], 'colors': []}

    for c in rmsd_dct['Cation']:
        rmsd_index = rmsd_dct['Cation'].index(c)
        comp_rmsd_gii_gs_dct['cations'].append(c)
        comp_rmsd_gii_gs_dct['rmsds'].append(rmsd_dct['R0'][rmsd_index])
        gii_gs_index = gs_dft_dct['Cation'].index(c)
        comp_rmsd_gii_gs_dct['gii_gss'].append(gs_dft_dct['R0'][gii_gs_index])
        if str(c.element) in alkali_metals:
            comp_rmsd_gii_gs_dct['labels'].append('Alkali metal')
            comp_rmsd_gii_gs_dct['colors'].append(use_colors[0])
        elif str(c.element) in alkaline_earth_metals:
            comp_rmsd_gii_gs_dct['labels'].append('Alkaline earth metal')
            comp_rmsd_gii_gs_dct['colors'].append(use_colors[1])
        elif str(c.element) in transition_metals:
            comp_rmsd_gii_gs_dct['labels'].append('Transition metal')
            comp_rmsd_gii_gs_dct['colors'].append(use_colors[2])
        elif str(c.element) in lanthanides:
            comp_rmsd_gii_gs_dct['labels'].append('Lanthanide')
            comp_rmsd_gii_gs_dct['colors'].append(use_colors[3])
        elif str(c.element) in post_transition_metals:
            comp_rmsd_gii_gs_dct['labels'].append('Post-transition metal')
            comp_rmsd_gii_gs_dct['colors'].append(use_colors[4])
        elif str(c.element) in metalloids:
            comp_rmsd_gii_gs_dct['labels'].append('Metalloid')
            comp_rmsd_gii_gs_dct['colors'].append(use_colors[5])

    compare_methods_df = pd.DataFrame(comp_rmsd_gii_gs_dct)
    #figure(figsize=(5, 5), dpi=500)
    #colors = ['#001219', '#1C2541', '#005F73', '#0A9396', '#38040E', '#800E13', '#BB3E03', '#CA6702']
    plt.tight_layout()
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6), sharey=False, sharex=False, dpi=500)
    for label in list(np.unique(compare_methods_df['labels'])):
        label_df = compare_methods_df[compare_methods_df['labels'] == label]
        print(label, len(label_df), np.round(np.mean(np.subtract(label_df['rmsds'], label_df['gii_gss'])), 3))
        scatter = ax1.scatter(label_df['gii_gss'], label_df['rmsds'],
                              c=label_df['colors'], label=label_df['labels'].iloc[0], marker='x')

    ax1.set_xlim([1.6, 2.4])
    ax1.set_ylim([1.6, 2.4])
    ax1.set_yticks(np.arange(1.6, 2.6, 0.2))
    ax1.set_xticks(np.arange(1.6, 2.6, 0.2))
    ax1.plot([1.6, 2.4], [1.6, 2.4], '--', c='black')
    ax1.legend(fontsize=9.7)
    #ax1.set_yticks(np.arange(0.8, 1.05, 0.05))
    #ax1.set_xticks(np.arange(0.8, 1.05, 0.05))
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_ylabel('$R_{0, \: RMSD} \enspace (\AA)$', fontsize=24)
    ax1.set_xlabel('$R_{0, \: GS - DFT} \enspace (\AA)$', fontsize=24)

    if name != None:
        plt.savefig(name, bbox_inches='tight', dpi=500)
    return

# Publication thumbnail

def pub_thumbnail(cmpd_giis, cmpd_energies, name=None):
    def f(x, y):
        y1 = 3.5
        x1 = 1
        return 1-(0.8*(abs(x-x1)+abs(y-y1))+np.sin(x+y)+np.cos(x-y))

    x = np.linspace(0, 6, 80)
    y = np.linspace(0, 5, 80)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    xo = 0.9
    yo = 3.9

    sxo = 4.35
    syo = 4.4

    fig, axes = plt.subplots()

    plt.contourf(X, Y, Z, 15, cmap='RdGy_r',alpha=0.7)#colors='black')
    plt.ylabel('$R^{A}_{0,GS-DFT} \: (\AA)$',fontsize=18)
    plt.xlabel('$R^{B}_{0,GS-DFT} \: (\AA)$',fontsize=18)
    plt.yticks([])
    plt.xticks([])
    plt.scatter([xo],[yo],s=550, alpha=1,marker=(5, 1),c='darkred')
    plt.plot([0,xo],[yo,yo],'--',lw=1,c='darkred')
    plt.plot([xo,xo],[0,yo],'--',lw=1,c='darkred')
    #plt.text(0.1,4.5,'$(R_{0,Opt}^{A-site}, \: R_{0,Opt}^{B-site})$',fontsize=14)
    #plt.text(1.0,4.5,'$R_{0,Opt}^{B-site}$',fontsize=14)

    x_fill = np.linspace(0, 6, 90)
    y_fill = list(1.5 + np.power(np.sqrt(3.5), np.linspace(0, 2, 30))) + [5 for i in range(60)]
    y2_fill = [5 for i in range(90)]
    plt.fill_between(x_fill, y_fill, y2_fill, color='blue', alpha=0.3, zorder=2)

    # Optimization route
    plt.text(4.45,4.45,'start',fontsize=15)
    plt.text(0.4, 4.5,'$p\geq C$',fontsize=15)
    rA = np.linspace(syo,yo,7)
    rB = np.linspace(sxo,xo,7)
    dr = [0,0.2,0.5,0.2,0.2,0.25,0]
    #plt.plot(rB,rA)
    plt.plot(rB,[x+dr[i] for i,x in enumerate(rA)],ls='-',lw=1.5,c='k')
    #plt.arrow(x, y, dx, dy, **kwargs)
    plt.arrow(rB[5], rA[5]+0.25, rB[6]-rB[5],rA[6]-rA[5]-dr[5],
              length_includes_head=True,
              head_width=0.2,color='k',lw=1)
              #ls='dotted')
    #cb = plt.colorbar(cmap='RdGy')

    sm = plt.cm.ScalarMappable(cmap='RdGy')
    sm.set_clim(vmin=1, vmax=0)
    cb = plt.colorbar(sm, alpha=0.7)
    #cb.ax.set_ylabel('P$_{(\Delta H^{DFT}_{d}}$$_{vs}$ $_{GII)}$',fontsize=18)
    cb.ax.set_ylabel('$GII_{GS - DFT} \enspace (v.u.)$',fontsize=18)
    #cb.ax.set_yticks([0,0.5,1])
    #cb.ax.set_yticklabels([0,0.5,1])
    #xis = [0.0.25,0.5,0.7]
    for t in cb.ax.get_yticklabels():
         t.set_fontsize(14)
     #t.set_label()
    #plt.colorbar(cmap='RdGy')
    #ax = inset_axes(axes,
    #                    width="50%", # width = 30% of parent_bbox
    #                    height="50%", # height : 1 inch
    #                    loc=4)
                        #bbox_to_anchor=(2, 4, 6, 5))
                        #loc=(0.1,0.1))


    axi = axes.inset_axes([0.32, 0.12, 0.65, 0.65])


    #inset_axes.set_axis_off()
    axi.set_yticks([])
    axi.set_xticks([])
    axi.set_xlabel('$GII \enspace (v.u.)$',fontsize=17)
    axi.set_ylabel('$\Delta H^{DFT}_{d}$ ${(\dfrac{eV}{atom})}$',fontsize=17)
    #axi.set_ylabel('$\Delta H^{DFT}_{d}$ (eV/atom)',fontsize=18)
    for i in range(len(cmpd_energies)):
        if cmpd_energies[i] == np.min(cmpd_energies):
            axi.scatter(cmpd_giis[i], cmpd_energies[i], s=70, alpha=0.8,
                                            facecolors='none', edgecolors='purple', linewidth=2)
        else:
            axi.scatter(cmpd_giis[i], cmpd_energies[i], color='purple', s=100, alpha=0.8,
                                                  marker='o', edgecolor='w')
    #axi.scatter(cmpd_giis, cmpd_e ,alpha=0.8,s=80,c='purple', edgecolor='white')
    xx =[min(cmpd_giis), np.mean(cmpd_giis), max(cmpd_giis)]
    slope, intercept, r_value, p_value, std_err = linregress(cmpd_giis, cmpd_energies)
    axi.plot(xx,[x*slope+intercept for x in xx],'k--',lw=1.5)
    axi.text(0.09,0.0255,'$GII_{GS - DFT}$',fontsize=15)
    p = np.round(pearsonr(cmpd_energies, cmpd_giis)[0], 3)
    axi.text(0.26, 0.0275, '$p=%s$' % (str(p)),fontsize=15)

    imfile = 'figures/New_ABO3.png'
    im = plt.imread(imfile)
    #ai = mpimg.imread(imfile)
    #imagebox = OffsetImage(ai, zoom=1)
    #ab = AnnotationBbox(imagebox, (0.4, 0.6), zorder=4)
    #axi.add_artist(ab)
    axii = axi.inset_axes([-0.0, 0.525, 0.45, 0.45])
    axii.set_yticks([])
    axii.set_xticks([])
    axii.spines['top'].set_visible(False)
    axii.spines['right'].set_visible(False)
    axii.spines['bottom'].set_visible(False)
    axii.spines['left'].set_visible(False)
    axii.patch.set_facecolor('None')
    axii.patch.set_alpha(0.0)
    axii.imshow(im)

    if name != None:
        plt.savefig(name, dpi=800, bbox_inches = "tight")
    return

# Use to plot different bond valence tolerance factors for different parameters
def compare_tbvs(cmpds_dct, rmsd_dct, gs_dft_dct, structures_energies, name=None):
    tbv_rmsds = []
    tbv_mods = []
    diffs = []

    for key in list(cmpds_dct.keys()):
        rmsd_tbv = calculate_tbv_for_composition(key, cmpds_dct, rmsd_dct)
        gs_dft_tbv = calculate_tbv_for_composition(key, cmpds_dct, gs_dft_dct)
        min_energy = np.min(structures_energies[key]['energies'])
        max_energy = np.max(structures_energies[key]['energies'])

        tbv_rmsds.append(rmsd_tbv)
        tbv_mods.append(gs_dft_tbv)
        diffs.append(np.subtract(min_energy, max_energy))

    matplotlib.rcParams.update({'font.size': 10})
    cmap = plt.cm.get_cmap('gist_rainbow_r', 20)
    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5), sharey=False, sharex=False, dpi=500)
    im = ax1.scatter(tbv_mods, tbv_rmsds, c=diffs, cmap=cmap, vmin=-1, vmax=0, edgecolor='black')
    ax1.set_xlim(0.79, 1.01)
    ax1.set_ylim(0.79, 1.01)
    ax1.plot([0.79, 1.01], (0.79, 1.01), '--', c='black')
    ax1.set_yticks(np.arange(0.8, 1.05, 0.05))
    ax1.set_xticks(np.arange(0.8, 1.05, 0.05))
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_ylabel('$t_{bv}, \enspace RMSD$', fontsize=14)
    ax1.set_xlabel('$t_{bv}, \enspace GS-DFT$', fontsize=14)
    cbar = fig.colorbar(im, ax=ax1, drawedges=True)
    cbar.dividers.set_color('black')
    cbar.dividers.set_linewidth(1)
    cbar.set_label('$Stabilization \: from \: BO_{6} \: Tilting, \: \dfrac{eV}{atom}$', rotation=90, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    if name != None:
        plt.savefig(name, bbox_inches = "tight")

    return

def plot_broken_bar(dcts_list, cations_list, unique_coords_list, counts_list, width=0.005,
                     colors = ['blue', 'purple', 'green', 'black', 'orange', 'cyan', 'maroon', 'lime'], name=None):

    cns = get_coordination_envs(unique_coords_list, counts_list)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)
    fig.tight_layout()
    max_val = 0

    for cation_ind, cation in enumerate(cations_list):
        y_val = ((cation_ind+1)*10, 9)
        cation_dct = dcts_list[cation_ind]
        unique_coords = unique_coords_list[cation_ind]
        counts = counts_list[cation_ind]
        for cn in cns:
            try:
                index = unique_coords.index(cn)
                count = counts[index]
            except:
                continue

            if count > 1:
                color = colors[cns.index(cn)]
                #print(colors_ind)
                val = float(cation_dct[cn])
                if val > max_val:
                    max_val = val
                alpha = count / np.max(counts)
                #ax.broken_barh([(val-width, 2*width)], y_val, facecolors=(color), label=cn, alpha=alpha)
                #print(val, y_val[0])
                ax.scatter(val, y_val[0], facecolor='none', color=color, label=cn, s=alpha*200)

    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Coordination Number', ncol=2)
    ax.set_xlim(0, max_val + 0.2)
    #ax.set_yticks([15+i*10 for i in range(len(cations_list))])
    ax.set_yticks([(i+1)*10 for i in range(len(cations_list))])
    ax.set_yticklabels([str(c) for c in cations_list])
    ax.set_xlabel('$Range \enspace of \enspace \overline{d}_{M-O}, \enspace \AA$')

    if name != None:
        plt.savefig(name, bbox_inches='tight')

    return

def compare_ideal_bond_lengths(dct, gs_dft_params, rmsd_params):

    gs_dft_gs_deviations, gs_dft_ls_deviations = get_gs_ls_deviations(dct, gs_dft_params)
    rmsd_dft_gs_deviations, rmsd_dft_ls_deviations = get_gs_ls_deviations(dct, rmsd_params)
    bins = np.arange(0, 0.56, 0.02)

    gs_v1 = np.histogram(np.abs(gs_dft_ls_deviations), bins=bins)
    gs_v2 = np.histogram(np.abs(gs_dft_gs_deviations), bins=bins)
    x1s, y1s = bar_plot(gs_v1)
    x2s, y2s = bar_plot(gs_v2)

    rmsd_v1 = np.histogram(np.abs(rmsd_dft_ls_deviations), bins=bins)
    rmsd_v2 = np.histogram(np.abs(rmsd_dft_gs_deviations), bins=bins)
    rx1s, ry1s = bar_plot(rmsd_v1)
    rx2s, ry2s = bar_plot(rmsd_v2)

    matplotlib.rcParams.update({'font.size': 16})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False, dpi=500)

    warnings.filterwarnings("ignore")
    ax1.bar(x2s, y2s, width=0.02, edgecolor='w', color='green', label='Ground state structures')
    ax1.bar(x1s, [-1*y1 for y1 in y1s], width=0.02, edgecolor='w', color='blue', label='Least stable structures')
    ax1.set_ylim(-700, 700)
    ax1.set_xlim(-0.01, 0.36)
    ax1.set_xlabel('$\| r^{BVM}_{ij} - \overline{r}_{ij} \|, \enspace \AA$', fontsize=20)
    ax1.set_yticklabels([500, 500, 250, 0, 250, 500])
    ax1.set_ylabel('$Count$', fontsize=20)
    ax1.legend(fontsize=12, loc='lower right')
    ax1.text(0.2, 100, '$R_{0,GS-DFT}$', fontsize=18)
    ax1.text(0.3, 550, 'a)', fontsize='large', weight='bold', fontfamily='serif')

    ax2.bar(rx2s, ry2s, width=0.02, edgecolor='w', color='green', label='Ground state structures')
    ax2.bar(rx1s, [-1*y1 for y1 in ry1s], width=0.02, edgecolor='w', color='blue', label='Least stable structures')
    ax2.set_xlim(-0.01, 0.36)
    ax2.set_ylim(-700, 700)
    ax2.set_xlim(-0.01, 0.36)
    ax2.set_yticklabels([500, 500, 250, 0, 250, 500])
    ax2.set_xlabel('$\| r^{BVM}_{ij} - \overline{r}_{ij} \|, \enspace \AA$', fontsize=20)
    ax2.legend(fontsize=12, loc='lower right')
    ax2.text(0.2, 100, '$R_{0,RMSD}$', fontsize=20)
    ax2.text(0.3, 550, 'b)', fontsize='large', weight='bold', fontfamily='serif')
    fig.savefig('figures/bond_length_comparison.png', bbox_inches = "tight")

    return
