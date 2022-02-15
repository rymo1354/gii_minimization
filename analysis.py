#@author: rymo1354
# date - 1/20/2022

from gii_calculator import GIICalculator
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Specie
from scipy.stats import pearsonr
import numpy as np
import sys
from tqdm import tqdm
import json


def get_cation_anion_pairs(structure):
    # Unique cation anion pairs in a structure
    pairs = []
    cations = [s for s in structure.species if np.sign(s.oxi_state) == 1]
    anions = [s for s in structure.species if np.sign(s.oxi_state) == -1]
    for cation in cations:
        for anion in anions:
            pairs.append((cation, anion))
    return pairs

def get_cation_anion_pair_index(cation, anion, dct):
    # Index of cation anion pair parameters in parameter dictionary
    cation_inds = [i for i in range(len(dct['Cation'])) if cation == dct['Cation'][i]]
    anion_inds = [i for i in range(len(dct['Anion'])) if anion == dct['Anion'][i]]
    ind = list(set(cation_inds) & set(anion_inds))[0]

    return ind

def get_parameters(dct, cmpd=None, pairs=None):
    if cmpd != None:
        entry = dct[cmpd]
    elif pairs != None:
        entry = {'Cation': [], 'Anion': [], 'R0': [], 'B': []}
        for pair in pairs:
            pair_ind = get_cation_anion_pair_index(pair[0], pair[1], dct)
            entry['Cation'].append(dct['Cation'][pair_ind])
            entry['Anion'].append(dct['Anion'][pair_ind])
            entry['R0'].append(dct['R0'][pair_ind])
            entry['B'].append(dct['B'][pair_ind])
    else:
        print('Must pass cmpd or pair, not both or neither')
        sys.exit(1)
    return entry

def correct_ordering(arr):
    lis = [0] * len(arr)
    for i in range(len(arr)):
        lis[i] = 1
    for i in range(1, len(arr)):
        for j in range(i):
            if (arr[i] >= arr[j] and
                lis[i] < lis[j] + 1):
                lis[i] = lis[j] + 1
    maximum = 0
    for i in range(len(arr)):
        if (maximum < lis[i]):
            maximum = lis[i]
    return maximum

# Used to get the data for the plotted distributions
def stats(structures_energies, param_dct, gii_upper_lim=0.2, pearson_lower_lim=0.7):
    cmpds = list(structures_energies.keys())

    dft_gs_giis = []
    correctly_ordered = []
    pearsons = []
    correctly_identified_gs = [0 for i in range(11)]
    cmpds_used = []

    for cmpd in tqdm(cmpds):
        opt_structures = structures_energies[cmpd]['structures']
        opt_energies = structures_energies[cmpd]['energies']

        if cmpd in list(param_dct.keys()):
            opt_params = get_parameters(param_dct, cmpd=cmpd)
        else:
            pairs = get_cation_anion_pairs(opt_structures[0])
            opt_params = get_parameters(param_dct, pairs=pairs)

        gii_calc = GIICalculator(opt_params)
        opt_giis = [gii_calc.GII(s) for s in opt_structures]

        rounded_energies = np.round(opt_energies, 3)
        rounded_giis = np.round(opt_giis, 3)

        zipped = zip(rounded_energies, rounded_giis)
        sort = sorted(zipped, key = lambda t: t[1])
        s_energies, s_giis = [sort[i][0] for i in range(len(sort))], [sort[i][1] for i in range(len(sort))]

        co = correct_ordering(s_energies)
        correctly_ordered.append(co)
        p = np.round(pearsonr(opt_energies, opt_giis)[0], 3)
        pearsons.append(p)
        dft_gs_giis.append(s_giis[0])
        if s_giis[0] > gii_upper_lim:
            print(cmpd, 'min_gii', s_giis[0])
        if p < pearson_lower_lim:
            print(cmpd, 'pearson', p)
        min_energy = np.min(s_energies)
        min_energy_indices = [i for i in range(len(s_energies)) if s_energies[i] == min_energy]
        min_index_min_energy = np.min(min_energy_indices)
        correctly_identified_gs[min_index_min_energy] += 1
        cmpds_used.append(cmpd)

    return dft_gs_giis, pearsons, correctly_ordered, correctly_identified_gs, cmpds_used

# Used to get the most frequent cation in the dataset for a given element
# For composition-specific parameters
def get_most_frequent(structures_energies, param_dct):
    cations = []
    elements = []
    for cmpd in list(structures_energies.keys()):
        entry = get_parameters(param_dct, cmpd=None, pairs=None)
        cations += entry['Cation']
    species, uniques = np.unique(cations, return_counts=True)
    for s in species:
        elements.append(s.element)
    unique_elements = list(np.unique(elements))
    most_frequent = []
    most_frequent_counts = []
    for el in unique_elements:
        most_specie = None
        most_counts = None
        for i in range(len(species)):
            if el == species[i].element:
                #print('Checking %s' % species[i])
                if most_specie == None or most_counts < uniques[i]:
                    #print('Replacing %s with %s' % (most_specie, species[i]))
                    most_specie = species[i]
                    most_counts = uniques[i]
        most_frequent.append(most_specie)
        most_frequent_counts.append(most_counts)
    return most_frequent, most_frequent_counts

def percentage_IQR_med(species_to_use, full_species_meds, full_species_IQRs, full_IQRs, full_meds):
    species_used = []
    med_indices = []
    iqr_indices = []

    percents = []
    for su in species_to_use:
        try:
            iqr_indices.append(full_species_IQRs.index(su))
            med_indices.append(full_species_meds.index(su))
            species_used.append(su)
            print(su)
        except:
            continue

    for i in range(len(med_indices)):
        species = species_used[i]
        med = full_meds[med_indices[i]]
        iqr = full_IQRs[iqr_indices[i]]
        print(species, med, iqr)
        percents.append(np.multiply(np.round(np.divide(iqr, med), 3), 100))

    return percents

def get_opt_MX(R0, B, nM, coord_num):
    return R0-B*np.log(nM/coord_num)

def tbv_calculator(A_species, B_species, X_species, param_dct,
                   A_fracs, B_fracs, X_fracs):
    ### Calculate tbv by taking the weighted means of the different species
    A_pairs = []
    A_weights = []
    for A_specie_ind in range(len(A_species)):
        for X_specie_ind in range(len(X_species)):
            A_pairs.append((A_species[A_specie_ind], X_species[X_specie_ind]))
            A_weights.append(np.multiply(A_fracs[A_specie_ind], X_fracs[X_specie_ind]))

    B_pairs = []
    B_weights = []
    for B_specie_ind in range(len(B_species)):
        for X_specie_ind in range(len(X_species)):
            B_pairs.append((B_species[B_specie_ind], X_species[X_specie_ind]))
            B_weights.append(np.multiply(B_fracs[B_specie_ind], X_fracs[X_specie_ind]))

    opt_AXs = []
    for A_pair in A_pairs:
        cation_anion_index = get_cation_anion_pair_index(A_pair[0], A_pair[1], param_dct)
        R0_A = param_dct['R0'][cation_anion_index]
        B_A = param_dct['B'][cation_anion_index]
        opt_AX = get_opt_MX(R0_A, B_A, A_pair[0].oxi_state, 12)
        opt_AXs.append(opt_AX)

    opt_BXs = []
    for B_pair in B_pairs:
        cation_anion_index = get_cation_anion_pair_index(B_pair[0], B_pair[1], param_dct)
        R0_B = param_dct['R0'][cation_anion_index]
        B_B = param_dct['B'][cation_anion_index]
        opt_BX = get_opt_MX(R0_B, B_B, B_pair[0].oxi_state, 6)
        opt_BXs.append(opt_BX)

    normalized_A_weights = [A_weight/np.sum(A_weights) for A_weight in A_weights]
    normalized_B_weights = [B_weight/np.sum(B_weights) for B_weight in B_weights]

    mean_opt_AX = np.mean(np.multiply(normalized_A_weights, opt_AXs))
    mean_opt_BX = np.mean(np.multiply(normalized_B_weights, opt_BXs))

    return mean_opt_AX/(np.sqrt(2)*mean_opt_BX)

def calculate_tbv_for_composition(composition, inputs_dct, params_dct):

    ''' Composition: string
        Inputs dictionary: {composition: {A_sites: [], B_sites: [], X_sites: []}}
        Dictionary of species for each, organized by site '''

    composition_object_dct = Composition(composition).fractional_composition.as_dict()
    A_site_elements = [str(a.element) for a in inputs_dct[composition]['A_sites']]
    B_site_elements = [str(b.element) for b in inputs_dct[composition]['B_sites']]
    X_site_elements = [str(x.element) for x in inputs_dct[composition]['X_sites']]

    A_specs = []
    A_fracs = []
    B_specs = []
    B_fracs = []
    X_specs = []
    X_fracs = []

    for key in composition_object_dct: # each element
        if key in A_site_elements:
            index = A_site_elements.index(key)
            A_specs.append(inputs_dct[composition]['A_sites'][index])
            A_fracs.append(composition_object_dct[key])
        elif key in B_site_elements:
            index = B_site_elements.index(key)
            B_specs.append(inputs_dct[composition]['B_sites'][index])
            B_fracs.append(composition_object_dct[key])
        elif key in X_site_elements:
            index = X_site_elements.index(key)
            X_specs.append(inputs_dct[composition]['X_sites'][index])
            X_fracs.append(composition_object_dct[key])

    tbv = tbv_calculator(A_specs, B_specs, X_specs, params_dct,
                       A_fracs, B_fracs, X_fracs)

    return tbv

def bond_distances_by_coordination(cation, anion, dct):
    final_dct = {}
    gii_calc = GIICalculator()
    cation_cmpds = [cmpd for cmpd in list(dct.keys()) if cation in dct[cmpd]['structures'][0].species]
    for cmpd in cation_cmpds:
        final_dct[cmpd] = {}
        structures = dct[cmpd]['structures']
        distances_dct = {}
        for s in structures:
            equiv_sites = gii_calc.get_equivalent_sites(s)
            for equiv_sites_list in equiv_sites:
                if equiv_sites_list[0].specie == cation: # Just for the cation of interest
                    site_index = s.index(equiv_sites_list[0])
                    neighbors = gii_calc.get_neighbors(s, site_index)
                    key = len(neighbors)
                    if key not in list(distances_dct.keys()):
                        distances_dct[key] = []

                    site_distances = []
                    for neighbor in neighbors:
                        distance = neighbor.nn_distance
                        site_distances += [distance for i in range(len(equiv_sites_list))] # Distances for all sites
                    mean_site_distance = np.mean(site_distances)
                    distances_dct[key].append(mean_site_distance)
        for key in list(distances_dct.keys()):
            distances_dct[key] = np.mean(distances_dct[key])
        final_dct[cmpd] = distances_dct

    return final_dct

def range_bond_distance_by_coordination(coord_dct):
    coords = []
    for cmpd in list(coord_dct.keys()):
        for coord in list(coord_dct[cmpd].keys()):
            coords.append(coord)
    unique_coords, counts = np.unique(coords, return_counts=True)

    coords_only = {}
    for cmpd in list(coord_dct.keys()):
        for coord in unique_coords:
            if coord in list(coord_dct[cmpd].keys()):
                mean_distance = coord_dct[cmpd][coord]
                if coord not in list(coords_only.keys()):
                    coords_only[coord] = []
                coords_only[coord].append(mean_distance)
    for coord in list(coords_only.keys()):
        coords_only[coord] = np.max(coords_only[coord]) - np.min(coords_only[coord])

    return coords_only, list(unique_coords), list(counts)

def get_coordination_envs(unique_coords_list, counts_list):
    coords_to_plot = []
    for uc_ind, uc in enumerate(unique_coords_list):
        for cn_ind, cn in enumerate(uc):
            if counts_list[uc_ind][cn_ind] > 1 and cn not in coords_to_plot:
                coords_to_plot.append(cn)

    return sorted(coords_to_plot, reverse=True)

def get_correct_gs_Ln3_Tm3(structures_energies, params_dct):
    three_plus_tms = [Specie('Sc', 3), Specie('Ti', 3), Specie('V', 3), Specie('Cr', 3),
                      Specie('Mn', 3), Specie('Fe', 3), Specie('Co', 3), Specie('Ni', 3),
                      Specie('Cu', 3), Specie('Zn', 3)]
    three_plus_lns = [Specie('La', 3), Specie('Ce', 3), Specie('Pr', 3), Specie('Nd', 3),
                      Specie('Sm', 3), Specie('Eu', 3), Specie('Gd', 3), Specie('Tb', 3),
                      Specie('Dy', 3), Specie('Ho', 3), Specie('Er', 3), Specie('Tm', 3),
                      Specie('Yb', 3), Specie('Lu', 3)]
    gs_total = 0
    gs_two_total = 0
    correct = 0

    gii_calc = GIICalculator(params_dct)
    for cmpd in tqdm(list(structures_energies.keys())):
        species = list(np.unique(structures_energies[cmpd]['structures'][0].species))
        has_tm = False
        has_ln = False
        for tm in three_plus_tms:
            if tm in species:
                has_tm = True
        for ln in three_plus_lns:
            if ln in species:
                has_ln = True

        if has_tm == True and has_ln == True:
            print(cmpd)
            opt_giis = [gii_calc.GII(s) for s in structures_energies[cmpd]['structures']]

            rounded_energies = np.round(structures_energies[cmpd]['energies'], 3)
            rounded_giis = np.round(opt_giis, 3)

            zipped = zip(rounded_energies, rounded_giis)
            sort = sorted(zipped, key = lambda t: t[1])
            s_energies, s_giis = [sort[i][0] for i in range(len(sort))], [sort[i][1] for i in range(len(sort))]
            if np.min(s_energies) == s_energies[0]: # if GII predicts the ground state energy
                gs_total += 1
            if np.min(s_energies) in s_energies[0:2]:
                gs_two_total += 1
            correct += 1
    return gs_total, gs_two_total, correct

def get_gs_and_ls(dct):
    short_dct = {}
    for cmpd in list(dct.keys()):
        short_dct[cmpd] = {}
        gs_ind = dct[cmpd]['energies'].index(np.min(dct[cmpd]['energies']))
        ls_ind = dct[cmpd]['energies'].index(np.max(dct[cmpd]['energies']))
        short_dct[cmpd]['gs'] = dct[cmpd]['structures'][gs_ind]
        short_dct[cmpd]['ls'] = dct[cmpd]['structures'][ls_ind]
    return short_dct

def get_deviation(structure, params_dct):
    deviations = []
    gii_calc = GIICalculator()
    for site in structure:
        if np.sign(site.specie.oxi_state) == 1:
            site_index = structure.index(site)
            neighbors = gii_calc.get_neighbors(structure, site_index)
            site_distances = []
            for neighbor in neighbors:
                distance = neighbor.nn_distance
                site_distances.append(distance)
            mean_distance = np.mean(site_distances)

            cation = site.specie
            anion = neighbors[0].specie
            ind = get_cation_anion_pair_index(cation, anion, params_dct)
            ideal = params_dct['R0'][ind]-params_dct['B'][ind]*np.log(cation.oxi_state/len(neighbors))
            deviations.append(ideal - mean_distance)
    return deviations

def get_gs_ls_deviations(dct, params_dct):
    short_dct = get_gs_and_ls(dct)
    gs_deviations = []
    ls_deviations = []

    for cmpd in tqdm(list(short_dct.keys())):
        gs = short_dct[cmpd]['gs']
        ls = short_dct[cmpd]['ls']
        gs_s_deviations = get_deviation(gs, params_dct)
        ls_s_deviations = get_deviation(ls, params_dct)

        gs_deviations += gs_s_deviations
        ls_deviations += ls_s_deviations

    return gs_deviations, ls_deviations

def bar_plot(h):
    xs = []
    ys = []
    for x_ind in range(len(h[0])):
        y = h[0][x_ind]
        x = np.round(h[1][x_ind] + 0.01, 2)
        xs.append(x)
        ys.append(y)
    return xs, ys
