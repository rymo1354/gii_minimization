#@author: rymo1354
# date - 1/20/2022

from gii_calculator import GIICalculator
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
