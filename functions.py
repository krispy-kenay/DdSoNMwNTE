#!/usr/bin/env python3
''' 0 Setup '''
import os
import sys
import time
import warnings
import chemparse

import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from pymatgen.core import Structure
from scipy.stats import linregress

warnings.filterwarnings("ignore")

''' ----- 1 Loading Data ----- '''

## Function to load .cif file into a pandas DataFrame combined with the pymatgen Structure object

def load_PCD_cif(folder_path):
    # Setting up console output variables
    print("Loading: PCD cifs")
    x, n, x_tot, stime = 0, 0, len(os.listdir(folder_path)), time.time()
    # Empty list to store data
    cif_data = []

    # Iterate over all files in folder
    for file in Path(folder_path).glob('*.cif'):
        x += 1 
        progress_bar(x, x_tot) # Visual progress bar output

        try:
            # Load .cif file as raw dictionary and pymatgen.core.structure dictionary
            dict = MMCIF2Dict(file)
            struc = Structure.from_file(file)
            dict_pymatgen = struc.as_dict()
            # Combining the two dictionaries into one
            dict.update(dict_pymatgen)
            # Turn it into a dataframe with one column and transpose, this way everything will be on the same row later
            temp = pd.DataFrame.from_dict(dict, orient='index')
            temp = temp.transpose()
            # Add database entry 
            temp.insert(0, 'database', 'PCD')
            # Append the dataframe into the list
            cif_data.append(temp)

        except:
            # Ignore .cifs that are missing too much data and count how many failed
            n += 1
    
    # Combine all the data into one dataframe
    df = pd.concat(cif_data)

    # Add PCD to the id as to not have conflicting index entries later
    df['data_'] = df['data_'].apply(lambda x: 'PCD_' + x)

    # Final output variables
    failed_n, ttime = str(int(n / x * 100)), str(int((time.time() - stime) / 60))
    
    # Print final output to see what percentage failed to load and the total time taken
    print("Final Report: " + failed_n + "% Failed to load, " + ttime + " minutes taken for entire operation")
    return df

## Function to load .cif file into a pandas DataFrame combined with the pymatgen Structure object

def load_ICSD_cif(folder_path):
    # Setting up console output variables
    print("Loading: ICSD cifs")
    x, n, x_tot, stime = 0, 0, len(os.listdir(folder_path)), time.time()
    # Empty list to store data
    cif_data = []

    # Iterate over all files in folder
    for file in Path(folder_path).glob('*.cif'):
        x += 1 
        progress_bar(x, x_tot) # Visual progress bar output

        try:
            # Load .cif file as raw dictionary and pymatgen.core.structure dictionary
            dict = MMCIF2Dict(file)
            struc = Structure.from_file(file)
            dict_pymatgen = struc.as_dict()
            # Combining the two dictionaries into one
            dict.update(dict_pymatgen)
            # Turn it into a dataframe with one column and transpose, this way everything will be on the same row later
            temp = pd.DataFrame.from_dict(dict, orient='index')
            temp = temp.transpose()
            # Add database entry 
            temp.insert(0, 'database', 'ICSD')
            # Append the dataframe into the list
            cif_data.append(temp)

        except:
            # Ignore .cifs that are missing too much data and count how many failed
            n += 1
    
    # Combine all the data into one dataframe
    df = pd.concat(cif_data)
    # Add ICSD to the id as to not have conflicting index entries later
    df['data_'] = df['data_'].replace('-ICSD','', regex=True)
    df['data_'] = df['data_'].apply(lambda x: 'ICSD_' + x)
    # Final output variables
    failed_n, ttime = str(int(n / x * 100)), str(int((time.time() - stime) / 60))

    # Print final output to see what percentage failed to load and the total time taken
    print("Final Report: " + failed_n + "% Failed to load, " + ttime + " minutes taken for entire operation")
    return df



''' ----- 2 Processing Data ----- '''
''' Cleaning '''

def keep_columns(df, custom_columns=[]):
    df_mod = df.copy()
    if custom_columns == []:
        custom_columns = [
            'data_',
            '_chemical_formula_sum',
            '_space_group_name_H-M_alt',
            '_space_group_IT_number',
            '_diffrn_ambient_temperature',
            '_cell_length_a',
            '_cell_length_b',
            '_cell_length_c',
            '_cell_angle_alpha',
            '_cell_angle_beta',
            '_cell_angle_gamma',
            '_cell_volume',
            '_diffrn_ambient_pressure',
            '_exptl_crystal_density_diffrn',
            '_atom_type_symbol',
            'database',
            '_journal_year',
            '_journal_volume',
            '_journal_page_first',
            '_journal_page_last',
            '@module',
            '@class',
            'charge',
            'lattice',
            'sites'
            ]
    df_mod = df_mod[custom_columns]
    return df_mod

def rename_columns(df, custom_dict={}):
    df_mod = df.copy()
    if custom_dict == {}:
        custom_dict = {
            '_chemical_formula_sum': 'FORMULA',
            '_space_group_name_H-M_alt': 'SPACEGROUP_SYM',
            '_space_group_IT_number': 'SPCAEGROUP_NO',
            '_diffrn_ambient_temperature': 'TEMPERATURE',      #K
            '_cell_length_a' : 'CELL_LENGTH_A',                     #A
            '_cell_length_b' : 'CELL_LENGTH_B',                     #A
            '_cell_length_c' : 'CELL_LENGTH_C',                     #A
            '_cell_angle_alpha' : 'ALPHA',              # ° 
            '_cell_angle_beta': 'BETA',                 # °
            '_cell_angle_gamma': 'GAMMA',               # °
            '_cell_volume' : 'VOLUME',                     #A³
            '_diffrn_ambient_pressure': 'PRESSURE',         #MPa
            '_exptl_crystal_density_diffrn': 'DENSITY', #g/cm³
            '_atom_type_symbol': 'ELEMENT_COUNT',
            'database': 'DATABASE',
            'data_':'ID',
            '_journal_year':'REFERENCE'
        }
    df_mod.rename(columns=custom_dict, inplace=True)
    df_mod.set_index('ID', inplace=True)
    return df_mod

def remove_characters(df):
    df_mod = df.copy()
    for column in df_mod.columns:
        if (column != 'DATABASE' and column != 'PRESSURE' and column != '@module' and column != '@class' and column != 'charge' and column != 'lattice' and column != 'sites'):
            df_mod[column] = df_mod[column].str.join(', ')
        if column == 'FORMULA':
            df_mod[column] = df_mod[column].str.replace(' |~','', regex=True)
            df_mod[column] = df_mod[column].apply(lambda x: chemparse.parse_formula(x))
            df_mod[column] = df_mod[column].apply(lambda x: ''.join(key + str(value) for key, value in x.items()))
        if column == 'SPACEGROUP_SYM':
            df_mod[column] = df_mod[column].str.replace(' |~|\(|\)','', regex=True)
            df_mod[column] = df_mod[column].str.replace('originchoice2',' O2')
        if column == 'ELEMENT_COUNT':
            df_mod[column] = df_mod[column].apply(lambda x: np.NaN if type(x) == float or type(x) == int else x.count(',') + 1)
    
    df_mod = df_mod.assign(REFERENCE = '(' + df_mod.REFERENCE.astype(str) + ') ' + df_mod._journal_volume.astype(str) + ', ' + df_mod._journal_page_first.astype(str) + '-' + df_mod._journal_page_last.astype(str))
    df_mod = df_mod.drop(columns=["_journal_volume","_journal_page_first","_journal_page_last"])

    return df_mod   



''' Manipulate Data '''

def fill_missing_values(df, rt=293.15, rp=101.325):
    # create copy as to not edit input dataframe
    df_mod = df.copy()
    # While importing, some entries become '?' instead of NaN, so we change that here
    df_mod = df_mod.replace('\?', np.NaN , regex=True)
    # Some temperature values had secondary entries, we remove them first here and convert to float
    df_mod['TEMPERATURE'] = df_mod['TEMPERATURE'].str.replace(r"\(.*?\)", "", regex=True)
    df_mod['TEMPERATURE'] = df_mod['TEMPERATURE'].astype(float)
    # columns to be edited
    temp_str = 'TEMPERATURE'
    press_str = 'PRESSURE'
    # replace Nan and '?' values with room conditions
    df_mod[temp_str] = df_mod[temp_str].fillna(rt)
    df_mod[temp_str] = df_mod[temp_str].replace('?',rt)
    df_mod[press_str] = df_mod[press_str].fillna(rt)
    df_mod[press_str] = df_mod[press_str].replace('?',rt)
    return df_mod

def remove_few_entries(df, num_entries):
    # create copy as to not edit input dataframe
    df_mod = df.copy()
    # Sort columns by FORMULA and SPACEGROUP_SYM, then remove ones with less than 3 entries
    df_mod = df_mod.groupby(['FORMULA','SPACEGROUP_SYM'])
    df_mod = df_mod.filter(lambda x: len(x) >= num_entries)
    # create dictionary with amount of entries and crystals
    entries_dict = {
        'kept entries' : len(df_mod),
        'removed entries' : (len(df) - len(df_mod)),
        'kept crystals' : len(df_mod.groupby(['FORMULA','SPACEGROUP_SYM'])),
        'removed crystals' :(len(df.groupby(['FORMULA','SPACEGROUP_SYM'])) - len(df_mod.groupby(['FORMULA','SPACEGROUP_SYM'])))
    }
    return df_mod, entries_dict


def grouped_histogram(df, title):
    hist = df.groupby(['FORMULA','SPACEGROUP_SYM']).size()
    max_num = max(hist)
    fig = hist.plot.hist(bins=max_num).set(title=title, xlim=(1,15))


def drop_NaN(df):
    df_mod = df.copy()
    # Drop all entries without temperature (just to be extra sure)
    df_mod = df_mod.dropna(axis=0, how='any',subset=['TEMPERATURE'])
    # Drop columns where all values are NaN
    df_mod = df_mod.dropna(axis=1, how='all')
    # Drop columns with 40% NaN values
    df_mod = df_mod.dropna(axis=1, thresh = int(df.shape[0]*0.2))
    # Drop entries with any NaN left
    df_mod = df_mod.dropna(axis=0, how='any')
    # create dictionary with removed entries length
    length_dict = {
        'kept entries' : len(df_mod),
        'removed entries' : (len(df) - len(df_mod))
    }

    return df_mod, length_dict

''' Calculate CTE '''

def temperature_bins(df, spacing):
    df_mod = df.copy()

    # 
    start = min(df_mod['TEMPERATURE'])
    end = max(df_mod['TEMPERATURE'])
    steps = round((end - start) / spacing)
    temp_resolution = 1.0

    for i in range(steps):
        df_mod['TEMPERATURE'].replace(to_replace = np.arange(i * spacing, (i + 1) * spacing,temp_resolution),
                                    value = np.mean([i * spacing, (i + 1) * spacing]),
                                    inplace = True)
    return df_mod

def average_temperature(df, a_str = ['CELL_LENGTH_A'], b_str = ['CELL_LENGTH_B'], c_str = ['CELL_LENGTH_C'], vol_str = ['VOLUME']):
    df[a_str] = df[a_str].astype(float)
    df[b_str] = df[b_str].astype(float)
    df[c_str] = df[c_str].astype(float)
    df[vol_str] = df[vol_str].astype(float)

    df_mod = df.groupby(['FORMULA','SPACEGROUP_SYM','TEMPERATURE']).mean()
    
    df_mod.reset_index(inplace=True)

    df_mod = df_mod.set_index(['FORMULA','SPACEGROUP_SYM'])

    df_mod = df_mod[['TEMPERATURE',a_str, b_str, c_str, vol_str]]
    
    return df_mod

def CTE_reg(df, temp_str = ['TEMPERATURE'], a_str = ['CELL_LENGTH_A'], b_str = ['CELL_LENGTH_B'], c_str = ['CELL_LENGTH_C'], vol_str = ['VOLUME']):
    CTE_strs = ['CTE '+a_str, 'CTE '+b_str, 'CTE '+c_str, 'CTE '+vol_str]
    R2_strs = ['R^2 '+a_str, 'R^2 '+b_str, 'R^2 '+c_str, 'R^2 '+vol_str]
    rename_dict = {temp_str: 'Temp',
                   a_str: 'a',
                   b_str: 'b',
                   c_str: 'c',
                   vol_str: 'Vol'}
    df_linreg = df.copy()
    df_linreg = df_linreg.rename(columns=rename_dict)

    'calculate logarithm'
    df_linreg['ln_V'] = np.log(df_linreg['Vol'])
    df_linreg['ln_a'] = np.log(df_linreg['a'])
    df_linreg['ln_b'] = np.log(df_linreg['b'])
    df_linreg['ln_c'] = np.log(df_linreg['c'])

    'do linear regression of all log values'
    df_V = df_linreg.groupby(['group'])
    #print(df_linreg.head(),df_linreg.shape)
    linreg_V = df_V.apply(lambda v: linregress(v.Temp, v.ln_V))
    #print(linreg_V[1])
    df_a = df_linreg.groupby(['group'])
    linreg_a = df_a.apply(lambda v: linregress(v.Temp, v.ln_a))
    df_b = df_linreg.groupby(['group'])
    linreg_b = df_b.apply(lambda v: linregress(v.Temp, v.ln_b))
    df_c = df_linreg.groupby(['group'])
    linreg_c = df_c.apply(lambda v: linregress(v.Temp, v.ln_c))
    
    'get slope and R value of the linear regression'
    l = len(df_V)
    slope_V = np.array([linreg_V[i].slope for i in range(l)])
    #print(slope_V)
    R_value_V = np.array([linreg_V[i].rvalue for i in range(l)])
    slope_a = np.array([linreg_a[i].slope for i in range(l)])
    R_value_a = np.array([linreg_a[i].rvalue for i in range(l)])
    slope_b = np.array([linreg_b[i].slope for i in range(l)])
    R_value_b = np.array([linreg_b[i].rvalue for i in range(l)])
    slope_c = np.array([linreg_c[i].slope for i in range(l)])
    R_value_c = np.array([linreg_c[i].rvalue for i in range(l)])

    'create dataframe with the group indizes'
    df_CTE = pd.DataFrame(linreg_a.index)

    'add them to an dataframe'
    CTE = [slope_a, slope_b, slope_c, slope_V]
    R2 = [R_value_a ** 2,R_value_b ** 2,R_value_b ** 2,R_value_V ** 2]
    for i in range(4):
        df_CTE[CTE_strs[i]] = CTE[i]
        df_CTE[R2_strs[i]] = R2[i]

    return df_CTE


''' X Miscellaneous '''
def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)