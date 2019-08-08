#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""

Predicting re-hospitalization within two years of patient admission in acute
major depression: A multimodal machine learning approach.

This script cleans the dataset prior to predictive modelling.

"""

__author__ = 'Micah Cearns'
__contact__ = 'micahcearns@gmail.com'
__date__ = 'Jan 2019'

import pandas as pd
import numpy as np
import os
from googletrans import Translator

os.chdir('/Users/MicahJackson/anaconda/pycharm_wd/hospitilization_pred')


def pgrs_import():
    """

    Import function to parse individual PGRS csv files into one pandas
    dataframe merged on idbidirect.

    :return: Merged pandas dataframe

    """

    # Creating a base df to inspect, debug and merge with
    anx_pgrs_df = pd.read_csv('Anxiety_case_control_2016_0.5.csv')
    anx_pgrs_df = anx_pgrs_df[['IID', 'SCORE']]
    anx_pgrs_df = anx_pgrs_df.rename(columns = {'SCORE': 'anxiety_pgrs_score',
                                                'IID': 'idbidirect'})
    anx_pgrs_df['idbidirect'] = (anx_pgrs_df['idbidirect']
                                 .str[8:]
                                 .astype(dtype = int))

    # if threshold is '0.5':
    #     # Files to read in from wd
    pgrs_files = [
        'IGAP_Alzheimer_0.5.csv',
        'MDD23andme_0.5.csv',
        'PGC_anorexia_snp_all_13May2016_0.5.csv',
        'PGC_ASD_AUD_5Mar2015_0.5.csv',
        'pgc_bip_full_2012_0.5.csv',
        'pgc_mdd_full_2012_0.5.csv',
        'PGC_MDD_NEW_2017_DATA_0.5.csv',
        'PGC_SCZ2_2014_0.5.csv'
    ]
    # elif threshold is '0.05':
    # pgrs_files = [
    #     'IGAP_Alzheimer_0.05.csv',
    #     'MDD23andme_0.05.csv',
    #     'PGC_anorexia_snp_all_13May2016_0.05.csv',
    #     'PGC_ASD_AUD_5Mar2015_0.05.csv',
    #     'pgc_bip_full_2012_0.05.csv',
    #     'pgc_mdd_full_2012_0.05.csv',
    #     'PGC_MDD_NEW_2017_DATA_0.05.csv',
    #     'PGC_SCZ2_2014_0.05.csv'
    # ]
    # elif threshold is '0.01':
    # pgrs_files = [
    #     'IGAP_Alzheimer_0.01.csv',
    #     'MDD23andme_0.01.csv',
    #     'PGC_anorexia_snp_all_13May2016_0.01.csv',
    #     'PGC_ASD_AUD_5Mar2015_0.01.csv',
    #     'pgc_bip_full_2012_0.01.csv',
    #     'pgc_mdd_full_2012_0.01.csv',
    #     'PGC_MDD_NEW_2017_DATA_0.01.csv',
    #     'PGC_SCZ2_2014_0.01.csv'
    # ]
    # else:
    #     raise ValueError('Must specify p-value threshold 0.5, 0.05, or 0.01')

    # Names for each pgrs score in the pandas dataframe
    pgrs_names = [
        'alzheimer_pgrs',
        'mdd_23me_pgrs',
        'anorexia_pgrs',
        'asd_pgrs',
        'pgc_bip_pgrs',
        'pgc_mdd_pgrs',
        'pgc_mdd_new_pgrs',
        'pgc_scz_pgrs'
    ]

    # Merging
    for f, n in zip(pgrs_files, pgrs_names):
        pgrs = pd.read_csv(f)
        pgrs = pgrs[['IID', 'SCORE']]
        pgrs = pgrs.rename(columns = {'SCORE': n, 'IID': 'idbidirect'})
        pgrs['idbidirect'] = (pgrs['idbidirect'].str[8:].astype(dtype = int))
        anx_pgrs_df = anx_pgrs_df.merge(right = pgrs, on = 'idbidirect')
        anx_pgrs_df = anx_pgrs_df.iloc[:1004]

    # Adding 10,000 to each ID to match with the clinical file
    anx_pgrs_df['idbidirect'] += 10000

    return anx_pgrs_df


def outlier_detector(data):

    """
    This is a function to filter out potential outliers from a dataset

    :param data: Raw pandas dataframe

    :return: List of column names with potential outlier values

    """

    # Finding quantiles
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    # Getting outlier columns
    outlier_cols = (pd.DataFrame(((data < (Q1 - 1.5 * IQR)) |
                                  (data > (Q3 + 1.5 * IQR)))
                    .sum())
                    .rename(columns = {0: 'n_outliers'}))
    # Getting outlier names
    outliers = list(outlier_cols[outlier_cols.n_outliers >= 1].index)

    return outliers


def to_boolean(data):

    """
    These following 3 functions below allow for fast conversion of a large
    feature space to their correct data type. This is important for
    imputation later on. I have manually checked and edited any values that
    have been incorrectly parsed.

    :param data: Pandas dataframe

    :return: Pandas dataframe with parsed dtypes

    """

    # Finding max values
    bool_vars_df = pd.DataFrame(data.max().sort_values(ascending=True))
    bool_vars_df.rename(columns={0: 'max'}, inplace=True)
    # Selecting those between 0-1
    bool_vars_df = bool_vars_df.loc[bool_vars_df['max'] == 1.0]
    bool_names = list(bool_vars_df.index)
    # Converting to booleans
    data[bool_names] = data[bool_names].apply(lambda x: x.astype(bool))

    return data


def to_category(data):

    """

    :param data:

    :return:

    """
  
    # Selecting everything that is not a boolean
    to_cat = data.loc[:, data.dtypes != bool]
    # Finding max values
    cat_vars_df = pd.DataFrame(to_cat.max().sort_values(ascending=True))
    cat_vars_df.rename(columns={0:'max'}, inplace=True)
    # Selecting those < 5 (likely to be categorical, but still needs manual
    # checking)
    cat_vars_df = cat_vars_df.loc[cat_vars_df['max'] <= 7.0]
    # Grabbing their names
    cat_names = list(cat_vars_df.index)
    data[cat_names] = data[cat_names].apply(lambda x: x.astype('category'))

    return data


def to_float(data):

    """

    :param data:

    :return:

    """
  
    # Selecting categorical variables
    cat_df = data.loc[:, data.dtypes == 'category']
    # Counting out the number of unique values, should not be much greater than
    # 7 if they are truly categorical and not continuous (maybe slightly
    # greater) due to incorrect values
    to_float_df = cat_df.loc[:, cat_df.apply(lambda x: x.nunique()) >= 7]
    # If greater, taking their name and converting to floats
    to_float_names = list(to_float_df.columns)
    data[to_float_names] = data[to_float_names].apply(lambda x: x.astype(float))

    return data


def na_percent(raw_df):
    """

    :param raw_df: Pandas dataframe

    :return: Dataframe with percentage proportions of missing data for each
             predictor

    """

    raw_df = raw_df.loc[:1003] # selecting cases only
    na_percent = pd.DataFrame(raw_df.isna().mean()*100,
                              columns = ['Percent_missing'])
  
    return na_percent


# def pgrs_import():
#
#     """
#
#     Import function to parse individual PGRS csv files into one pandas
#     dataframe merged on idbidirect.
#
#     :return: Merged pandas dataframe
#
#     """
#
#     # ID matching df
#     id_match_df = pd.read_excel('BiDirect_ID_BMI_pheno.xlsx')
#     id_match_df = id_match_df[['idbidirect', 'idgenlab']]
#     id_match_df = id_match_df.rename(columns = {'idgenlab': 'IID'})
#     # Creating a base df to inspect, debug and merge with
#     anx_pgrs_df = pd.read_csv('Anxiety_case_control_2016_0.5.csv')
#     anx_pgrs_df = anx_pgrs_df[['IID', 'SCORE']]
#     anx_pgrs_df = anx_pgrs_df.rename(columns = {'SCORE': 'anxiety_pgrs_score'})
#     # Merging
#     id_match_df = pd.merge(right = id_match_df,
#                            left = anx_pgrs_df,
#                            on = 'IID')
#     id_match_df = id_match_df[['IID', 'idbidirect', 'anxiety_pgrs_score']]
#
#     # Files to read in from wd
#     pgrs_files = [
#         'IGAP_Alzheimer_0.5.csv',
#         'MDD23andme_0.5.csv',
#         'PGC_anorexia_snp_all_13May2016_0.5.csv',
#         'PGC_ASD_AUD_5Mar2015_0.5.csv',
#         'pgc_bip_full_2012_0.5.csv',
#         'pgc_mdd_full_2012_0.5.csv',
#         'PGC_MDD_NEW_2017_DATA_0.5.csv',
#         'PGC_SCZ2_2014_0.5.csv'
#     ]
#     # Names for each pgrs score in the pandas dataframe
#     pgrs_names = [
#         'alzheimer_pgrs',
#         'mdd_23me_pgrs',
#         'anorexia_pgrs',
#         'asd_pgrs',
#         'pgc_bip_pgrs',
#         'pgc_mdd_pgrs',
#         'pgc_mdd_new_pgrs',
#         'pgc_scz_pgrs'
#     ]
#     # Merging
#     for f, n in zip(pgrs_files, pgrs_names):
#         pgrs = pd.read_csv(f)
#         pgrs = pgrs[['IID', 'SCORE']]
#         pgrs = pgrs.rename(columns = {'SCORE': n})
#         id_match_df = id_match_df.merge(right = pgrs, on = 'IID')
#
#     return id_match_df


def build_english_codebook(df, raw_codebook):
    """

    :return:
    """

    # Cleaning up the final df names
    predictors = df.columns.tolist()
    pred_df = pd.DataFrame(data = predictors, columns = ['Predictors'])[0:424]
    pred_df['Predictors'] = pred_df['Predictors'].str.replace(pat = 's0_',
                                                              repl = '')

    # Processing the codebook
    codebook = raw_codebook[['Variablenname', 'Variablenlabel']]
    codebook = codebook.rename(
        columns = {'Variablenname': 'Predictors',
                   'Variablenlabel': 'Predictor_labels'}
    )
    # Merging the dataframes
    merged_df = pd.merge(left = pred_df,
                         right = codebook,
                         how = 'left',
                         on = 'Predictors')

    # Removing special characters
    merged_df['Predictor_labels'] = (merged_df['Predictor_labels']
                                     .replace(to_replace = np.nan,
                                              value = 'fehlt'))
    merged_df['Predictor_labels'] = (merged_df['Predictor_labels']
                                     .replace(regex = '\W+',
                                              value = ' '))

    # Translating
    translator = Translator()
    merged_df['English_labels'] = (
        merged_df['Predictor_labels']
        .apply(translator
               .translate, src = 'de', dest = 'en')
        .apply(getattr, args = ('text',))
    )

    english_codebook = merged_df
    # english_codebook.to_csv('bidirect_translated_vars.csv')

    return english_codebook


if __name__ == '__main__':

    # Reading in data
    raw_df = pd.read_csv('bidirect_2018.csv', low_memory = False)
    raw_df.drop('Unnamed: 0', axis = 1, inplace = True)
    bidirect_codebook = pd.read_excel(io = 'Codebook-BiDirect.xlsx',
                                      sheet_name = 'Variablenbeschreibung')

    # Removing any missing data codes so they can be imputed properly later
    raw_df.replace(to_replace = [-99, -999], value = np.nan)

    # Only selecting those with MDD
    raw_df = raw_df.loc[raw_df.s0_l_kohorte == 1]

    # Filtering out those who are missing relapse data
    raw_df['s2_dp_episode'].replace([2,3], np.nan, inplace = True)
    raw_df = raw_df[raw_df['s2_dp_episode'].isna() == False]

    # Setting up the outcome variable, the logic is that a patient must have
    # relapsed and been re-hospitalized at least once for one of the relapsers
    # in between baseline and their 2 year follow up
    raw_df['relapse'] = (
        np.where((raw_df['s2_dp_episode'] == 1)
                 & (raw_df['s2_dp_episoden_bek'] >= 1), 1, 0)
    )

    # Saving the outcome to merge back in later once the rest of the follow
    # up variables are lost
    outcome = raw_df[['idbidirect', 'relapse']]

    # Selecting baseline predictors
    raw_df = raw_df.loc[:, 'idbidirect':'s0_psychchip']

    # Adding back in relapse status
    raw_df = raw_df.merge(right = outcome, on = 'idbidirect')

    # Merging with PGRS scores
    pgrs_df = pgrs_import()
    raw_df = raw_df.merge(right = pgrs_df, on = 'idbidirect')

    # Structural imaging
    surf_df = pd.read_csv('CorticalMeasuresENIGMA_SurfAvg.csv')
    thick_df = pd.read_csv('CorticalMeasuresENIGMA_ThickAvg.csv')
    vol_df = pd.read_csv('LandRvolumes.csv')

    # Preparing imaging data
    # Removing str elements that don't match with the clinical data in IDs
    surf_df['SubjID'] = surf_df['SubjID'].str[13:]
    thick_df['SubjID'] = thick_df['SubjID'].str[13:]
    vol_df['SubjID'] = vol_df['SubjID'].str[13:]

    # Merging the dataframes together
    imaging_df = surf_df.merge(right = thick_df, on ='SubjID')
    imaging_df = (imaging_df
                  .merge(right = vol_df,
                         on = 'SubjID')
                  .rename(columns = {'SubjID': 'idbidirect'}))

    # Converting to int so I can merge with the clinical data file
    imaging_df['idbidirect'] = imaging_df.idbidirect.astype(dtype = int)

    # Adding in the imaging data to the raw_df
    raw_df = raw_df.merge(right = imaging_df, on = 'idbidirect')

    # List of IDs to drop with problematic imaging data
    ids = [10202, 10272, 10374, 10389, 10599, 10735, 10806,
           10820, 10879, 11004, 11006, 10290, 10168, 10353]
    for x in ids:
        raw_df = raw_df[raw_df.idbidirect != x]

    # raw_df = [raw_df[raw_df.idbidirect != x] for x in ids]
    # raw_df.to_csv('full_cohort_with_imaging.csv')

    # Subsetting out variables for prediction of re-hospitalization
    imaging_vars = [
         'idbidirect',
         'Lhippo',  # mean hippocampal volume
         'Rhippo',
         'L_medialorbitofrontal_thickavg',  # medial orbitofrontal
         'R_medialorbitofrontal_thickavg',
         'L_fusiform_thickavg',  # Fusifrom gyrus
         'R_fusiform_thickavg',
         'L_insula_thickavg',  # Insula
         'R_insula_thickavg',
         'L_rostralanteriorcingulate_thickavg',  # Rostral anterior
         'R_rostralanteriorcingulate_thickavg',
         'L_posteriorcingulate_thickavg',  # Posterior cingulate cortex
         'R_posteriorcingulate_thickavg',
         'L_middletemporal_thickavg',  # Left middle temporal gyrus
         'R_inferiortemporal_thickavg', # Right inferior temporal gyrus
         'R_caudalanteriorcingulate_thickavg',  # Right caudal ACC
         'ICV']  # Total intracranial volume

    demo_and_general_health_vars = [
        's0_ges_fk',
        's0_sf1',
        's0_sf2',
        's0_migra1',
        's0_migra4m',
        's0_migra4v',
        's0_szstand',
        's0_sz_part',
        's0_job_f1',
        's0_szfam1',
        's0_szlage',
        's0_szeink',
        's0_szfam3_1',
        's0_szfam3_2',
        's0_szfam3_4',
        's0_szfam3_5',
        's0_szfam3_6',
        's0_szfam3_7',
        's0_szfam3_8',
        's0_szfam3_9',
        's0_szfam3_10',
        's0_szfam3f_11',
        's0_szfam3f_12',
        's0_szfam3f_13',
        's0_dx_chschmerz',
        's0_rls_f1',
        's0_rfdiat',
        's0_sm_f1',
        's0_sz_mde1',
        's0_minia1',
        's0_minia2',
        's0_alter1',
        's0_sex',
        's0_weight',
        's0_height',
        's0_waist',
        's0_edu_f1',
        's0_edu_f2',
        's0_bildungsgrad'
    ]

    serum_vars = [
        's0_andrix',
        's0_b17o',
        's0_shbg',
        's0_testosteron',
        's0_cholesterol',
        's0_ft3',
        's0_hdl',
        's0_ft4',
        's0_tsh',
        's0_hscrp'
    ]

    # Ham, CES-D, alcohol consumption
    clin_vars = list(raw_df.loc[:, 's0_pm_a2':'s0_al_f8'].columns)
    print(raw_df.loc[:, 's0_pm_a2':'s0_al_f8'])

    # Pain sensitivity questionnaire, eq5d, and childhood trauma, and psqi
    # sleep quality
    pain_vars = list(raw_df.loc[:, 's0_psq1':'s0_ctssum'].columns)

    # Cardiovascular predictors, ECG and bioelectrical impedance analysis
    cardio_vars = [
         's0_ekg_hr',
         's0_bia_bmi',
         's0_bia_ecm_bcm_index',
         's0_bia_grundumsatz',
         's0_bia_handwiderstand',
         's0_bia_kfett_k_kg',
         's0_bia_koerperwasser',
         's0_bia_magermasse'
    ]

    # Health changes and Stroop test
    health_vars = list(raw_df.loc[:, 's0_s_lsstair':'s0_stroop'].columns)

    # Medication taken at baseline assesment
    medication_vars = list(raw_df.loc[:, 's0_med_a02':'s0_med_v06'].columns)

    # IPAQ physical activity
    exercise_vars = list(raw_df.loc[:, 's0_ipaq_f1':
                                       's0_ipaq_total_met_100hours'].columns)

    pgrs_vars = [
        'anxiety_pgrs_score',
        'alzheimer_pgrs',
        'mdd_23me_pgrs',
        'anorexia_pgrs',
        'asd_pgrs',
        'pgc_bip_pgrs',
        'pgc_mdd_pgrs',
        'pgc_mdd_new_pgrs',
        'pgc_scz_pgrs'
    ]

    outcome = ['relapse']

    # Merging lists
    final_vars = (imaging_vars
                  + serum_vars
                  + pgrs_vars
                  + demo_and_general_health_vars
                  + clin_vars
                  + pain_vars
                  + cardio_vars
                  + health_vars
                  + medication_vars
                  + exercise_vars
                  + outcome)

    # Selecting from df
    raw_df = raw_df[final_vars]

    # Inspecting
    print(raw_df.columns)

    # Dropping unnecessary columns
    drop_one = list(raw_df.loc[:, 's0_d_ishi_t02':'s0_d_smell12'].columns)
    drop_two = list(raw_df.loc[:, 's0_s_version':'s0_basic_n'].columns)
    drop_three = list(raw_df.loc[:, 's0_emo_valence':
                                    's0_emo_aroudfneg'].columns)
    drop_four = ['s0_s_lsnutri']

    final_drop = (drop_one + drop_two + drop_three + drop_four)
    raw_df.drop(final_drop, axis = 1, inplace = True)

    # Going through all strings and finding those that are actually meant to be
    # numerical yet have random string data accidentally placed within column
    # cells
    obj_names = list(raw_df.select_dtypes(include = [object]))

    # Dropping strings, all noise vars with meta data
    raw_df.drop(obj_names, axis = 1, inplace = True)

    # Finding potential outliers that are based off of genuine coding errors
    potential_outliers = outlier_detector(raw_df)

    # Making a subsetted raw_df for inspection that contains potential outliers
    outlier_raw_df = raw_df[potential_outliers]

    raw_df['s0_bildungsgrad'].replace(
        to_replace = [1991.0,
                      1974.0,
                      1987.0,
                      1969.0,
                      1979.0,
                      1990.0,
                      1995.0,
                      2002.0,
                      1968.0],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_ipaq_category'].replace(
        to_replace = [3559.500000,
                      946.500000,
                      678.000000,
                      1039.500000,
                      1386.000000,
                      0.482500,
                      3298.000000,
                      0.292667,
                      5586.000000,
                      1410.000000,
                      1120.000000],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_pi_n_inpatient'].replace(
        to_replace = [320040.0,
                      320111.0,
                      99],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_hamd_17total'].replace(
        to_replace = [18002.0,
                      19220.0],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_pm_mela_lif'].replace(
        to_replace = [99.0,
                      5.0,
                      2.0],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_subtype_mini_prob'].replace(
        to_replace = 99.0,
        value = np.nan,
        inplace = True
    )

    raw_df['s0_psqi4'].replace(
        to_replace = [82800.0,
                      75600.0],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_ks_wmigspks'].replace(
        to_replace = [18648.0,
                      19047.0],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_ipaq_f7m'].replace(
        to_replace = [1756.0,
                      2895.0],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_ipaq_total_met'].replace(
        to_replace = 0.0,
        value = np.nan,
        inplace = True
    )

    raw_df['s0_sex'].replace(
        to_replace = [172,
                      37.41168501],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_cesd20'].replace(
        to_replace = 40,
        value = np.nan,
        inplace = True
    )

    raw_df['s0_cesd2c'].replace(
        to_replace = 40,
        value = np.nan,
        inplace = True
    )

    raw_df['s0_cesd3c'].replace(
        to_replace = 18,
        value = np.nan,
        inplace = True
    )

    raw_df['s0_al_f4_s'].replace(
        to_replace = 12,
        value = np.nan,
        inplace = True
    )

    raw_df['s0_eq5d_1'].replace(
        to_replace = [10, 1],
        value = np.nan,
        inplace = True
    )

    raw_df['s0_eq5d_5'].replace(
        to_replace = 75,
        value = np.nan,
        inplace = True
    )

    raw_df['s0_psqi6'].replace(
        to_replace = [20, 13],
        value = np.nan,
        inplace = True
    )

    # Dropping zero variance variables
    raw_df = raw_df.loc[:, raw_df.apply(pd.Series.nunique) != 1]

    # Removing variables with more than 20% NA
    raw_df = raw_df[raw_df.columns[raw_df.isnull().mean() < 0.2]]

    # Parsing to the correct data type. This is important for the ML pipeline
    # as the imputation strategy depends on the dtype of each feature
    raw_df = to_category(raw_df)
    raw_df = to_float(raw_df)
    raw_df = to_boolean(raw_df)

    # Try subsetting variables here from the excel clean up file
    raw_df.columns = raw_df.columns.str.replace(r's0_', '')
    print(raw_df.columns)

    # Now select the subset
    raw_df = raw_df[['Lhippo',
                     'Rhippo',
                     'L_medialorbitofrontal_thickavg',
                     'R_medialorbitofrontal_thickavg',
                     'L_fusiform_thickavg',
                     'R_fusiform_thickavg',
                     'L_insula_thickavg',
                     'R_insula_thickavg',
                     'L_rostralanteriorcingulate_thickavg',
                     'R_rostralanteriorcingulate_thickavg',
                     'L_posteriorcingulate_thickavg',
                      'R_posteriorcingulate_thickavg',
                     'L_middletemporal_thickavg',
                     'R_inferiortemporal_thickavg',
                     'R_caudalanteriorcingulate_thickavg',
                     'andrix',
                     'b17o',
                     'shbg',
                     'testosteron',
                     'cholesterol',
                     'ft3',
                     'hdl',
                     'ft4',
                     'tsh',
                     'hscrp',
                     'anxiety_pgrs_score',
                     'alzheimer_pgrs',
                     'mdd_23me_pgrs',
                     'anorexia_pgrs',
                     'asd_pgrs',
                     'pgc_bip_pgrs',
                     'pgc_mdd_pgrs',
                     'pgc_mdd_new_pgrs',
                     'pgc_scz_pgrs',
                     'sex',
                     'sf1',
                     'job_f1',
                     'szlage',
                     'szeink',
                     'rfdiat',
                     'sm_f1',
                     'sz_mde1',
                     'minia1',
                     'minia2',
                     'alter1',
                     'weight',
                     'edu_f1',
                     'edu_f2',
                     'ph_hamd1',
                     'ph_ids8',
                     'ph_hamd2',
                     'ph_ids4',
                     'ph_hamd3',
                     'ph_hamd4',
                     'ph_hamd5',
                     'ph_hama6',
                     'ph_hama7',
                     'ph_hama8',
                     'ph_hamd9',
                     'ph_hama10',
                     'ph_hamd11',
                     'ph_ids12',
                     'ph_ids14',
                     'ph_hamd12',
                     'ph_hamd13',
                     'ph_ids30',
                     'ph_hamd14',
                     'ph_hamd15',
                     'ph_hama16',
                     'ph_hamd17',
                     'ph_hama18',
                     'ph_hama19',
                     'ph_hama20',
                     'ph_hama21',
                     'ph_hama22',
                     'ph_hama23',
                     'ph_hama24',
                     'ph_hama25',
                     'ph_hamd26',
                     'ph_ids29',
                     'ph_hamd27',
                     'ph_hamd28',
                     'ph_hamd29',
                     'ph_hama30',
                     'ph_hamd31',
                     'pi_n_epi_inpatient',
                     'pi_n_episodes',
                     'pi_n_inpatient',
                     'pi_age_1episode',
                     'pi_dat_prevepi',
                     'pi_dat_1inpatient',
                     'pi_curepi_year',
                     'hamd_17total',
                     'hamd_17n',
                     'hama_14total',
                     'hama_14n',
                     'pm_a2',
                     'pm_lifa7_a',
                     'pm_lifa7_b',
                     'pm_lifa7_c',
                     'pm_lifa7_d',
                     'pm_lifa7_e',
                     'pm_lifa7_f',
                     'pm_lifepis_n',
                     'pm_lif_age',
                     'pm_d1_a',
                     'pm_d2_a',
                     'pm_o1_a',
                     'pm_a4',
                     'cesd1',
                     'cesd2',
                     'cesd3',
                     'cesd4',
                     'cesd5',
                     'cesd6',
                     'cesd7',
                     'cesd8',
                     'cesd9',
                     'cesd10',
                     'cesd11',
                     'cesd12',
                     'cesd13',
                     'cesd14',
                     'cesd15',
                     'cesd16',
                     'cesd17',
                     'cesd18',
                     'cesd19',
                     'cesd20',
                     'cesdsum',
                     'cesd2c',
                     'cesd3c',
                     'cesd_dp',
                     'cesd_wb',
                     'cesd_so',
                     'cesd_ip',
                     'eq5d_2',
                     'eq5d_3',
                     'eq5d_4',
                     'eq5d_5',
                     'eq5d_vas',
                     'eq5dscore',
                     'diet_score',
                     'al_f1',
                     'al_f3_b',
                     'al_f4_b',
                     'al_f5',
                     'al_f8',
                     'basicsum',
                     'alkohol',
                     'ks_allks',
                     'ks_mig',
                     'psq_mean',
                     'dx_chschmerz',
                     'cts_1',
                     'cts_2',
                     'cts_3',
                     'cts_4',
                     'cts_5',
                     'ctssum',
                     'psqi7',
                     'psqilaten',
                     'psqidurat',
                     'psqihse',
                     'psqidistb',
                     'psqidaydys',
                     'psqi_sum',
                     'ekg_hr',
                     'bia_bmi',
                     'bia_ecm_bcm_index',
                     'bia_grundumsatz',
                     'bia_handwiderstand',
                     'bia_kfett_k_kg',
                     'bia_koerperwasser',
                     'bia_magermasse',
                     's_lsstair',
                     's_lswalk',
                     's_lsweight',
                     's_lsalc',
                     's_lssmoke',
                     'med_a03',
                     'med_a10',
                     'med_a11',
                     'med_a12',
                     'med_c07',
                     'med_c08',
                     'med_c09',
                     'med_c10',
                     'med_g03',
                     'med_g04',
                     'med_h02',
                     'med_h03',
                     'med_l04',
                     'med_m03',
                     'med_n02aa',
                     'med_n02ab',
                     'med_n02ae',
                     'med_n02ax',
                     'med_n02ba',
                     'med_n02bb',
                     'med_n02be',
                     'med_n02bg',
                     'med_n02cc',
                     'med_n03ae',
                     'med_n03af',
                     'med_n03ag',
                     'med_n03ax',
                     'med_n05ab',
                     'med_n05ad',
                     'med_n05af',
                     'med_n05ah',
                     'med_n05al',
                     'med_n05an',
                     'med_n05ax',
                     'med_n05ba',
                     'med_n05cd',
                     'med_n05cf',
                     'med_n05ch',
                     'med_n05cm',
                     'med_n06aa',
                     'med_n06ab',
                     'med_n06af',
                     'med_n06ax',
                     'med_v06',
                     'ipaq_f1d',
                     'ipaq_f2h',
                     'ipaq_f3d',
                     'ipaq_f4h',
                     'ipaq_f5d',
                     'ipaq_f6h',
                     'ipaq_f7h',
                     'ipaq_total_met',
                     'ipaq_category'] + outcome]

    # # Biomarker only and clinical only dfs
    # imaging_df_cleaned = raw_df[imaging_vars + outcome]
    # serum_df_cleaned = raw_df[serum_vars + outcome]
    # cardio_df_cleaned = raw_df[cardio_vars + outcome]
    # pgrs_df_cleaned = raw_df[pgrs_vars + outcome]
    #
    # biomarkers = (serum_vars
    #               + imaging_vars
    #               + cardio_vars
    #               + pgrs_vars
    #               + outcome)
    #
    # biomarkers_cleaned = raw_df[biomarkers]
    # full_clin_vars_only = raw_df.drop(biomarkers, axis = 1)
    # clin_df_cleaned = raw_df[list(full_clin_vars_only.columns) + outcome]

    # Exporting
    raw_df.to_csv('subset_multi_df_cleaned_psqi7.csv')
    raw_df.to_pickle('subset_multi_df_cleaned_psqi7.pkl')

    # biomarkers_cleaned.to_pickle('subset_biomarkers_cleaned.pkl')
    # imaging_df_cleaned.to_pickle('subset_imaging_df_cleaned.pkl')
    # serum_df_cleaned.to_pickle('subset_serum_df_cleaned.pkl')
    # cardio_df_cleaned.to_pickle('subset_cardio_df_cleaned.pkl')
    # pgrs_df_cleaned.to_pickle('subset_pgrs_df_cleaned.pkl')
    # clin_df_cleaned.to_pickle('subset_clin_df_cleaned.pkl')





