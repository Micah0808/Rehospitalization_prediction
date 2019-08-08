# Manipulation and plotting
import numpy  as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, levene
import matplotlib.pyplot as plt
from   matplotlib.colors import ListedColormap
import seaborn as sns
import os
sns.set_style("whitegrid")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)

# Setting working directory
os.chdir('/Users/MicahJackson/anaconda/pycharm_wd/hospitilization_pred')


def desc_stats(data = str, pkl=True):

    if pkl is True:

        desc_df = pd.read_pickle(data)

    else:

        desc_df = pd.read_csv(data)

    full_df = desc_df
    desc_df = desc_df[['relapse', 's0_alter1', 's0_hamd_17total', 's0_cesdsum',
                       's0_pi_n_epi_inpatient', 's0_sex']]

    desc_df = desc_df.apply(lambda x: x.astype(float))

    desc_df = desc_df.rename(columns = {'s0_alter1':'Age'})

    case = desc_df.loc[desc_df['relapse'] == 1]
    control = desc_df.loc[desc_df['relapse'] == 0]

    return (np.round(case.describe(), decimals = 2),
            np.round(control.describe(), decimals = 2),
            desc_df, full_df)


if __name__ == '__main__':

    case, control, desc_df, full_df = desc_stats(
        data = 'multi_cleaned_hospital_bidirect_df.pkl',
        pkl = True
    )

    case.to_csv('comorbid_MDD_desc.csv')
    control.to_csv('comorbid_No_MDD_desc.csv')

    # Women relapse
    desc_df[(desc_df.s0_sex == 2) & (desc_df.relapse == 1.0)].shape

    # Women no relapse
    desc_df[(desc_df.s0_sex == 2) & (desc_df.relapse == 0.0)].shape

    # Men relapse
    desc_df[(desc_df.s0_sex == 1) & (desc_df.relapse == 1.0)].shape

    # Men no relapse
    desc_df[(desc_df.s0_sex == 1) & (desc_df.relapse == 0.0)].shape

# =============================================================================
# Cases
#        relapse     Age  s0_hamd_17total  s0_cesdsum  n_epi_inpatient  s0_sex
# count    102.0  102.00           101.00      102.00           101.00  102.00
# mean       1.0   49.03            15.33       31.30             2.06    1.58
# std        0.0    7.32             6.59       12.97             2.00    0.50
# min        1.0   34.96             0.00        1.00             0.00    1.00
# 25%        1.0   43.08            11.00       22.25             1.00    1.00
# 50%        1.0   49.51            16.00       33.00             1.00    2.00
# 75%        1.0   55.34            20.00       41.00             2.00    2.00
# max        1.0   63.96            27.00       56.00            10.00    2.00

# =============================================================================
# Controls
#         relapse     Age   s0_hamd_17total  s0_cesdsum  n_epi_inpatient s0_sex
# count    278.0    278.00         278.00      276.00          274.00    278.00
# mean       0.0     49.91          12.71       25.40            1.42      1.61
# std        0.0      7.38           6.33       11.50            0.90      0.49
# min        0.0     35.15           0.00        0.00            0.00      1.00
# 25%        0.0     44.23           8.00       16.75            1.00      1.00
# 50%        0.0     49.86          13.00       26.00            1.00      2.00
# 75%        0.0     55.73          17.00       34.25            2.00      2.00
# max        0.0     65.37          33.00       48.00            6.00      2.00

    # Subsetting vars for t-tests
    df_continuous = desc_df[['Age',
                             's0_hamd_17total',
                             's0_cesdsum',
                             's0_pi_n_epi_inpatient',
                             'relapse']]

    # Running independent samples t-tests
    cont_vars = df_continuous.set_index('relapse')

    t_test_results = ttest_ind(a = cont_vars.loc[True],
                               b = cont_vars.loc[False],
                               equal_var = True,
                               nan_policy = 'omit')

    cont_p_vals = np.round([2.99986564e-01,
                            6.79767065e-04,
                            8.03177308e-05,
                            2.60316412e-03], decimals = 4)

    # Medications
    med_df = full_df[['s0_med_c07',
                      's0_med_n02cc',
                      's0_med_n05ab',
                      's0_med_n05ad',
                      's0_med_n05af',
                      's0_med_n05ah',
                      's0_med_n05al',
                      's0_med_n05an',
                      's0_med_n05ax',
                      's0_med_n05ba',
                      's0_med_n05cd',
                      's0_med_n05cf',
                      's0_med_n05ch',
                      's0_med_n06aa',
                      's0_med_n06ab',
                      's0_med_n06af',
                      's0_med_n06ax',
                      'relapse']]

    med_df = med_df.rename(
        columns = {'s0_med_c07':'Beta_blocking_agents',
                   's0_med_n02cc':'Selective_serotonin_agonists',
                   's0_med_n05ab':'Phenothiazines_with_piperazine_structure',
                   's0_med_n05ad':'Butyrophenone_derivates',
                   's0_med_n05af':'Thioxanthene_derivates',
                   's0_med_n05ah':'Diazepines_oxazepined_thiazepines_oxepines',
                   's0_med_n05al':'Benzamides',
                   's0_med_n05an':'Lithium',
                   's0_med_n05ax':'Other_antipsychotics',
                   's0_med_n05ba':'Benzodiazepine_derivates',
                   's0_med_n05cd':'Benzodiazepine_derivates_2',
                   's0_med_n05cf':'Benzodiazepine_related_drugs',
                   's0_med_n05ch':'Melatonin_receptor_agonists',
                   's0_med_n06aa':'Non_selective_momoamine_reuptake_inhibitors',
                   's0_med_n06ab':'Selective_serotonin_reuptake_inhibitors',
                   's0_med_n06af':'Monoamine_oxidase_inhibitors_non_selective',
                   's0_med_n06ax':'Other_antidepressants'}
    )

    med_df = med_df.apply(lambda x: x.astype(int))

    med_df['Benzodiazepines'] = (med_df['Benzodiazepine_derivates']
                                 + med_df['Benzodiazepine_derivates_2'])

    med_df['Selective_serotonin_reuptake_inhibitors'] = (
        med_df['Selective_serotonin_reuptake_inhibitors']
        + med_df['Selective_serotonin_agonists']
    )

    med_df['Non_selective_momoamine_reuptake_inhibitors'] = (
        med_df['Non_selective_momoamine_reuptake_inhibitors']
        + med_df['Monoamine_oxidase_inhibitors_non_selective']
    )

    med_df['Other_antipsychotics'] = (
        med_df['Other_antipsychotics']
        + med_df['Benzamides']
        + med_df['Thioxanthene_derivates']
        + med_df['Phenothiazines_with_piperazine_structure']
    )

    med_df['Other_antidepressants'] = (
        med_df['Other_antidepressants']
        + med_df['Melatonin_receptor_agonists']
    )

    med_df['Benzodiazepines'] = (
        med_df['Benzodiazepines']
        + med_df['Benzodiazepine_related_drugs']
    )

    med_df = med_df.drop(['Benzodiazepine_derivates',
                          'Benzodiazepine_derivates_2',
                          'Selective_serotonin_agonists',
                          'Monoamine_oxidase_inhibitors_non_selective',
                          'Benzamides',
                          'Thioxanthene_derivates',
                          'Phenothiazines_with_piperazine_structure',
                          'Melatonin_receptor_agonists',
                          'Benzodiazepine_related_drugs'],
                         axis = 1)

    # How many are currently taking a psychotropic medication of some kind?
    med_df['med_load'] = (
        med_df['Beta_blocking_agents']
        + med_df['Butyrophenone_derivates']
        + med_df['Diazepines_oxazepined_thiazepines_oxepines']
        + med_df['Lithium']
        + med_df['Other_antipsychotics']
        + med_df['Non_selective_momoamine_reuptake_inhibitors']
        + med_df['Selective_serotonin_reuptake_inhibitors']
        + med_df['Other_antidepressants']
        + med_df['Benzodiazepines']
    )

    # 352/380. Specifically, 92.6% of patients.
    med_df['treated'] = np.where(med_df['med_load'] >= 1, 1, 0)

    # 334/380. 87.9 are currently on some form of antidepressant.
    med_df['anti_deps'] = np.where(
        (med_df['Beta_blocking_agents'] == 1) |
        (med_df['Non_selective_momoamine_reuptake_inhibitors'] == 1) |
        (med_df['Selective_serotonin_reuptake_inhibitors'] == 1) |
        (med_df['Other_antidepressants'] == 1), 1, 0
    )

    # 153/380. 40.3% on some for of antipsychotic.
    med_df['anti_psychotics'] = np.where(
        (med_df['Diazepines_oxazepined_thiazepines_oxepines'] == 1) |
         (med_df['Other_antipsychotics'] == 1) |
         (med_df['Butyrophenone_derivates'] == 1), 1, 0
    )

    # Setting up for chi2 tests
    independent_ordinal_vars = (med_df
                                .drop('relapse', axis = 1)
                                .dropna(axis = 0)
                                .apply(lambda x: x.astype(int)))

    # Running chi2 tests
    chi_square = []
    p_val = []
    dof_list = []

    for o in independent_ordinal_vars:

        tab = pd.crosstab(med_df['relapse'], independent_ordinal_vars[o])
        chi2, p, dof, expected = chi2_contingency(observed = tab,
                                                  correction = False)
        chi_square.append(chi2)
        p_val.append(p)
        dof_list.append(dof)

    # Now let's get these counts in a dataframe that we can append p-values to
    ordinal_counts_by_group = (med_df
                               .dropna(axis = 0)
                               .apply(lambda x: x.astype(int))
                               .groupby('relapse')
                               .sum()
                               .transpose()
                               .reset_index()
                               .rename(columns = {'index':'Medication',
                                                  0:'Rehosp? No',
                                                  1:'Rehosp? Yes'}))

    p_val_round = list(np.round(p_val, decimals = 3))
    ordinal_counts_by_group['P'] = p_val_round

    ordinal_counts_by_group.to_csv('rehosp_medication_summary.csv')


















