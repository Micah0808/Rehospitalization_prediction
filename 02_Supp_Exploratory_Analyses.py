#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""

Chi square and t-tests for the final predictors from the
re-hospitilization models

"""

__author__ = 'Micah Cearns'
__contact__ = 'micahcearns@gmail.com'
__date__ = 'Jan 2019'

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, levene, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.eval_measures import aic
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import os

# Setting working directory
os.chdir('/Users/MicahJackson/anaconda/pycharm_wd/hospitilization_pred')


def logit_results_to_dataframe(results):

    """
    Take the result of an statsmodel results table and transforms it
    into a dataframe

    :param results: Statsmodels summary object

    :return: Pandas dataframe object

    """

    ''''''

    pvals = results.pvalues
    coef = results.params
    conf = results.conf_int()
    conf['OR'] = coef
    conf.columns = ['2.5%', '97.5%', 'OR']
    OR = np.exp(conf)
    results_df = pd.DataFrame({'pvals':pvals, 'coef':coef})
    results_df = results_df.join(OR)

    # Re-ordering
    results_df = results_df[['coef',
                             'pvals',
                             'OR',
                             '2.5%',
                             '97.5%']]

    return results_df

if __name__ == '__main__':

    # =========================================================================
    # Testing for significance between ML models
    # =========================================================================
    # Outer CV loop scores from the repeated nested cross-validation
    multi = [0.68, 0.78, 0.63, 0.73, 0.45, 0.82, 0.77, 0.82, 0.61, 0.46]
    clin = [0.64, 0.61, 0.6, 0.7, 0.42, 0.74, 0.77, 0.74, 0.56, 0.5]
    bio = [0.61, 0.63, 0.59, 0.75, 0.6, 0.59, 0.3, 0.6, 0.56, 0.49]
    serum = [0.58, 0.58, 0.71, 0.44, 0.45, 0.6, 0.5, 0.56, 0.46, 0.56]
    smri = [0.47, 0.63, 0.7, 0.58, 0.59, 0.51, 0.37, 0.67, 0.49, 0.67]
    cardio = [0.67, 0.68, 0.36, 0.74, 0.45, 0.57, 0.56, 0.63, 0.59, 0.36]
    pgrs = [0.42, 0.73, 0.44, 0.36, 0.51, 0.5, 0.34, 0.48, 0.74, 0.53]

    # Testing for an omnibus effect between models in the outer cv loop
    kruskal(multi, clin, bio, serum, smri, cardio, pgrs)

    # Testing for pairwise differences
    model_names = ['Clin', 'Bio', 'Serum', 'sMRI', 'Cardio', 'PGRS']
    model_score_lists = [clin, bio, serum, smri, cardio, pgrs]
    for name, model in zip(model_names, model_score_lists):
        print('Multi vs', name, np.round(mannwhitneyu(multi, model), 4))
        
    # Multi - Clin: [36.    0.15]
    # Multi - Bio: [24.      0.0267]
    # Multi - Serum: [1.90e+01 0.0104]
    # Multi - sMRI: [26.5     0.0409]
    # Multi - Cardio: [24.5     0.0292]
    # Multi - PGRS: [1.95e+01 0.0116]

    # Correcting for multiple comparisons
    # Non-corrected p-values
    raw_p = [0.15, 0.0267, 0.0104, 0.0409, 0.0292, 0.0116]
    # Correcting with the Benjamin and Hochberg method
    multipletests(pvals = raw_p, alpha = 0.05, method = 'fdr_bh')
    # Corrected p-values
    cor_p = [0.15, 0.0438, 0.0348, 0.04908, 0.0438, 0.0348]

    # =========================================================================
    # Logistic regression
    # =========================================================================
    df = pd.read_pickle(path = 'multi_df_cleaned.pkl')
    # Selecting variables for logit regression from repeated nested cv analysis
    logit_df = df[['s0_cesd5',
                   's0_med_n05ah',
                   's0_psqi7',
                   's0_cesd3',
                   's0_med_h03',
                   's0_al_f1',
                   's0_pi_n_epi_inpatient',
                   's0_psqi_sum',
                   'Rhippo',
                   'ICV',
                   's0_cholesterol',
                   's0_sex',
                   's0_alter1',
                   's0_sm_f1',
                   's0_bia_bmi',
                   'relapse']]

    # Translating from German to English
    logit_df = logit_df.rename(
        columns = {'s0_cesd5':'CESD_5_trouble_concentrating',
                   's0_med_n05ah':'Benzodiazepines',
                   's0_psqi7':'PSQI_7_taken_sleeping_pills',
                   's0_cesd3':'CES_D_3_not_get_rid_of_mood',
                   's0_med_h03':'Thyroid_medication',
                   's0_al_f1':'Drink_alcohol_more_than_once_a_week',
                   's0_pi_n_epi_inpatient':'N_previous_admissions',
                   's0_psqi_sum':'PSQI_sleep_quality_index_global_score',
                   'Rhippo':'Mean_right_hippocampal_volume',
                   'ICV':'Intracranial_volume',
                   's0_cholesterol':'Cholesterol',
                   's0_sex':'Sex',
                   's0_alter1':'Age',
                   's0_bia_bmi':'BMI',
                   's0_sm_f1':'Smoke'}
    )

    # Creating a medication index to control for in the analysis
    medication_confounds = df[
        ['s0_med_c07',
         's0_med_n02cc',
         's0_med_n05ab',
         's0_med_n05ad',
         's0_med_n05af',
         's0_med_n05ah',
         's0_med_n05al',
         's0_med_n05an',
         's0_med_n05ax',
         's0_med_n05ba',
         's0_med_n05cf',
         's0_med_n05ch',
         's0_med_n05cm',
         's0_med_n06ab',
         's0_med_n06af',
         's0_med_n06ax',
         's0_med_n06ba']
    ]
    
    medication_confounds = medication_confounds.apply(lambda x: x.astype(int))
    medication_confounds['med_index'] = (
            medication_confounds['s0_med_c07']
          + medication_confounds['s0_med_n02cc']
          + medication_confounds['s0_med_n05ab']
          + medication_confounds['s0_med_n05ad']
          + medication_confounds['s0_med_n05af']
          + medication_confounds['s0_med_n05ah']
          + medication_confounds['s0_med_n05al']
          + medication_confounds['s0_med_n05an']
          + medication_confounds['s0_med_n05ax']
          + medication_confounds['s0_med_n05ba']
          + medication_confounds['s0_med_n05cf']
          + medication_confounds['s0_med_n05ch']
          + medication_confounds['s0_med_n05cm']
          + medication_confounds['s0_med_n06ab']
          + medication_confounds['s0_med_n06af']
          + medication_confounds['s0_med_n06ax']
          + medication_confounds['s0_med_n06ba']
    )

    # Predictors
    cols = logit_df.columns
    # Adding to the main df for analysis
    logit_df['med_index'] = medication_confounds['med_index']
    logit_df['relapse'] = logit_df['relapse'].astype(int)

    # Preparing the data for the model
    X = logit_df.drop('relapse', axis = 1)
    y = logit_df['relapse']
    X = X.apply(lambda x: x.astype(float))
    X = (X - X.mean()) / X.std()  # Standardising due to multicollinearity
    X['relapse'] = y  # Adding back in the dep var
    logit_df = X  # Naming back to df now after standardising the ind vars
    logit_df = logit_df.dropna(axis = 0)  # Drop missings

    # Logistic regression model
    model = smf.logit(formula = 'relapse ~ C(CESD_5_trouble_concentrating)'
                                '+ C(Benzodiazepines)'
                                '+ C(PSQI_7_taken_sleeping_pills)'
                                '+ C(CES_D_3_not_get_rid_of_mood)'
                                '+ C(Thyroid_medication)'
                                '+ C(Drink_alcohol_more_than_once_a_week)'
                                '+ N_previous_admissions'
                                '+ PSQI_sleep_quality_index_global_score'
                                '+ Mean_right_hippocampal_volume'
                                '+ Intracranial_volume'
                                '+ Cholesterol'
                                '+ C(Sex)'
                                '+ Age'
                                '+ C(Smoke)'
                                '+ BMI'
                                '+ med_index',
                      data = logit_df)

    result = model.fit()
    main_effect_aic = aic(llf = -152.36, nobs = 322, df_modelwc = 22)
    print(result.summary())
    print(np.exp(result.params))
    print(np.exp(result.conf_int()))
    print(main_effect_aic)

    # Getting a dataframe of results
    results_df = logit_results_to_dataframe(results = result)
    logit_p_vals = result.pvalues  # Adding in corrected p values
    corrected_log_p_vals = multipletests(pvals = logit_p_vals,
                                         alpha = 0.05,
                                         method = 'fdr_bh')
    corrected_log_p_vals = list(corrected_log_p_vals[1])
    results_df['fdr_pvals'] = corrected_log_p_vals

    # Re-ordering the output and exporting
    results_df = results_df[['coef',
                             'pvals',
                             'fdr_pvals',
                             'OR',
                             '2.5%',
                             '97.5%']]

    results_df = results_df.apply(lambda x: x.round(decimals = 4))
    results_df.to_csv('multi_log_reg_results.csv')

    # =========================================================================
    # First, does the well-established trend of smaller hippocampal volumes
    # hold up in the full dataset?
    # =========================================================================
    full_cohort_df = pd.read_csv('full_cohort_with_imaging.csv')
    mdd_cohort = full_cohort_df.loc[full_cohort_df['s0_l_kohorte'] == 1]
    no_mdd_cohort = full_cohort_df.loc[full_cohort_df['s0_l_kohorte'] == 3]
    print(np.round(mdd_cohort['Rhippo'].mean(), decimals = 2))    # Vol 4033.07
    print(np.round(no_mdd_cohort['Rhippo'].mean(), decimals = 2)) # Vol 4066.39

    full_cohort_df['s0_l_kohorte'] = (full_cohort_df['s0_l_kohorte']
                                      .replace({1:1, 3:0}))

    full_cohort_df = full_cohort_df[['Rhippo',
                                     'ICV',
                                     's0_sex',
                                     's0_age',
                                     's0_l_kohorte']]

    # Standardising the independent vars
    full_cohort_df = full_cohort_df.dropna(axis = 0)
    X = full_cohort_df.drop('Rhippo', axis = 1)
    y = full_cohort_df['Rhippo']
    X = (X - X.mean()) / X.std()
    full_cohort_df = X
    full_cohort_df['Rhippo'] = y

    # Model controlling for basic covariates
    ols_hipp_model = smf.ols(formula = 'Rhippo ~ '
                                       'C(s0_l_kohorte) '
                                       '+ s0_sex'
                                       '+ s0_age'
                                       '+ ICV',
                             data = full_cohort_df).fit()

    print(ols_hipp_model.summary())

    # Running independent samples t-tests
    t_test_df = full_cohort_df.set_index('s0_l_kohorte')
    t_test_df = t_test_df[['Rhippo']]
    t_test_df = t_test_df.dropna(axis = 0)

    levene(t_test_df['Rhippo'].loc[True],
           t_test_df['Rhippo'].loc[False],
           center = 'median')

    t_test_results = ttest_ind(a = t_test_df['Rhippo'].loc[True],
                               b = t_test_df['Rhippo'].loc[False],
                               equal_var = False,
                               nan_policy = 'omit')

    Ttest_indResult(statistic = -1.2055968862752673,
                    pvalue = 0.22825820500830207)

    # CONCLUSION: The effect is in the right direction but not significant.

    # UNIVARIATE MODELS
    # Can increases in hippocampal volume be explained by severity at baseline,
    # age, sex, medication use, previous admissions in univariate models?
    meds = list(df.loc[:, 's0_med_a02':'s0_med_v06'].columns)
    outcomes = ['relapse', 'Rhippo']
    final_preds = meds + outcomes
    uni_df = df[final_preds]
    uni_df = uni_df.apply(lambda x: x.astype(float))
    uni_df = (uni_df - uni_df.mean()) / uni_df.std()
    corr = uni_df.corr()
    hippo_corr = (pd.DataFrame(corr['Rhippo']
                  .drop('Rhippo'))
                  .sort_values(by = 'Rhippo',
                               ascending = False))

    # Interestingly, benzos are associated with an increase in hippo volumes
    hipp_model = df[['s0_hamd_17total',
                     's0_sex',
                     's0_alter1',
                     's0_pi_n_epi_inpatient',
                     's0_testosteron',
                     's0_andrix',
                     's0_bia_grundumsatz',
                     's0_height',
                     's0_ft3',
                     's0_med_n05cf', # Benzos
                     's0_med_n05ch', # Melatonin receptor agonists
                     's0_med_d07', # Corticosteroids (dermatological)
                     's0_med_h02', # Corticosteroids for systemic use
                     'anxiety_pgrs_score',
                     'relapse',
                     'Rhippo',
                     's0_med_n06ab',
                     's0_med_n06aa',
                     's0_med_c07',
                     's0_med_n06ax']]

    hipp_model = hipp_model.apply(lambda x: x.astype(float))
    hipp_model['anti_dep_load'] = (hipp_model['s0_med_n06ab']
                                   + hipp_model['s0_med_n06aa']
                                   + hipp_model['s0_med_c07']
                                   + hipp_model['s0_med_n06ax'])

    hipp_model = hipp_model.dropna(axis = 0)
    hipp_model = hipp_model.rename(
        columns = {'Rhippo':'Right_hippo_vol',
                   's0_hamd_17total':'HAMD_17',
                   's0_sex':'Sex',
                   's0_alter1':'Age',
                   's0_testosteron':'Testosteron',
                   's0_andrix':'Free_testosterone',
                   's0_bia_grundumsatz':'Basic_metabolic_rate',
                   's0_height':'Height',
                   's0_med_n05cf':'Benzodiazepines'}
    )

    hipp_model['Sex'] = hipp_model['Sex'].astype(int)
    X = hipp_model.drop('Right_hippo_vol', axis = 1)
    y = hipp_model['Right_hippo_vol']
    X = (X - X.mean()) / X.std()
    hipp_model = X
    hipp_model['Right_hippo_vol'] = y
    
    # Benzo on Rhippo model by Sex
    ols_hipp_model = smf.ols(formula = 'Right_hippo_vol ~ '
                                       'C(Benzodiazepines) * C(Sex)'
                                       '+ Age',
                             data = hipp_model).fit()
    print(ols_hipp_model.summary())

    # Outputting as a latex file
    beginningtex = """\\documentclass{report}
    \\usepackage{booktabs}
    \\begin{document}"""
    endtex = "\end{document}"

    f = open('benzo_hippo.tex', 'w')
    f.write(beginningtex)
    f.write(ols_hipp_model.summary().as_latex())
    f.write(endtex)
    f.close()

    # =========================================================================
    # Visualising r hippo stratified by gender and medication use
    # =========================================================================
    wom_benz = df.loc[(df['s0_sex'] == 2) & (df['s0_med_n05cf'] == 1)]
    wom_benz = wom_benz['Rhippo']
    wom_benz = wom_benz.dropna(axis = 0)
    # wom_benz:
    # count
    # 38.000000
    # mean
    # 4017.897368
    # std
    # 397.030160
    # min
    # 3293.500000
    # 25 % 3707.575000
    # 50 % 4049.550000
    # 75 % 4242.375000
    # max
    # 4791.400000

    wom_no_benz = df.loc[(df['s0_sex'] == 2) & (df['s0_med_n05cf'] == 0)]
    wom_no_benz = wom_no_benz['Rhippo']
    wom_no_benz = wom_no_benz.dropna(axis = 0)
    # wom_no_benz:
    # count
    # 189.000000
    # mean
    # 3846.580423
    # std
    # 336.511635
    # min
    # 2789.900000
    # 25 % 3616.500000
    # 50 % 3842.500000
    # 75 % 4075.700000
    # max
    # 4673.500000

    men_benz = df.loc[(df['s0_sex'] == 1) & (df['s0_med_n05cf'] == 1)]
    men_benz = men_benz['Rhippo']
    men_benz = men_benz.dropna(axis = 0)
    # men_benz
    # count
    # 28.000000
    # mean
    # 4257.650000
    # std
    # 376.356933
    # min
    # 3497.500000
    # 25 % 3981.925000
    # 50 % 4243.350000
    # 75 % 4411.825000
    # max
    # 5108.300000

    men_no_benz = df.loc[(df['s0_sex'] == 1)
                       & (df['s0_med_n05cf'] == 0)
                       & (df['Rhippo'] >= 3000.0)]
    men_no_benz = men_no_benz['Rhippo']
    men_no_benz = men_no_benz.dropna(axis = 0)
    # men_no_benz
    # count
    # 122.000000
    # mean
    # 4330.281967
    # std
    # 393.932262
    # min
    # 3399.100000
    # 25 % 4043.375000
    # 50 % 4301.250000
    # 75 % 4610.525000
    # max
    # 5518.900000
    
    # Plotting out the distributions
    # Women
    sns.distplot(a=wom_benz,
                 bins=25,
                 hist=True,
                 kde=True,
                 rug=False,
                 fit=None,
                 hist_kws=None,
                 kde_kws=None,
                 rug_kws=None,
                 fit_kws=None,
                 vertical=False,
                 norm_hist=False,
                 axlabel=None,
                 label='Female: Benzo = True',
                 ax=None)

    sns.distplot(a=wom_no_benz,
                 bins=25,
                 hist=True,
                 kde=True,
                 rug=False,
                 fit=None,
                 hist_kws=None,
                 kde_kws=None,
                 rug_kws=None,
                 fit_kws=None,
                 vertical=False,
                 norm_hist=False,
                 axlabel=None,
                 label='Female: Benzo = False',
                 ax=None)

    # Title, label, legend
    plt.title("""Female right hippocampal volume stratified by 
                 benzodiazepine use""",
              fontdict = {'fontsize': 22})
    plt.xlabel('Right hippocampal volume', fontdict = {'fontsize':16})
    plt.legend(loc = 'upper right')
    plt.show()

    # Men
    sns.distplot(a=men_benz,
                 bins=25,
                 hist=True,
                 kde=True,
                 rug=False,
                 fit=None,
                 hist_kws=None,
                 kde_kws=None,
                 rug_kws=None,
                 fit_kws=None,
                 vertical=False,
                 norm_hist=False,
                 axlabel=None,
                 label='Male: Benzo = True',
                 ax=None)

    sns.distplot(a=men_no_benz,
                 bins=25,
                 hist=True,
                 kde=True,
                 rug=False,
                 fit=None,
                 hist_kws=None,
                 kde_kws=None,
                 rug_kws=None,
                 fit_kws=None,
                 vertical=False,
                 norm_hist=False,
                 axlabel=None,
                 label='Male: Benzo = False',
                 ax=None)

    # Title, labels, legends
    plt.title("""Male right hippocampal volume stratified by 
                 benzodiazepine use""",
              fontdict = {'fontsize': 22})
    plt.xlabel('Right hippocampal volume', fontdict = {'fontsize':16})
    plt.legend(loc = 'upper right')
    plt.show()

    # There is a significant gender and benzo effect on hippocampal volume.
    # Let's look at relapse proportions.

    # 44.7% of women on benzos are re-hospitalized
    female_benzo = df.loc[(df['s0_sex'] == 2) & (df['s0_med_n05cf'] == True)]
    # 22% of women NOT on benzos are hospitalized
    female_no_benzo = df.loc[(df['s0_sex'] == 2) & (df['s0_med_n05cf'] == False)]

    # Now in a logit model - not significant
    hipp_model['relapse'] = hipp_model['relapse'].astype(int)
    benz_relapse = smf.logit(formula = 'relapse ~ C(Benzodiazepines) * C(Sex)',
                             data = hipp_model).fit()
    benz_result = benz_relapse.summary()
    print(benz_result)

    # First, what are the overall gender differences?
    men_hippo = df.loc[(df['s0_sex']) == 1]
    print(np.round(men_hippo['Rhippo'].mean(), decimals = 2))
    print(np.round(men_hippo['Rhippo'].std(), decimals = 2))
    
    wom_hippo = df.loc[(df['s0_sex']) == 2]
    print(np.round(wom_hippo['Rhippo'].mean(), decimals = 2))
    print(np.round(wom_hippo['Rhippo'].std(), decimals = 2))

    ols_anti_model = smf.ols(formula = 'Rhippo ~ '
                                       '+ s0_sex'
                                       '+ s0_alter1'
                                       '+ ICV',
                             data = df).fit()

    print(ols_anti_model.summary())

    # Not let's look at the differences in r-hippo between men who relapse
    # and those who do not, vs women who relapse and those who do not.
    men_relapse = df.loc[(df['s0_sex'] == 1) & (df['relapse'] == True)]
    men_no_relapse = df.loc[(df['s0_sex'] == 1) & (df['relapse'] == False)]
    print(np.round(men_relapse['Rhippo'].mean(), decimals = 2))
    print(np.round(men_no_relapse['Rhippo'].mean(), decimals = 2))
    # 4397.93
    # 4244.42

    wom_relapse = df.loc[(df['s0_sex'] == 2) & (df['relapse'] == True)]
    wom_no_relapse = df.loc[(df['s0_sex'] == 2) & (df['relapse'] == False)]
    print(np.round(wom_relapse['Rhippo'].mean(), decimals = 2))
    print(np.round(wom_no_relapse['Rhippo'].mean(), decimals = 2))
    # 3954.67
    # 3847.37

    # Regardless of gender, all who relapse have higher r-hippo
    # It's also obvious that the r-hippo of men is much larger than that of
    # women. Thus, increases in R-hippo may be a proxy for men relapsing,
    # as well as the gender * benzo relationship

    # What are the proportions of mean and women in the whole dataset?
    df['s0_sex'].value_counts()
    # 151 men, 39.7%
    # 229 women, 60.3%

    # How many of each sex are re-hospitalized?
    # 43 men out of 102 who relapse. That is 42.2% of those who relapse are men.
    men_relapse = df.loc[(df['s0_sex'] == 1) & (df['relapse'] == True)]

    # These men have a mean r-hippo volume of 4398.0
    np.round(men_relapse['Rhippo'].mean())

    # 59 out of 102 who relapse. That is 57.8% of those who relapse are women.
    wom_relapse = df.loc[(df['s0_sex'] == 2) & (df['relapse'] == True)]

    # These women have a mean r-hippo of 3955
    np.round(wom_relapse['Rhippo'].mean())

    # Some anti-psychotic meds have been shown to be positively associated
    # with increases in hippocampal volume. Let's explore if there is a
    # relationship there.
    df['anti_psychotic_index'] = np.where((df['s0_med_n05ab'] == 1)
                                           | (df['s0_med_n05ad'] == 1)
                                           | (df['s0_med_n05af'] == 1)
                                           | (df['s0_med_n05ah'] == 1)
                                           | (df['s0_med_n05al'] == 1), 1, 0)
    
    anti_psych_yes = df.loc[df['anti_psychotic_index'] == 1]
    print(np.round(anti_psych_yes['Rhippo'].mean(), decimals = 2))
    print(np.round(anti_psych_yes['Rhippo'].std(), decimals = 2))

    anti_psych_no = df.loc[df['anti_psychotic_index'] == 0]
    print(np.round(anti_psych_no['Rhippo'].mean(), decimals = 2))
    print(np.round(anti_psych_no['Rhippo'].std(), decimals = 2))


    ols_anti_model = smf.ols(formula = 'Rhippo ~ '
                                       'C(anti_psychotic_index)'
                                       '+ s0_sex'
                                       '+ s0_alter1',
                             data = df).fit()

    print(ols_anti_model.summary())

    # =========================================================================
    # What if we remove those on antipsychotics and benzodiazepines? Does this
    # change hippocampal volumes?
    # =========================================================================
    df['anti_psychotic_index'] = np.where((df['s0_med_n05ab'] == 1)
                                     | (df['s0_med_n05ad'] == 1)
                                     | (df['s0_med_n05af'] == 1)
                                     | (df['s0_med_n05ah'] == 1)
                                     | (df['s0_med_n05al'] == 1), 1, 0)

    # Men
    men_cases = df.loc[(df['relapse'] == True) & (df['s0_sex'] == 1.0)]
    men_control = df.loc[(df['relapse'] == False) & (df['s0_sex'] == 1.0)]

    # Small difference when just looking at the full sample of men.
    print(np.round(men_cases['Rhippo'].describe(), decimals = 2))
    # count      43.00
    # mean     4397.93
    # std       380.40
    # min      3626.80
    # 25%      4140.85
    # 50%      4362.70
    # 75%      4693.35
    # max      5518.90

    print(np.round(men_control['Rhippo'].describe(), decimals = 2))
    # count     108.00
    # mean     4244.42
    # std       567.25
    # min         0.00
    # 25%      3975.62
    # 50%      4268.20
    # 75%      4562.15
    # max      5274.60

    # Now let's drop those on anti_psychotics and benzodiazepines
    clean_sample = df
    clean_sample = clean_sample.loc[(df['s0_med_n05cf'] == 0)
                                  & (df['anti_psychotic_index'] == 0)]

    men_clean_cases = df.loc[(clean_sample['relapse'] == True)
                                   & (df['s0_sex'] == 1.0)]

    men_clean_control = df.loc[(clean_sample['relapse'] == False)
                                     & (df['s0_sex'] == 1.0)]

    # Becomes large when we drop those on benzos and antipsychotics.
    print(np.round(men_clean_cases['Rhippo'].describe(), decimals = 2))
    # count      13.00
    # mean     4684.56
    # std       330.72
    # min      4132.10
    # 25%      4476.60
    # 50%      4706.80
    # 75%      4820.60
    # max      5518.90

    print(np.round(men_clean_control['Rhippo'].describe(), decimals = 2))
    # count      62.00
    # mean     4228.11
    # std       677.83
    # min         0.00
    # 25%      3933.12
    # 50%      4286.95
    # 75%      4562.98
    # max      5274.60

    # Women
    # Now let's look at women
    wom_sample = df
    wom_cases = wom_sample.loc[(wom_sample['relapse'] == True)
                               & (df['s0_sex'] == 2.0)]

    wom_control = wom_sample.loc[(wom_sample['relapse'] == False)
                                 & (df['s0_sex'] == 2.0)]

    # Small difference when just looking at the full sample of women.
    print(np.round(wom_cases['Rhippo'].describe(), decimals = 2))
    # count      59.00
    # mean     3954.67
    # std       364.93
    # min      3264.20
    # 25%      3699.50
    # 50%      3893.20
    # 75%      4204.80
    # max      4791.40

    print(np.round(wom_control['Rhippo'].describe(), decimals = 2))
    # count     168.00
    # mean     3847.37
    # std       344.53
    # min      2789.90
    # 25%      3605.08
    # 50%      3840.20
    # 75%      4092.60
    # max      4673.50

    wom_clean_sample = wom_sample.loc[(df['s0_med_n05cf'] == 0) 
                                      & (df['anti_psychotic_index'] == 0)]

    wom_clean_cases = wom_clean_sample.loc[(wom_clean_sample['relapse'] == True) 
                                           & (wom_clean_sample['s0_sex'] == 2.0)]

    wom_clean_control = (
        wom_clean_sample.loc[(wom_clean_sample['relapse'] == False) 
                             & (wom_clean_sample['s0_sex'] == 2.0)]
    )

    # The opposite effect compared to what happened with men, when we remove
    # those on antipsychotics and benzodiazepines, the effect of larger
    # hippocampal volume in relapsing women mostly disappears.
    print(np.round(wom_clean_cases['Rhippo'].describe(), decimals = 2))
    # count      20.00
    # mean     3870.46
    # std       285.73
    # min      3297.30
    # 25%      3696.90
    # 50%      3843.25
    # 75%      4029.95
    # max      4513.80

    print(np.round(wom_clean_control['Rhippo'].describe(), decimals = 2))
    # count     108.00
    # mean     3828.33
    # std       320.00
    # min      2988.20
    # 25%      3596.45
    # 50%      3818.10
    # 75%      4037.50
    # max      4618.70

    # =========================================================================
    # THYROID MEDICATION
    # =========================================================================
    # Why does thyroid medication use appear to be protective against
    # re-hospitalization?

    # Thyroid medication: s0_med_h03

    # Is it because those who have been diagnosed with a thyroid condition
    # are not taking their medication?

    # These participants have a diagnosis and are currently taking thyroid meds
    thy_df = pd.read_csv('multi_with_thy.csv')
    thy_yes_meds_df = thy_df.loc[(thy_df['s0_dx_thyp'] == True)
                               & (thy_df['s0_med_h03'] == True)]
    thy_yes_meds_df = thy_yes_meds_df[['s0_dx_thyp', 's0_med_h03']]
    print(thy_yes_meds_df)

    # These ones have a diagnosis but are not currently taking their
    # medications
    thy_no_meds_df = thy_df[['s0_dx_thyp',
                             's0_med_h03',
                             's0_tsh',
                             's0_ft3',
                             's0_ft4']]

    thy_no_meds_df = (
        thy_no_meds_df.loc[(thy_no_meds_df['s0_dx_thyp'] == True) 
                           & (thy_no_meds_df['s0_med_h03'] == False)]
    )
    print(thy_no_meds_df)

    # What ranges are their thyroid levels in? Have they recovered? Thus, no
    # meds?
    print(np.round(thy_no_meds_df.describe(), decimals = 2))
    # Healthy reference ranges
    # tsh = 0.4 to 4.0 mU/l
    # t3  = 3.5 to 7.8 pmol/l
    # t4  = 9.0 to 25.0 pmol/l

    # Pretty much all are in the healthy range, thus, it appears they have
    # recovered and do not need to be on thyroid medications. Four have t3
    # levels below 3.5, however, the lowest is only 3.08.
    print(thy_no_meds_df.loc[thy_no_meds_df['s0_tsh'] >= 0.04])

    # Are there any patients without a diagnosis taking them as a
    # poly-therapeutic treatment strategy for their mood disorder?
    thy_poly_df = thy_df[['s0_dx_thyp',
                          's0_med_h03',
                          's0_tsh',
                          's0_ft3',
                          's0_ft4']]

    thy_poly_df = thy_poly_df.loc[(thy_poly_df['s0_dx_thyp'] == False) 
                                  & (thy_poly_df['s0_med_h03'] == True)]
    print(np.round(thy_poly_df.describe(), decimals = 2))
    # There are 5 participants taking thyroid medications who do not have a
    # diagnosis of hyper or hypo-thyroidism.

    # ========================================================================
    # CHOLESTEROL
    # ========================================================================
    # Memication use
    meds = list(df.loc[:, 's0_med_a02':'s0_med_v06'].columns)  # Med list
    outcomes = ['relapse', 's0_cholesterol']  # Outcomes for uni corrs
    final_preds = meds + outcomes
    uni_df = df[final_preds]
    uni_df = uni_df.apply(lambda x: x.astype(float))
    uni_df = (uni_df - uni_df.mean()) / uni_df.std()  # Standardising
    corr = uni_df.corr()
    cholesterol_corr = (pd.DataFrame(corr['s0_cholesterol']
                          .drop('s0_cholesterol'))
                          .sort_values(by = 's0_cholesterol',
                                       ascending = False))

    # Medication top associations with cholesterol

    # s0_med_g03:     0.134555 Sex hormones and modulators of the genital system
    # s0_med_h02:     0.109895 Corticosteroids for systemic use
    # s0_med_n06ax:   0.099468 Other antidepressants
    # s0_med_c01:     0.090558 Cardiac therapy

    # s0_med_r01:   - 0.095073 Nasal preparations
    # s0_med_n06af: - 0.096860 Monoamine oxidase inhibitors, non-selective
    # s0_med_s01:   - 0.113628 Ophthalmologicals
    # s0_med_c10:   - 0.207575 Lipid modifying agents

    chol_df = df[['s0_med_g03',
                  's0_med_h02',
                  's0_med_n06ax',
                  's0_med_c01',
                  's0_med_r01',
                  's0_med_n06af',
                  's0_med_s01',
                  's0_med_c10',
                  's0_ph_hamd11', # Appetite
                  'pgc_mdd_pgrs',
                  's0_sex',
                  's0_alter1',
                  's0_bia_bmi',
                  's0_cholesterol']]

    chol_df = chol_df.dropna(axis = 0)
    X = chol_df.drop('s0_cholesterol', axis = 1)
    y = chol_df['s0_cholesterol']
    X = X.apply(lambda x:x.astype(float))
    X = (X - X.mean()) / X.std()
    chol_model = X
    chol_model['s0_cholesterol'] = y

    chol_ols_model = smf.ols(formula = 's0_cholesterol ~ s0_med_c10'
                                       '+ s0_sex'
                                       '+ s0_alter1'
                                       '+ s0_bia_bmi',
                             data = df).fit()

    chol_med_results = chol_ols_model.summary()
    print(chol_med_results)

    # Sex hormones and modulators of the genital system - Sig pos relationship,
    # no gender interactions
    # Corticosteroids for systemic use - Sig pos relationship, no gender
    # Other antidepressants - Nothing
    # Cardiac therapy - Nothing
    # Nasal preparations - Nothing
    # MAO - Nothing
    # Ophthalmologicals - Sig negative relationship
    # Lipid modifying agents - Highly sig neg relationship after
    # controlling for age and gender

