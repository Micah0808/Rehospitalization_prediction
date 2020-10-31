#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""

Visualization of predictors from Bidirect rehospitalization prediction

"""

__author__ = 'Micah Cearns'
__contact__ = 'micahcearns@gmail.com'
__date__ = 'Jan 2019'

# Imputation
from MICE import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Cross validation and hyperparameter optimisation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.svm import SVC

# Model scoring and calibration
from sklearn import metrics
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

# Data cleaning
import pandas as pd
import numpy as np

# Misc
from tempfile import mkdtemp
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.4)
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)


def data_prep(data, y, dropna = False):
    """
    
    Basic data preparation function, prepares X matrix and y vector,
    sets up pseudo random number for cross-validation

    :param data: Dataframe to be partitioned into an X feature matrix and y
                 outcome vector
    :param y: Which feature to convert to a class label vector
    :param dropna: If True, drop observations with missing datas

    :return: X - Feature matrix
             y - Vector of class labels
             rand_state - Pseudo random number for repeatable cross-validation
                          results in later analysis
    """


    rand_state = 10
    df = pd.read_pickle(data)
    
    if dropna is True:
        df.dropna(axis = 0, inplace = True)
    else:
        pass

    X = df.drop(y, axis = 1)
    y = df[y]

    return (X, y, rand_state)

# Distribution plot
def plot_dist(x, y, title, xaxis):
    """
    
    Seaborn distribution plots for continuous variables selected by the
    Elastic Net in the ML analysis
    
    :param x: Independent variable
    :param y: Dependant variable for stratification
    :param title: Plot title
    :param xaxis: X axis label
    
    :return: Seaborn distribution plot
    
    """

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(13, 8))

    # Case plot
    sns.distplot(x[y == True],
                 bins=5,
                 hist=True,
                 kde=True,
                 rug=False,
                 fit=None,
                 hist_kws=None,
                 kde_kws=None,
                 rug_kws=None,
                 fit_kws=None,
                 color='#E17174',
                 vertical=False,
                 norm_hist=False,
                 axlabel=None,
                 label='True',
                 ax=None)

    # Control plot
    sns.distplot(x[y == False],
                 bins=5,
                 hist=True,
                 kde=True,
                 rug=False,
                 fit=None,
                 hist_kws=None,
                 kde_kws=None,
                 rug_kws=None,
                 fit_kws=None,
                 color='#71ACE1',
                 vertical=False,
                 norm_hist=False,
                 axlabel=None,
                 label='False',
                 ax=None)

    plt.title(title, fontdict = {'fontsize': 26})
    plt.xlabel(xaxis)
    plt.legend(loc='upper right')

    return


# Count plot
def plot_counts(x, y, title, order, yaxis, alpha, saturation):
    """
    
    Seaborn count plots for categorical variables selected by the 
    Elastic Net in the ML analysis
    
    :param x: Independent variable
    :param y: Dependant variable for stratification
    :param title: Plot title
    :param order: Order list for categorical variables
    :param yaxis: X axis label
    :param alpha: Opacity of the bars
    :param saturation: Saturation level of the bars
    
    :return: Seaborn distribution plot
    
    
    """

    # Initializing the matplotlib figure
    f, ax = plt.subplots(figsize=(13, 8))
    colors = ['#71ACE1', '#E17174']  # Color palette

    # Counts by case/control status
    sns.countplot(y=x,
                  hue=y,
                  data=df,
                  order=order,
                  palette=colors,
                  alpha=alpha,
                  saturation=saturation)

    # Title / labels
    plt.title(title, fontdict={'fontsize': 26})
    plt.xlabel('Counts')
    plt.ylabel(yaxis)

    return


if __name__ == '__main__':

    os.chdir('/Users/MicahJackson/anaconda/pycharm_wd/hospitilization_pred')
    df = pd.read_pickle(path = 'multi_cleaned_hospital_bidirect_df.pkl')
    df = df.rename(columns = {'relapse': 'Rehospitalized'})

    df = df[['s0_pi_n_epi_inpatient',
             's0_cesd5',
             's0_med_n05ah',
             's0_psqi7',
             's0_cesd3',
             's0_med_h03',
             's0_al_f1',
             's0_psqi_sum',
             'Rhippo',
             's0_cholesterol',
             'Rehospitalized']]

    # PSQI Sum
    df['s0_psqi_sum'].dropna(axis=0, inplace=True)

    plot_dist(x=df.s0_psqi_sum,
              y=df.Rehospitalized,
              title='PSQI sleep quality index (global score)',
              xaxis='Total score')

    # R hippo
    rhippo_df = df.loc[df['Rhippo'] >= 2000.0]
    rhippo_df['Rhippo'].dropna(axis=0, inplace=True)

    plot_dist(x=rhippo_df.Rhippo,
              y=rhippo_df.Rehospitalized,
              title='Mean right hippocampal volume',
              xaxis='Mean volume')

    # Cholesterol
    df['s0_cholesterol'].dropna(axis=0, inplace=True)

    plot_dist(x=df.s0_cholesterol,
              y=df.Rehospitalized,
              title='Cholesterol (mmol/l)',
              xaxis='Cholesterol (mmol/l)')

    # N inpatient episodes
    plot_dist(x=df.s0_pi_n_epi_inpatient,
              y=df.Rehospitalized,
              title='Number of inpatient episodes',
              xaxis='N')

    # CES-D 5
    df['s0_cesd5'] = df['s0_cesd5'].replace({0.0: 'Hardly',
                                             1.0: 'Sometimes',
                                             2.0: 'More often',
                                             3.0: 'Mostly'})

    plot_counts(x='s0_cesd5',
                y='Rehospitalized',
                title='CES-D 5: Last week I had trouble concentrating',
                order=['Hardly', 'Sometimes', 'More often', 'Mostly'],
                yaxis='',
                alpha=0.5,
                saturation=1)

    # Antipsychotics
    plot_counts(x='s0_med_n05ah',
                y='Rehospitalized',
                title='Currently use diazepines, oxazepines, thiazepines or oxepines?',
                order=None,
                yaxis='',
                alpha=0.5,
                saturation=1)

    # PSQI-7
    df['s0_psqi7'] = df['s0_psqi7'].replace({0.0: 'Not at all',
                                             1.0: 'Less than once',
                                             2.0: 'Once or twice',
                                             3.0: 'Three or more'})

    plot_counts(x='s0_psqi7',
                y='Rehospitalized',
                title= """How often during the last 4 weeks have you had 
                difficulties staying awake?""",
                order=['Not at all', 'Less than once', 'Once or twice', 'Three or more'],
                yaxis='',
                alpha=0.5,
                saturation=1)

    # CES-D 3: Could not get rid of my troubling mood
    df['s0_cesd3'] = df['s0_cesd3'].replace({0.0: 'Hardly',
                                             1.0: 'Sometimes',
                                             2.0: 'More often',
                                             3.0: 'Mostly'})

    plot_counts(x='s0_cesd3',
                y='Rehospitalized',
                title='CES-D 3: Could not get rid of my troubling mood',
                order=['Hardly', 'Sometimes', 'More often','Mostly'],
                yaxis='',
                alpha=0.5,
                saturation=1)

    # Alcohol
    df['s0_al_f1'] = df['s0_al_f1'].replace({False:'<= 1 a week',
                                             True:'>= 2 a week'})

    plot_counts(x='s0_al_f1',
                y='Rehospitalized',
                title='How often do you drink an alcoholic beverage?',
                order=[ '<= 1 a week', '>= 2 a week'],
                yaxis='',
                alpha=0.5,
                saturation=1)

    # Outer 10-fold cross-validation loop scores
    f_multi_auc = [0.820843, 
                   0.826107, 
                   0.892954, 
                   0.742999, 
                   0.756549, 
                   0.827010, 
                   0.777326,
                   0.777778, 
                   0.712963, 
                   0.700463]

    multi_auc = [0.787716, 
                 0.816810, 
                 0.815733, 
                 0.630388, 
                 0.686422, 
                 0.838362, 
                 0.654095,
                 0.745690, 
                 0.608454, 
                 0.645161]

    serum_auc = [0.712284, 
                 0.716595, 
                 0.831897, 
                 0.676724, 
                 0.730603, 
                 0.877155, 
                 0.543103,
                 0.724138, 
                 0.579533, 
                 0.692396]

    cardio_auc = [0.725216, 
                  0.728448, 
                  0.789871, 
                  0.667026, 
                  0.614224, 
                  0.581897, 
                  0.691810,
                  0.741379, 
                  0.640712, 
                  0.509217]

    sMRI_auc = [0.532328, 
                0.523707, 
                0.544181, 
                0.494612, 
                0.629310, 
                0.549569, 
                0.647629,
                0.563578, 
                0.611791, 
                0.662442]

    snp_auc = [0.465517, 
               0.549569, 
               0.573276, 
               0.520474, 
               0.302802, 
               0.540948, 
               0.559267,
               0.451509, 
               0.585095, 
               0.550691]

    # Error bar plots for all trained classifiers
    # Setting figsize
    plt.figure(figsize=(13, 9))

    # Plotting
    plt.errorbar(x=1,
                 y=np.array(snp_auc).mean(),
                 yerr=np.array(snp_auc).std(),
                 elinewidth=8,
                 alpha=0.5,
                 fmt='o',
                 capsize=15,
                 capthick=1)

    plt.errorbar(x=2,
                 y=np.array(sMRI_auc).mean(),
                 yerr=np.array(sMRI_auc).std(),
                 elinewidth=8,
                 alpha=0.5,
                 fmt='o',
                 capsize=15,
                 capthick=1)

    plt.errorbar(x=3,
                 y=np.array(cardio_auc).mean(),
                 yerr=np.array(cardio_auc).std(),
                 elinewidth=8,
                 alpha=0.5,
                 fmt='o',
                 capsize=15,
                 capthick=1)

    plt.errorbar(x=4,
                 y=np.array(serum_auc).mean(),
                 yerr=np.array(serum_auc).std(),
                 elinewidth=8,
                 alpha=0.5,
                 fmt='o',
                 capsize=15,
                 capthick=1)

    plt.errorbar(x=5,
                 y=np.array(multi_auc).mean(),
                 yerr=np.array(multi_auc).std(),
                 elinewidth=8,
                 alpha=0.5,
                 fmt='o',
                 capsize=15,
                 capthick=1)

    plt.errorbar(x=6,
                 y=np.array(f_multi_auc).mean(),
                 yerr=np.array(f_multi_auc).std(),
                 elinewidth=8,
                 alpha=0.5,
                 fmt='o',
                 capsize=15,
                 capthick=1)

    # Hiding arbitrary axis ticks
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])

    # Labels
    plt.title("""AUCs and SDs for our C-Multi, Multi, Serum, Cardio,
                 Imaging, and SNP Models""")

    plt.legend(['SNP model',
                'sMRI model',
                'Cardio model',
                'Serum model',
                'Multi model',
                'C-Multi model'])

    plt.ylabel('AUC Prediction Rate')
    plt.xlabel('Modality')
    plt.show()

