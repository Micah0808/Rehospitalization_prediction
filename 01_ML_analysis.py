# !/usr/bin/env python
# _*_ coding: utf-8 _*_


"""

Predicting re-hospitalization within two years of admission for acute
major depression: A multimodal machine learning approach.

"""

__author__ = 'Micah Cearns'
__contact__ = 'micahcearns@gmail.com'
__date__ = 'Jan 2019'

# Misc
from tempfile import mkdtemp
from shutil import rmtree
from time import time
from tqdm import tqdm
import pickle
import os


# Data manipulation
import numpy as np
import pandas as pd

# Imputation
from MICE import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel

# Cross validation and hyperparameter optimisation
from sklearn.model_selection import (cross_validate,
                                     cross_val_predict,
                                     RepeatedStratifiedKFold,
                                     GridSearchCV,
                                     RandomizedSearchCV,
                                     train_test_split)

# Classifiers
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNetCV

# Pipeline architecture
from sklearn.pipeline import Pipeline

# Model metrics
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             brier_score_loss)

# Calibration
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
sns.set_style("whitegrid")

# Setting working directory
os.chdir('/Users/MicahJackson/anaconda/pycharm_wd/hospitilization_pred')


def pandas_config():
    """

    Pandas configuration

    :return: Configured Pandas
    """
    options = {
        'display': {
            'max_columns': None,
            'max_colwidth': 25,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 14,
            'max_seq_items': 50,  # Max length of printed sequence
            'precision': 4,
            'show_dimensions': False},  # Controls SettingWithCopyWarning
        'mode': {
            'chained_assignment': None
            }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)

    return


def data_prep(data, y, dropna=False):
    """

    Basic data preparation function, prepares the feature matrix X and the
    outcome vector y, sets up pseudo random number for cross-validation.

    :param data: Dataframe to be partitioned into an X feature matrix and y
                 outcome vector

    :param y: Which feature to convert to an outcome vector

    :param dropna: If True, drop observations with missing data

    :return: X - Feature matrix
             y - Vector of class labels
             rand_state - Pseudo random number for repeatable cross-validation
                          results in later analysis
    """

    rand_state = 10  # Setting random state for later cv
    df = pd.read_pickle(data)  # Reading in data
    if dropna is True:
        df.dropna(axis = 0, inplace = True)
    else:
        pass
    X = df.drop(y, axis = 1)  # Assigning the feature space to X
    y = df[y]  # Class labels to predict

    return X, y, rand_state


def multi_metric_scorer():
    """

    Scoring dictionary for outer loop multi metric scoring.

    :return: Dictionary of scoring metrics
    """

    scoring = {'AUC': 'roc_auc',
               'Accuracy': 'accuracy',

               'Balanced_accuracy': make_scorer(
                   recall_score,
                   pos_label = None,
                   average = 'macro',
                   sample_weight = None
               ),
               'Sensitivity': make_scorer(
                   recall_score,
                   pos_label = 1,
                   average = 'binary',
                   sample_weight = None
               ),
               'Specificity': make_scorer(
                   recall_score,
                   pos_label = 0,
                   average = 'binary',
                   sample_weight = None
               ),
               'F1': make_scorer(
                   f1_score, average = 'weighted'
               ),
               'PPV': make_scorer(
                   precision_score,
                   pos_label = 1,
                   average = 'binary'
               ),
               'NPV': make_scorer(
                   precision_score, pos_label = 0, average = 'binary'
               ),
               'Brier_score': 'brier_score_loss'}

    return scoring


def pipeline_estimator(X,
                       y,
                       estimator,
                       params,
                       scorer,
                       inner_cv=10,
                       inner_repeats=5,
                       outer_cv=10,
                       metric='roc_auc',
                       selector='enet',
                       grid_search=True,
                       n_iter=60,
                       probas=False,
                       n_jobs=1,
                       random_state=10):
    """

    Pipeline for estimator training and testing. To avoid data leakage,
    pipeline architecture ensures that all transformations are conducted
    within the same cross-validation folds in a nested-scheme.

    :param X: Feature matrix for prediction
    :param y: Vector of outcome labels to learn and predict
    :param estimator: Final model for fitting after transformations
    :param params: Hyperparameter dictionary
    :param scorer: Scoring dictionary for outer loop multi metric scoring
    :param inner_cv: Number of inner loop cross-validation folds
    :param inner_repeats: Number of inner loop cross-validation repeats
    :param outer_cv: Number of outer loop cross-validation folds
    :param metric: Metric to optimize in model training
    :param selector: Feature selection method, Elastic net or f-tests
    :param probas: If true, use cross_val_predict and return a vector of
                   predicted probabilities, if False, run cross_validate
    :param n_jobs: How many cores to run for model fitting
    :param random_state: Pseudo random number for repeatable cross-validation
                         results

    :return: scores - Train and test scores
             tuned_pipe - Fitted and tuned pipeline
             rkfold - rkfold object
             cachedir - cache directory to avoid repeat computation
    """

    # Setting up repeated cross validation
    rkfold = RepeatedStratifiedKFold(
        n_splits = inner_cv,
        n_repeats = inner_repeats,
        random_state = random_state
    )
    # MICE imputation
    imputer = IterativeImputer(
        n_nearest_features = 10,
        min_value = -5000.0,
        max_value = 5000.0,
        random_state = random_state
    )
    # Mean centering the data for the linear svm
    scaler = StandardScaler(with_mean = True, with_std = True)

    # Feature selection
    if selector is 'enet':
        selector = SelectFromModel(
            estimator = ElasticNetCV(
                cv = 5,
                n_alphas = 10,
                alphas = np.arange(0.1, 1.1, 0.1),
                max_iter = -1,
                verbose = False,
                random_state = random_state
            ),
            threshold = -np.inf,
            max_features = 25,
        )
        # selector = SelectFromModel(
        #     estimator = ElasticNet(max_iter = -1,
        #                            random_state = random_state),
        #     threshold = -np.inf,
        #     max_features = 25,
        # )
    elif selector is 'f-test':
        selector = SelectKBest(score_func = f_classif)
    else:
        pass

    # sampler = SMOTE(random_state = random_state)

    # Temp directory to avoid repeat computation
    cachedir = mkdtemp()
    # Pipeline steps
    pipe_params = [
        ('imputer', imputer),
        ('scaler', scaler),
        # ('selector', selector),
        ('clf', estimator)
    ]
    pipe = Pipeline(pipe_params, memory = cachedir)

    if grid_search is True:
        # Establishing a grid search for hyperparameter optimisation
        tuned_pipe = GridSearchCV(
            estimator = pipe,
            cv = rkfold,
            param_grid = params,
            scoring = metric,
            refit = True,
            n_jobs = n_jobs,
            iid = True,
            verbose = False
        )
    elif grid_search is False:
        tuned_pipe = RandomizedSearchCV(
            estimator = pipe,
            cv = rkfold,
            param_distributions = params,
            scoring = metric,
            n_iter = n_iter,
            refit = True,
            n_jobs = n_jobs,
            iid = True,
            verbose = False,
            random_state = random_state
        )
    else:
        raise ValueError('Must specify grid_search = True or False')

    if probas is True:  # Outer cross validation loop
        scores = cross_val_predict(
            estimator = tuned_pipe,
            X = X,
            y = y,
            cv = outer_cv,
            n_jobs = n_jobs,
            method = 'predict_proba'  # Returns probs, not [0,1]
        )
    elif probas is False:
        scores = cross_validate(
            estimator = tuned_pipe,
            X = X,
            y = y,
            scoring = scorer,
            cv = outer_cv,
            return_train_score = True,
            n_jobs = n_jobs
        )
    else:
        raise ValueError('Must specify True or False for probas')

    return scores, tuned_pipe, rkfold, cachedir


def reliability_curve(tuned_pipe, X, y):
    """

    :param tuned_pipe: Grid search instance from pipeline estimator
    :param X: Feature matrix X in the form of a Pandas dataframe
    :param y: Outcome vector y, in the form of a Pandas series

    :return: Reliability curve
    """

    # Calibrated pipeline
    sigmoid = CalibratedClassifierCV(
        base_estimator = tuned_pipe, method = 'sigmoid', cv = 10
    )
    pred_probas = sigmoid.fit(X, y).predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true = y, y_prob = pred_probas, normalize = False, n_bins = 10
    )
    # Brier score loss
    bs = brier_score_loss(
        y_true = y, y_prob = pred_probas, sample_weight = None, pos_label = 1
    )
    fig = plt.figure(1, figsize = (10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label = "Perfectly calibrated")
    ax1.plot(mean_predicted_value,
             fraction_of_positives, "s-", label = "%s (%1.3f)" %
                                                  ('Brier Score', bs))
    ax2.hist(pred_probas,
             range = (0, 1),
             bins = 10,
             label = 'Multimodal SVM + Sigmoid calibration',
             histtype = "step",
             lw = 2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc = "lower right")
    ax1.set_title('Calibration plot  (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc = "upper right", ncol = 2)
    plt.tight_layout()
    plt.show()

    return


def serialize_model(model, X, y):
    """

    Function to serialize the trained model for open sourcing on the Photon
    AI repository

    To load model and predict:
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    Scoring:
    score = pickle_model.score(X_test, y_test)
    print("Test score: {0:.2f} %".format(100 * score))

    Get a prediction array for each target:
    y_pred = pickle_model.predict(X_test)

    :param model: Gridsearch object with nested pipeline
    :param X: Predictor matrix
    :param y: Outcome vector {0,1}

    :return: Serialized model file in pickle format saved to the working
             directory
    """

    # Fitting the model to the full dataset
    model.fit(X, y)
    # Pickling
    pkl_filename = 'rehosp_model.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    return


def train_test_scores(estimator_scores):
    """

    Scoring function for cross-validated pipeline results

    :param estimator_scores: Train and test scores from pipeline estimator

    :return: train_results - Train partition results
             test_results - Test partition results
    """

    # Converting the dictionary of scores from cross_validate to a dataframe
    # and dropping unnecessary rows
    scores_df = (pd
                 .DataFrame
                 .from_dict(estimator_scores)
                 .drop(['fit_time', 'score_time'], axis = 1))
    # Getting mean scores and standard deviations from repeated cv
    scores_mean = np.abs(scores_df.mean() * 100)
    scores_std = np.abs(scores_df.std() * 100)
    # Returning results as pandas dataframe
    results = pd.DataFrame({'Accuracy': scores_mean,
                            'Standard Deviation': scores_std})
    # Sub-setting train and test results into their own dataframes
    train_results = np.round(results.iloc[list(range(1, 19, 2))], decimals = 4)
    test_results = np.round(results.iloc[list(range(0, 18, 2))], decimals = 4)
    # Returning Brier scores back to a value between 0 and 1
    train_results.iloc[8] = (train_results.iloc[8]/100)
    test_results.iloc[8] = (test_results.iloc[8]/100)

    return train_results, test_results, scores_df


def kbest(X, y, select_method, pipeline):
    """

    Elastic net or univariate f-test based feature selection.

    :param X: Feature matrix for prediction
    :param y: Vector of outcome labels
    :param select_method: The elastic net of f-tests
    :param pipeline: Pipeline estimator for fitting with final selected
                     features

    :return: kbest_df - Dataframe of final selected features
             params_df - Dataframe of the selected hyperparameters
             best_inner_cv_test_score - Best grid search score for model
             selection
    """

    pipe = pipeline.fit(X = X, y = y)  # Fit the tuned pipe to the whole dataset
    # if select_method is 'enet':
    #     coefs = (pipe
    #              .best_estimator_
    #              .named_steps['selector']
    #              .estimator_  # This is an attribute from SelectFromModel
    #              .coef_[pipe
    #                     .best_estimator_
    #                     .named_steps['selector']
    #                     .get_support()])  # Integer index of selected feats
    # elif select_method is 'f-test':
    #     coefs = (pipe
    #              .best_estimator_
    #              .named_steps['selector']
    #              .scores_[pipe
    #                       .named_steps['selector']
    #                       .get_support()])
    # else:
    #     raise ValueError("""Must specify feature selection technique
    #                      in select method""")

    # Getting weights from the linear SVM
    svm_weights = pipe.best_estimator_.named_steps['clf'].coef_.ravel()
    # # Getting feature names
    # names = (X
    #          .columns
    #          .values[pipe
    #                  .best_estimator_
    #                  .named_steps['selector']
    #                  .get_support()])
    names = X.columns.values
    names_scores = list(zip(names, svm_weights))  # coefs
    kbest_df = (pd
                .DataFrame(data = names_scores,
                           columns = ['Features', 'Weights'])
                .sort_values(by = 'Weights', ascending = False))
    optimal_params = pipeline.best_params_  # Getting the tuned parameters
    params_df = pd.DataFrame.from_dict(
        data = optimal_params, orient = 'index', columns = ['Parameters']
    )
    best_inner_cv_test_score = pipeline.best_score_  # Best hyper-param score
    # if select_method is 'enet':
    #     # Filtering out zeroed coefficients from the elastic net that were not
    #     # removed in SelectFromModel
    #     kbest_df = kbest_df.loc[(kbest_df['Coefs'] != 0.000000)
    #                             | kbest_df['Coefs'] != -0.000000]
    #     # Final constant regularization value (alpha) used in the elastic net
    #     alpha = (np
    #              .round(pipe
    #                     .best_estimator_
    #                     .named_steps['selector']
    #                     .estimator_
    #                     .coef_,
    #                     decimals = 4))

        # alpha = (np
        #          .round(pipe
        #                 .best_estimator_
        #                 .named_steps['selector']
        #                 .estimator_
        #                 .alpha_,
        #                 decimals = 4))
        # # The grid of alphas used for fitting each l1 ratio
        # alphas = (np
        #           .round(pipe
        #                  .best_estimator_
        #                  .named_steps['selector']
        #                  .estimator_
        #                  .alphas_,
        #                  decimals = 4))

    #     return pipe, kbest_df, params_df, best_inner_cv_test_score, alpha
    # else:
    return pipe, kbest_df, params_df, best_inner_cv_test_score


def manual_perm_test(model: 'Fitted sklearn estimator',
                     X: 'Pandas df',
                     y: 'Pandas series',
                     true_score: float,
                     n_permutations: int=100,
                     plot: bool=True,
                     clf: bool=False) -> 'p-value, null_counts':
    """

    :param model: Fitted sklearn estimator object
    :param X: Feature matrix for prediction
    :param y: Vector of outcome labels
    :param true_score: Final score from the outer cross-validation loop
    :param n_permutations: Number of permutations to build the null distribution
    :param plot: If true, plot the null distribution and fitted estimator score
    :param clf: If True, model param is a classification estimator and not an
                a regression estimator

    :return: p_value - p value for the fitted estimator
             null_counts - The number of times a null score that was greater or
             equal to the fitted estimator score appears in the null
             distribution
    """

    scores = []  # Empty list for null distribution scores
    n_perms = range(1, n_permutations, 1)  # Range of values to permute
    for n in tqdm(n_perms, desc = 'Permutation test'):  # tqdm for progress bar
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify = y, test_size = 0.90, random_state = n
        )
        model.fit(X_train, y_train)
        y_test_perm = np.random.permutation(y_test)  # Permuting class labels
        chance_scores = round(model.score(X = X_test, y = y_test_perm), 4)
        scores.append(chance_scores)

    # Converting to a pandas dataframe
    perm_scores_df = pd.DataFrame(data = scores, columns = ['null_dist'])
    perm_scores_df['null_dist'] *= 100
    null_counts = (
        perm_scores_df  # Counts greater than or equal to our test set score
        .loc[(perm_scores_df['null_dist']) >= true_score]
        .count()
        .iloc[0]
    )
    p_value = (null_counts + 1) / (n_permutations + 1)
    p_value = np.round(p_value, decimals = 5)

    if plot is True:  # Plotting a histogram of permutation scores
        plt.figure(figsize = (10, 10))
        sns.distplot(a = perm_scores_df['null_dist'],
                     hist = True,
                     label = 'Permutation scores')
        ylim = plt.ylim()
        if clf is False:
            # True classifier score and p-value
            plt.plot(2 * [true_score],
                     ylim,
                     '--g',
                     linewidth = 3,
                     label = 'R2 score %s (pvalue : %s)' %
                             (true_score, p_value))
        else:
            plt.plot(2 * [true_score],
                     ylim,
                     '--g',
                     linewidth = 3,
                     label = 'Multimodal AUC score: %s (pvalue = %s)' %
                             (true_score, p_value))
            n_classes = np.unique(y).size
            chance = 2 * [100. / n_classes]
            plt.plot(chance,
                     ylim,
                     '--k',
                     linewidth = 3,
                     label = 'Null model mean AUC score: %s' % 50.00)
        # Plotting distribution and results
        plt.ylim(ylim)
        plt.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.38))
        plt.tight_layout()

        if clf is False:
            plt.xlabel(xlabel = 'R2 Scores')
        else:
            plt.xlabel(xlabel = 'AUC Scores')
        plt.title(label = 'Null Distribution')
        plt.savefig('quadratic_null_dist.png', dpi = 300, bbox_inches = 'tight')
        plt.show()

    return p_value, null_counts


def main():

    print(__doc__)
    os.environ['PYTHONWARNINGS'] = 'ignore'
    start_time = time()
    pandas_config()
    # # Preparing data
    # X, y, rand_state = data_prep(
    #     data = 'multi_cleaned_hospital_bidirect_df.pkl',
    #     y = 'relapse',
    #     dropna = False
    # )

    # Preparing data
    X, y, rand_state = data_prep(
        data = 'subset_multi_df_cleaned_psqi7.pkl',
        y = 'relapse',
    )
    X = X[['pi_n_epi_inpatient',
           'cesd5',
           'psqi_sum',
           'med_n05ah',
           'psqi7',
           'Rhippo',
           'cesd3',
           'cholesterol',
           'med_h03',
           'al_f1']]

    # Linear SVM
    svc = SVC(
        C = 0.001,
        kernel = 'linear',
        class_weight = 'balanced',
        probability = True,  # Implements Platt scaling
        random_state = rand_state
    )
    # Hyperparameter grid
    svc_params = {
        'clf__C': [0.001, 0.01],  # 0.1, 1.0
        # 'selector__estimator__l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
        # 'selector__estimator__alpha': np.arange(0.2, 1.2, 0.2),
    }
    # 'selector__estimator__l1_ratio': [0.1, 0.5, 0.9]}

    # 'selector__max_features': [5, 10, 15]}
    # svc_params = {'clf__C': [0.001, 0.001, 0.1, 1.0]}

    # Setting up multi-metric evaluation
    scorer = multi_metric_scorer()

    # Running pipeline
    scores, tuned_pipe, rkfold, cachedir = pipeline_estimator(
        X = X,
        y = y,
        estimator = svc,
        params = svc_params,
        scorer = scorer,
        inner_cv = 2,
        inner_repeats = 2,
        outer_cv = 2,
        metric = 'roc_auc',
        selector = 'enet',
        grid_search = True,
        n_iter = 2,
        probas = False,
        n_jobs = -2,
        random_state = rand_state
    )
    # reliability_curve(tuned_pipe = tuned_pipe, X = X, y = y)
    # Train / test results
    train_results, test_results, scores_df = train_test_scores(
        estimator_scores = scores
    )
    # Clearing the cache directory
    rmtree(cachedir)
    # # Serializing the model
    # serialize_model(model = tuned_pipe, X = X, y = y)

    # Kbest features from the elastic net
    pipe, top_features, params_df, best_inner_cv_test_score = kbest(
        X = X, y = y, select_method = 'enet', pipeline = tuned_pipe
    )
    # Using a permutation test to test the significance of the classifier
    p_value, null_counts = manual_perm_test(
        model = pipe,
        X = X,
        y = y,
        true_score = 67.74,
        n_permutations = 10000,
        plot = True,
        clf = True
    )
    # Output
    stop_time = time()
    run_time = np.round((stop_time - start_time) / 60, decimals = 2)
    best_inner_cv_test_score = np.round(best_inner_cv_test_score, decimals = 4)
    output = (('Best inner CV score = {score}'
               .format(score = best_inner_cv_test_score)), '',
              'Train results:', train_results, '',
              'Test results:', test_results, '',
              'Features selected by the Elastic Net:', top_features, '',
              'Hyperparameters selected in CV:', params_df, '',
              # 'Final elastic net alpha value:', alpha, '',
              # 'Elastic net alphas searched in CV:', alphas, '',
              'p value:', p_value, '',
              'Null counts:', null_counts, '',
              'Run time = {timer} minutes'.format(timer = run_time))

    return print(*output, sep = '\n')


if __name__ == '__main__':
    print(main())

    
    
    
