
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler  # estandarizacion de variables
from gplearn.genetic import SymbolicTransformer                               # variables simbolicas
from sklearn.svm import SVC


# -------------------------------------------------------------------------- Data Scaling/Transformation -- #
# -------------------------------------------------------------------------- --------------------------- -- #

def data_scaler(p_data, p_trans):
    """
    Estandarizar (a cada dato se le resta la media y se divide entre la desviacion estandar) se aplica a
    todas excepto la primera columna del dataframe que se use a la entrada

    Parameters
    ----------
    p_trans: str
        Standard: Para estandarizacion (restar media y dividir entre desviacion estandar)
        Robust: Para estandarizacion robusta (restar mediana y dividir entre rango intercuartilico)

    p_data: pd.DataFrame
        Con datos numericos de entrada

    Returns
    -------
    p_datos: pd.DataFrame
        Con los datos originales estandarizados

    """

    if p_trans == 'Standard':

        # estandarizacion de todas las variables independientes
        lista = p_data[list(p_data.columns[1:])]

        # armar objeto de salida
        p_data[list(p_data.columns[1:])] = StandardScaler().fit_transform(lista)

    elif p_trans == 'Robust':

        # estandarizacion de todas las variables independientes
        lista = p_data[list(p_data.columns[1:])]

        # armar objeto de salida
        p_data[list(p_data.columns[1:])] = RobustScaler().fit_transform(lista)

    elif p_trans == 'Scale':

        # estandarizacion de todas las variables independientes
        lista = p_data[list(p_data.columns[1:])]

        p_data[list(p_data.columns[1:])] = MaxAbsScaler().fit_transform(lista)

    return p_data


# ------------------------------------------------------------------------------ Autoregressive Features -- #
# --------------------------------------------------------------------------------------------------------- #

def f_autoregressive_features(p_data, p_nmax):
    """
    Creacion de variables de naturaleza autoregresiva (resagos, promedios, diferencias)

    Parameters
    ----------
    p_data: pd.DataFrame
        Con columnas OHLCV para construir los features

    p_nmax: int
        Para considerar n calculos de features (resagos y promedios moviles)

    Returns
    -------
    r_features: pd.DataFrame
        Con dataframe de features (timestamp + co + co_d + features)

    """

    # reasignar datos
    data = p_data.copy()

    # pips descontados al cierre
    data['co'] = (data['close'] - data['open']) * 10000

    # pips descontados alcistas
    data['ho'] = (data['high'] - data['open']) * 10000

    # pips descontados bajistas
    data['ol'] = (data['open'] - data['low']) * 10000

    # pips descontados en total (medida de volatilidad)
    data['hl'] = (data['high'] - data['low']) * 10000

    # clase a predecir
    data['co_d'] = [1 if i > 0 else -1 for i in list(data['co'])]

    # ciclo para calcular N features con logica de "Ventanas de tamaño n"
    for n in range(0, p_nmax):

        # rezago n de Open Interest
        data['lag_vol_' + str(n + 1)] = data['volume'].shift(n + 1)

        # rezago n de Open - Low
        data['lag_ol_' + str(n + 1)] = data['ol'].shift(n + 1)

        # rezago n de High - Open
        data['lag_ho_' + str(n + 1)] = data['ho'].shift(n + 1)

        # rezago n de High - Low
        data['lag_hl_' + str(n + 1)] = data['hl'].shift(n + 1)

        # promedio movil de open-high de ventana n
        data['ma_vol_' + str(n + 1)] = data['volume'].rolling(n + 1).mean()

        # promedio movil de open-high de ventana n
        data['ma_ol_' + str(n + 1)] = data['ol'].rolling(n + 1).mean()

        # promedio movil de ventana n
        data['ma_ho_' + str(n + 1)] = data['ho'].rolling(n + 1).mean()

        # promedio movil de ventana n
        data['ma_hl_' + str(n + 1)] = data['hl'].rolling(n + 1).mean()

    # asignar timestamp como index
    data.index = pd.to_datetime(data.index)
    # quitar columnas no necesarias para modelos de ML
    r_features = data.drop(['open', 'high', 'low', 'close', 'hl', 'ol', 'ho', 'volume'], axis=1)
    # borrar columnas donde exista solo NAs
    r_features = r_features.dropna(axis='columns', how='all')
    # borrar renglones donde exista algun NA
    r_features = r_features.dropna(axis='rows')
    # convertir a numeros tipo float las columnas
    r_features.iloc[:, 2:] = r_features.iloc[:, 2:].astype(float)
    # reformatear columna de variable binaria a 0 y 1
    r_features['co_d'] = [0 if i <= 0 else 1 for i in r_features['co_d']]
    # resetear index
    r_features.reset_index(inplace=True, drop=True)

    return r_features


# ------------------------------------------------------------------------------------ Hadamard Features -- #
# --------------------------------------------------------------------------------------------------------- #

def f_hadamard_features(p_data, p_nmax):
    """
    Creacion de variables haciendo un producto hadamard entre todas las variables

    Parameters
    ----------
    p_data: pd.DataFrame
        Con columnas OHLCV para construir los features

    p_nmax: int
        Para considerar n calculos de features (resagos y promedios moviles)

    Returns
    -------
    r_features: pd.DataFrame
        Con dataframe de features con producto hadamard

    """

    # ciclo para crear una combinacion secuencial
    for n in range(p_nmax):

        # lista de features previos
        list_hadamard = ['lag_vol_' + str(n + 1),
                         'lag_ol_' + str(n + 1),
                         'lag_ho_' + str(n + 1),
                         'lag_hl_' + str(n + 1)]

        # producto hadamard con los features previos
        for x in list_hadamard:
            p_data['h_' + x + '_' + 'ma_ol_' + str(n + 1)] = p_data[x] * p_data['ma_ol_' + str(n + 1)]
            p_data['h_' + x + '_' + 'ma_ho_' + str(n + 1)] = p_data[x] * p_data['ma_ho_' + str(n + 1)]
            p_data['h_' + x + '_' + 'ma_hl_' + str(n + 1)] = p_data[x] * p_data['ma_hl_' + str(n + 1)]

    return p_data


# ------------------------------------------------------------------ MODEL: Symbolic Features Generation -- #
# --------------------------------------------------------------------------------------------------------- #

def symbolic_features(p_x, p_y):
    """
    Funcion para crear regresores no lineales

    Parameters
    ----------
    p_x: pd.DataFrame
        with regressors or predictor variables
        p_x = data_features.iloc[0:30, 3:]

    p_y: pd.DataFrame
        with variable to predict
        p_y = data_features.iloc[0:30, 1]

    Returns
    -------
    score_gp: float
        error of prediction

    """

    # funcion de generacion de variables simbolicas
    model = SymbolicTransformer(function_set=["sub", "add", 'inv', 'mul', 'div', 'abs', 'log'],
                                population_size=12000, hall_of_fame=300, n_components=30,
                                generations=4, tournament_size=600,  stopping_criteria=.75,
                                const_range=None, init_method='half and half', init_depth=(4, 20),
                                metric='pearson', parsimony_coefficient=0.001,
                                p_crossover=0.4, p_subtree_mutation=0.3, p_hoist_mutation=0.1,
                                p_point_mutation=0.2, p_point_replace=0.2,
                                verbose=1, random_state=None, n_jobs=-1, feature_names=p_x.columns,
                                warm_start=True)

    # SymbolicTransformer fit
    model_fit = model.fit_transform(p_x, p_y)

    # output data of the model
    data = pd.DataFrame(np.round(model_fit, 6))

    # parameters of the model
    model_params = model.get_params()

    # best programs dataframe
    best_programs = {}
    for p in model._best_programs:
        factor_name = 'sym_' + str(model._best_programs.index(p))
        best_programs[factor_name] = {'raw_fitness': p.raw_fitness_, 'reg_fitness': p.fitness_, 
                                      'expression': str(p), 'depth': p.depth_, 'length': p.length_}

    # formatting, drop duplicates and sort by reg_fitness
    best_programs = pd.DataFrame(best_programs).T
    best_programs = best_programs.drop_duplicates(subset = ['expression'])
    best_programs = best_programs.sort_values(by='reg_fitness', ascending=False)

    # results
    results = {'fit': model_fit, 'params': model_params, 'model': model, 'data': data,
               'best_programs': best_programs, 'details': model.run_details_}


    return results


# -------------------------- MODEL: Multivariate Linear Regression Model with ELASTIC NET regularization -- #
# --------------------------------------------------------------------------------------------------------- #

def ols_elastic_net(p_data, p_params):
    """
    Funcion para ajustar varios modelos lineales
    Parameters
    ----------
    p_data: dict
        Diccionario con datos de entrada como los siguientes:

        p_x: pd.DataFrame
            with regressors or predictor variables
            p_x = data_features.iloc[0:30, 3:]

        p_y: pd.DataFrame
            with variable to predict
            p_y = data_features.iloc[0:30, 1]

    p_params: dict
        Diccionario con parametros de entrada para modelos, como los siguientes

        p_alpha: float
                alpha for the models
                p_alpha = alphas[1e-3]

        p_iter: int
            Number of iterations until stop the model fit process
            p_iter = 1e6

        p_intercept: bool
            Si se incluye o no el intercepto en el ajuste
            p_intercept = True

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

    """

    x_train = p_data['train_x']
    y_train = p_data['train_y']

    x_test = p_data['test_x']
    y_test = p_data['test_y']

    # p_params = {'alpha': .1, 'ratio': 95}

    # Fit model
    en_model = ElasticNet(alpha=p_params['alpha'], l1_ratio=p_params['ratio'],
                          max_iter=200000, fit_intercept=False, 
                          tol=1e-2, warm_start=False, random_state=123)

    # model fit
    en_model.fit(x_train, y_train)

    # fitted train values
    p_y_train = en_model.predict(x_train)
    p_y_train_d = [1 if i > 0 else -1 for i in p_y_train]
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})

    # Confussion matrix
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])
    # Probabilities of class in train data
    probs_train = [0]*len(x_train)

    # Accuracy rate
    acc_train = accuracy_score(list(y_train), p_y_train_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_train, tpr_train, thresholds_train = roc_curve(list(y_train), probs_train, pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_train = roc_auc_score(list(y_train), probs_train)

    # fitted test values
    p_y_test = en_model.predict(x_test)
    p_y_test_d = [1 if i > 0 else -1 for i in p_y_test]
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])
    # Probabilities of class in test data
    probs_test = [0]*len(x_test)

    # Accuracy rate
    acc_test = accuracy_score(list(y_test), p_y_test_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_test, tpr_test, thresholds_test = roc_curve(list(y_test), probs_test, pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_test = roc_auc_score(list(y_test), probs_test)

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': en_model, 'intercept': en_model.intercept_, 'coef': en_model.coef_,
                'metrics': {'train': {'acc': acc_train, 'tpr': tpr_train, 'fpr': fpr_train,
                                      'probs': probs_train, 'auc': auc_train},
                            'test': {'acc': acc_test, 'tpr': tpr_test, 'fpr': fpr_test,
                                     'probs': probs_test, 'auc': auc_test}},
                'params': p_params}

    return r_models


# --------------------------------------------------------- MODEL: Least Squares Support Vector Machines -- #
# --------------------------------------------------------------------------------------------------------- #

def l1_svm(p_data, p_params):
    """
    L1 Support Vector Machines

    Parameters
    ----------
    p_data
    p_params

    Returns
    -------

    References
    ----------
    https://scikit-learn.org/stable/modules/svm.html#

    """

    x_train = p_data['train_x']
    y_train = p_data['train_y']

    x_test = p_data['test_x']
    y_test = p_data['test_y']

    # ------------------------------------------------------------------------------ FUNCTION PARAMETERS -- #
    # model hyperparameters
    # C, kernel, degree (if kernel = poly), gamma (if kernel = {rbf, poly, sigmoid},
    # coef0 (if kernel = {poly, sigmoid})

    # computations parameters
    # shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape,
    # break_ties, random_state

    # model function
    svm_model = SVC(C=p_params['c'], kernel=p_params['kernel'], gamma=p_params['gamma'],

                    degree=p_params['degree'], coef0=p_params['coef0'],
                    shrinking=True, probability=True, tol=1e-5, cache_size=4000,
                    class_weight=None, verbose=False, max_iter=100000, decision_function_shape='ovr',
                    break_ties=False, random_state=None)
    
    # save adjusted model
    model = svm_model

    # model fit
    svm_model.fit(x_train, y_train)

    # fitted train values
    p_y_train_d = svm_model.predict(x_train)
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])
    # Probabilities of class in train data
    probs_train = svm_model.predict_proba(x_train)

    # Accuracy rate
    acc_train = accuracy_score(list(y_train), p_y_train_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_train, tpr_train, thresholds_train = roc_curve(list(y_train), probs_train[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_train = roc_auc_score(list(y_train), probs_train[:, 1])

    # fitted test values
    p_y_test_d = svm_model.predict(x_test)
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])
    # Probabilities of class in test data
    probs_test = svm_model.predict_proba(x_test)

    # Accuracy rate
    acc_test = accuracy_score(list(y_test), p_y_test_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_test, tpr_test, thresholds_test = roc_curve(list(y_test), probs_test[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_test = roc_auc_score(list(y_test), probs_test[:, 1])

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': model,
                'metrics': {'train': {'acc': acc_train, 'tpr': tpr_train, 'fpr': fpr_train,
                                      'probs': probs_train, 'auc': auc_train},
                            'test': {'acc': acc_test, 'tpr': tpr_test, 'fpr': fpr_test,
                                     'probs': probs_test, 'auc': auc_test}},
                'params': p_params}

    return r_models


# --------------------------------------------------------------------------- Divide the data in T-Folds -- #
# --------------------------------------------------------------------------- ----------------------------- #

def t_folds(p_data, p_period):
    """
    Function to separate in T-Folds the data, considering not having filtrations (Month and Quarter)

    Parameters
    ----------
    p_data : pd.DataFrame
        DataFrame with data

    p_period : str
        'month': monthly data division
        'quarter' quarterly data division

    Returns
    -------
    m_data or q_data : 'period_'

    References
    ----------
    https://web.stanford.edu/~hastie/ElemStatLearn/

    """

    # data scaling by standarization
    p_data.iloc[:, 1:] = data_scaler(p_data=p_data.copy(), p_trans='Standard')

    # For monthly separation of the data
    if p_period == 'month':
        # List of months in the dataset
        months = list(set(time.month for time in list(p_data['timestamp'])))
        # List of years in the dataset
        years = list(set(time.year for time in list(p_data['timestamp'])))
        m_data = {}
        # New key for every month_year
        for j in years:
            m_data.update({'m_' + str('0') + str(i) + '_' + str(j) if i <= 9 else str(i) + '_' + str(j):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.month == i) &
                                      (pd.to_datetime(p_data['timestamp']).dt.year == j)]
                           for i in months})
        return m_data

    # For quarterly separation of the data
    elif p_period == 'quarter':
        # List of quarters in the dataset
        quarters = list(set(time.quarter for time in list(p_data['timestamp'])))
        # List of years in the dataset
        years = set(time.year for time in list(p_data['timestamp']))
        q_data = {}
        # New key for every quarter_year
        for y in sorted(list(years)):
            q_data.update({'q_' + str('0') + str(i) + '_' + str(y) if i <= 9 else str(i) + '_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y) &
                                      (pd.to_datetime(p_data['timestamp']).dt.quarter == i)]
                           for i in quarters})
        return q_data

    # For quarterly separation of the data
    elif p_period == 'semester':
        # List of years in the dataset
        years = set(time.year for time in list(p_data['timestamp']))
        s_data = {}
        # New key for every quarter_year
        for y in sorted(list(years)):
            # y = sorted(list(years))[0]
            s_data.update({'s_' + str('0') + str(1) + '_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y) &
                                      ((pd.to_datetime(p_data['timestamp']).dt.quarter == 1) |
                                      (pd.to_datetime(p_data['timestamp']).dt.quarter == 2))]})

            s_data.update({'s_' + str('0') + str(2) + '_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y) &
                                      ((pd.to_datetime(p_data['timestamp']).dt.quarter == 3) |
                                       (pd.to_datetime(p_data['timestamp']).dt.quarter == 4))]})

        return s_data

    # In the case a different label has been receieved
    return 'Error: verify parameters'
