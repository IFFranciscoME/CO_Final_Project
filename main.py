
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import functions as fn
import data as dt
import visualizations as vs
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------- Exploratory Data Description -- #
# ------------------------------------------------------------------------- ---------------------------- -- #

# data description
table_1 = dt.ohlc_data.describe()

# general data
data_ohlc = dt.ohlc_data.copy()

# -- ------------------------------------------------------------------------- Train and Test Data Folds -- #
# -- ------------------------------------------------------------------------- ------------------------- -- #

# training data set
train_ohlc = dt.ohlc_data.copy().iloc[0:932, :]

# testing data set
test_ohlc = dt.ohlc_data.copy().iloc[932:-1, :]

# -- ----------------------------------------------------------------- PLOT 2: Time Series Block T-Folds -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

# dates for every fold in order to construct the 2nd plot
dates_folds = [train_ohlc.iloc[-1, 0]]

# Similar to plot_1 but with vertical lines for data division
plot_1 = vs.g_ohlc(p_ohlc=dt.ohlc_data, p_theme=dt.theme_plot_2, p_vlines=dates_folds)

# show plot in explorer
# plot_1.show()

# --------------------------------------------------------------------------------- Feature Engineering -- #
# --------------------------------------------------------------------------------- -------------------- -- #

# ----------------------------------------------------------------- Autoregressive independent varibales -- #
# For the autoregressive feature engineering process
p_memory = 7

# Data with autoregressive variables
data_ar = fn.f_autoregressive_features(p_data=data_ohlc, p_nmax=p_memory)

# Dependent variable (Y) separation
data_y = data_ar['co_d'].copy()

# Timestamp separation
data_timestamp = data_ar['timestamp'].copy()

# Independent variables (x1, x2, ..., xn)
data_ar = data_ar.drop(['timestamp', 'co', 'co_d'], axis=1, inplace=False)

# -- ------------------------------------------------------------------------ Hadamard Product Variables -- #
# Data with Hadamard product variables
data_had = fn.f_hadamard_features(p_data=data_ar, p_nmax=p_memory)

# -- -------------------------------------------------------------------------------- Symbolic Variables -- #

# -- ------------------------------------------------------------------------------------ Generated Data -- #

# # Symbolic features generator
# fun_sym = fn.symbolic_features(p_x=data_had, p_y=data_y)
#
# # variables
# data_sym = fun_sym['data']
# data_sym.columns = ['sym_' + str(i) for i in range(0, len(fun_sym['data'].iloc[0, :]))]
#
# # symbolic expressions (equations) for the generated variables
# eq_sym = [i.__str__() for i in list(fun_sym['model'])]
#
# # save founded symbolic features
# dt.data_save_load(p_data_objects={'features': data_sym, 'equations': eq_sym},
#                   p_data_action='save', p_data_file='files/oc_symbolic_features.dat')

# -- -------------------------------------------------------------------- Load Previously Generated Data -- #
# Load previously generated variables (for reproducibility purposes)
data_sym = dt.data_save_load(p_data_objects=None, p_data_action='load',
                             p_data_file='oc_symbolic_features.dat')

# ----------------------------------------------------------------------------------- Data Concatenation -- #
# ----------------------------------------------------------------------------------- ------------------ -- #

# datos para utilizar en la siguiente etapa
data = pd.concat([data_ar.copy(), data_had.copy(), data_sym['features'].copy()], axis=1)

# model data
model_data = dict()

# Whole data separation for train and test
xtrain, xtest, ytrain, ytest = train_test_split(data, data_y, test_size=.2, shuffle=False)

# Data vision inside the dictionary
model_data['train_x'] = xtrain
model_data['train_y'] = ytrain
model_data['test_x'] = xtest
model_data['test_y'] = ytest

# ------------------------------------------------------------------------------------ Predictive Models -- #
# ------------------------------------------------------------------------------------ ----------------- -- #

# -- --------------------------------------------------------------------------------------- Elastic Net -- #
en_parameters = {'alpha': 0.1, 'ratio': .9}
elastic_net = fn.ols_elastic_net(p_data=model_data, p_params=en_parameters)

# Model accuracy (in of sample)
in_en_acc = elastic_net['metrics']['train']['acc']
print(in_en_acc)

# Model accuracy (out of sample)
out_en_acc = elastic_net['metrics']['test']['acc']
print(out_en_acc)

# -- --------------------------------------------------------------------------- Support Vector Machines -- #
svm_parameters = {'kernel': 'linear', 'gamma': 'auto', 'c': 0.1}
ls_svm = fn.ls_svm(p_data=model_data, p_params=svm_parameters)

# Model accuracy (in sample)
in_svm_acc = ls_svm['metrics']['train']['acc']
print(in_svm_acc)

# Model accuracy (out of sample)
out_svm_acc = ls_svm['metrics']['test']['acc']
print(out_svm_acc)
