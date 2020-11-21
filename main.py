
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
from data import data_train, data_test
import visualizations as vs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------------------- Data Visualization -- #
# ----------------------------------------------------------------------------------- ------------------ -- #

# grafica OHLC
p_theme = {'color_1': '#ABABAB', 'color_2': '#ABABAB', 'color_3': '#ABABAB', 'font_color_1': '#ABABAB',
           'font_size_1': 12, 'font_size_2': 16}
p_dims = {'width': 1450, 'height': 800}
p_vlines = [data_train['timestamp'].head(1), data_train['timestamp'].tail(1)]
p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

# cargar funcion importada desde GIST de visualizaciones
ohlc_train = vs.g_ohlc(p_ohlc=data_train, p_theme=p_theme, p_dims=p_dims,
                       p_vlines=p_vlines, p_labels=p_labels)

# grafica OHLC
p_theme = {'color_1': '#ABABAB', 'color_2': '#ABABAB', 'color_3': '#ABABAB', 'font_color_1': '#ABABAB',
           'font_size_1': 12, 'font_size_2': 16}
p_dims = {'width': 1450, 'height': 800}
p_vlines = [data_test['timestamp'].head(1), data_test['timestamp'].tail(1)]
p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

# cargar funcion importada desde GIST de visualizaciones
ohlc_test = vs.g_ohlc(p_ohlc=data_test, p_theme=p_theme, p_dims=p_dims,
                      p_vlines=p_vlines, p_labels=p_labels)

# ------------------------------------------------------------------------- Exploratory Data Description -- #
# ------------------------------------------------------------------------- ---------------------------- -- #

# tabla de descripcion de datos
data_train.describe()

# tabla de descripcion de datos
data_test.describe()

# --------------------------------------------------------------------------------- Feature Engineering -- #
# --------------------------------------------------------------------------------- -------------------- -- #

# ----------------------------------------------------------------- Autoregressive independent varibales -- #
# For the autoregressive feature engineering process
p_memory = 7

# Data with autoregressive variables
data_ar = fn.f_autoregressive_features(p_data=data_train, p_nmax=p_memory)

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
# Lista de operaciones simbolicas
fun_sym = fn.symbolic_features(p_x=data_had, p_y=data_y)

# variables
data_sym = fun_sym['data']
data_sym.columns = ['sym_' + str(i) for i in range(0, len(fun_sym['data'].iloc[0, :]))]

# ecuaciones de todas las variables
equations = [i.__str__() for i in list(fun_sym['model'])]

# ----------------------------------------------------------------------------------- Data Concatenation -- #
# ----------------------------------------------------------------------------------- ------------------ -- #

# datos para utilizar en la siguiente etapa
data = pd.concat([data_ar.copy(), data_had.copy(), data_sym.copy()], axis=1)

# Whole data separation for train and test
xtrain, xtest, ytrain, ytest = train_test_split(data, data_y, test_size=.2, shuffle=False)
model_data = dict()

# Data vision inside the dictionary
model_data['train_x'] = xtrain
model_data['train_y'] = ytrain
model_data['test_x'] = xtest
model_data['test_y'] = ytest

# ------------------------------------------------------------------------------------ Predictive Models -- #
# ------------------------------------------------------------------------------------ ----------------- -- #

# -- --------------------------------------------------------------------------------------- Elastic Net -- #
en_parameters = {'alpha': 0.7, 'ratio': .1}
elastic_net = fn.ols_elastic_net(p_data=model_data, p_params=en_parameters)

# Model accuracy (in of sample)
in_en_acc = round((elastic_net['results']['matrix']['train'][0, 0] +
                   elastic_net['results']['matrix']['train'][0, 1])/len(ytrain), 4)

print(in_en_acc)

# Model accuracy (out of sample)
out_en_acc = round((elastic_net['results']['matrix']['test'][0, 0] +
                    elastic_net['results']['matrix']['test'][0, 1])/len(ytest), 4)
print(out_en_acc)

# -- --------------------------------------------------------------------------- Support Vector Machines -- #
svm_parameters = {'kernel': 'linear', 'gamma': 'scale', 'C': 0.5}
ls_svm = fn.ls_svm(p_data=model_data, p_params=svm_parameters)

# Model accuracy (in sample)
in_svm_acc = round((ls_svm['results']['matrix']['train'][0, 0] +
                    ls_svm['results']['matrix']['train'][0, 1])/len(ytrain), 4)

print(in_svm_acc)

# Model accuracy (out of sample)
out_svm_acc = round((ls_svm['results']['matrix']['test'][0, 0] +
                     ls_svm['results']['matrix']['test'][0, 1])/len(ytest), 4)

print(out_svm_acc)
