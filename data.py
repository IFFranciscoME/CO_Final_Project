
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import pickle
from os import listdir, path
from os.path import isfile, join

# ---------------------------------------------------------------------------- Historical Prices Reading -- #
# ---------------------------------------------------------------------------- ------------------------- -- #

# the price in the file is expressed as the USD to purchas one MXN
# if is needed to convert to the inverse, the MXN to purchas one USD, uncomment the following line
mode = 'MXN_USD'

# path in order to read files
main_path = 'files/'
abspath_f = path.abspath(main_path)
files_f = sorted([f for f in listdir(abspath_f) if isfile(join(abspath_f, f))])
price_data = {}

# iterative data reading
for file_f in files_f:
    # file_f = files_f[3]
    data_f = pd.read_csv(main_path + file_f)
    data_f['timestamp'] = pd.to_datetime(data_f['timestamp'])

    # swap since original is wrong
    low = data_f['low'].copy()
    high = data_f['high'].copy()
    data_f['high'] = low
    data_f['low'] = high

    if mode == 'MXN_USD':
        data_f['open'] = round(1/data_f['open'], 5)
        data_f['high'] = round(1/data_f['high'], 5)
        data_f['low'] = round(1/data_f['low'], 5)
        data_f['close'] = round(1/data_f['close'], 5)

    years_f = set([str(datadate.year) for datadate in list(data_f['timestamp'])])
    for year_f in years_f:
        price_data['MP_D_' + year_f] = data_f

# whole data sets integrated
ohlc_data = pd.concat([price_data[list(price_data.keys())[0]], price_data[list(price_data.keys())[1]],
                       price_data[list(price_data.keys())[2]], price_data[list(price_data.keys())[3]]])

# reset index
ohlc_data.reset_index(inplace=True, drop=True)

# ----------------------------------------------------------------------- Hyperparameters for the Models -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# data dictionary for models and their respective hyperparameter value candidates
models = {
    'logistic-elasticnet': {
        'label': 'logistic-elasticnet',
        'params': {'ratio': [0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
                   'c': [1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5]}},
    'ls-svm': {
        'label': 'ls-svm',
        'params': {'c': [1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5],
                   'kernel': ['linear', 'linear', 'linear', 'linear', 'linear',
                              'rbf', 'rbf', 'rbf', 'rbf', 'rbf'],
                   'gamma': ['scale', 'scale', 'scale', 'scale', 'scale',
                             'auto', 'auto', 'auto', 'auto', 'auto']}}}

# ------------------------------------------------------------------------------------- Themes for plots -- #
# ------------------------------------------------------------------------------------- ---------------- -- #

# Plot_1 : Original Historical OHLC prices
theme_plot_1 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 20, 'font_axis': 16, 'font_ticks': 12},
                    p_dims={'width': 1600, 'height': 800},
                    p_labels={'title': 'Precios OHLC',
                              'x_title': 'Fechas', 'y_title': 'Futuros USD/MXN'})

# Plot_2 : Timeseries T-Folds blocks without filtration
theme_plot_2 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 20, 'font_axis': 16, 'font_ticks': 12},
                    p_dims={'width': 1600, 'height': 800},
                    p_labels={'title': 'T-Folds por Bloques Sin Filtraciones',
                              'x_title': 'Fechas', 'y_title': 'Futuros USD/MXN'})

# Plot_3 Observed Class vs Predicted Class
theme_plot_3 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 20, 'font_axis': 16, 'font_ticks': 12},
                    p_dims={'width': 1600, 'height': 800},
                    p_labels={'title': 'Clasificaciones',
                              'x_title': 'Fechas', 'y_title': 'Clasificacion'})

# Plot_4 ROC of models
theme_plot_4 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 20, 'font_axis': 16, 'font_ticks': 12},
                    p_dims={'width': 1600, 'height': 800},
                    p_labels={'title': 'ROC',
                              'x_title': 'FPR', 'y_title': 'TPR'})

# Plot_5 AUC Timeseries of models
theme_plot_5 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 20, 'font_axis': 16, 'font_ticks': 12},
                    p_dims={'width': 1600, 'height': 800},
                    p_labels={'title': 'AUC por periodo (Test Data)',
                              'x_title': 'Periodos', 'y_title': 'AUC'})


# -------------------------------------------------------------------------------------------- Save Data -- #
# -------------------------------------------------------------------------------------------- --------- -- #

def data_save_load(p_data_objects, p_data_action, p_data_file):
    """
    Save or load data in pickle format for offline use

    Parameters
    ----------
    p_data_objects: dict
        with data objects to be saved

    p_data_action: str
        'save' to data saving or 'load' to data loading

    p_data_file: str
        with the name of the pickle file

    Returns
    -------
    Message if data file is saved or data objects if data file is loaded

    """

    # if saving is required
    if p_data_action == 'save':
        # define and create file
        pick = p_data_file
        with open(pick, "wb") as f:
            pickle.dump(p_data_objects, f)

        # Return message
        return 'Data saved in' + p_data_file + 'file'

    # if loading is required
    elif p_data_action == 'load':

        # read the file
        with open(p_data_file, 'rb') as handle:
            loaded_data = pickle.load(handle)

        # return loaded data
        return loaded_data
