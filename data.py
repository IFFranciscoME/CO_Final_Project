
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

# ----------------------------------------------------------------------------------- Lectura de precios -- #
# ----------------------------------------------------------------------------------- ------------------ -- #

data_train = pd.read_csv('files/MP_D_2019.csv')
data_train['timestamp'] = pd.to_datetime(data_train['timestamp'])

data_test = pd.read_csv('files/MP_D_2020.csv')
data_test['timestamp'] = pd.to_datetime(data_test['timestamp'])

