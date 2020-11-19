
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
from os import listdir, path
from os.path import isfile, join


# ----------------------------------------------------------------------------------- Lectura de precios -- #
# ----------------------------------------------------------------------------------- ------------------ -- #

abspath_f = path.abspath('files/')
files_f = sorted([f for f in listdir(abspath_f) if isfile(join(abspath_f, f))])
price_data = {}

for file_f in files_f:
    data_f = pd.read_csv('files/' + file_f)
    data_f['timestamp'] = pd.to_datetime(data_f['timestamp'])
    years_f = set([str(datadate.year) for datadate in list(data_f['timestamp'])])
    for year_f in years_f:
        price_data['MP_D_' + year_f] = data_f
