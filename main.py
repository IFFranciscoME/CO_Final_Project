
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""


import functions as fn
from data import price_data
import visualizations as vs

# primeras pruebas sera con 1 solo periodo
data = price_data[list(price_data.keys())[9]]

# ---------------------------------------------------------------------------------- datos para proyecto -- #
# ---------------------------------------------------------------------------------- ------------------- -- #

# datos iniciales para hacer pruebas
datos = data

# ------------------------------------------------------------------------------- visualizacion de datos -- #
# ------------------------------------------------------------------------------- ---------------------- -- #

# grafica OHLC
p_theme = {'color_1': '#ABABAB', 'color_2': '#ABABAB', 'color_3': '#ABABAB', 'font_color_1': '#ABABAB',
           'font_size_1': 12, 'font_size_2': 16}
p_dims = {'width': 1450, 'height': 800}
p_vlines = [datos['timestamp'].head(1), datos['timestamp'].tail(1)]
p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

# cargar funcion importada desde GIST de visualizaciones
ohlc = vs.g_ohlc(p_ohlc=datos, p_theme=p_theme, p_dims=p_dims, p_vlines=p_vlines, p_labels=p_labels)

# mostrar grafica
# ohlc.show()

# ----------------------------------------------------------------------- analisis exploratorio de datos -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# tabla de descripcion de datos
datos.describe()

# --------------------------------------------------------------- ingenieria de variables autoregresivas -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# funcion para generar variables autoregresivas
datos_arf = fn.f_autoregressive_features(p_data=datos, p_nmax=30)

# Visualizacion: head del DataFrame
datos_arf.head(5)

# --------------------------------------------------------------------- ingenieria de variables hadamard -- #
# --------------------------------------------------------------------- -------------------------------- -- #

# funcion para generar variables con producto hadamard
datos_had = fn.f_hadamard_features(p_data=datos_arf, p_nmax=29)

# Visualizacion: head del DataFrame
datos_had.head(5)

# ------------------------------------------------------------------- ingenieria de variables simbolicas -- #
# ------------------------------------------------------------------- ---------------------------------- -- #

# Lista de operaciones simbolicas
fun_sym = fn.symbolic_features(p_x=datos_had.iloc[:, 3:], p_y=datos_had.iloc[:, 2])

# variables
datos_sym = fun_sym['data']
datos_sym.columns = ['sym_' + str(i) for i in range(0, len(fun_sym['data'].iloc[0, :]))]

# ecuaciones de todas las variables
equaciones = [i.__str__() for i in list(fun_sym['model'])]

# ------------------------------------------------------------------------------------ ELASTIC NET MODEL -- #
# ------------------------------------------------------------------------------------ ----------------- -- #

# model 1 result
model_1 = fn.Elastic_Net(p_data=0, p_params=0)
