
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


# -- -------------------------------------------------------- PLOT: OHLC Price Chart with Vertical Lines -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_ohlc(p_ohlc, p_theme, p_dims, p_vlines=None, p_labels=None):
    """

    Timeseries Candlestick with OHLC prices and figures for trades indicator

    Requirements
    ------------
    numpy
    pandas
    plotly

    Parameters
    ----------
    p_ohlc: pd.DataFrame
        that contains the following float or int columns: 'timestamp', 'open', 'high', 'low', 'close'
    p_theme: dict
        with the theme for the visualizations
    p_dims: dict
        with sizes for visualizations
    p_vlines: list
        with the dates where to visualize the vertical lines, format = pd.to_datetime('2020-01-01 22:15:00')
    p_labels: dict
        with main title and both x and y axes

    Returns
    -------
    fig_g_ohlc: plotly
        objet/dictionary to .show() and plot in the browser

    Debugging
    ---------
    p_ohlc = price_data
    p_theme = p_theme
    p_dims = p_dims
    p_vlines = [pd.to_datetime('2020-01-01 22:35:00'), pd.to_datetime('2020-01-01 22:15:00')]
    p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}
    """

    # default value for lables to use in main title, and both x and y axis
    if p_labels is None:
        p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

    # tick values calculation for simetry in y axes
    y0_ticks_vals = np.arange(min(p_ohlc['low']), max(p_ohlc['high']),
                              (max(p_ohlc['high']) - min(p_ohlc['low'])) / 10)
    y0_ticks_vals = np.append(y0_ticks_vals, max(p_ohlc['high']))
    y0_ticks_vals = np.round(y0_ticks_vals, 4)

    # instantiate a figure object
    fig_g_ohlc = go.Figure()

    # Add layer for OHLC candlestick chart
    fig_g_ohlc.add_trace(
        go.Candlestick(name='ohlc',
                       x=p_ohlc['timestamp'],
                       open=p_ohlc['open'],
                       high=p_ohlc['high'],
                       low=p_ohlc['low'],
                       close=p_ohlc['close'],
                       opacity=0.7))

    # Layout for margin, and both x and y axes
    fig_g_ohlc.update_layout(
        margin=go.layout.Margin(l=50, r=50, b=20, t=50, pad=20),
        xaxis=dict(title_text=p_labels['x_title'], rangeslider=dict(visible=False)),
        yaxis=dict(title_text=p_labels['y_title']))

    # Color and font type for text in axes
    fig_g_ohlc.update_layout(
        xaxis=dict(titlefont=dict(color=p_theme['color_1']),
                   tickfont=dict(color=p_theme['color_1'],
                                 size=p_theme['font_size_1'])),
        yaxis=dict(zeroline=False, automargin=True,
                   titlefont=dict(color=p_theme['color_1']),
                   tickfont=dict(color=p_theme['color_1'],
                                 size=p_theme['font_size_1']),
                   showgrid=True))

    # Size of final plot according to desired dimensions
    fig_g_ohlc.layout.autosize = True
    fig_g_ohlc.layout.width = p_dims['width']
    fig_g_ohlc.layout.height = p_dims['height']

    # Dynamically add vertical lines according to the provided list of x dates.
    shapes_list = list()
    for i in p_vlines:
        shapes_list.append({'type': 'line', 'fillcolor': p_theme['color_1'],
                            'line': {'color': p_theme['color_1'], 'dash': 'dashdot'},
                            'x0': i, 'x1': i, 'xref': 'x',
                            'y0': min(p_ohlc['low']), 'y1': max(p_ohlc['high']), 'yref': 'y'})

    # Update layout
    fig_g_ohlc.update_layout(shapes=shapes_list)

    # Update layout for the background
    fig_g_ohlc.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                                    yaxis=dict(tickvals=y0_ticks_vals, zeroline=False, automargin=True,
                                               tickfont=dict(color='grey', size=p_theme['font_size_1'])),
                                    xaxis=dict(tickfont=dict(color='grey', size=p_theme['font_size_1'])))

    # Update layout for the y axis
    fig_g_ohlc.update_yaxes(showgrid=True, gridwidth=.25, gridcolor='lightgrey')
    fig_g_ohlc.update_xaxes(showgrid=False, rangebreaks=[dict(pattern="day of week",
                                                              bounds=['sat', 'sun'])])

    # Format to main title
    fig_g_ohlc.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=50, pad=20),
                               title=dict(x=0.5, text='<b> ' + p_labels['title'] + ' </b>'),
                               legend=go.layout.Legend(x=.3, y=-.15, orientation='h', font=dict(size=15)))

    return fig_g_ohlc
