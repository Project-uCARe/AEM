# Copyright (c) 2021, TNO
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================

# Module xymatrix.py
# Module that transforms measurement data to 2D-matrices which will be used to generate the Augmented Emission Map (AEM).
# This module also contains the plotting function for the 2D-matrices

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from functools import reduce
import pandas as pd


def clean_array(dict_input):
    """
    This function remove unnecessary rows and columns where there are only zeros. The function scans the matrix of
    count_var from bottom to top and from right to left and deletes the row or column as long as the sum of row or column
    is zero. The expected input is a dict. The output is a shrunk dict, if the mentioned conditions were met.
    """

    array = dict_input['count_var']
    vars = ['z_var', 'count_var', 'z_var_ave', 'std_var', 'Q25_var', 'Q75_var']
    for row in reversed(range(0, array.shape[0])):
        if sum(array[row]) == 0:
            for var in vars:
                dict_input[var] = np.delete(dict_input[var], row, 0)
            continue
        else:
            break

    for col in reversed(range(0, array.shape[1])):
        if sum(array[:, col]) == 0:
            for var in vars:
                dict_input[var] = np.delete(dict_input[var], col, 1)
            continue
        else:
            break

    print('Cleaning 2D matrix from excessive 0s')
    print('2D matrix size reduced from {} to {}'.format(array.shape,dict_input['count_var'].shape))
    return dict_input


# Calculate standard deviation
def std(x):
    return np.std(x, ddof=1)


# Plot function for plotting 2D matrix z_var as a heatmap
def plot(z_var, min_x, max_x, min_y, max_y, min_z, max_z, x_sig_name, y_sig_name, z_sig_name,
         v_scale):
    # Plot the Z variable in the color bar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    H = z_var.copy()
    if v_scale == 'log':
        plt.imshow(H, interpolation='nearest', cmap='jet', extent=(min_x, max_x, max_y, min_y), aspect='auto',
                   norm=LogNorm(vmin=min_z, vmax=max_z))
    elif v_scale == 'linear':
        plt.imshow(H, interpolation='nearest', cmap='bwr', extent=(min_x, max_x, max_y, min_y), aspect='auto',
                   norm=Normalize(vmin=-40, vmax=40))

    ax = plt.gca()
    ax.set_xlabel(x_sig_name)
    ax.set_ylabel(y_sig_name)

    ax.invert_yaxis()  # 0 at the bottom

    cbar = plt.colorbar()  # show colorbar z axis
    cbar.set_label(z_sig_name)

    return ax



def getmatrix(data, z_sig, x_sig, y_sig, min_x, max_x, min_y, max_y, x_bins, y_bins, bin_threshold):
    """
    Convert measurement data onto 2D matrix format.
    Current implementation will return 2D matrices of data counts per bin, mean values, std values, 25th quartile values, and 75th quartile values
    """

    y_axis = np.linspace(min_y, max_y, y_bins)
    x_axis = np.linspace(min_x, max_x, x_bins)

    data = data[data[x_sig] < max_x]
    data = data[data[x_sig] > min_x]
    data = data[data[y_sig] < max_y]
    data = data[data[y_sig] > min_y]

    data_bin_sum = data.groupby([np.digitize(data[y_sig], y_axis), np.digitize(data[x_sig], x_axis)], sort=True,
                                group_keys=True).agg('sum')
    # Count variable
    data_bin_sum['count'] = \
        data.groupby([np.digitize(data[y_sig], y_axis), np.digitize(data[x_sig], x_axis)], sort=True,
                     group_keys=True).agg('count')[x_sig]

    # standard deviation
    data_bin_sum['std'] = \
        data.groupby([np.digitize(data[y_sig], y_axis), np.digitize(data[x_sig], x_axis)], sort=True,
                     group_keys=True).agg(std)[z_sig]

    # 25th and 75th quantile
    data_bin_sum['Q25'] = \
        data.groupby([np.digitize(data[y_sig], y_axis), np.digitize(data[x_sig], x_axis)], sort=True,
                     group_keys=True).quantile(q=0.25)[z_sig]

    data_bin_sum['Q75'] = \
        data.groupby([np.digitize(data[y_sig], y_axis), np.digitize(data[x_sig], x_axis)], sort=True,
                     group_keys=True).quantile(q=0.75)[z_sig]

    z_var = np.zeros((y_bins, x_bins))
    count_var = np.zeros((y_bins, x_bins))
    z_var_ave = np.zeros((y_bins, x_bins))
    std_var = np.zeros((y_bins, x_bins))
    Q25_var = np.zeros((y_bins, x_bins))
    Q75_var = np.zeros((y_bins, x_bins))

    # Initialize x-labels and y-labels for csv output of the matrix
    x_axis_csv = []
    y_axis_csv = []

    if data_bin_sum.empty:
        return z_var, count_var, z_var_ave, std_var, Q25_var, Q75_var, x_axis_csv, y_axis_csv, pd.DataFrame()

    for yy_ctr, temp_y in data_bin_sum.groupby(level=0):
        yy = yy_ctr - 1
        for xx_ctr, temp_x in temp_y.groupby(level=1):
            xx = xx_ctr - 1

            # Skip empty cell and too few data
            if len(temp_x['count'].values) == 0:
                continue
            elif temp_x['count'].values[0] < bin_threshold:  # apply threshold of minimum x datapoints per bin.
                continue

            z_var[yy][xx] = temp_x[z_sig].values[0]
            count_var[yy][xx] = temp_x['count'].values[0]
            std_var[yy][xx] = temp_x['std'].values[0]
            Q25_var[yy][xx] = temp_x['Q25'].values[0]
            Q75_var[yy][xx] = temp_x['Q75'].values[0]

    z_var_ave = np.divide(z_var, count_var, out=np.zeros_like(z_var), where=(count_var > 0))

    maps_dict = {'z_var': z_var, 'count_var': count_var, 'z_var_ave': z_var_ave, 'std_var': std_var, 'Q25_var': Q25_var,
                 'Q75_var': Q75_var}

    # clean zero's from arrays:
    maps_dict = clean_array(maps_dict)
    x_axis = x_axis[1:maps_dict['z_var_ave'].shape[1] + 1]  # reduce the size of the x axis after the array is cleaned
    y_axis = y_axis[1:maps_dict['z_var_ave'].shape[0] + 1]  # reduce the size of the y axis after the array is cleaned

    x_axis_csv = x_axis.tolist()
    y_axis_csv = y_axis.tolist()

    ## Make Table for .map.txt outputfile
    std_var_df = pd.DataFrame(data=maps_dict['std_var'], index=y_axis, columns=x_axis)
    z_var_ave_df = pd.DataFrame(data=maps_dict['z_var_ave'], index=y_axis, columns=x_axis)
    count_var_df = pd.DataFrame(data=maps_dict['count_var'], index=y_axis, columns=x_axis)
    Q25_var_df = pd.DataFrame(data=maps_dict['Q25_var'], index=y_axis, columns=x_axis)
    Q75_var_df = pd.DataFrame(data=maps_dict['Q75_var'], index=y_axis, columns=x_axis)

    df_finalz1 = pd.DataFrame()
    df_finalz2 = pd.DataFrame()
    df_finalz3 = pd.DataFrame()
    df_finalz4 = pd.DataFrame()
    df_finalz5 = pd.DataFrame()

    for key, value in z_var_ave_df.items():
        df_temp = pd.DataFrame()
        df_temp['Y'] = value.index  # CO2_mF [g/s]
        df_temp['X'] = key  # velocity_filtered [km/h]
        df_temp['Z1'] = z_var_ave_df[key].values  # avgNOx [mg/s]
        df_temp = df_temp[['X', 'Y', 'Z1']]
        df_finalz1 = df_finalz1.append(df_temp)

    for key, value in std_var_df.items():
        df_temp = pd.DataFrame()
        df_temp['Y'] = value.index  # CO2_mF [g/s]
        df_temp['X'] = key  # velocity_filtered [km/h]
        df_temp['Z2'] = std_var_df[key].values  # stdNOx [mg/s]
        df_temp = df_temp[['X', 'Y', 'Z2']]
        df_finalz2 = df_finalz2.append(df_temp)

    for key, value in Q25_var_df.items():
        df_temp = pd.DataFrame()
        df_temp['Y'] = value.index  # CO2_mF [g/s]
        df_temp['X'] = key  # velocity_filtered [km/h]
        df_temp['Z3'] = Q25_var_df[key].values  # 25th quantile[mg/s]
        df_temp = df_temp[['X', 'Y', 'Z3']]
        df_finalz3 = df_finalz3.append(df_temp)

    for key, value in Q75_var_df.items():
        df_temp = pd.DataFrame()
        df_temp['Y'] = value.index  # CO2_mF [g/s]
        df_temp['X'] = key  # velocity_filtered [km/h]
        df_temp['Z4'] = Q75_var_df[key].values  # 75th quantile[mg/s]
        df_temp = df_temp[['X', 'Y', 'Z4']]
        df_finalz4 = df_finalz4.append(df_temp)

    for key, value in count_var_df.items():
        df_temp = pd.DataFrame()
        df_temp['Y'] = value.index  # CO2_mF [g/s]
        df_temp['X'] = key  # velocity_filtered [km/h]
        df_temp['Z5'] = count_var_df[key].values  # count
        df_temp = df_temp[['X', 'Y', 'Z5']]
        df_finalz5 = df_finalz5.append(df_temp)

    dfs = [df_finalz1, df_finalz2, df_finalz3, df_finalz4, df_finalz5]
    map_tbl_df = reduce(lambda left, right: pd.merge(left, right, on=['X', 'Y']), dfs)
    map_tbl_df = map_tbl_df[map_tbl_df['X'] > 0]

    return maps_dict['z_var'], maps_dict['count_var'], maps_dict['z_var_ave'], \
           maps_dict['std_var'], maps_dict['Q25_var'], maps_dict['Q75_var'], \
           x_axis_csv, y_axis_csv, map_tbl_df
