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

# Module graph_output.py
# Module to generate the figures for visual representation of the Augemented Emission Map (AEM)


from scripts import xymatrix
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def create_plot_settings_dict(emission_name):
    """
    Create a dictionary containing the plot settings for the AEM figures
    """
    title_dict = {'z_var_ave': 'Average {} emission [mg/s]'.format(emission_name),
                  'std_var': 'std {} emission per bin [mg/s]'.format(emission_name),
                  'Q25_var': '{} emission 0.25 quantile per bin [mg/s]'.format(emission_name),
                  'Q75_var': '{} emission 0.75 quantile per bin [mg/s]'.format(emission_name),
                  'count_var': 'Amount of {} data per bin [-]'.format(emission_name),
                  }
    minz_dict = {'z_var_ave': 1e-2,
                 'std_var': 1e-2,
                 'Q25_var': 1e-2,
                 'Q75_var': 1e-2,
                 'count_var': 1,
                 }

    figure_dict = {'z_var_ave': 'MEAN',
                   'std_var': 'STD',
                   'Q25_var': 'Q25',
                   'Q75_var': 'Q75',
                   'count_var': 'COUNT',
                   }
    return title_dict, minz_dict, figure_dict


def graphs(outpath, engine_taxonomy, V_CO2_emission, RPM_CO2_emission, ):
    """
    Produce AEMS figures and save them in the designated outpath
    """
    # Make folder if it doesnt exist yet
    if not os.path.exists('{}/emissiongraphs'.format(str(outpath), str(engine_taxonomy))):
        os.makedirs('{}/emissiongraphs'.format(str(outpath), str(engine_taxonomy)))

    if bool(V_CO2_emission):
        for i in V_CO2_emission:
            hours_of_data = ' - {:.1f} hours of data'.format(V_CO2_emission[i]['total_time'])
            title_dict, minz_dict, figure_dict = create_plot_settings_dict(i)
            for j in ['z_var_ave', 'std_var', 'Q25_var', 'Q75_var', 'count_var']:
                V_CO2_filename = '{}-V-CO2-{}_{}-map.png'.format(str(engine_taxonomy), str(i), str(figure_dict[j]))
                ax = xymatrix.plot(V_CO2_emission[i][j],
                                   min_x=V_CO2_emission[i]['min_x'], max_x=V_CO2_emission[i]['max_x'],
                                   min_y=V_CO2_emission[i]['min_y'], max_y=V_CO2_emission[i]['max_y'],
                                   min_z=minz_dict[j], max_z=np.nanmax(V_CO2_emission[i][j]).max(),
                                   x_sig_name='Vehicle Speed [km/h]', y_sig_name='CO2 emission [g/s]',
                                   z_sig_name=title_dict[j], v_scale='log')

                ax.set_title(title_dict[j] + '\n' + engine_taxonomy + hours_of_data)

                ax.set_xlim([0, V_CO2_emission[i]['map_tbl_df']['X'].max()])
                ax.set_ylim([0, V_CO2_emission[i]['map_tbl_df']['Y'].max()])
                plt.savefig(Path('{}/emissiongraphs/{}'.format(str(outpath), V_CO2_filename)))

    if bool(RPM_CO2_emission):
        for i in RPM_CO2_emission:
            hours_of_data = ' - {:.1f} hours of data'.format(RPM_CO2_emission[i]['total_time'])
            title_dict, minz_dict, figure_dict = create_plot_settings_dict(i)
            for j in ['z_var_ave', 'std_var', 'Q25_var', 'Q75_var', 'count_var']:
                RPM_CO2_filename = '{}-RPM-CO2-{}_{}-map.png'.format(str(engine_taxonomy), str(i), str(figure_dict[j]))
                ax = xymatrix.plot(RPM_CO2_emission[i][j],
                                   min_x=RPM_CO2_emission[i]['min_x'], max_x=RPM_CO2_emission[i]['max_x'],
                                   min_y=RPM_CO2_emission[i]['min_y'], max_y=RPM_CO2_emission[i]['max_y'],
                                   min_z=minz_dict[j], max_z=np.nanmax(RPM_CO2_emission[i][j]).max(),
                                   x_sig_name='Engine Speed [RPM]', y_sig_name='CO2 emission [g/s]',
                                   z_sig_name=title_dict[j], v_scale='log')

                ax.set_title(title_dict[j] + '\n' + engine_taxonomy + hours_of_data)
                ax.set_xlim([0, RPM_CO2_emission[i]['map_tbl_df']['X'].max()])
                ax.set_ylim([0, RPM_CO2_emission[i]['map_tbl_df']['Y'].max()])
                plt.savefig(Path('{}/emissiongraphs/{}'.format(str(outpath), RPM_CO2_filename)))
