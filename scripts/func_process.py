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

# Module func_process.py
# Module to read and process the CSV input file into 2D matrices, which are needed to generate the Augmented Emission Map

import pandas as pd
import numpy as np
from scripts import func_flex_bin, map_output, graph_output


def process_data(datafile, engine_taxonomy_code, available_pollutants, veh_cnt, bin_threshold,
                 start_mileage, method, DOI, org_name, version_ctrl, outpath):
    # Read dataset =========================================================================================================
    df = pd.read_csv(datafile)  # columns: ['velocity', 'RPM', 'CO2', 'NOx', 'NH3']

    # Emission maps: Velocity, CO2, EmissionComponent=======================================================================
    V_CO2_emission = {}
    if 'CO2' in df and 'velocity' in df:
        for emission in available_pollutants:
            z_sig = emission  # mg/s
            x_sig = 'velocity'  # km/h
            y_sig = 'CO2'  # g/s
            min_x = 0
            max_x = np.nanmax(df[x_sig]).max()
            min_y = 0
            max_y = np.nanmax(df[y_sig]).max()

            # Starting point of bin size
            x_scale = 5
            y_scale = 0.2

            # smoothing factor
            N = 2

            if z_sig in df.columns:
                maps = func_flex_bin.flexible_bin_sizing_average(df, z_sig, x_sig, y_sig, min_x, max_x,
                                                                 min_y, max_y, x_scale, y_scale, bin_threshold)
                V_CO2_emission.update({z_sig: maps})

    # Emission maps: EngineSpeed, CO2, EmissionComponent====================================================================
    RPM_CO2_emission = {}
    if 'CO2' in df and 'RPM' in df:
        for emission in available_pollutants:
            z_sig = emission  # mg/s
            x_sig = 'RPM'  # rpm
            y_sig = 'CO2'  # g/s
            min_x = 0
            max_x = np.nanmax(df[x_sig]).max()
            min_y = 0
            max_y = np.nanmax(df[y_sig]).max()

            # Starting point of bin size
            x_scale = 50
            y_scale = 0.2

            if z_sig in df.columns:
                maps = func_flex_bin.flexible_bin_sizing_average(df, z_sig, x_sig, y_sig, min_x, max_x,
                                                                 min_y, max_y, x_scale, y_scale, bin_threshold)
                RPM_CO2_emission.update({z_sig: maps})

    # Creating .map.txt outputfile
    map_output.emission_map(outpath=outpath, engine_taxonomy=engine_taxonomy_code, veh_cnt=veh_cnt,
                            start_mileage=start_mileage, method=method, DOI=DOI,
                            org_name=org_name, version_ctrl=version_ctrl,
                            V_CO2_emission=V_CO2_emission, RPM_CO2_emission=RPM_CO2_emission)

    # Creating graphs for visual check
    graph_output.graphs(outpath=outpath, engine_taxonomy=engine_taxonomy_code,
                        V_CO2_emission=V_CO2_emission, RPM_CO2_emission=RPM_CO2_emission)
