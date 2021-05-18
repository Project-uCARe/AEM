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

# Module map_output.py
# Module to generate the Augmented Emission Map (AEM) text files. Text files contain information
# on the data sources, as described in D1.2 Report

import csv
import numpy as np
import time
import os


def emission_map(outpath, engine_taxonomy, veh_cnt, start_mileage, method, DOI,
                 org_name, version_ctrl, RPM_CO2_emission, V_CO2_emission):

    # Define filename for map text file
    basemap_filename = '{}.{}-{}.map.txt'.format(str(engine_taxonomy), str(org_name),str(version_ctrl))

    # Check if start_mileage is a valid number
    if not isinstance(start_mileage, int) and not isinstance(start_mileage, float):
        start_mileage = np.nan

    # Check which maps are available
    map_list = []
    if bool(V_CO2_emission):
        for i in V_CO2_emission:
            map_list.append('VEHICLE SPEED - CO2 - MEAN {} - STD - Q25 - Q75 - COUNT'.format(i))

    if bool(RPM_CO2_emission):
        for i in RPM_CO2_emission:
            map_list.append('ENGINE SPEED - CO2 - MEAN {} - STD - Q25 - Q75 - COUNT'.format(i))

    if not os.path.exists('{}/emissionmaps'.format(str(outpath), str(engine_taxonomy))):
        os.makedirs('{}/emissionmaps'.format(str(outpath), str(engine_taxonomy)))

    print('Saving map data to: {}/emissionmaps/{}'.format(str(outpath), basemap_filename))

    with open('{}/emissionmaps/{}'.format(str(outpath), basemap_filename), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE)
        writer.writerow(["################################################################################"])
        writer.writerow(["# START META"])
        writer.writerow(["#"])
        writer.writerow(["# ID: {}".format(str(engine_taxonomy))])

        if bool(RPM_CO2_emission):
            writer.writerow(["# TOTAL KM: {}".format(RPM_CO2_emission[i]['total_km'])])  # Any emissions will do
            writer.writerow(["# TOTAL TIME [h]: {} ".format(RPM_CO2_emission[i]['total_time'])])
        elif bool(V_CO2_emission):
            writer.writerow(["# TOTAL KM: {}".format(V_CO2_emission[i]['total_km'])])  # Any emissions will do
            writer.writerow(["# TOTAL TIME [h]: {} ".format(V_CO2_emission[i]['total_time'])])

        writer.writerow(["# NUMBER OF VEHICLES: {}".format(veh_cnt)])
        writer.writerow(['# AVERAGE MILEAGE OF VEHICLES [km]: {}'.format(
            'n/a' if np.isnan(start_mileage) else int(np.round(start_mileage)))])
        writer.writerow(["# REFERENCE DOI: {}".format(DOI)])
        writer.writerow(["# AVAILABLE MAPS: {}".format(', '.join(map_list))])
        writer.writerow(["# END META"])
        f.close()

    time.sleep(1)

    if bool(V_CO2_emission):
        for i in V_CO2_emission:
            # write VEHICLE SPEED - CO2 - NOX - STD NOx - Q25 - Q75 - COUNT
            with open('{}/emissionmaps/{}'.format(str(outpath), basemap_filename), 'a', newline='') as f:
                writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE)
                writer.writerow(["################################################################################"])
                writer.writerow(["# START VEHICLE SPEED - CO2 - MEAN {} - STD - Q25 - Q75 - COUNT".format(i)])
                writer.writerow(["#"])
                writer.writerow(["# NOTES: [{}]".format(method)])
                writer.writerow(['# XLABEL: {}'.format('Vehicle Speed upper bin limit [km/h]')])
                writer.writerow(['# YLABEL: {}'.format('CO2 upper bin limit [g/s]')])
                writer.writerow(['# Z1LABEL: Average {} emissions [mg/s]'.format(i)])  # map_tbl_df
                writer.writerow(['# Z2LABEL: Standard deviation {} emissions [mg/s]'.format(i)])
                writer.writerow(['# Z3LABEL: 0.25 quantile {} emissions [mg/s]'.format(i)])
                writer.writerow(['# Z4LABEL: 0.75 quantile {} emissions [mg/s]'.format(i)])
                writer.writerow(['# Z5LABEL: Count per bin [#]'])
                writer.writerow(['# START DATA VEHICLE SPEED - CO2 - MEAN {} - STD - Q25 - Q75 - COUNT'.format(i)])
                f.close()
            time.sleep(1)

            V_CO2_emission[i]['map_tbl_df'].to_csv('{}/emissionmaps/{}'.format(str(outpath), basemap_filename), mode='a', header=True, index=False)

            time.sleep(1)

            with open('{}/emissionmaps/{}'.format(str(outpath), basemap_filename), 'a', newline='') as f:
                writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE)
                writer.writerow(["# END VEHICLE SPEED - CO2 - MEAN {} - STD - Q25 - Q75 - COUNT".format(i)])
                f.close()

    if bool(RPM_CO2_emission):
        for i in RPM_CO2_emission:
            # example: write ENGINE SPEED - CO2 - NOX - STD NOx - COUNT
            with open('{}/emissionmaps/{}'.format(str(outpath), basemap_filename), 'a', newline='') as f:
                writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE)
                writer.writerow(["################################################################################"])
                writer.writerow(["# START ENGINE SPEED - CO2 - MEAN {} - STD - Q25 - Q75 - COUNT".format(i)])
                writer.writerow(["#"])
                writer.writerow(["# NOTES: [{}]".format(method)])
                writer.writerow(['# XLABEL: {}'.format('Engine Speed upper bin limit [rpm]')])
                writer.writerow(['# YLABEL: {}'.format('CO2 upper bin limit [g/s]')])
                writer.writerow(['# Z1LABEL: Average {} emissions [mg/s]'.format(i)])  # map_tbl_df
                writer.writerow(['# Z2LABEL: Standard deviation {} emissions [mg/s]'.format(i)])
                writer.writerow(['# Z3LABEL: 0.25 quantile {} emissions [mg/s]'.format(i)])
                writer.writerow(['# Z4LABEL: 0.75 quantile {} emissions [mg/s]'.format(i)])
                writer.writerow(['# Z5LABEL: Count per bin [#]'])
                writer.writerow(['# START DATA ENGINE SPEED - CO2 - MEAN {} - STD - Q25 - Q75 - COUNT'.format(i)])
                f.close()
            time.sleep(1)

            RPM_CO2_emission[i]['map_tbl_df'].to_csv('{}/emissionmaps/{}'.format(str(outpath), basemap_filename), mode='a', header=True, index=False)
            time.sleep(1)

            with open('{}/emissionmaps/{}'.format(str(outpath), basemap_filename), 'a', newline='') as f:
                writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE)
                writer.writerow(['# END ENGINE SPEED - CO2 - MEAN {} - STD - Q25 - Q75 - COUNT'.format(i)])
                f.close()

    print('Map data successfully saved to: {}/emissionmaps/{}'.format(str(outpath), basemap_filename))
