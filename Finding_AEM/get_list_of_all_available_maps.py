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


# This script downloads a list of available maps on Zenodo.
# Please ensure that the necessary packages included in the requirements.txt are installed correctly

# TODO (1) create and fill in (line 203) your personal Zenodo access token
# TODO (2) fill in (line 206) the absolute path of the output folder e.g. r'C:\Users\me\output'


import time

import pandas as pd
import requests

from path import Path


def get_list_of_all_available_maps(ACCESS_TOKEN, search_query):
    # --- search Zenodo for search query ---------------------------------------
    response = requests.get('https://zenodo.org/api/records',
                            params={'q': search_query, 'communities': 'ucare', 'size': 500,
                                    'access_token': ACCESS_TOKEN})
    response = response.json()

    hits = response['hits']
    hits = hits['hits']
    list_out = []
    emission_maps = pd.DataFrame()

    for i in range(0, len(hits), 1):
        # --- per search result get record data --------------------------------
        engine = hits[i]
        create_date = engine['created'].split('T')[0]
        metadata = engine['metadata']
        title = metadata['title']

        doi = engine['doi']
        updated = engine['updated'].split('T')[0]
        revision = engine['revision']
        files = engine['files']
        t_last_check = time.time()
        t_allowance = 1
        for k in range(0, len(files), 1):
            file = files[k]
            aem_file_name = file['key']
            link = file['links']
            link = link['self']

            # --- read and check file name -------------------------------------
            taxonomycode = aem_file_name.split('.')[0]
            if 'map.txt' in aem_file_name:
                version = ''
                creator = ''
                if '-v' in aem_file_name:
                    version = aem_file_name.split('-v')[1]
                    version = version.split('.')[0]
                    version = 'v{}'.format(version)
                    creator = aem_file_name.split('-v')[0].split('.')[1]
                if not '-v' in aem_file_name:
                    creator = aem_file_name.split('.')[1]
                    if creator == 'map':
                        creator = 'not in filename'
                print(taxonomycode)
                splitResult = taxonomycode.split('_')  # split on underscores
                if not len(splitResult) == 5:
                    print('--- invalid file name')
                    continue
                fuel_type = splitResult[0]
                euro_class = splitResult[1]
                engine_displacement = splitResult[2]
                power = splitResult[3]
                alliance = splitResult[4]

                # --- add rate limiter: can only request 60 per minute ---------
                t_current = time.time()
                t_time_passed = t_current - t_last_check
                t_last_check = t_current
                t_allowance += t_time_passed
                if t_allowance > 1:
                    t_allowance = 1
                if t_allowance < 1:
                    time.sleep(1)

                # --- get AEM map.txt file to scrape ---------------------------
                response = requests.get(link)
                t_allowance -= 1
                data = response.text

                # --- initiate scrape storing variables ---
                maps = ''
                ID = ''
                TOTAL_KM = ''
                TOTAL_TIME = ''
                NUM_VEHICLES = ''
                pollutants_coldstart = ''
                pollutants_deterioration = ''
                notes = []

                if '�' in data:
                    data = data.replace('�', '')
                    print('--- cold start')

                # --- start scraping -------------------------------------------
                for line in data.splitlines():
                    if 'AVAILABLE MAPS' in line:
                        line_maps = line.strip()
                        [i_map, n_maps] = line_maps.split(': ')
                        maps = n_maps

                    if 'ID' in line:
                        line_id = line.strip()
                        [i_id, n_id] = line_id.split(':')
                        ID = n_id

                    if 'NOTES' in line:
                        line_label = line.strip()
                        a_note = line_label.split('NOTES: ')[1].strip('[').strip(']')
                        notes.append(a_note)

                    if 'TOTAL KM' in line:
                        line_km = line.strip()
                        [i_km, n_km] = line_km.split(':')
                        TOTAL_KM = n_km

                    if 'TOTAL TIME [h]' in line:
                        line_time = line.strip()
                        [i_time, n_time] = line_time.split(':')
                        n_res = n_time.replace('hours of data', '')
                        TOTAL_TIME = n_res

                    if 'NUMBER OF VEHICLES' in line:
                        line_veh = line.strip()
                        [i_veh, n_veh] = line_veh.split(':')
                        NUM_VEHICLES = n_veh
                    if 'AVAILABLE COLD START' in line:
                        line_maps = line.strip()
                        [i_map, n_maps] = line_maps.split(':')
                        pollutants_coldstart = n_maps

                    if 'AVAILABLE DETERIORATION' in line:
                        line_maps = line.strip()
                        [i_map, n_maps] = line_maps.split(':')
                        pollutants_deterioration = n_maps

                    if 'NOTES' in line:
                        line_label = line.strip()
                        a_note = line_label.split('NOTES: ')[1].strip('[').strip(']')
                        notes.append(a_note)

                # --- check if engine code has been correctly parsed ---
                if ID == '':
                    print('wait')
                    ID = 'error parsing file'

                # --- create entry for final output ----------------------------

                out_dict = {'filename': title, 'doi': doi, 'taxonomycode': ID, 'creator': creator,
                            'version_filename': version, 'createdate': create_date,
                            'updated': updated, 'revision': revision, 'available_basemaps': maps,
                            'total_km': TOTAL_KM, 'total_time': TOTAL_TIME,
                            'number_of_vehicles': NUM_VEHICLES, 'cold_start': pollutants_coldstart,
                            'deterioration': pollutants_deterioration, 'notes': notes,
                            'fuel_type': fuel_type, 'euro_class': euro_class,
                            'engine_displacement': engine_displacement, 'power': power,
                            'alliance': alliance, 'link': link}

                list_out.append(out_dict)

            emission_maps = pd.DataFrame(list_out)

    return emission_maps


if __name__ == '__main__':
    # TODO (1) create an access token on Zenodo (profile -> applications -> new token):
    ACCESS_TOKEN = 'ACCESS2CODE'  # <insert access code here>
    output_folder = Path(r'path')  # TODO (2) <insert path here>
    string_to_search = 'Augmented emission maps '

    df_emission_maps = get_list_of_all_available_maps(ACCESS_TOKEN, string_to_search)
    # save a csv file in the supplied output folder, dated per accessed date
    df_emission_maps.to_csv(output_folder / 'all_maps_on_Zenodo_{}.csv'.format(
        time.strftime("%d-%m-%Y_%H%M%S", time.localtime(time.time()))),
                            index=False)

    print(f'AEM list saved to {output_folder}')
