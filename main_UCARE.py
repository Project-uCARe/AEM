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


# This is the main script to generate the Augmented Emission Maps (AEM) from the project uCARe.
# Please ensure that the necessary packages includded in the requirements.txt are installed correctly

from scripts import func_process

if __name__ == '__main__':
    # Input fields==========================================================================================================
    from pathlib import Path

    # Setting working directories
    working_dir = Path.cwd()
    outpath = working_dir / 'outputdata'

    # Input parameters for the header
    datafile = working_dir / 'inputdata/dataset_example.csv'
    engine_taxonomy_code = 'D_6_1499_70_EXA'  # example: D_6_1499_70_EXA --> Diesel_Euro6_1499cc_70kW_Example alliance
    available_pollutants = ['NOx', 'NH3']  # pollutants which are in the datafile
    veh_cnt = 10  # number of vehicles which are used to gather data
    method = 'Chassis dynamometer'  # method which is used to gather the data (Chassis dynamometer, PEMS, etc)
    DOI = '10.5281/zenodo.3669986'  # Digital Object Identifier for the open research data repository (Zenodo)
    start_mileage = 12345.67  # Example average start mileage [km]. If not applicable, fill in string "N/A"

    # Threshold values for valid data per bin
    BIN_THRESHOLD = 5  # if number of data per bin < BIN_THRESHOLD, it will be removed from map.txt output

    # Parameters for filenaming
    org_name = 'ABC'
    version_ctrl = '1'

    func_process.process_data(datafile=datafile, engine_taxonomy_code=engine_taxonomy_code,
                              available_pollutants=available_pollutants, veh_cnt=veh_cnt,
                              bin_threshold=BIN_THRESHOLD, start_mileage=start_mileage, method=method,
                              DOI=DOI, org_name=org_name, version_ctrl=version_ctrl,
                              outpath=outpath)

    print('Finished processing {}'.format(datafile))

    print('done')
