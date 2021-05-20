import glob, os, re, csv, time
import pandas as pd
import numpy as np
from datetime import datetime
from functools import reduce
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from pathlib import Path


def convert_object_to_float(x):
    """
    Convert input object to float

    :param x: input series/signal as datatype not a float
    :return: same series/signal with datatype float.
    """
    if isinstance(x, str):
        x = x.strip('\'')
        try:
            return float(x)

        except ValueError:
            # print('Non-number value %s was converted to NaN' % x)
            return np.nan

        except:
            print(
                'An unexpected error has occurred finding and converting erroneous data from dataframe')
            return np.nan

    else:
        return x


def correct_data_errors(df):
    """
    Whenever possible convert data type object to float and set it to NaN. Expected input is:
    df: pd.DataFrame
    """
    # check if the dataframe contains erroneous data, and correct it
    df_ = df.copy()
    # loop through columns that have 'object' as datatype
    for col in df.select_dtypes(include=['object']):
        df_[col] = df[col].apply(convert_object_to_float)
    return df_


def parse_textfile(input_text):
    """This funtion converts text into a list of available emission maps, a dict of emission data and a dict of
    metadata. The expected input is:
    input_text: str
     """
    list_available_maps = ''
    dict_data = {}
    dict_meta = {}
    start_data_row = -1
    end_data_row = -1
    for line in input_text:
        if '# AVAILABLE MAPS:' in line:
            list_available_maps = line.replace('# AVAILABLE MAPS:', '')
            list_available_maps = re.split(',', list_available_maps)
            list_available_maps = [avail_map.lstrip() for avail_map in list_available_maps]
        if "# TOTAL KM:" in line:
            if any(char.isdigit() for char in line):
                dict_meta['TOTAL_KM'] = float(line.replace('# TOTAL KM: ', ''))
            else:
                dict_meta['TOTAL_KM'] = line.replace('# TOTAL KM: ', '')
        if "# TOTAL TIME [h]:" in line:
            line = line.replace('# TOTAL TIME [h]: ', '')
            line = line.replace(' hours of data', '')
            if any(char.isdigit() for char in line):
                dict_meta['TOTAL_TIME [h]'] = float(line)
            else:
                dict_meta['TOTAL_TIME [h]'] = line.replace(' ', '')
        if "# NUMBER OF VEHICLES:" in line:
            if any(char.isdigit() for char in line):
                dict_meta['NUMBER_OF_VEHICLES'] = float(line.replace('# NUMBER OF VEHICLES:', ''))
            else:
                dict_meta['NUMBER_OF_VEHICLES'] = line.replace('# NUMBER OF VEHICLES: ', '')
        if "# AVERAGE MILEAGE OF VEHICLES [km]:" in line:
            if any(char.isdigit() for char in line):
                dict_meta['AVERAGE_MILEAGE [km]'] = float(line.replace('# AVERAGE MILEAGE OF VEHICLES [km]: ', ''))
            else:
                dict_meta['AVERAGE_MILEAGE [km]'] = line.replace('# AVERAGE MILEAGE OF VEHICLES [km]: ', '')

        if "# AVAILABLE MAPS:" in line:
            dict_meta['AVAILABLE_MAPS'] = list_available_maps

    for map_subsection in list_available_maps:
        for index, line in enumerate(input_text):
            if "# NOTES:" in line:
                notes = line.replace('# NOTES:', '')

            if "# START DATA {}".format(map_subsection) in line:
                start_data_row = index + 2
            elif "# END {}".format(map_subsection) in line:
                end_data_row = index
            if (start_data_row != -1) & (end_data_row != -1):
                df_data = [sub.split(',') for sub in input_text[start_data_row:end_data_row]]
                df_column_name = input_text[start_data_row - 1].split(',')
                df = pd.DataFrame(df_data, columns=df_column_name)
                # dict_data.update({map_subsection:{'start_data_row':start_data_row,
                #                                   'end_data_row':end_data_row,
                #                                   'df':df}
                dict_data.update({map_subsection: df})
                dict_meta.update({map_subsection: notes})
                start_data_row = -1
                end_data_row = -1
    return list_available_maps, dict_data, dict_meta


def map_textfile_to_dict(input_filename):
    """This function takes the lead in parsing a .map.txt file and converts the read data into a dictionary.
    Expected input:
    input_filename: str
    """
    input_text = open(f'{input_filename}').readlines()
    input_text_processed = [line.strip('\n') for line in input_text]

    # Get list of available maps and dictionary containing data
    list_available_maps, dict_data, dict_meta = parse_textfile(input_text_processed)

    map_dict = {}

    for map_subsection in list_available_maps:

        # Get dataframe
        df = dict_data[map_subsection]

        # Convert all non numbers to numbers
        df = correct_data_errors(df)

        # GET X & Y keys
        x_keys = df['X'].unique()
        y_keys = df['Y'].unique()

        # df group
        df_group = df.groupby(['X', 'Y'])

        # Setup empty matrices
        z_var_ave = np.zeros((len(y_keys), len(x_keys)))
        std_var = np.zeros((len(y_keys), len(x_keys)))
        q25_var = np.zeros((len(y_keys), len(x_keys)))
        q75_var = np.zeros((len(y_keys), len(x_keys)))
        count_var = np.zeros((len(y_keys), len(x_keys)))

        # Loop to assign values to 2d array
        for group_key in df_group.groups.keys():
            x_place = np.where(x_keys == group_key[0])
            y_place = np.where(y_keys == group_key[1])
            z_var_ave[y_place, x_place] = df_group.get_group(group_key)['Z1'].values[0]
            std_var[y_place, x_place] = df_group.get_group(group_key)['Z2'].values[0]
            q25_var[y_place, x_place] = df_group.get_group(group_key)['Z3'].values[0]
            q75_var[y_place, x_place] = df_group.get_group(group_key)['Z4'].values[0]
            count_var[y_place, x_place] = df_group.get_group(group_key)['Z5'].values[0]
        map_dict[map_subsection] = {'z_var_ave': z_var_ave, 'count_var': count_var, 'std_var': std_var,
                                    'q25_var': q25_var,
                                    'q75_var': q75_var, 'max_x': df['X'].max(), 'max_y': df['Y'].max(),
                                    'binsizes': (df['X'][0], df['Y'][0]),
                                    'notes': dict_meta[map_subsection]}
    map_dict['meta'] = {key: dict_meta[key] for key in list(dict_meta.keys())[0:4]}

    return map_dict


def find_duo_taxcode(source_A, source_B):
    """This function searches two folders to find filenames with matching taxonomy codes. It disregards the suffixes
        after the taxonomy code in the filenames. Expected inputs are:
        source_A and source_B: str
    """

    def get_filelist(source):
        """This subfunction searches a directory for taxonomy codes an returns it as a list."""
        list = []
        for file in glob.glob('{}/*.map.txt'.format(source)):
            taxcode = file.split('.', 5)[0]
            taxcode = taxcode.split('\\')[1]
            list.append(taxcode)
            del taxcode
        print("Found {} maps in {}".format(len(list), source))
        return list

    # get all taxonomy codes from source_A and source_B
    list_A = get_filelist(source_A)
    list_B = get_filelist(source_B)

    # get all the duo's
    list_duos = list(set(list_A) & set(list_B))
    print("These taxonomy codes are present in both folders:", *list_duos, sep='\n')
    return list_duos


def parse_file(taxcode, source):
    """This function will parse a textfile and will return it as dictionary."""
    file = glob.glob('{}/{}*.map.txt'.format(source, taxcode))[0]
    map_dict = map_textfile_to_dict(file)
    print("Loaded {}".format(file))
    return map_dict


def find_duo_maps(dict_A, dict_B):
    """This function finds the maps that are to be combined. Expected inputs are:
    dict_A and dict_B: dictionaries"""
    maps_A = list(dict_A.keys())
    maps_B = list(dict_B.keys())
    maps_to_combine = list(set(maps_A) & set(maps_B))
    maps_to_combine.remove('meta')
    print("Map(s) that occur(s) in both sources:\n{}".format(maps_to_combine))
    return maps_to_combine


def convert_resolution(map_A, map_B):
    """This function takes the lead in converting the lowest resolution map to the desired higher resolution.
     The desired higher resolution is based on the resolution of the map that has highest resolution. The expected
     inputs of this function are:
     map_A and map_B: dict.
     The output is a dictionary with the converted map.
     """

    # The RR is the so-called 'circle of confusion. It determines the radius of the circle around a selected bin.
    # All bins within that circle are taken into account for determination of weighing factors and bin values.
    RR = 1.5

    # Initialize dictionary for the converted map
    converted_map = {}

    def get_proc_dict(map_old, map_new, target_res):
        """This subfunction prepares the parameters which are needed to convert the resolution in spline_v3(). The
        expected inputs are:
        map_old and map_new: dict
        target_res: tuple(y,x)"""

        # Initialize a dictionary for the converted map.
        map_dict_proc = {}

        # Get parameters from the maps
        xmax_old = map_old['max_x']
        ymax_old = map_old['max_y']
        xmax_new = map_new['max_x']
        ymax_new = map_new['max_y']

        # Determine the binsize of the low-resolution map
        dx = xmax_old / map_old['z_var_ave'].shape[1]
        dy = ymax_old / map_old['z_var_ave'].shape[0]

        # Convert all matrices in the map
        for var in ['z_var_ave', 'count_var', 'std_var', 'q25_var', 'q75_var']:
            if var == 'count_var':  # the count_var requires different treatment in the conversion process.
                cnt_flag = True
            else:
                cnt_flag = False

            map_dict_proc[var] = spline_v3(input_matrix=map_old[var], xmax=xmax_old, ymax=ymax_old, xxmax=xmax_new,
                                           yymax=ymax_new, dx=dx, dy=dy, target_resolution=target_res, RR=RR,
                                           cnt_flag=cnt_flag)
            print("{} is processed".format(var))

        map_dict_proc['binsizes'] = ((xmax_new / target_res[1]), (ymax_new / target_res[0]))

        return map_dict_proc

    # Determine the highest resolution
    if map_A['z_var_ave'].size > map_B['z_var_ave'].size:
        target_res = map_A['z_var_ave'].shape  # (rows,cols)
        converted_map['B'] = get_proc_dict(map_B, map_A, target_res)
    elif map_A['z_var_ave'].size < map_B['z_var_ave'].size:
        target_res = map_B['z_var_ave'].shape  # (rows,cols)
        converted_map['A'] = get_proc_dict(map_A, map_B, target_res)
    else:
        return "The resolution is identical or there is an error in one of the maps"

    return converted_map


def spline_v3(input_matrix, xmax, ymax, xxmax, yymax, dx, dy, target_resolution, RR, cnt_flag):
    """This function creates the high resolution matrix and calculates the values through spline interpolation for each
     bin within that high-res matrix. In this process the matrix of the low resolution is used as datasource. The
     expected inputs are:
     input_matrix: ndarray
     xmax, ymax, xxmax, yymax, dx,dy: float
     target_resolution: tuple
     RR: float
     cnt_flag: boolean

     The output is a high resolution ndarray.
     """

    # input table
    INCOL = input_matrix.shape[1]  # nr of columns of input table
    INROW = input_matrix.shape[0]  # nr of rows of input table
    xmin = 0
    xmax = xmax  # maximum value on the x-axis of the input table
    ymin = 0
    ymax = ymax  # maximum value on the y-axis of the input table

    # output table
    OUTCOL = target_resolution[1]  # desired nr of columns of output table
    OUTROW = target_resolution[0]  # desired nr of rows of output table
    xxmin = 0
    xxmax = xxmax  # maximum value on the x-axis of the output table
    yymin = 0
    yymax = yymax  # maximum value on the y-axis of the output table

    RR = RR  # "Circle of Confusion"

    A = input_matrix

    # make empty matrix with target resolution
    B = np.zeros((OUTROW, OUTCOL))

    for j in range(0, OUTROW):  # loop over the rows of the new matrix
        yy = yymin + (yymax - yymin) * j / (OUTROW)
        iy = np.floor((INROW) * (yy - ymin) / (ymax - ymin))  # row location on the old grid
        y = ymin + (ymax - ymin) * iy / (INROW)

        for i in range(0, OUTCOL):  # loop over the columns of the new matrix
            xx = xxmin + (xxmax - xxmin) * i / (OUTCOL)
            ix = np.floor((INCOL) * (xx - xmin) / (xmax - xmin))  # column location on the old grid
            x = xmin + (xmax - xmin) * ix / (INCOL)

            z = 0.0
            w = 0.0

            # including a square around the datapoint
            for ii in range(-2, 3):
                for jj in range(-2, 3):

                    r = (y + dy * jj - yy) * (y + dy * jj - yy) / (dy * dy) + (x + dx * ii - xx) * (
                            x + dx * ii - xx) / (dx * dx)
                    if r < RR:
                        rr = 0.3 * (RR - r) * (RR - r)
                    else:
                        rr = 0.0

                    ixx = ix + ii  # column location on the old grid including square around the datapoint
                    iyy = iy + jj  # row location on the old grid including square around the datapoint

                    if (ixx >= 0) and (ixx < INCOL) and (iyy >= 0) and (iyy < INROW):
                        z += rr * A[int(iyy), int(ixx)]  # multiply rr with a value from the old matrix
                        w += rr

            # weighed result
            if w > 0:
                if cnt_flag:
                    B[j, i] = int(np.ceil((z / w) * ((INCOL * INROW) / (OUTCOL * OUTROW))))
                else:
                    B[j, i] = z / w  # Assign calculated value to the new matrix

        else:
            B[j, i] = 0.0

    return B


def clean_array(dict_input):
    """This function remove unnecessary rows and columns where there are only zeros. The function scans the matrix of
    count_var from bottom to top and from right to left and deletes the row or column as long as the sum of row or column
    is zero. The expected input is a dict. The output is a shrunk dict, if the mentioned conditions were met.
    """

    array = dict_input['count_var']

    for row in reversed(range(0, array.shape[0])):
        if sum(array[row]) == 0:
            for var in ['z_var_ave', 'count_var', 'std_var', 'q25_var', 'q75_var']:
                dict_input[var] = np.delete(dict_input[var], row, 0)
            continue
        else:
            break

    for col in reversed(range(0, array.shape[1])):
        if sum(array[:, col]) == 0:
            for var in ['z_var_ave', 'count_var', 'std_var', 'q25_var', 'q75_var']:
                dict_input[var] = np.delete(dict_input[var], col, 1)
            continue
        else:
            break
    return dict_input


def nrule_combining(map_A, map_B):
    """This function combines two maps (matrices) of equal size using the n-rule. The weighfactor of a map is determined
    by a combination of the total count of datapoints of map_A and 2% of map_B.

    The expected inputs are:
    map_A and map_B: dict

    The output is a combined and cleaned dict."""

    combined_nrule_map = {}

    N_a = np.sum(map_A['count_var'])
    N_b = np.sum(map_B['count_var'])
    n_A = np.sqrt(N_a + (0.02 * N_b))
    n_B = np.sqrt(N_b + (0.02 * N_a))

    factor_A = n_A / (n_A + n_B)
    factor_B = n_B / (n_A + n_B)

    for var in ['z_var_ave', 'count_var', 'std_var', 'q25_var', 'q75_var']:
        print("processing {}".format(var))
        combined_nrule_map[var] = map_A[var] * factor_A + map_B[var] * factor_B
        if var == "count_var":
            combined_nrule_map[var] = np.ceil(combined_nrule_map['count_var'])

    try:
        combined_nrule_map['binsizes'] = map_A['binsizes']
    except:
        combined_nrule_map['binsizes'] = map_B['binsizes']

    combined_nrule_map_cln = clean_array(combined_nrule_map)

    return combined_nrule_map_cln


def num(s):
    """Function will try to convert the variable to a float. If not possible it will return the original variable."""
    try:
        return float(s)
    except:
        return s


def get_uncombined_maps(map_dict_A, map_dict_B, maps_to_combine):
    """This function looks for emission maps that did not match and are therefore not combined. These maps will also
    be saved in the final outputfile that contains combined and uncombined maps. Expected inputs are:
    map_dict_A and map_dict_B: dict
    maps_to_combine: list

    The expected output are two dictionaries containing both combined and uncombined maps.
    """
    uncombined_list_A = set(map_dict_A.keys()) - set(maps_to_combine)
    uncombined_list_B = set(map_dict_B.keys()) - set(maps_to_combine)

    uncombined_A = {key: map_dict_A[key] for key in uncombined_list_A}
    uncombined_B = {key: map_dict_B[key] for key in uncombined_list_B}
    uncombined_maps = {}
    uncombined_maps.update(uncombined_A)
    uncombined_maps.update(uncombined_B)

    del map_dict_A, map_dict_B

    combined_meta = {}
    for key in set(uncombined_A['meta']) & set(uncombined_B['meta']):
        sumlist = [num(uncombined_A['meta'][key]), num(uncombined_B['meta'][key])]
        sumlist_dig = [float(i) for i in sumlist if type(i) == float or i.isdigit()]

        if len(sumlist_dig) == 0:
            combined_meta[key] = 'n/a'
            continue
        elif len(sumlist_dig) == 1:
            combined_meta[key] = sumlist_dig[0]
        ele = sum(sumlist_dig)
        combined_meta[key] = ele

    uncombined_maps['meta'] = combined_meta

    del uncombined_list_A, uncombined_list_B, uncombined_A, uncombined_B, combined_meta

    # clean the notes of the uncombined maps
    for map in uncombined_maps.keys():
        if map != 'meta':
            uncombined_maps[map]['notes'] = re.sub(r"[ \[\]]", "", uncombined_maps[map]['notes'])

    # clean the array's
    for map in uncombined_maps.keys():
        if map != 'meta':
            print("Clean {} with shape {}".format(map, uncombined_maps[map]['z_var_ave'].shape))
            cleaned_map = clean_array(uncombined_maps[map])
            print("Cleaned to shape {}\n".format(cleaned_map['z_var_ave'].shape))
            uncombined_maps[map] = cleaned_map

    return uncombined_maps


def convert_array_to_df(emission_map):
    """
    This function converts the emission map dict to a DataFrame where
    - 'emission_map' is a dictionary containing at least 'z_var_ave', 'count_var','std_var','q25_var' and'q75_var
    """

    def reform_df(df, nr):
        """This subfunction will reform the format of the dataframe is such a way that it can be saved in the .map.txt
        file later on. The expected input is:
        df: pd.DataFrame
        nr: int

        The output is a dataframe that can contains all data of one map and is ready to be written to a .map.txt file"""

        df_temp_fin = pd.DataFrame()
        for key, value in df.items():
            df_temp = pd.DataFrame()
            df_temp['Y'] = value.index  # CO2_mF [g/s]
            df_temp['X'] = key  # velocity_filtered [km/h]
            df_temp['Z{}'.format(nr)] = df[key].values  # avgNOx [mg/s]
            df_temp = df_temp[['X', 'Y', 'Z{}'.format(nr)]]
            df_temp_fin = df_temp_fin.append(df_temp)
        return df_temp_fin

    numbering = {'z_var_ave': 1, 'std_var': 2, 'q25_var': 3,
                 'q75_var': 4, 'count_var': 5}

    map_df = []
    for var in numbering.keys():
        if type(emission_map[var]) == np.ndarray:
            map = emission_map[var]

            x_axis = np.arange(emission_map['binsizes'][0],
                               emission_map['binsizes'][0] * map.shape[1] + 1,
                               emission_map['binsizes'][0])
            y_axis = np.arange(emission_map['binsizes'][1],
                               (emission_map['binsizes'][1] * map.shape[0]) + emission_map['binsizes'][1],
                               emission_map['binsizes'][1])

            # check if shape of axis and indices are the same
            if map.shape[1] != len(x_axis):
                x_axis = x_axis[:map.shape[1]]
            elif map.shape[0] != len(y_axis):
                y_axis = y_axis[:map.shape[0]]

            ## Make Table for .map.txt outputfile
            df = pd.DataFrame(data=map, index=y_axis, columns=x_axis)
            reformed_df = reform_df(df, numbering[var])

            map_df.append(reformed_df)

    final_df = reduce(lambda left, right: pd.merge(left, right, on=['X', 'Y']), map_df)

    return final_df


def write_to_maptxt(map_dict, output_loc, taxcode, version_ctrl, organisation):
    """This function will write the new .map.txt file of the emission map that contains both combined maps and
    uncombined maps. The expected inputs are:
    map_dict: dict
    output_loc, taxcode and version_ctrl: str

    The output is a new .map.txt file in the specified folder"""

    # convert emission maps to df
    for variant in ['combined_maps', 'uncombined_maps']:
        for emission_map in map_dict[variant]:
            map_dict[variant][emission_map]['map_df'] = convert_array_to_df(
                emission_map=map_dict[variant][emission_map])

    map_list = list(map_dict['combined_maps'].keys()) + list(map_dict['uncombined_maps'].keys())
    t = [map_dict['combined_maps'][map]['notes'] for map in map_dict['combined_maps'].keys()]

    # Get only 1 NOTES from one of the maps
    notes = np.unique(t)[0]
    notes_meta = notes + ". Combined maps: {}".format(','.join(list(map_dict['combined_maps'].keys())))

    # start writing the .map.txt file
    date = datetime.now().strftime("%d-%m-%Y")
    output_dir = "{}/maps/{}".format(output_loc, date)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(r'{}/{}.{}-{}.map.txt'.format(output_dir, str(taxcode), organisation, version_ctrl), 'w',
              newline='') as f:

        # ============================ META =========================================
        writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE)
        writer.writerow(["################################################################################"])
        writer.writerow(["# START META"])
        writer.writerow(["#"])
        writer.writerow(["# ID: {}".format(str(taxcode))])
        writer.writerow(["# NOTES: [{}]".format(notes_meta)])
        writer.writerow(["# TOTAL KM: {:}".format(map_dict['meta']['TOTAL_KM'])])
        writer.writerow(["# TOTAL TIME [h]: {:} ".format(map_dict['meta']['TOTAL_TIME [h]'])])
        writer.writerow(["# NUMBER OF VEHICLES: {}".format(map_dict['meta']['NUMBER_OF_VEHICLES'])])
        if type(map_dict['meta']['AVERAGE_MILEAGE [km]']) == float:
            map_dict['meta']['AVERAGE_MILEAGE [km]'] = str(int(map_dict['meta']['AVERAGE_MILEAGE [km]']))
        else:
            map_dict['meta']['AVERAGE_MILEAGE [km]'] = 'n/a'
        writer.writerow(['# AVERAGE MILEAGE OF VEHICLES [km]: {}'.format(map_dict['meta']['AVERAGE_MILEAGE [km]'])])
        writer.writerow(["# REFERENCE DOI: {}".format('10.5281/zenodo.4268034')])
        writer.writerow(["# AVAILABLE MAPS: {}".format(', '.join(map_list))])
        writer.writerow(["# END META"])
        f.close()
    time.sleep(1)

    # ============================ MAPS =========================================
    for maptype in ['combined_maps', 'uncombined_maps']:
        for mapname in map_dict[maptype]:
            emission = str.replace(mapname.split("-")[2], " MEAN ", '')
            with open(r'{}/{}.{}-{}.map.txt'.format(output_dir, str(taxcode), organisation, version_ctrl), 'a',
                      newline='') as f:
                writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE)
                writer.writerow(["################################################################################"])
                writer.writerow(["# START {}".format(mapname)])
                writer.writerow(["#"])
                # writer.writerow(["# NOTES: [{}]".format(re.sub(r"[\'{\}]", '', str(map_dict_comb)).split(":")[1])])
                writer.writerow(["# NOTES: [{}]".format(map_dict[maptype][mapname]['notes'])])
                if 'VEHICLE SPEED' in mapname:
                    writer.writerow(['# XLABEL: Vehicle Speed upper bin limit [km/h]'])
                elif 'ENGINE SPEED' in mapname:
                    writer.writerow(['# XLABEL: Engine Speed upper bin limit [RPM]'])
                writer.writerow(['# YLABEL: CO2 upper bin limit [g/s]'])
                writer.writerow(['# Z1LABEL: Average {}emissions [mg/s]'.format(emission)])  # map_tbl_df
                writer.writerow(['# Z2LABEL: Standard deviation {}emissions [mg/s]'.format(emission)])
                writer.writerow(['# Z3LABEL: 0.25 quantile {}emissions [mg/s]'.format(emission)])
                writer.writerow(['# Z4LABEL: 0.75 quantile {}emissions [mg/s]'.format(emission)])
                writer.writerow(['# Z5LABEL: Count per bin [#]'])
                writer.writerow(['# START DATA {}'.format(mapname)])
                f.close()
            time.sleep(1)
            # write df data
            map_dict[maptype][mapname]['map_df'].to_csv(
                r'{}/{}.{}-{}.map.txt'.format(output_dir, str(taxcode), organisation, version_ctrl),
                mode='a', header=True, index=False)

            with open(r'{}/{}.{}-{}.map.txt'.format(output_dir, str(taxcode), organisation, version_ctrl), 'a',
                      newline='') as f:
                writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE)
                writer.writerow(['# END {}'.format(mapname)])
                f.close()

    return


def make_graphs(map_dict, output_loc, taxcode, version_ctrl, organisation):
    """This function will create new plots of the converted maps. The expected inputs are:
    map_dict: dict
    output_loc, taxcode and version_ctrl: str

    The output are PNG graphs of the converted maps in the specified folder."""

    def plot(z_var_ave, min_x, max_x, min_y, max_y, min_z, max_z, x_sig_name, y_sig_name, z_sig_name,
             v_scale):
        # Plot the Z variable in the color bar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        H = z_var_ave.copy()
        if v_scale == 'log':
            plt.imshow(H, interpolation='nearest', cmap='jet', extent=(min_x, max_x, max_y, min_y), aspect='auto',
                       norm=LogNorm(vmin=min_z, vmax=max_z))
        elif v_scale == 'linear':
            plt.imshow(H, interpolation='nearest', cmap='jet', extent=(min_x, max_x, max_y, min_y), aspect='auto',
                       norm=Normalize(vmin=min_z, vmax=max_z))

        ax = plt.gca()
        ax.set_xlabel(x_sig_name)
        ax.set_ylabel(y_sig_name)

        ax.invert_yaxis()  # 0 at the bottom

        cbar = plt.colorbar()  # show colorbar z axis
        cbar.set_label(z_sig_name)

        return ax

    date = datetime.now().strftime("%d-%m-%Y")
    comb = 'combined_maps'

    for emission_map in map_dict[comb]:
        for var in ['z_var_ave', 'count_var', 'std_var', 'q25_var', 'q75_var']:
            min_x = map_dict[comb][emission_map]['map_df']['X'].min()
            max_x = map_dict[comb][emission_map]['map_df']['X'].max()
            min_y = map_dict[comb][emission_map]['map_df']['Y'].min()
            max_y = map_dict[comb][emission_map]['map_df']['Y'].max()
            min_z = 1e-2
            max_z = np.nanmax(map_dict[comb][emission_map][var])
            if 'ENGINE SPEED' in emission_map.split('-')[0]:
                x_sig_name = 'Engine Speed [RPM]'
            elif 'VEHICLE SPEED' in emission_map.split('-')[0]:
                x_sig_name = 'Vehicle Speed [km/h]'
            else:
                x_sig_name = emission_map.split('-')[0]

            emission = emission_map.split('-')[2].split(' ')[2]
            if 'z_var_ave' in var:
                z_sig_name = "Average {} emission [mg/s]".format(emission)
            elif 'count_var' in var:
                z_sig_name = 'Count [-]'
            elif 'std_var' in var:
                z_sig_name = 'Standard deviation [mg/s]'
            elif 'q' in var:
                z_sig_name = '{}th quartile [mg/s]'.format(var.split('_')[0][1:])
            ax = plot(map_dict[comb][emission_map][var],
                      min_x=min_x, max_x=max_x,
                      min_y=min_y, max_y=max_y,
                      min_z=min_z, max_z=max_z,
                      x_sig_name=x_sig_name, y_sig_name='CO2 emission [g/s]',
                      z_sig_name=z_sig_name, v_scale='log')

            if comb == 'combined_maps':
                title = "{} -{} {} combined".format(taxcode, map_dict[comb][emission_map]['notes'].split(',')[0],
                                                    map_dict[comb][emission_map]['notes'].split(',')[1])
            else:
                title = "{}-{}".format(taxcode, map_dict[comb][emission_map]['notes'])
            ax.set_title(title)

            output_dir = "{}/graphs/{}".format(output_loc, date)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            plt.savefig(Path(r'{}/{}_{}_{}_{}.{}-{}.png'.format(output_dir, str(taxcode), var, emission,
                                                                emission_map.split("-")[0].strip(" "), organisation,
                                                                version_ctrl)))

            plt.clf()
    return
