"""This script will:
    - look in two folders for identical taxonomy codes in .map.txt files (basemaps). This is the input.
    - parse the .map.txt files with identical taxonomy codes
    - Check the resolution of both maps
    - convert the lowest resolution map to the higher resolution, taking into account the binsizes.
    - combine the two maps into one new map using the n-rule
    - write a new .map.txt file containing the combined data and create new graphs.
"""

# import functions
from helpers import *

if __name__ == '__main__':
    # ==============START INPUT FIELDS==============
    # Parameters for filenaming
    org_name = 'TNO'
    version_ctrl = '1'
    # ==============END INPUT FIELDS==============

    folder_A = r'source_A'
    folder_B = r'source_B'

    # the maps wil be saved in /combined_maps/maps/<date>/ and the graphs will be save in /combined_maps/graphs/<date>/
    output_folder = r'combined_maps'

    # Find common taxonomy codes in input folders
    duo_taxonomy = find_duo_taxcode(folder_A, folder_B)

    # loop over the taxonomy codes that have a basemap of two sources (taxonomy duo)
    for counter, duo in enumerate(duo_taxonomy):

        # make an empty dict for the combined map
        combined_duo = {}

        print('\nProcessing {}\n{} of {}'.format(duo, counter+1, len(duo_taxonomy)))

        # Parse the textfiles into dictionaries
        taxcode = duo
        map_dict_A = parse_file(taxcode, folder_A)
        map_dict_B = parse_file(taxcode, folder_B)

        # For a certain taxonomy code, check which basemaps are present in both sources (map duo)
        maps_to_combine = find_duo_maps(map_dict_A, map_dict_B)

        # create an empty dictionary for combined maps per taxonomy code
        combined_duo[duo] = {}
        combined_duo[duo]['combined_maps'] = {}

        # loop over the basemaps to combine
        for mapname in maps_to_combine:

            # Check resolutions and convert lowest-resolution map to highest resolution with adaptable weighing function method.
            converted_map = convert_resolution(map_dict_A[mapname], map_dict_B[mapname])

            # combine the highest resolution map with the converted map using the n-rule
            if 'A' in converted_map.keys():
                combined_duo[duo]['combined_maps'][mapname] = nrule_combining(map_A=converted_map['A'],
                                                                              map_B=map_dict_B[mapname])

            elif 'B' in converted_map.keys():
                combined_duo[duo]['combined_maps'][mapname] = nrule_combining(map_A=map_dict_A[mapname],
                                                                              map_B=converted_map['B'])
                # combined_duo[duo]['converted_map'] = converted_map['B']  # for plotting

            # Combine and clean the NOTES from both emission maps
            t = [map_dict_A[mapname]['notes'], map_dict_B[mapname]['notes']]
            notes = "{}".format(re.sub(r"[\[\]]", "", ",".join([item for item in t])))

            combined_duo[duo]['combined_maps'][mapname].update({'notes': notes})

        del converted_map, mapname

        # Add the non-combined maps of the input maps to the output map
        print('Adding the uncombinable maps to the output map...')
        uncombined_maps = get_uncombined_maps(map_dict_A, map_dict_B, maps_to_combine)
        combined_duo[duo]['uncombined_maps'] = uncombined_maps

        del uncombined_maps, map_dict_A, map_dict_B, maps_to_combine

        combined_duo[duo]['meta'] = combined_duo[duo]['uncombined_maps']['meta']

        del combined_duo[duo]['uncombined_maps']['meta']

        # Rewrite the .map.txt output file.
        #   - in the notes it's mentioned which maps are combined, inlcuding names of sources.
        print('Creating the .map.txt output of {}.'.format(duo))

        # create a function which accounts for the (for example) P_4_ALL_ALL_ALL dictionary, to write
        # a .map.txt file for the combined map.
        write_to_maptxt(map_dict=combined_duo[duo], output_loc=output_folder, taxcode=duo, version_ctrl=version_ctrl, organisation=org_name)
        print('Finished map.txt file of {}\n'.format(duo))

        # Create graphs of the combined maps
        print('Creating the graphs output of {}.'.format(duo))
        make_graphs(map_dict=combined_duo[duo], output_loc=output_folder, taxcode=duo, version_ctrl=version_ctrl, organisation=org_name)
        print("Finished graphs of {}".format(duo))
    print('done')
