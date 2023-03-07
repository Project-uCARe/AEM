Steps to follow for combining two emission maps of different sources with identical taxonomy codes:

1.Make sure you have installed Python3 including packages (see requirements)
2.Copy all contents of the Sharepoint folder "5. Script for combining emission maps" to your local directory.
3.In the folder "source_A" place all basemap (.map.txt) files of one source. You can check the example file "D_6_1499_70_EXA.ABC-1.map.txt". Don't forget to remove the example file when running the code.
4.In the folder "source_B" place all basemap (.map.txt) files of the other source you wish to combine with. You can check the example file "D_6_1499_70_EXA.ABC-1.map.txt". Don't forget to remove the example file when running the code.
4.Open "combine_maps.py". Enter the "org_name" (e.g. EMPA, LAT, TNO, etc.) and the "version_ctrl". This can be found at line number 16 and 17.
5.Run "combine_maps.py" in the preferred IDE. In the command prompt, this can be also executed by typing "python combine_maps.py" when the working directory is set on "4. Script for combining emission maps".
6.The combined emission maps txt files will be placed in "\combined_maps\maps\<date>" and the corresponding graphs will be placed in "\combined_maps\graphs\<date>".