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

# Module func_flex_bin.py
# Module to determine the bin size used to generate Augmented Emission Map (AEM), depending on data availability
# The algorithm is based on the "Island Counting Problem", where the solution is publicly available on :
# https://leetcode.com/problems/number-of-islands/discuss/121164/Python-BFS-and-DFS-solution

import collections
import numpy as np
from scripts import xymatrix
import pandas as pd

# Threshold value to determine the sufficient map area coverage for deciding bin size
THRESHOLD = 90.


def flexible_bin_sizing_average(df, z_sig, x_sig, y_sig, min_x, max_x, min_y, max_y, x_scale, y_scale, bin_threshold):
    """
    Function to determine bin size to accomodate sparse measurement data
    The output is dictionary containing average values of emission per bin and parameters required for plotting and
    writing into an output text file
     """
    x_bins = int((max_x - min_x) / float(x_scale))
    y_bins = int((max_y - min_y) / float(y_scale))

    z_var, count_var, z_var_ave, std_var, Q25_var, Q75_var, x_axis_csv, y_axis_csv, map_tbl_df = \
        xymatrix.getmatrix(df, z_sig, x_sig, y_sig, min_x, max_x, min_y, max_y, x_bins, y_bins, bin_threshold)

    pct_area = maxAreaOfIsland(z_var_ave.copy()) / np.count_nonzero(z_var_ave.copy()) * 100
    print('Starting % area is {:.2f}'.format(pct_area))

    while pct_area < THRESHOLD:
        x_scale = x_scale * np.sqrt(2)
        y_scale = y_scale * np.sqrt(2)

        x_bins = int((max_x - min_x) / float(x_scale))
        y_bins = int((max_y - min_y) / float(y_scale))

        z_var, count_var, z_var_ave, std_var, Q25_var, Q75_var, x_axis_csv, y_axis_csv, map_tbl_df = \
            xymatrix.getmatrix(df, z_sig, x_sig, y_sig, min_x, max_x, min_y, max_y, x_bins, y_bins, bin_threshold)

        pct_area = maxAreaOfIsland(z_var_ave.copy()) / np.count_nonzero(z_var_ave.copy()) * 100
        print('Current % area is {:.2f} with x_scale:{:.2f}, y_scale:{:.2f}'.format(pct_area, x_scale, y_scale))

    maps_dict = {'z_var': z_var, 'count_var': count_var, 'z_var_ave': z_var_ave,
                 'std_var': std_var, 'Q25_var': Q25_var, 'Q75_var': Q75_var,
                 'x_axis_csv': x_axis_csv, 'y_axis_csv': y_axis_csv, 'x_bins': len(x_axis_csv),
                 'y_bins': len(y_axis_csv),
                 'min_x': min_x, 'max_x': x_axis_csv[-1], 'min_y': min_y, 'max_y': y_axis_csv[-1],
                 'total_km': round((df['velocity'].sum() / 3600), 1),
                 'total_time': round(len(df['velocity']) / 3600, 1),
                 'map_tbl_df': map_tbl_df
                 }
    return maps_dict


def numIslands(grid):
    """
    Islands counting algorithm
    This function are publicly available from: https://leetcode.com/problems/number-of-islands/discuss/121164/Python-BFS-and-DFS-solution
    """
    if not len(grid):
        return 0

    nofisland = 0
    visited = {}
    nrow = len(grid)
    ncol = len(grid[0])

    q = collections.deque()
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for x in range(nrow):
        for y in range(ncol):
            if grid[x][y] > 0 and (x, y) not in visited:
                nofisland += 1
                visited[(x, y)] = True
                q.append((x, y))

                while len(q) > 0:
                    tempx, tempy = q.popleft()
                    for dx, dy in dirs:
                        xx, yy = tempx + dx, tempy + dy
                        if 0 <= xx < nrow and 0 <= yy < ncol and grid[xx][yy] > 0 and (xx, yy) not in visited:
                            q.append((xx, yy))
                            visited[(xx, yy)] = True
    return nofisland


def maxAreaOfIsland(grid):
    """
    Calculate the area of an island
    This function are publicly available from: https://leetcode.com/problems/number-of-islands/discuss/121164/Python-BFS-and-DFS-solution
    """

    def is_valid(row, col):
        if row < 0 or col < 0 or \
                row >= len(grid) or col >= len(grid[0]) or grid[row][col] == 0:
            return False
        else:
            return True

    def bfs(row, col):
        if not is_valid(row, col):
            return
        count = 0
        queue = collections.deque()
        queue.append((row, col))
        while queue:
            nrow, ncol = queue.popleft()
            if not is_valid(nrow, ncol):
                continue
            grid[nrow][ncol] = 0
            count += 1
            ## get all of the neighbours
            queue.append((nrow + 1, ncol))
            queue.append((nrow - 1, ncol))
            queue.append((nrow, ncol - 1))
            queue.append((nrow, ncol + 1))
        return count

    total_area = sum([sum(grid[i]) for i in range(len(grid))])

    if total_area == 0:
        return 0
    max_area = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] > 0:
                area = bfs(row, col)
                if area > max_area:
                    max_area = area
    return max_area
