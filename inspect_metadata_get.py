# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:52:10 2024

@author: yusufqq
"""

from pylibCZIrw import czi as pyczi
from matplotlib import pyplot as plt
import time
def get_metadata(czidoc, channel_to_use):
    
    # Read out some metadata
    metadata = czidoc.metadata['ImageDocument']['Metadata'] # These two layers are pro-forma, there is nothing else in here
    
    Pixels_in_x = float(czidoc.total_bounding_rectangle[2])
    Pixels_in_y = float(czidoc.total_bounding_rectangle[3])
    # try:
    #     Pixel_size_nm = float(metadata['Scaling']['Items']['Distance'][0]['Value']) * 1E9
    #     Pixel_dwell_time_us = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel'][channel_to_use]['LaserScanInfo']['PixelTime']) * 1E6
    #     Frame_time_s = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel'][channel_to_use]['LaserScanInfo']['FrameTime'])
    #     line_time_ms = Pixel_dwell_time_us*Pixels_in_x*1e-3
    # except:
    #     Pixel_size_nm = float(metadata['Scaling']['Items']['Distance'][0]['Value']) * 1E9
    #     Pixel_dwell_time_us = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel']['LaserScanInfo']['PixelTime']) * 1E6
    #     Frame_time_s = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel']['LaserScanInfo']['FrameTime'])
    #     line_time_ms = Pixel_dwell_time_us*Pixels_in_x*1e-3

    return  metadata



if __name__ == '__main__':
    metadata_path = r'X:\AlNahas_Kareem\Raquel_mastersproject\20260220_sfcs\New-09_xy.czi'
    # with pyczi.open_czi(metadata_path) as czidoc:
    #     # Pixel_size_nm,Pixel_dwell_time_us,line_time_ms = get_metadata(czidoc, 0)
    #     # total_bounding_rectangle = czidoc.total_bounding_rectangle
    #     metadata = czidoc.metadata['ImageDocument']['Metadata']
    #     # data_frame = czidoc.read(roi = total_bounding_rectangle,
    #     #                          plane = {'C':0})
    #     # data_frame = data_frame.reshape([data_frame.shape[0], data_frame.shape[1]])
    #     # print(Pixel_size_nm)
    #     # print(Pixel_dwell_time_us)
    #     # print(line_time_ms)

    path = "your_file.czi"
    with pyczi.open_czi(metadata_path) as czidoc:
        metadata = get_metadata(czidoc, 0)
        # dictionary: {scene_index: (x, y, w, h)}
        # data_frame = czidoc.read(scene = 1, plane = {'C':0})
        # scene_boxes = czi.scenes_bounding_rectangle
        # print(data_frame)          # see all regions

    pass