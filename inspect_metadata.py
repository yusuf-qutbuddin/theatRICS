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
    try:
        Pixel_size_nm = float(metadata['Scaling']['Items']['Distance'][0]['Value']) * 1E9
        Pixel_dwell_time_us = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel'][channel_to_use]['LaserScanInfo']['PixelTime']) * 1E6
        Frame_time_s = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel'][channel_to_use]['LaserScanInfo']['FrameTime'])
        line_time_ms = Pixel_dwell_time_us*Pixels_in_x*1e-3
    except:
        Pixel_size_nm = float(metadata['Scaling']['Items']['Distance'][0]['Value']) * 1E9
        Pixel_dwell_time_us = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel']['LaserScanInfo']['PixelTime']) * 1E6
        Frame_time_s = float(metadata['Information']['Image']['Dimensions']['Channels']['Channel']['LaserScanInfo']['FrameTime'])
        line_time_ms = Pixel_dwell_time_us*Pixels_in_x*1e-3

    return Pixel_size_nm, Pixel_dwell_time_us, line_time_ms, metadata



if __name__ == '__main__':
    # metadata_path = r'Z:\_Data\positions_image.czi'
    # with pyczi.open_czi(metadata_path) as czidoc:
    #     _,_,_,metadata = get_metadata(czidoc, 0)
    #     total_bounding_rectangle = czidoc.total_bounding_rectangle
            
    #     data_frame = czidoc.read(roi = total_bounding_rectangle,
    #                              scene = 2,
    #                              plane = {'C':0})
    #     data_frame = data_frame.reshape([data_frame.shape[0], data_frame.shape[1]])
    #     print(data_frame.mean())
    #     plt.imshow(data_frame)
    pass