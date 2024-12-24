import os

Config = {}

Config['files'] = {'data_folder': os.getenv('SATA', 'data/')}
Config['scan_setting'] = {'interval':6}

video_config = {
    '2.mp4': {'start':110},
    '4-2.mp4': {'end':532},
    '5-2.mp4': {'start':23, 'end':396},
    '8-2.mp4': {'end':960},
    '9.mp4': {'start':32, 'end':1334}
}