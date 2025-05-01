import os

Config = {}

Config['files'] = {'data_folder': os.getenv('SATA', 'data/')}
Config['scan_setting'] = {'interval':6}

video_config = {
    '2.mp4': {'start':110},
    '4-2.mp4': {'end':532},
    '5-2.mp4': {'start':23, 'end':396},
    '8-2.mp4': {'end':960},
    '9.mp4': {'start':32, 'end':1334},
    '6.mp4': {'end':618},
    '11.mp4': {'end':293},
    '12.mp4': {'start':152, 'end':1181},
    '7.mp4':{'end':1367},
    '10.mp4':{'start':20},
    '13.mp4':{'end':801},
    '14.mp4':{'start':19, 'end':196},
    '15.mp4':{'start':262, 'end':789},
    '17.mp4':{'start':25},
    '18.mp4':{'start':18, 'end':257}
}