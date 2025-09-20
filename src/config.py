import os

Config = {}

Config['files'] = {'data_folder': os.getenv('SATA', './')}
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
    '7.mp4': {'end':1367}, ####
    '10.mp4': {'start':20},
    '13.mp4': {'end':801},
    '14.mp4': {'start':19, 'end':196},
    '15.mp4': {'start':262, 'end':789},
    '17.mp4': {'start':25},
    '18.mp4': {'start':18, 'end':257},
    '16.mp4': {'start':284}, ####
    '19.mp4': {'end':610},
    '21.mp4': {'start':30, 'end':798},
    '22.mp4': {'start':35, 'end':650},
    '23.mp4': {'start':30, 'end':660}, ####
    '28.mp4': {'start':10, 'end':15*60+55},
    '29.mp4': {'start':40, 'end':12*60+30},
    '30.mp4': {'start':30, 'end':15*60},
    '31-1.mp4': {'start':10, 'end':12*60+16},
    '31-2.mp4': {'start':28*60},
    '34.mp4': {'end': 3*60+36}, 
    '36-1.mp4': {'end': 2*60+24}, 
    '36-2.mp4': {'start':7*60+4}, 
    '37.mp4': {'start':10}, 
    '38-1.mp4': {'start':16, 'end': 8*60}, 
    '38-2.mp4': {'start':8*60+55, 'end': 13*60+15}, 
    '39.mp4': {'start':7, 'end':16*60+15}, 
    '40.mp4': {'end':6*60+55}, 
    '41.mp4': {'end':18*60+30}, 
    '42.mp4': {'end':14*60}, 
    '43.mp4': {'start':60, 'end':12*60}, 
    '45.mp4': {'start':5, 'end':12*60+15}, 
    '46.mp4': {'start':5, 'end':13*60+2}
}