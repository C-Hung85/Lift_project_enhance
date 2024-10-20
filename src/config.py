import os

Config = {}

Config['files'] = {'data_folder': os.getenv('SATA', 'data/')}
Config['scan_setting'] = {'interval':3}