import sys
import json
import os

def sanity_check(settings):
    if not os.path.exists(settings['data_path']):
        sys.exit('Error: data_path=\'{}\' does not exist!'.format(settings['data_path']))

    # TODO: add more stuff here as needed.

print('Loading settings file \'{}\'...'.format(sys.argv[1]))
with open(sys.argv[1]) as settings_file:   
    settings = json.load(settings_file)

sanity_check(settings)