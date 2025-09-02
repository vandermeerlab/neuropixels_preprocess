analog_channels = {'XA0': [1, 3], 'XA1': [1, 3], 'XA2': [1, 3], 'XA3': [1, 3], 'XA4': [1, 3], 'XA5': [1, 3], 'XA6': [1,3], 'XA7': [1,3]}
digital_channels = ['XD1', 'XD4']
num_analog_channels = 8 # Change after looking at meta-files

# Other parameters
catgt_path = 'D:\\CatGT-win' # The location where CatGT.exe exists
source_path = 'E:\\' # The parent directory of the session folder
run_name = 'M539-2024-08-14' # The thing that is suffixed by the top level 'g0' folder
dest_path = 'E:\\odor-pixels\\Cohort5\\M539\\' # Where catgt'

command_prefix = '{}/CatGT.exe -dir={} -run={} -g=0 -t=0 -prb_fld -t_miss_ok -ni'.format(catgt_path, source_path, run_name)

analog_params = '';
for key in analog_channels:
    analog_params += '-xa=\'0,0,{},{},{},0\' -xia=\'0,0,{},{},{},0\' '.format(key[2:],analog_channels[key][0], analog_channels[key][1], \
    key[2:],analog_channels[key][0], analog_channels[key][1])

digital_params = '';
for item in digital_channels:
    digital_params += '-xd=\'0,0,{},{},0\' -xid=\'0,0,{},{},0\' '.format(num_analog_channels, item[2:], num_analog_channels, item[2:])

command_suffix = '-dest={}'.format(dest_path)

print('{} {} {} {}'.format(command_prefix, analog_params, digital_params, command_suffix))