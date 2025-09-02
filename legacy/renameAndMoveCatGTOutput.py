import os
import shutil
import pathlib

# Parameters
src_path = 'E:\\odor-pixels\\Cohort5\\M539\\catgt_M539-2024-08-14_g0\\' # Location where all sync pulses and edge extraction files live
dest_path = 'E:\\odor-pixels\\Cohort5\\M539\\Training\\M539-2024-08-14\\'# Location where

# Analog and digital channels to be renamed and moved
analog_channels = ['XA0', 'XA1', 'XA2', 'XA3', 'XA4', 'XA5', 'XA6']
digital_channels = ['XD1','XD4']


# Parameter sanity checker
file_list = []
for root, dirs, files in os.walk(src_path):
    for file in files:
       file_list.append(file)
analog_in_files = []
analog_out_files = []
for ch in analog_channels:
    this_on_file = [x for x in file_list if 'nidq.xa_'+ch[2:] in x and '0.txt' in x]
    assert len(this_on_file) == 1, 'Something wrong with ON events from channel {}, expected 1, found {}'.format(ch, this_on_file)
    analog_in_files.extend(this_on_file)
    analog_out_files.append(ch+'_ON.txt')
    this_off_file = [x for x in file_list if 'nidq.xia_'+ch[2:] in x and '0.txt' in x]
    assert len(this_off_file) == 1, 'Something wrong with OFF events from channel {}, expected 1, found {}'.format(ch, this_off_file)
    analog_in_files.extend(this_off_file)
    analog_out_files.append(ch+'_OFF.txt')
digital_in_files = []
digital_out_files = []
for ch in digital_channels:
    this_on_file = [x for x in file_list if 'nidq.xd' in x and ch[2:]+'_0.txt' in x]
    assert len(this_on_file) == 1, 'Something wrong with ON events from channel {}, expected 1, found {}'.format(ch, this_on_file)
    digital_in_files.extend(this_on_file)
    digital_out_files.append(ch+'_ON.txt')
    this_off_file = [x for x in file_list if 'nidq.xid' in x and ch[2:]+'_0.txt' in x]
    assert len(this_off_file) == 1, 'Something wrong with OFF events from channel {}, expected 1, found {}'.format(ch, this_off_file)
    digital_in_files.extend(this_off_file)
    digital_out_files.append(ch+'_OFF.txt')  

# Create destination folders if they don't exist
pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True)

for infile, outfile in zip(analog_in_files,analog_out_files):
	shutil.copy(src_path+infile, dest_path+outfile)


for infile, outfile in zip(digital_in_files,digital_out_files):
	shutil.copy(src_path+infile, dest_path+outfile)