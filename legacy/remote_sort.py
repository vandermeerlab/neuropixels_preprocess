import spikeinterface.full as si
import numpy as np
import os
import json
import requests

remote_recording_file = 'M489-2024-08-05_g0_tcat.imec0.ap.bin' # CHANGE THIS ACCORDINGLY
remote_recording_stream = 'imec0.ap' # CHANGE THIS ACCORDINGLY
remote_incoming_path = '/home/kleaman/incoming_ks2/'  # DO NOT CHANGE
remote_ks2_path = '/home/kleaman/Kilosort/'   # DO NOT CHANGE
remote_si_working_folder = '/home/kleaman/incoming_ks2/'  # DO NOT CHANGE
remote_ks_working_folder = '/ks2_t/'   # DO NOT CHANGE

f = open(remote_incoming_path + 'remote_params.json', 'r')
remote_params = json.load(f)
f.close()

bad_channel_ids = np.asarray(remote_params["bad_channels"])
ks2_5_params = remote_params["ks2_5_params"]

rec = si.read_spikeglx(remote_incoming_path, stream_name = remote_recording_stream)

rec = si.highpass_filter(rec, freq_min=400.)
rec  = rec.remove_channels(bad_channel_ids)

si.Kilosort2_5Sorter.set_kilosort2_5_path(remote_ks2_path)

job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
rec = rec.save(folder=remote_si_working_folder + 'si_preprocess', format='binary', **job_kwargs)

si.run_sorter('kilosort2_5', rec, output_folder=remote_ks_working_folder+'ks2_5_output', verbose=True, remove_existing_folder=False, **ks2_5_params )

# Webhook for Slack Channel
webhook = "https://hooks.slack.com/services/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"

user_dict = {"kleaman": "U05NKC7CE1W", "manishm":"UMPLQT80M"}
# Ready to send a message
payload = {"text": "<@{}>, your spike-sorting finished running on Deimos".format(user_dict[os.getlogin()])}
#print(payload)
requests.post(webhook, json.dumps(payload))
