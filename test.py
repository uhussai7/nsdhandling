from nsdhandling import core


# df=core.DataFetcher()

# df.load_exp_design()

# imgs_out=df.load_stimuli([0])

# print(imgs_out[0].shape)


stim_data=core.StimData()
stim_data.load_stimuli_raw(['1'])
nsd_data=core.NsdData(stim_data,1)
nsd_data.load_from_raw()