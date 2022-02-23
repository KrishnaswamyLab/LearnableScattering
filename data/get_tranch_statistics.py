import numpy as np

tranch = "JBCD_subset.npy"
tranch_name, _ = tranch.split('.')

tranch_dict = np.load(tranch, allow_pickle=True).item()
tranch_dict_keys = list(tranch_dict.keys())

list_dict = {}

for entry in tranch_dict[tranch_dict_keys[0]].keys():
    if type(tranch_dict[tranch_dict_keys[0]][entry]) == float or type(tranch_dict[tranch_dict_keys[0]][entry]) == int:
        list_dict[entry] = []

for smi in tranch_dict_keys:
    for prop in tranch_dict[smi].keys():
        if type(tranch_dict[smi][prop]) == float or type(tranch_dict[smi][prop]) == int:
            list_dict[prop].append(tranch_dict[smi][prop])


stats_dict = {}

for entry in list_dict.keys():
    prop_list = list_dict[entry]
    try:
        mean = np.mean(prop_list)
    except:
        print(entry)
        raise Exception
    std = np.std(prop_list)
    stat = {}
    stat['mean'] = mean
    stat['std'] = std
    stats_dict[entry] = stat

print(stats_dict['BertzCT']['mean'])
print(stats_dict['BertzCT']['std'])

np.save(tranch_name + '_stats.npy', stats_dict)




