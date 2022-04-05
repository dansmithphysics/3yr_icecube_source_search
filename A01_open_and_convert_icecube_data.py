import glob
import numpy as np

data_files = glob.glob("./data/3year-data-release/IC*-events.txt")

data_day = np.array([])
data_sigmas = np.array([])
data_ra = np.array([])
data_dec = np.array([])

for data_file_name in data_files:
    print("Loading filename: %s" % data_file_name)
    f = open(data_file_name)

    data = np.loadtxt(data_file_name, dtype='float')
    data_day = np.append(data_day, data[:, 0])
    data_sigmas = np.append(data_sigmas, data[:, 2])
    data_ra = np.append(data_ra, data[:, 3])
    data_dec = np.append(data_dec, data[:, 4])

np.savez("processed_data/output_icecube_data.npz",
         data_day=data_day,
         data_sigmas=data_sigmas,
         data_ra=data_ra,
         data_dec=data_dec)
