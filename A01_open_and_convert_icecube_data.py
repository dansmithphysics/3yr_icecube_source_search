import glob
import numpy as np


def main(file_names):
    """
    Converts the IceCube data from fixed-width
    text files to numpy pickle files.

    Parameters
    ----------
    file_names : array_like
        List of strings to IceCube data file locations.
    """

    data_files = file_names
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


if(__name__ == "__main__"):

    file_names = glob.glob("./data/3year-data-release/IC*-events.txt")
    main(file_names)
