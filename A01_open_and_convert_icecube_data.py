import glob
import numpy as np
import pandas as pd


def main(raw_icecube_file_names, output_file_name):
    """
    Converts the IceCube data from fixed-width
    text files to numpy pickle files.

    Parameters
    ----------
    raw_icecube_file_names : array_like
        List of strings to IceCube data file locations.
    output_file_name : array_like
        Name of output pickled IceCube data file locations.
    """

    data_files = raw_icecube_file_names

    df = pd.DataFrame()
    
    for data_file_name in data_files:
        print("Loading filename: %s" % data_file_name)
        
        df_cur = pd.read_fwf(data_file_name)
        df_cur.drop(columns="#", inplace=True)
        df = pd.concat([df, df_cur])

    df.to_pickle(output_file_name)


if(__name__ == "__main__"):

    raw_icecube_file_names = glob.glob("./data/3year-data-release/IC*-events.txt")
    output_file_name = "processed_data/output_icecube_data.pkl"
    main(raw_icecube_file_names, output_file_name)
