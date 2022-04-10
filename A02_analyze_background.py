import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate


def main(icecube_file_name, output_file_name, n_dec_pts=1000):
    """
    Looking to calculate B_i, the background PDF of the search.
    This is done by scrambling data in a 6 degree
    declination angle band in the sky.

    Parameters
    ----------
    icecube_file_name : str
        IceCube pickle file location.
    output_file_name : str
        Output file name for processed background PDF.

    Returns
    -------
    sweep_dec : array_like
        Array of declinations used to calculate PDF

    B_i : array_like
        The background PDF, at each step in sweep_dec
    """
    
    # Load up the IceCube data
    icecube_data = np.load(icecube_file_name,
                           allow_pickle=True)
    data_dec = np.array(icecube_data["data_dec"])

    # size of bins
    size_of_band = 3.0
    sweep_lowerlimit = -87.0
    sweep_upperlimit = 87.0

    # sweep over different sin decs to calculate the B_i at that point
    sweep_dec = np.linspace(sweep_lowerlimit, sweep_upperlimit, n_dec_pts)

    # Count number of entries in bin
    entries_in_bands = np.abs(data_dec[:, np.newaxis] - sweep_dec) < size_of_band
    entries_in_bands = np.sum(entries_in_bands, axis=0)

    solid_angles = (2.0 * np.pi *
                    np.sin(np.deg2rad(size_of_band)) *
                    np.cos(np.deg2rad(sweep_dec)))
    event_per_solid_angle = entries_in_bands / solid_angles
    
    f_integrand = scipy.interpolate.interp1d(np.sin(np.deg2rad(sweep_dec)),
                                             event_per_solid_angle,
                                             kind='cubic',
                                             bounds_error=False,
                                             fill_value=0.0)

    # to perform the average, integrate over result and divide it out
    sweep_counts_norm, err = scipy.integrate.quad(f_integrand,
                                                  np.sin(np.deg2rad(sweep_lowerlimit)),
                                                  np.sin(np.deg2rad(sweep_upperlimit)),
                                                  limit=1000)

    # equation 2.2 in the paper
    P_B = event_per_solid_angle / sweep_counts_norm
    B_i = P_B / (2.0 * np.pi)

    np.savez(output_file_name,
             dec=sweep_dec,
             B_i=B_i)

    return sweep_dec, B_i


if(__name__ == "__main__"):
    icecube_file_name = "processed_data/output_icecube_data.npz"
    output_file_name = "processed_data/output_icecube_background_count.npz"
    sweep_dec, B_i = main(icecube_file_name, output_file_name)
    
    plt.figure()
    plt.plot(sweep_dec, B_i, color='black')
    plt.axvline(-87.0, color='red')
    plt.axvline(87.0, color='red', label="Declincation Limit of Search")
    plt.xlabel(r"Declination [$^\circ$]")
    plt.ylabel("Background PDF")
    plt.xlim(-90.0, 90.0)
    plt.ylim(0.0, 0.15)
    plt.grid()
    plt.legend()

    plt.savefig("./plots/A02_analyze_background.png", dpi=300)
    
    plt.show()
