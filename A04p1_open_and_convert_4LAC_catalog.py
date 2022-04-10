import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units


def open_and_convert_catalog(fits_file_name, output_file_name, verbose=False):
    """
    Opens the 4LAC FITS file and converts it to a numpy pickle file.

    Parameters
    ----------
    fits_file_name : str
        File location of the 4LAC catalog FITS file.
    output_file_name : str
        Output file location of numpy pickle file.
    """

    hdul = fits.open(fits_file_name)

    if(verbose):
        print(hdul.info())
        print(hdul[1].columns.info())

    # Its possible this isn't right, but I think it is
    cat_names = hdul[1].data.field(0).tolist()
    cat_ra = hdul[1].data.field(1).tolist()
    cat_dec = hdul[1].data.field(2).tolist()
    cat_flux1000 = hdul[1].data.field(6).tolist()
    cat_z = hdul[1].data.field(30).tolist()
    cat_type = hdul[1].data.field(19).tolist()
    cat_var_index = hdul[1].data.field(34).tolist()

    np.savez(output_file_name,
             cat_names=cat_names,
             cat_ra=cat_ra,
             cat_dec=cat_dec,
             cat_type=cat_type,
             cat_flux1000=cat_flux1000,
             cat_z=cat_z,
             cat_var_index=cat_var_index)


def plot_catalog(catalog_file_name):
    """
    Creates some diagnostic plots of the
    4LAC catalog.

    Parameters
    ----------
    catalog_file_name : str
        File location of pickle 4LAC catalog file.
    """

    catalog_data = np.load(catalog_file_name)
    cat_var_index = catalog_data['cat_var_index']
    cat_flux1000 = catalog_data['cat_flux1000']
    cat_ra = catalog_data['cat_ra']
    cat_dec = catalog_data['cat_dec']

    plt.figure()
    plt.hist(cat_var_index,
             log=True,
             range=(0.0, 100.0),
             bins=100)
    plt.xlabel("Measured Variability Index of Objects in Catalog")

    plt.figure()
    plt.scatter(cat_ra, cat_dec)
    plt.xlabel("ra")
    plt.ylabel("dec.")

    coords = SkyCoord(ra=cat_ra,
                      dec=cat_dec,
                      unit='degree')
    ra = coords.ra.wrap_at(180 * units.deg).radian
    dec = coords.dec.radian
    color_map = plt.cm.Spectral_r

    ax = plt.subplot(111, projection="aitoff")
    image = ax.hexbin(ra, dec,
                      cmap=color_map,
                      gridsize=512,
                      mincnt=1,
                      bins='log')
    ax.set_xlabel('R.A.')
    ax.set_ylabel('decl.')
    ax.grid(True)
    cbar = plt.colorbar(image, spacing='uniform', extend='max')

    plt.figure()
    plt.hist(cat_flux1000,
             log=True,
             range=(-1.0e-9, 1.0e-9),
             bins=1000)
    plt.xlabel("Measured Flux of Objects in Catalog")

    plt.show()


if(__name__ == "__main__"):

    fits_file_name = "./data/table_4LAC.fits"
    output_file_name = "./processed_data/4LAC_catelogy.npz"
    open_and_convert_catalog(fits_file_name,
                             output_file_name)

    plot_catalog(output_file_name)
