import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units


def open_and_convert_catalog(file_name, output_file_name):
    hdul = fits.open(file_name)

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

    
def plot_catalog(file_name):

    catalog_data = np.load(file_name)
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

    open_and_convert_catalog("./data/table_4LAC.fits",
                             "./processed_data/4LAC_catelogy.npz")

    plot_catalog("./processed_data/4LAC_catelogy.npz")
