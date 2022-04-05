import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import IceCubeAnalysis
import cProfile


def prepare_skymap_coordinates(step_size):
    """
    Returns the RA and Dec for each point, and a map with the index
    """

    ra_sweep = np.arange(0, 360, step_size)
    dec_sweep = np.arange(-90, 90, step_size)

    ra_len = len(ra_sweep)
    dec_len = len(dec_sweep)

    total_pts = dec_len * ra_len

    index_map = np.zeros((total_pts, 2), dtype='int')

    ras = np.zeros(total_pts)
    decs = np.zeros(total_pts)

    i_source = 0
    for iX in range(ra_len):
        for iY in range(dec_len):
            index_map[i_source] = [iX, iY]
            ras[i_source] = ra_sweep[iX]
            decs[i_source] = dec_sweep[iY]
            i_source += 1

    return ras, decs, index_map, ra_len, dec_len


def main():

    use_parallel = True
    n_cpu = 4

    step_size = 5.0  # Degrees step on the sky

    sourcesearch_ = IceCubeAnalysis.SourceSearch("./processed_data/output_icecube_data.npz")
    sourcesearch_.load_background("./processed_data/output_icecube_background_count.npz")

    #  This is the coordinate of each point on the sky we are checking.
    cat_ra, cat_dec, index_map, ra_len, dec_len = prepare_skymap_coordinates(step_size)

    N_sky_pts = len(cat_ra)

    print("Number of IceCube events: \t %i" % sourcesearch_.N)
    print("Number of skypoints to calc: \t %i" % N_sky_pts)

    # In the equations in the paper, these are s indices, the index of source direction
    cord_s = np.stack((cat_ra, cat_dec), axis=1)

    start_time = time.time()

    if(use_parallel):
        pool = Pool(n_cpu)

        args_for_multiprocessing = [(np.array(cord_s[i_source]), i_source) for i_source in range(N_sky_pts)]
        results = pool.starmap(sourcesearch_.job_submission,
                               args_for_multiprocessing)

        pool.close()
    else:
        results = []
        for i_source in range(N_sky_pts):
            results += [sourcesearch_.job_submission(cord_s[i_source], 
                                                     i_source)]

    end_time = time.time()

    if(use_parallel):
        print("Using parallel, time passed was: \t %f" % (end_time - start_time))
    else:
        print("Using nonparallel, time passed was: \t %f" % (end_time - start_time))
        
    data_map = np.zeros((ra_len, dec_len))
    n_s_map = np.zeros((ra_len, dec_len))

    for i_source in range(N_sky_pts):
        if(np.abs(cat_dec[i_source]) > 87.0):
            continue

        ns, del_ln_L = results[i_source]
        i_ra, i_dec = index_map[i_source]
        n_s_map[i_ra, i_dec] = ns
        data_map[i_ra, i_dec] = del_ln_L
    
    np.save("./processed_data/calculated_fit_likelihood_map_allsky.npy", data_map)
    np.save("./processed_data/calculated_fit_ns_map_allsky.npy", n_s_map)

    data_map_pos = data_map[data_map > 0.0]
    data_map_neg = data_map[data_map < 0.0]
    data_map_zero = data_map[data_map == 0.0]

    plt.figure()
    plt.hist(np.sqrt(2.0 * data_map_pos.flatten()),
             range=(-6.0, 6.0), bins=120, log=True,
             label="Positive TS")
    plt.hist(-np.sqrt(2.0 * -data_map_neg.flatten()),
             range=(-6.0, 6.0), bins=120, log=True,
             label="Negative TS")
    plt.hist(np.sqrt(2.0 * data_map_zero.flatten()),
             range=(-6.0, 6.0), bins=120, log=True,
             label="Zero TS")
    plt.xlabel("$\sqrt{2 \Delta \ln \mathcal{L}}$")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.hist(n_s_map.flatten(),
             range=(-30.0, 30.0), bins=60, log=True)
    plt.xlabel("$n_s$")
    plt.grid()

    plt.figure()
    plt.title("$\sqrt{2 \Delta \ln \mathcal{L}}$")
    plt.imshow(np.sqrt(2.0 * np.abs(data_map)).transpose())
    plt.xlabel("RA Index")
    plt.ylabel("Dec Index")

    plt.figure()
    plt.title("$n_s$")
    plt.imshow(n_s_map.transpose())
    plt.xlabel("RA Index")
    plt.ylabel("Dec Index")

    plt.show()


if(__name__ == "__main__"):
    #import cProfile, pstats
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats()
