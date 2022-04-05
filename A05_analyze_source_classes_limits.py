import time
import scipy.integrate
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import IceCubeAnalysis


def load_catalog(file_name, allowed_names):

    catelog_data = np.load(file_name, allow_pickle=True)
    cat_ra = catelog_data["cat_RA"]
    cat_dec = catelog_data["cat_Dec"]
    cat_names = catelog_data["cat_names"]
    cat_type = catelog_data["cat_type"]
    cat_flux1000 = catelog_data["cat_flux1000"]

    allowed_names_mask = np.zeros(len(cat_dec))
    for i in range(len(allowed_names)):
        allowed_names_mask[cat_type == allowed_names[i]] = 1

    allowed = np.logical_and(np.abs(cat_dec) < 87.0,
                             allowed_names_mask)

    cat_ra = cat_ra[allowed]
    cat_dec = cat_dec[allowed]
    cat_names = cat_names[allowed]
    cat_flux1000 = cat_flux1000[allowed]

    cat_ra = np.deg2rad(cat_ra)
    cat_dec = np.deg2rad(cat_dec)

    return cat_ra, cat_dec, cat_names, cat_flux1000


def load_weights(weights_type, cat_flux1000):
    if(weights_type == 'flat'):
        cat_flux_weights = 1e-9 * np.ones(len(cat_flux1000))
    elif(weights_type == 'flux'):
        cat_flux_weights = cat_flux1000
    elif(weights_type == 'dist'):
        cat_flux_weights = np.zeros(len(cat_flux1000))
        cat_flux_weights = 1e-2 / np.power(cat_DL, 2.0)
        cat_flux_weights[cat_DL[i] == -10.] = 0.0
    else:
        print("Weights not known: %s" % weights)
        exit()
    return cat_flux_weights


def load_Aeff(file_name):

    icecube_Aeff_integrated = np.load(file_name, allow_pickle=True)

    f_Aeff_dec_integration = scipy.interpolate.interp1d(icecube_Aeff_integrated['dec'],
                                                        icecube_Aeff_integrated['Aeffintegrated'],
                                                        kind='cubic',
                                                        bounds_error=False,
                                                        fill_value="extrapolate")

    return f_Aeff_dec_integration


def calculate_span(E1, E2, alpha, cat_flux_weights, n_entries=30):
    sum_of_interest = np.sum(np.power(E1 / E2, alpha) 
                             * np.power(E2, 2.0)
                             * cat_flux_weights
                             / (4.0 * np.pi))

    para_min = 1e-20 / sum_of_interest
    para_max = 1e-9 / sum_of_interest

    para_span = np.power(10.0, np.linspace(np.log10(para_min), np.log10(para_max), n_entries))
    return para_span


def para_loop(given_para, sourcesearch_, cart_s, cat_flux_weights, T, E1, E2, alpha, f_Aeff_dec_integration_):
    given_ns = given_para * cat_flux_weights * T * np.power(E1, alpha) * f_Aeff_dec_integration_
    N_sources = cart_s.shape[-1]

    results = np.zeros(N_sources)

    for i_source in range(N_sources):
        results[i_source] = sourcesearch_.test_statistic_at_point(cart_s[:, i_source],
                                                                  given_ns[i_source])

    current_flux = np.sum(given_para * cat_flux_weights * np.power(E1 / E2, alpha))
    sweep_flux = np.power(E2, 2.0) * current_flux / (4.0 * np.pi)

    print("Finished", given_para)

    return sweep_flux, results


def main():
    use_parallel = True
    n_cpu = 20

    sourcesearch_ = IceCubeAnalysis.SourceSearch("./processed_data/output_icecube_data.npz")
    sourcesearch_.load_background("./processed_data/output_icecube_background_count.npz")

    # Parameters of the problem
    alpha = 2.0
    allowed_names = ['BLL', 'bll', 'FSRQ', 'fsrq']
    weights_type = 'flat'

    # The time used in integration, in seconds
    T = (10.0 * 365.25 * 24.0 * 3600.0)

    # The energy bounds used in integration.
    E1 = 100.0
    E2 = 30.0
    
    # Load integrate detector affective volume, a function of zenith angle
    f_Aeff_dec_integration = load_Aeff("processed_data/output_icecube_AffIntegrated_%s.npz" % alpha)

    # Load catalog data
    cat_ra, cat_dec, cat_names, cat_flux1000 = load_catalog("./processed_data/4LAC_catelogy.npz", allowed_names)
    cat_sin_dec = np.sin(cat_dec)

    # Load flux weights used when setting limits.
    cat_flux_weights = load_weights(weights_type, cat_flux1000)

    N_sources = len(cat_ra)

    print("Number of Sources:\t %i" % len(cat_ra))
    print("Number of Events:\t %i" % sourcesearch_.N)

    # The cartesian positions of all sources in the catalog.
    # Easier for great circle distance calculations
    cart_x_s = np.sin(np.pi/2.0 - cat_dec) * np.cos(cat_ra)
    cart_y_s = np.sin(np.pi/2.0 - cat_dec) * np.sin(cat_ra)
    cart_z_s = np.cos(np.pi/2.0 - cat_dec)
    cart_s = np.array([cart_x_s, cart_y_s, cart_z_s])

    # Calculate the points that we will then loop over
    parameterized_span = calculate_span(E1, E2, alpha, cat_flux_weights)

    sweep_test_stats = np.zeros(len(parameterized_span))
    sweep_flux = np.zeros(len(parameterized_span))

    test_stat_each_source = np.zeros((len(parameterized_span), N_sources))

    start_time = time.time()

    if(use_parallel):
        pool = Pool(n_cpu)        

        args_for_multiprocessing = [(given_para, sourcesearch_, cart_s, cat_flux_weights, T, E1, E2, alpha, f_Aeff_dec_integration(cat_dec)) 
                                    for i_given_para, given_para in enumerate(parameterized_span)]

        parallel_results = pool.starmap(para_loop,
                                        args_for_multiprocessing)
        parallel_results =  [list(t) for t in zip(*parallel_results)]

        test_stat_each_source = np.stack(parallel_results[1], axis=0)
        sweep_test_stats = np.sum(parallel_results[1], axis=1)

        sweep_flux = parallel_results[0]

        pool.close()
    else:
        for i_given_para, given_para in enumerate(parameterized_span):
            given_ns = given_para * cat_flux_weights * T * np.power(E1, alpha) * f_Aeff_dec_integration(cat_dec)
            results = np.zeros(N_sources)

            S_i = np.stack([sourcesearch_.Si_likelihood(cart_s[:, i_source]) for i_source in range(N_sources)])
            B_i = sourcesearch_.f_B_i(np.pi / 2.0 - np.arccos(cart_s[2, :]))

            for i_source in range(N_sources):
                results[i_source] = sourcesearch_.test_statistic_at_point(cart_s[:, i_source],
                                                                          given_ns[i_source],
                                                                          S_i[i_source],
                                                                          B_i[i_source])

            test_stat_each_source[i_given_para] = results
            sweep_test_stats[i_given_para] = np.sum(results)

            current_flux = np.sum(given_para * cat_flux_weights * np.power(E1 / E2, alpha))
            sweep_flux[i_given_para] = np.power(E2, 2.0) * current_flux / (4.0 * np.pi)
            
            print("%i \t %.2E \t %.2E \t %.2E" % (i_given_para,
                                                  parameterized_span[i_given_para],
                                                  sweep_flux[i_given_para],
                                                  sweep_test_stats[i_given_para]))

    end_time = time.time()

    if(use_parallel):
        for i in range(len(parameterized_span)):
            print("%i \t %.2E \t %.2E \t %.2E" % (i, parameterized_span[i], sweep_flux[i], sweep_test_stats[i]))

        print("Using parallel, time passed was: \t %f" % (end_time - start_time))
    else:
        print("Using nonparallel, time passed was: \t %f" % (end_time - start_time))

    calculated_values = sweep_flux != 0


    plt.semilogx(np.array(sweep_flux)[sweep_test_stats < 1e3],
                 sweep_test_stats[sweep_test_stats < 1e3],
                 color="black",
                 label="3 yr. IceCube Data")

    for i_source in range(N_sources):
        plt.semilogx(np.array(sweep_flux)[test_stat_each_source[:, i_source] < 1e3],
                     test_stat_each_source[:, i_source][test_stat_each_source[:, i_source] < 1e3],
                     color="red",
                     alpha=0.1)

    plt.axhline(-3.85,
                color="red",
                linestyle="--",
                label="95% Confidence Level")

    plt.xlabel("F_v, [TeV / cm^2 / s / sr] at 30 TeV")
    plt.ylabel("$2 \Delta \ln \mathcal{L}$")

    #plt.xlim(np.min(sweep_flux[calculated_values]),
    #         np.max(sweep_flux[calculated_values]))
    plt.ylim(-10.0, 10.0)

    plt.grid()
    plt.legend()

    plt.show()


if(__name__ == "__main__"):
    main()
