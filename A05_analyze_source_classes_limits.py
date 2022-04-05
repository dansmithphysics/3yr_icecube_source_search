import time
import scipy.integrate
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import IceCubeAnalysis


def load_catalog(file_name, allowed_names):

    catelog_data = np.load(file_name, allow_pickle=True)
    cat_ra = catelog_data["cat_ra"]
    cat_dec = catelog_data["cat_dec"]
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


def calculate_span(E1, E2, alpha, cat_flux_weights, n_entries=40):
    sum_of_interest = np.sum(np.power(E1 / E2, alpha) 
                             * np.power(E2, 2.0)
                             * cat_flux_weights
                             / (4.0 * np.pi))

    para_min = 1e-13 / sum_of_interest
    para_max = 1e-9 / sum_of_interest

    para_span = np.power(10.0, np.linspace(np.log10(para_min), np.log10(para_max), n_entries))
    return para_span


def source_loop(parameterized_span, sourcesearch_, cord_s, cat_flux_weight, T, E1, E2, alpha, f_Aeff_dec_integration_,
                close_point_cut=10, significance_cut=1e-10):

    sweep_fluxes = np.zeros(len(parameterized_span))
    ts_results = np.zeros(len(parameterized_span))

    S_i = sourcesearch_.Si_likelihood(cord_s, close_point_cut=close_point_cut)
    B_i = sourcesearch_.f_B_i(cord_s[1])

    non_zero_S_i = (S_i > significance_cut)
    S_i = S_i[non_zero_S_i]
    N_zeros = sourcesearch_.N - len(S_i)

    for i_given_para, given_para in enumerate(parameterized_span):
        given_ns = given_para * cat_flux_weight * T * np.power(E1, alpha) * f_Aeff_dec_integration_
        
        ts_results[i_given_para] = sourcesearch_.test_statistic_at_point(cord_s,
                                                                      given_ns,
                                                                      S_i, B_i, N_zeros)

        current_flux = given_para * cat_flux_weight * np.power(E1 / E2, alpha)
        sweep_fluxes[i_given_para] = np.power(E2, 2.0) * current_flux / (4.0 * np.pi)        

    return sweep_fluxes, ts_results


def main():
    use_parallel = True
    n_cpu = 4

    sourcesearch_ = IceCubeAnalysis.SourceSearch("./processed_data/output_icecube_data.npz")
    sourcesearch_.load_background("./processed_data/output_icecube_background_count.npz")

    # Parameters of the problem
    alpha = 2.0
    allowed_names = ['BLL', 'bll', 'FSRQ', 'fsrq', 'BCU', 'bcu']
    weights_type = 'flat'
    
    # The time used in integration, in seconds
    T = (3.0 * 365.25 * 24.0 * 3600.0)

    # The energy bounds used in integration.
    E1 = 100.0
    E2 = 30.0
    
    # Load integrate detector affective volume, a function of zenith angle
    f_Aeff_dec_integration = load_Aeff("processed_data/output_icecube_AffIntegrated_%s.npz" % alpha)

    # Load catalog data
    cat_ra, cat_dec, cat_names, cat_flux1000 = load_catalog("./processed_data/4LAC_catelogy.npz", allowed_names)

    # Load flux weights used when setting limits.
    cat_flux_weights = load_weights(weights_type, cat_flux1000)

    N_sources = len(cat_ra)

    print("Number of Sources:\t %i" % len(cat_ra))
    print("Number of Events:\t %i" % sourcesearch_.N)

    # The cordesian positions of all sources in the catalog.
    # Easier for great circle distance calculations
    cord_s = np.stack((cat_ra, cat_dec), axis=1)
    
    # Calculate the points that we will then loop over
    parameterized_span = calculate_span(E1, E2, alpha, cat_flux_weights)

    sweep_ts = np.zeros(len(parameterized_span))
    sweep_flux = np.zeros(len(parameterized_span))

    sweep_ts_each_source = np.zeros((len(parameterized_span), N_sources))

    start_time = time.time()

    if(use_parallel):
        args_for_multiprocessing = [(parameterized_span, sourcesearch_, cord_s[i_source],
                                     cat_flux_weights[i_source],
                                     T, E1, E2, alpha,
                                     f_Aeff_dec_integration(cat_dec[i_source]))
                                    for i_source in range(len(cord_s))]

        pool = Pool(n_cpu)
        parallel_results = pool.starmap(source_loop,
                                        args_for_multiprocessing)
        pool.close()
        
        parallel_results = [list(t) for t in zip(*parallel_results)]

        sweep_ts_each_source = np.stack(parallel_results[1], axis=1)
        sweep_ts = np.sum(parallel_results[1], axis=0)
        sweep_flux = np.sum(parallel_results[0], axis=0)

    else:  
        for i_source in range(len(cord_s)):
            sweep_fluxes_, ts_results_ = source_loop(parameterized_span,
                                                     sourcesearch_,
                                                     cord_s[i_source],
                                                     cat_flux_weights[i_source],
                                                     T, E1, E2, alpha,
                                                     f_Aeff_dec_integration(cat_dec[i_source]))
            sweep_flux += sweep_fluxes_
            sweep_ts += ts_results_
            sweep_ts_each_source[:, i_source] = ts_results_
                
    end_time = time.time()

    if(use_parallel):
        print("Using parallel, time passed was: \t %f" % (end_time - start_time))
    else:
        print("Using nonparallel, time passed was: \t %f" % (end_time - start_time))

    sweep_flux *= 1000.0 # convert TeV to GeV
        
    calculated_values = sweep_flux != 0

    plt.semilogx(np.array(sweep_flux)[sweep_ts < 1e3],
                 sweep_ts[sweep_ts < 1e3],
                 color="black",
                 label="3 yr. IceCube Data")

    for i_source in range(N_sources):
        plt.semilogx(np.array(sweep_flux)[sweep_ts_each_source[:, i_source] < 1e3],
                     sweep_ts_each_source[:, i_source][sweep_ts_each_source[:, i_source] < 1e3],
                     color="red",
                     alpha=0.1)

    plt.axhline(-3.85,
                color="red",
                linestyle="--",
                label="95% Confidence Level")

    plt.xlabel(r"$E^2_\nu dN_\nu/dE_\nu$ [GeV / cm$^2$ / s / sr] at 30 TeV")
    plt.ylabel("$2 \Delta \ln \mathcal{L}$")
    #plt.xlim(5e-10, 1e-6)
    plt.ylim(-10.0, 10.0)
    plt.grid()
    plt.legend()
    plt.show()


if(__name__ == "__main__"):
    main()
