import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import IceCubeAnalysis

def main(source_class_names, alpha=2.0, weights_type='flat', use_parallel=False, n_cpu=4, var_index_cut=None):

    sourcesearch_ = IceCubeAnalysis.SourceSearch("./processed_data/output_icecube_data.npz")
    sourcesearch_.load_background("./processed_data/output_icecube_background_count.npz")
    
    # The time used in integration, in seconds
    T = (3.0 * 365.25 * 24.0 * 3600.0)

    # The energy bounds used in integration.
    E1 = 100.0
    E2 = 30.0

    Aeff_filename = "processed_data/output_icecube_AffIntegrated_%s.npz" % alpha
    class_search = IceCubeAnalysis.SourceClassSearch(T, E1, E2, alpha, sourcesearch_, Aeff_filename)
    
    class_search.load_4lac("./processed_data/4LAC_catelogy.npz", source_class_names, weights_type)
    
    if(var_index_cut is not None):
        class_search.var_index_cut(var_index_cut)
    
    print("Number of Sources:\t %i" % class_search.N)
    print("Number of Events:\t %i" % sourcesearch_.N)
    
    start_time = time.time()

    if(use_parallel):
        args_for_multiprocessing = np.arange(class_search.N)
                                     
        pool = Pool(n_cpu)
        parallel_results = pool.map(class_search.source_loop,
                                    args_for_multiprocessing)
        pool.close()
        
        parallel_results = [list(t) for t in zip(*parallel_results)]

        sweep_ts_each_source = np.stack(parallel_results[1], axis=1)
        sweep_ts = np.sum(parallel_results[1], axis=0)
        sweep_flux = np.sum(parallel_results[0], axis=0)

    else:  
        # Calculate the points that we will then loop over
        parameterized_span = class_search.calculate_span()

        sweep_ts = np.zeros(len(parameterized_span))
        sweep_flux = np.zeros(len(parameterized_span))
        sweep_ts_each_source = np.zeros((len(parameterized_span), class_search.N))

        for i_source in range(class_search.N):
            sweep_fluxes_, ts_results_ = class_search.source_loop(i_source)

            sweep_flux += sweep_fluxes_
            sweep_ts += ts_results_
            sweep_ts_each_source[:, i_source] = ts_results_
                
    end_time = time.time()

    if(use_parallel):
        print("Using parallel, time passed was: \t %f" % (end_time - start_time))
    else:
        print("Using nonparallel, time passed was: \t %f" % (end_time - start_time))

    sweep_flux *= 1000.0 # convert TeV to GeV    

    return sweep_flux, sweep_ts, sweep_ts_each_source
    

if(__name__ == "__main__"):

    source_class_names = ['FSRQ', 'bll', 'FSRQ', 'fsrq', 'BCU', 'bcu']
    
    weights_types = ['flat', 'flux', 'dist']
    colors = ['orange', 'green', 'blue']
    labels = ['Flat', r'$\gamma$-Ray', r'1/D$_L^2$']

    alphas = [2.0, 2.5]
    linestyles = ['-', '--']
    
    plt.figure(figsize=(5, 4))
    for i_alpha, alpha in enumerate(alphas):
        
        for i_weights_type, weights_type in enumerate(weights_types):
            sweep_flux, sweep_ts, sweep_ts_each_source = main(source_class_names=source_class_names, alpha=alpha,                 
                                                              weights_type=weights_type, use_parallel=True, var_index_cut=None)#18.48)

            plt.semilogx(np.array(sweep_flux)[sweep_ts < 1e3],
                         sweep_ts[sweep_ts < 1e3],
                         linestyle=linestyles[i_alpha],
                         color=colors[i_weights_type])
            
    for i_weights_type, weights_type in enumerate(weights_types):            
        plt.plot([], [], color=colors[i_weights_type], label=labels[i_weights_type])

    for i_alpha, alpha in enumerate(alphas):
        plt.plot([], [], color='black', linestyle=linestyles[i_alpha], label=r"$\alpha=$" + str(alpha))
            
    plt.axhline(-3.85,
                color="red",
                linestyle="-.",
                label="95% CL")
            
    plt.xlabel(r"$E^2_\nu dN_\nu/dE_\nu$ [GeV / cm$^2$ / s / sr] at 30 TeV")
    plt.ylabel("$2 \Delta \ln \mathcal{L}$")
    plt.xlim(5e-10, 5e-7)
    plt.ylim(-10.0, 4.0)
    plt.grid()
    plt.legend()

    plt.show()
