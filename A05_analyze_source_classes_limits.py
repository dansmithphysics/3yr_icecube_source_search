import time
import scipy.integrate
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import IceCubeAnalysis


class SourceClassSearch:

    def __init__(self, T, E1, E2, alpha, sourcesearch, Aeff_filename):
        self.T = T
        self.E1 = E1
        self.E2 = E2
        self.alpha = alpha
        self.sourcesearch = sourcesearch
        self.load_Aeff(Aeff_filename)


    def load_4lac(self, catalog_filename, source_names, weights_type):
        self.load_catalog(catalog_filename, source_names)
        self.load_weights(weights_type)
        self.N = len(self.cat_ra)


    def load_Aeff(self, file_name):
        icecube_Aeff_integrated = np.load(file_name, allow_pickle=True)

        self.f_Aeff_dec_integration = scipy.interpolate.interp1d(icecube_Aeff_integrated['dec'],
                                                            icecube_Aeff_integrated['Aeffintegrated'],
                                                            kind='cubic',
                                                            bounds_error=False,
                                                            fill_value="extrapolate")


    def load_catalog(self, file_name, allowed_names):

        catelog_data = np.load(file_name, allow_pickle=True)
        cat_ra = catelog_data["cat_ra"]
        cat_dec = catelog_data["cat_dec"]
        cat_names = catelog_data["cat_names"]
        cat_type = catelog_data["cat_type"]
        cat_flux1000 = catelog_data["cat_flux1000"]
        cat_z = catelog_data["cat_z"]
        
        allowed_names_mask = np.zeros(len(cat_dec))
        for i in range(len(allowed_names)):
            allowed_names_mask[cat_type == allowed_names[i]] = 1

        allowed = np.logical_and(np.abs(cat_dec) < 87.0,
                                 allowed_names_mask)

        cat_ra = cat_ra[allowed]
        cat_dec = cat_dec[allowed]
        cat_names = cat_names[allowed]
        cat_flux1000 = cat_flux1000[allowed]
        cat_z = cat_z[allowed]
        
        self.cat_ra = cat_ra
        self.cat_dec = cat_dec
        self.cat_names = cat_names
        self.cat_flux1000 = cat_flux1000
        self.cat_z = cat_z  # Missing entries are -inf
        
        # Calculate the luminosity distance to source        
        self.cat_DL = -10 * np.ones(len(cat_ra))  # Missing entries are -10
        non_zero_entries = np.logical_not(np.isinf(self.cat_z))
        self.cat_DL[non_zero_entries] = np.array([self.luminosity_distance_from_redshift(z) for z in self.cat_z[non_zero_entries]])

        
    def luminosity_distance_from_redshift(self, z):
        # Values taken from https://arxiv.org/pdf/1807.06209.pdf
        omega_m = 0.3111
        omega_lambda = 0.6889
        H0 = 67.66  # km / s / Mpc
        c = 3e5  # km / s
        integrand = lambda zp : 1.0 / np.sqrt(omega_m * np.power(1 + zp, 3) + omega_lambda)
        luminosity_distance = c * (1 + z) / H0 * scipy.integrate.quad(integrand, 0, z)[0]
        return luminosity_distance

        
    def load_weights(self, weights_type):
        if(weights_type == 'flat'):
            cat_flux_weights = np.ones(len(self.cat_flux1000))
        elif(weights_type == 'flux'):
            cat_flux_weights = self.cat_flux1000
        elif(weights_type == 'dist'):
            cat_flux_weights = 1.0 / np.power(self.cat_DL, 2.0)
            cat_flux_weights[self.cat_DL == -10.] = 0.0  # Missing entries have a weight of zero, so aren't calculated
        else:
            print("Weights not known: %s" % weights)
            exit()
        self.cat_flux_weights = cat_flux_weights

        
    def calculate_span(self, n_entries=40):
        sum_of_interest = np.sum(np.power(self.E1 / self.E2, self.alpha) 
                                 * np.power(self.E2, 2.0)
                                 * self.cat_flux_weights
                                 / (4.0 * np.pi))

        para_min = 1e-13 / sum_of_interest
        para_max = 1e-9 / sum_of_interest

        para_span = np.power(10.0, np.linspace(np.log10(para_min), np.log10(para_max), n_entries))
        return para_span


    def source_loop(self, i_source, close_point_cut=10, significance_cut=1e-10, n_entries=40):

        parameterized_span = self.calculate_span(n_entries)
        
        sweep_fluxes = np.zeros(len(parameterized_span))
        ts_results = np.zeros(len(parameterized_span))

        S_i = self.sourcesearch.Si_likelihood([self.cat_ra[i_source], self.cat_dec[i_source]],
                                              close_point_cut=close_point_cut)
        B_i = self.sourcesearch.f_B_i(self.cat_dec[i_source])

        non_zero_S_i = (S_i > significance_cut)
        S_i = S_i[non_zero_S_i]
        N_zeros = self.sourcesearch.N - len(S_i)
        
        for i_given_para, given_para in enumerate(parameterized_span):
            given_ns = given_para * self.cat_flux_weights[i_source] * self.T * np.power(self.E1, self.alpha) * self.f_Aeff_dec_integration(self.cat_dec[i_source])
        
            ts_results[i_given_para] = self.sourcesearch.test_statistic_at_point([self.cat_ra[i_source], self.cat_dec[i_source]],
                                                                                 given_ns,
                                                                                 S_i, B_i, N_zeros)

            current_flux = given_para * self.cat_flux_weights[i_source] * np.power(self.E1 / self.E2, self.alpha)
            sweep_fluxes[i_given_para] = np.power(self.E2, 2.0) * current_flux / (4.0 * np.pi)

        return sweep_fluxes, ts_results


def main(source_class_names, alpha=2.0, weights_type='flat', use_parallel=False, n_cpu=4):

    sourcesearch_ = IceCubeAnalysis.SourceSearch("./processed_data/output_icecube_data.npz")
    sourcesearch_.load_background("./processed_data/output_icecube_background_count.npz")
    
    # The time used in integration, in seconds
    T = (3.0 * 365.25 * 24.0 * 3600.0)

    # The energy bounds used in integration.
    E1 = 100.0
    E2 = 30.0

    Aeff_filename = "processed_data/output_icecube_AffIntegrated_%s.npz" % alpha
    class_search = SourceClassSearch(T, E1, E2, alpha, sourcesearch_, Aeff_filename)
    
    class_search.load_4lac("./processed_data/4LAC_catelogy.npz", source_class_names, weights_type)
    
    print("Number of Sources:\t %i" % class_search.N)
    print("Number of Events:\t %i" % sourcesearch_.N)
    
    # Calculate the points that we will then loop over
    parameterized_span = class_search.calculate_span()

    sweep_ts = np.zeros(len(parameterized_span))
    sweep_flux = np.zeros(len(parameterized_span))
    sweep_ts_each_source = np.zeros((len(parameterized_span), class_search.N))

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

    source_class_names = ['BLL', 'bll', 'FSRQ', 'fsrq', 'BCU', 'bcu']
    
    weights_types = ['flat', 'flux', 'dist']
    colors = ['orange', 'green', 'blue']
    labels = ['Flat', r'$\gamma$-Ray', r'1/D$_L^2$']

    alphas = [2.0, 2.5]
    linestyles = ['-', '--']
    
    plt.figure(figsize=(5, 4))
    for i_alpha, alpha in enumerate(alphas):
        
        for i_weights_type, weights_type in enumerate(weights_types):
            sweep_flux, sweep_ts, sweep_ts_each_source = main(source_class_names=source_class_names, alpha=alpha,                 
                                                              weights_type=weights_type, use_parallel=True)

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
