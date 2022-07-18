import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.integrate
from scipy.optimize import minimize_scalar


class SourceSearch:
    """
    A class that handles the likelihood
    computations for sources within the IceCube
    track data.

    Attributes
    ----------
    N : int
        Number of events in IceCube Data
    cord_i : array_like
        Array of (ra, dec) of IceCube track data
    AngErr[deg] : array_like
        Standard deviation of IceCube track data in degrees
    sindec : array_like
        Pre-computed sin of the declination of IceCube track
        data, used to speed up S_i calculation.
    cosdec : array_like
        Pre-computed cos of the declination of IceCube track
        data, used to speed up S_i calculation.
    f_B_i : scipy function
        Function of the background PDF's dependance on declination
    """

    def __init__(self, icecube_file_name):
        """
        Loads up the IceCube data.

        Parameters
        ----------
        icecube_file_name : str
            IceCube pickle file location.
        """

        self.df = pd.read_pickle(icecube_file_name)
        self.df = self.df[self.df["AngErr[deg]"] != 0]    

        # Compute these sin/cos once to save computation time later
        self.df["sindec"] = np.sin(np.deg2rad(self.df["Dec[deg]"]))
        self.df["cosdec"] = np.cos(np.deg2rad(self.df["Dec[deg]"]))

        
    def Si_likelihood(self, cord_s, close_point_cut=None):
        """
        Calculates the signal PDF at a given
        point in the sky.

        Parameters
        ----------
        cord_s : array_like
            The (ra, dec) position on sky that is being tested.
        close_point_cut : float
            Remove data events that are further than
            close_point_cut degrees away.
            Speeds up computation considerably.

        Returns
        -------
        S_i : array_like
            The signal PDF of each event in the dataset.
        """

        cord_i_ = np.stack((self.df["RA[deg]"], self.df["Dec[deg]"]), axis=1)
        
        if(close_point_cut is None):
            close_points = np.ones(len(self.df)).astype('bool')
        else:
            close_points = np.sum(np.square(cord_s - cord_i_), axis=1) < np.square(close_point_cut)
            
        cosA = (self.df["sindec"][close_points] * np.sin(np.deg2rad(cord_s[1]))
                + self.df["cosdec"][close_points] * np.cos(np.deg2rad(cord_s[1]))
                * np.cos(np.deg2rad(cord_i_[close_points, 0] - cord_s[0])))
        great_dists = np.arccos(cosA)

        # This has to be in radians.
        AngErr_ = np.deg2rad(self.df["AngErr[deg]"][close_points])
        S_i = 1.0 / (2.0 * np.pi * AngErr_ * AngErr_)
        S_i *= np.exp(-0.5 * np.square(great_dists / AngErr_))

        return S_i

    def calculate_likelihood(self, n_s, S_i, B_i, N_zeros=0):
        """
        Calculates the test statistic for a given
        number of clustered neutrinos (n_s) and
        given signal pdf (S_i), background pdf (B_i).

        Parameters
        ----------
        n_s : float
            The number of neutrinos from the source tested.
        S_i : array_like
            The signal PDF of each event in the dataset.
        B_i : float
            The background PDF of the source being tested.
        N_zeros : int
            The number of S_i points that were removed from S_i
            due to being too small. Removing S_i points that
            have nearly zero contribution greatly speeds up computation.

        Returns
        -------
        out : float
            The calculated likelihood. Zero if n_s is below zero.
        """

        result_ = n_s / len(self.df) * S_i + (1.0 - n_s / len(self.df)) * B_i

        if(np.any(result_ <= 0)):
            return 0.0
        else:
            return np.sum(np.log(result_)) + N_zeros * np.log((1.0 - n_s / len(self.df)) * B_i)

    def test_statistic_at_point(self, cord_s, n_s, S_i=None, B_i=None, N_zeros=0):
        """
        Calculates the test statistic at point

        Parameters
        ----------
        cord_s : array_like
            The cordesian position on sky that is being tested.
        n_s : float
            The number of neutrinos from the source tested.
        S_i : array_like
            The signal PDF of each event in the dataset.
        B_i : float
            The background PDF of the source being tested.
        N_zeros : int
            The number of S_i points that were removed from S_i
            due to being too small. Removing S_i points that
            have nearly zero contribution greatly speeds up computation.

        Returns
        -------
        out : float
            The calculated test statistic.
        """

        if(S_i is None):
            S_i = self.Si_likelihood(cord_s)
        if(B_i is None):
            B_i = self.f_B_i(cord_s[1])

        del_ln_L_n_s = self.calculate_likelihood(n_s, S_i, B_i, N_zeros)
        del_ln_L_0 = self.calculate_likelihood(0.0, S_i, B_i, N_zeros)

        return 2.0 * (del_ln_L_n_s - del_ln_L_0)

    def load_background(self, background_file_name):
        """
        Loads the preprocessed background PDF.
        The background pdf is loaded to a scipy interpolate function.

        Parameters
        ----------
        background_file_name : str
            File location of background pdf.
        """

        df_bg = pd.read_pickle(background_file_name)

        self.f_B_i = scipy.interpolate.interp1d(df_bg['Dec[deg]'],
                                                df_bg['B_i'],
                                                kind='cubic',
                                                bounds_error=False,
                                                fill_value='extrapolate')

    def job_submission(self, cord_s, i_source, close_point_cut=None, significance_cut=1e-10):
        """
        Function that handles the parallelization of the all-sky map.
        Computes the max-likelihood number of neutrinos from the source.

        Parameters
        ----------
        cord_s : array_like
            The (ra, dec) position on sky that is being tested.
        i_source : int
            The integer of the source being tested, used only for print outs.
        close_point_cut : float
            Remove S_i of data events that are further than
            close_point_cut degrees away.
            Speeds up computation considerably.
        significance_cut : float
            Remove S_i of data events that produce a significance
            lower than significance_cut.
            Speeds up computation considerably.

        Parameters
        ----------
        n_s : float
            Max likelihood number of neutrinos from source being tested.
        del_ln_L : float
            The max likelihood from source being tested.
        """

        S_i = self.Si_likelihood(cord_s, close_point_cut=close_point_cut)
        B_i = self.f_B_i(cord_s[1])

        non_zero_S_i = (S_i > significance_cut)
        S_i = S_i[non_zero_S_i]
        N_zeros = len(self.df) - len(S_i)

        # Before jumping in, we check to make sure the n_s will be positive
        slope = (self.calculate_likelihood(0.05, S_i, B_i, N_zeros) -
                 self.calculate_likelihood(0.0, S_i, B_i, N_zeros))

        if(slope > 0):

            def _calculate_likelihood(n_s):
                result = -self.calculate_likelihood(n_s, S_i, B_i, N_zeros)
                return result

            res = minimize_scalar(_calculate_likelihood,
                                  bounds=(0, 200),
                                  method='bounded')
            n_s = res.x
        else:
            n_s = 0

        del_ln_L = (self.calculate_likelihood(n_s, S_i, B_i, N_zeros) -
                    self.calculate_likelihood(0.0, S_i, B_i, N_zeros))

        if(i_source % 1000 == 0):
            print("%i) \t n_s = \t %f" % (i_source, n_s))

        return n_s, del_ln_L


class SourceClassSearch:
    """
    A class that handles the likelihood
    computations for source classes.

    Attributes
    ----------
    T : float
        Time used for analysis
    E1 : float
    E2 : float
    alpha : float
        The spectrum of the neutrino flux being tested.
    sourcesearch : class
        The SourceSearch class that handles the IceCube data
        and computes the likelihood given a point in the sky.
    N : float
        The number of sources in the class
    f_Aeff_dec_integration : scipy function
        Function of the energy integrated effective area.
    cat_ra : array_like
        The RA of sources in class of interest, from 4LAC catalog.
    cat_dec : array_like
        The declination of sources in class of interest, from 4LAC catalog.
    cat_names : array_like
        The name of sources in class of interest, from 4LAC catalog.
    cat_flux1000 : array_like
        The gamma-ray flux of sources in class of interest, from 4LAC catalog.
    cat_var_index : array_like
        The gamma-ray variability metric of sources in class of interest, from 4LAC catalog.
    cat_z : array_like
        The redshift of sources in class of interest, from 4LAC catalog.
    cat_DL : array_like
        The luminosity distance of sources in class of interest, calculated from cat_z.
    cat_flux_weights : array_like
        The weighting used for the source class. Options are 'flat' for
        equal weight, 'flux' to weight against the gamma-ray flux, and
        'dist' to weight against the luminosity distance.
    """

    def __init__(self, T, E1, E2, alpha, sourcesearch, Aeff_file_name):
        """
        Initializer

        Parameters
        ----------
        T : float
            Time used for analysis
        E1 : float
        E2 : float
        alpha : float
            The spectrum of the neutrino flux being tested.
        sourcesearch : class
            The SourceSearch class that handles the IceCube data
            and computes the likelihood given a point in the sky.
        Aeff_file_name
            Pickle file location of pre-processed effective area.
        """

        self.T = T
        self.E1 = E1
        self.E2 = E2
        self.alpha = alpha
        self.sourcesearch = sourcesearch
        self.load_Aeff(Aeff_file_name)

    def load_4lac(self, catalog_file_name, source_class_names, weights_type):
        """
        Loads the 4LAC catalog.

        Parameters
        ----------
        catalog_file_name : str
            File location of pickled 4LAC catalog.
        source_class_names : array_like
            Names of source classes used in calculation.
        weights_type : str
            The weighting used for the source class. Options are 'flat' for
            equal weight, 'flux' to weight against the gamma-ray flux, and
            'dist' to weight against the luminosity distance.
        """
        self.df_cat = pd.read_pickle(catalog_file_name)
        # Modify the non-blazar AGN catalog to match the paper

        df_ = pd.DataFrame({"Source_Name":"Custom 3C 411",
                            "Flux1000":3.5e-12 / 0.011636,
                            "RAJ2000":305.5333,
                            "DEJ2000":10.0197,
                            "Variability_Index":0.0,
                            "CLASS":"rdg",
                            "Redshift":0.457}, index=[0])                                
        self.df_cat = pd.concat([self.df_cat, df_])
        
        df_ = pd.DataFrame({"Source_Name":"Custom Cen B",
                            "Flux1000":2.5471e-09,
                            "RAJ2000":206.59,
                            "DEJ2000":-60.4461,
                            "Variability_Index":6.528250,
                            "CLASS":"rdg",
                            "Redshift":0.0129}, index=[0])        
        self.df_cat = pd.concat([self.df_cat, df_])
        
        # Merge the two Cens
        self.df_cat.loc[self.df_cat["Source_Name"] == "4FGL J1325.5-4300", "Flux1000"] += self.df_cat.loc[self.df_cat["Source_Name"] == "4FGL J1324.0-4330e", "Flux1000"]
        self.df_cat.drop(self.df_cat[self.df_cat["Source_Name"] == "4FGL J1324.0-4330e"].index, inplace=True)
        
        allowed_names_mask = np.zeros(len(self.df_cat))
        for i in range(len(source_class_names)):
            allowed_names_mask[self.df_cat["CLASS"] == source_class_names[i]] = 1

        self.df_cat = self.df_cat[allowed_names_mask.astype(bool)]
        self.df_cat = self.df_cat[np.abs(self.df_cat["DEJ2000"]) < 87.0]
        
        # Calculate the luminosity distance to source
        self.df_cat["DL"] = np.array([self.luminosity_distance_from_redshift(z) if np.logical_not(np.isinf(z)) else -10.0 for z in self.df_cat["Redshift"]])
        
        self.load_weights(weights_type)

    def load_Aeff(self, aeff_file_name):
        """
        Loads the pre-processed effective area.

        Parameters
        ----------
        aeff_file_name : str
            File location of the pickled pre-processed effective area.
        """
        icecube_Aeff_integrated = np.load(aeff_file_name,
                                          allow_pickle=True)

        self.f_Aeff_dec_integration = scipy.interpolate.interp1d(icecube_Aeff_integrated['dec'],
                                                                 icecube_Aeff_integrated['Aeffintegrated'],
                                                                 kind='cubic',
                                                                 bounds_error=False,
                                                                 fill_value="extrapolate")

    def var_index_cut(self, var_index_cut):
        """
        Performs a cut on the source class based on its variability index.

        Parameters
        ----------
        var_index_cut : float
            Removes events that have a variability index greater
            than var_index_cut. If None, no cut is performed.
        """

        self.df_cat = self.df_cat[self.df_cat["Variability_Index"] < var_index_cut]
        
    def luminosity_distance_from_redshift(self, z):
        """
        Calculates the luminosity distance from the red shfit.
        Values of constants of nature taken
        from https://arxiv.org/pdf/1807.06209.pdf

        Parameters
        ----------
        z : float
            Red shift.

        Returns
        -------
        out : float
            The luminosity distance.
        """

        omega_m = 0.3111
        omega_lambda = 0.6889
        H0 = 67.66  # km / s / Mpc
        c = 3e5  # km / s
        integrand = lambda zp : 1.0 / np.sqrt(omega_m * np.power(1 + zp, 3) + omega_lambda)
        luminosity_distance = c * (1 + z) / H0 * scipy.integrate.quad(integrand, 0, z)[0]
        return luminosity_distance

    def load_weights(self, weights_type):
        """
        Loads the weights used to calculate the number of neutrinos
        attributed to each source in a class.

        Parameters
        ----------
        weights_type : str
            The weighting used for the source class. Options are 'flat' for
            equal weight, 'flux' to weight against the gamma-ray flux, and
            'dist' to weight against the luminosity distance.
        """
        if(weights_type == 'flat'):
            cat_flux_weights = np.ones(len(self.df_cat))
        elif(weights_type == 'flux'):
            cat_flux_weights = self.df_cat["Flux1000"]
        elif(weights_type == 'dist'):
            cat_flux_weights = 1.0 / np.power(self.df_cat["DL"], 2.0)
            cat_flux_weights[self.df_cat["DL"] == -10.] = 0.0  # Missing entries have a weight of zero, so aren't calculated
        else:
            print("Weights not known: %s" % weights)
            exit()
        self.df_cat["flux_weights"] = cat_flux_weights

    def calculate_span(self, n_entries=40):
        """
        Helper function to determine which fluxes points to use
        in the sweep of flux produced by source class.

        Parameters
        ----------
        n_entries : int
            The number of points of flux to sweep over.

        Returns
        -------
        para_span : array_like
            The flux points to be used to compute likelihood that
            neutrinos originate from the source class.
        """

        sum_of_interest = np.sum(np.power(self.E1 / self.E2, self.alpha)
                                 * np.power(self.E2, 2.0)
                                 * self.df_cat["flux_weights"]
                                 / (4.0 * np.pi))

        para_min = 1e-13 / sum_of_interest
        para_max = 1e-9 / sum_of_interest

        para_span = np.power(10.0, np.linspace(np.log10(para_min), np.log10(para_max), n_entries))
        return para_span

    def source_loop(self, i_source, close_point_cut=None, significance_cut=1e-10, n_entries=40):
        """
        Function used to compute the likelihood that a single source
        from the source class produced neutrinos, at a sweep over neutrino fluxes.
        For parallelization, it's faster to calculate the sweep for each source
        as opposed to each point in the sweep over neutrino fluxes.

        Parameters
        ----------
        i_source : int
            The index of the source from the class being testing.
        close_point_cut : float
            Remove S_i of data events that are further than
            close_point_cut degrees away.
            Speeds up computation considerably.
        significance_cut : float
            Remove S_i of data events that produce a significance
            lower than significance_cut.
            Speeds up computation considerably.
        n_entries : int
            The number of points of flux to sweep over.

        Returns
        -------
        sweep_fluxes : array_like
            The neutrino flux for which the likelihood was calculated.
        ts_results : array_like
            The likelihood that the single source produced neutrinos
            for the given flux in sweep_fluxes.
        """

        parameterized_span = self.calculate_span(n_entries)

        sweep_fluxes = np.zeros(len(parameterized_span))
        ts_results = np.zeros(len(parameterized_span))

        S_i = self.sourcesearch.Si_likelihood([self.df_cat.iloc[i_source]["RAJ2000"], self.df_cat.iloc[i_source]["DEJ2000"]],
                                              close_point_cut=close_point_cut)
        B_i = self.sourcesearch.f_B_i(self.df_cat.iloc[i_source]["DEJ2000"])

        non_zero_S_i = (S_i > significance_cut)
        S_i = S_i[non_zero_S_i]
        N_zeros = len(self.sourcesearch.df) - len(S_i)

        for i_given_para, given_para in enumerate(parameterized_span):
            given_ns = given_para * self.df_cat.iloc[i_source]["flux_weights"] * self.T * np.power(self.E1, self.alpha) * self.f_Aeff_dec_integration(self.df_cat.iloc[i_source]["DEJ2000"])

            ts_results[i_given_para] = self.sourcesearch.test_statistic_at_point([self.df_cat.iloc[i_source]["RAJ2000"], self.df_cat.iloc[i_source]["DEJ2000"]],
                                                                                 given_ns,
                                                                                 S_i, B_i, N_zeros)

            current_flux = given_para * self.df_cat.iloc[i_source]["flux_weights"] * np.power(self.E1 / self.E2, self.alpha)
            sweep_fluxes[i_given_para] = np.power(self.E2, 2.0) * current_flux / (4.0 * np.pi)

        return sweep_fluxes, ts_results


def prepare_skymap_coordinates(step_size):
    """
    Prepares the coordinates for the all-sky search.

    Parameters
    ----------
    step_size : float
        The steps in degrees taken in RA and Dec to
        calculate the all-sky map.

    Returns
    -------
    cords : array_like
        The (ra, dec) of each point in the sky
        that will be tested to perform the all-sky search.
    ra_len : int
        The number of RA steps.
    dec_len : int
        The number of declination steps.
    """

    ra_sweep = np.arange(0, 360, step_size)
    dec_sweep = np.arange(-90, 90, step_size)

    ra_len = len(ra_sweep)
    dec_len = len(dec_sweep)

    total_pts = dec_len * ra_len

    ras = np.zeros(total_pts)
    decs = np.zeros(total_pts)

    i_source = 0
    for iX in range(ra_len):
        for iY in range(dec_len):
            ras[i_source] = ra_sweep[iX]
            decs[i_source] = dec_sweep[iY]
            i_source += 1

    cords = np.stack((ras, decs), axis=1)

    return cords, ra_len, dec_len
