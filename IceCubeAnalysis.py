import numpy as np
import scipy.interpolate
from scipy.optimize import minimize_scalar


class SourceSearch:


    def __init__(self, icecube_filename):
        
        data_ra, data_dec, data_sigmas, data_file_year = self.load_icecube_data(icecube_filename)
    
        self.N = len(data_sigmas)
        self.cord_i = np.stack((data_ra, data_dec), axis=1)
        self.data_file_year = data_file_year
        self.data_sigmas = data_sigmas

        # Compute these sin/coss once to save computation time later
        self.sindec = np.sin(np.deg2rad(self.cord_i[:, 1]))
        self.cosdec = np.cos(np.deg2rad(self.cord_i[:, 1]))
        
        
    def Si_likelihood(self, cord_s, close_point_cut=None):
        """
        Calculates the signal PDF at a given
        point in the sky.

        Parameters
        ----------
        cord_s : array_like
            The (ra, dec) position on sky that is being tested.
        """
        
        if(close_point_cut is None):
            close_points = np.ones(self.N).astype('bool')
        else:
            close_points = np.sum(np.square(cord_s - self.cord_i), axis=1) < np.square(close_point_cut)
        
        cosA = (self.sindec[close_points] * np.sin(np.deg2rad(cord_s[1]))
                + self.cosdec[close_points] * np.cos(np.deg2rad(cord_s[1])) * np.cos(np.deg2rad(self.cord_i[close_points, 0] - cord_s[0])))
        great_dists = np.arccos(cosA)

        # This has to be in radians.
        data_sigmas_ = np.deg2rad(self.data_sigmas[close_points])
        S_i = 1.0 / (2.0 * np.pi * data_sigmas_ * data_sigmas_)
        S_i *= np.exp(-0.5 * np.square(great_dists / data_sigmas_))
        
        return S_i


    def calculate_likelihood(self, n_s, S_i, B_i, N_zeros=0):
        """
        Calculates the test statistic for a given
        number of clustered neutrinos (n_s) and
        given signal pdf (S_i), background pdf (B_i),
        and the total number of events (N).

        N_zeros : int
            Cheat 

        """

        result_ = n_s / self.N * S_i + (1.0 - n_s / self.N) * B_i

        if(np.any(result_ <= 0)):
            return 0.0
        else:
            return np.sum(np.log(result_)) + N_zeros * np.log((1.0 - n_s / self.N) * B_i)


    def test_statistic_at_point(self, cord_s, n_s, S_i=None, B_i=None, N_zeros=0):
        """
        Calculates the test statistic at point
        
        Parameters
        ----------
        cord_s : array_like
            The cordesian position on sky that is being tested.
        n_s : int
            The number of neutrinos being tested for that point
        """

        if(S_i is None):
            S_i = self.Si_likelihood(cord_s)
        if(B_i is None):
            B_i = self.f_B_i(cord_s[1])

        del_ln_L_n_s = self.calculate_likelihood(n_s, S_i, B_i, N_zeros)
        del_ln_L_0 = self.calculate_likelihood(0.0, S_i, B_i, N_zeros)

        return 2.0 * (del_ln_L_n_s - del_ln_L_0)


    def load_icecube_data(self, file_name):
        icecube_data = np.load(file_name, allow_pickle=True)

        data_sigmas = np.array(icecube_data["data_sigmas"])
        data_ra = np.array(icecube_data["data_ra"])
        data_dec = np.array(icecube_data["data_dec"])
        data_file_year = np.array(icecube_data["data_file_year"])

        allowed_entries = data_sigmas != 0.0
        data_ra = data_ra[allowed_entries]
        data_dec = data_dec[allowed_entries]
        data_sigmas = data_sigmas[allowed_entries]
        data_file_year = data_file_year[allowed_entries]

        return data_ra, data_dec, data_sigmas, data_file_year


    def job_submission(self, cord_s, i_source, close_point_cut=10, significance_cut=1e-10):

        S_i = self.Si_likelihood(cord_s, close_point_cut=close_point_cut)
        B_i = self.f_B_i(cord_s[1])
        
        non_zero_S_i = (S_i > significance_cut)
        S_i = S_i[non_zero_S_i]
        N_zeros = self.N - len(S_i)
                    
        # Before jumping in, we check to make sure the n_s will be positive
        slope = (self.calculate_likelihood(0.05, S_i, B_i, N_zeros) - self.calculate_likelihood(0.0, S_i, B_i, N_zeros))

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
        
        del_ln_L = (self.calculate_likelihood(n_s, S_i, B_i, N_zeros) - self.calculate_likelihood(0.0, S_i, B_i, N_zeros))

        if(i_source % 1000 == 0):
            print("%i) \t n_s = \t %f" % (i_source, n_s))

        return n_s, del_ln_L


    def load_background(self, file_name):
        data_bg = np.load(file_name, allow_pickle=True)

        self.f_B_i = scipy.interpolate.interp1d(data_bg['dec'],
                                                data_bg['B_i'],
                                                kind='cubic',
                                                bounds_error=False,
                                                fill_value="extrapolate")
