import glob
import scipy.interpolate
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt


def load_Aeff(input_file_names, output_file_name, alpha=2.5):
    
    data_E = np.array([])
    data_dec = np.array([])
    data_Aeff = np.array([])

    for data_file in input_file_names:
        data = np.loadtxt(data_file)
        
        data_E = np.append(data_E, (data[:, 0] + data[:, 1]) / 2.0)
        data_dec = np.append(data_dec, (data[:, 2] + data[:, 3]) / 2.0)
        data_Aeff = np.append(data_Aeff, data[:, 4])

    data_E = np.power(10.0, np.array(data_E)) / 1000.0  # convert to TeV
    data_Aeff = 10000.0 * np.array(data_Aeff) # convert to cm^2

    # So, now have to integrate Aeff as function of declination
    unique_decs = np.unique(data_dec)
    x_dec_steps = np.zeros(len(unique_decs))
    y_integrate_steps = np.zeros(len(unique_decs))

    for i_unique_decs, unique_decs in enumerate(unique_decs):

        cur_E_max = data_E[data_dec == unique_decs]
        cur_Aeff = data_Aeff[data_dec == unique_decs]

        allowed_events = cur_Aeff != 0.0
        cur_E_max = cur_E_max[allowed_events]
        cur_Aeff  = cur_Aeff[allowed_events]
              
        argsort_ = np.argsort(cur_E_max)
        cur_E_max = cur_E_max[argsort_]
        cur_Aeff = cur_Aeff[argsort_]

        # so the Aeff is different year on year
        # better ways to solve it, but for now just average
        unique_energies = np.unique(cur_E_max)
        unique_Aeff = np.zeros(len(unique_energies))
        for i in range(len(unique_Aeff)):
            unique_Aeff[i] = np.mean(cur_Aeff[cur_E_max == unique_energies[i]])

        

        # functional time
        f_integrand = scipy.interpolate.interp1d(unique_energies, 
                                                 np.power(unique_energies, -alpha) * unique_Aeff,
                                                 kind='linear',
                                                 bounds_error=False,
                                                 fill_value=0)
    
        #plt.loglog(cur_E_max, np.power(cur_E_max, -alpha) * cur_Aeff)
        #plt.loglog(unique_energies, np.power(unique_energies, -alpha) * unique_Aeff)

        integrated_Aeff, int_Aeff_error = scipy.integrate.quad(f_integrand,
                                                               np.min(unique_energies),
                                                               np.max(unique_energies),
                                                               limit=5000)
        
        x_dec_steps[i_unique_decs] = unique_decs
        y_integrate_steps[i_unique_decs] = integrated_Aeff

        print("%i \t %.2f \t %.2f \t %.2f \t %.2f " % (i_unique_decs,
                                                       unique_decs,
                                                       np.min(cur_E_max),
                                                       np.max(cur_E_max),
                                                       integrated_Aeff))
    #plt.show()
    #exit()

    # sort steps, just in case
    x_dec_steps, y_integrate_steps = zip(*sorted(zip(x_dec_steps, y_integrate_steps)))
    x_dec_steps = np.array(x_dec_steps)
    y_integrate_steps = np.array(y_integrate_steps)

    x_dec_steps = x_dec_steps[y_integrate_steps > 0.0]
    y_integrate_steps = y_integrate_steps[y_integrate_steps > 0.0]

    x_dec_steps = x_dec_steps[np.logical_not(np.isinf(y_integrate_steps))]
    y_integrate_steps = y_integrate_steps[np.logical_not(np.isinf(y_integrate_steps))]

    np.savez(output_file_name,
             dec=x_dec_steps,
             Aeffintegrated=y_integrate_steps)
        
    return x_dec_steps, y_integrate_steps


if(__name__ == "__main__"):

    input_file_names = glob.glob("./data/3year-data-release/*Aeff.txt")
    alpha = 2.5

    output_file_name = "./processed_data/output_icecube_AffIntegrated_%s.npz" % alpha
    
    x_cos_dec_steps, y_integrate_steps = load_Aeff(input_file_names, output_file_name, alpha)

    plt.plot(x_cos_dec_steps, y_integrate_steps)
    plt.show()
    exit()
