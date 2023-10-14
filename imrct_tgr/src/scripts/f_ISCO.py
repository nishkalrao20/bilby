import lal
import numpy as np
from pesummary.io import read
import lalinference.imrtgr.nrutils as nr

# Constants
G_SI = lal.G_SI  #6.67430*1e-11 
c_SI = lal.C_SI  #299792458. 
Msun_SI = lal.MSUN_SI  #1.989*1e30

data = read('git_overlap/src/data/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5')
injection = data.samples_dict['C01:IMRPhenomXPHM']   # Loading the GW150914 Posterior distributions
nmap = np.argmax(injection['log_likelihood'])   # Maximum A Posteriori values
injection_parameters = {}
for key in injection.keys():
    injection_parameters[key] = injection[key][nmap]

'''import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({"text.usetex": True,
    "font.family": "sans-serif",
    "axes.formatter.use_mathtext": True
})
for idx, i in enumerate(data.samples_dict.keys()):
    plt.hist(data.samples_dict[i]['mass_1'], label=i.split(':')[-1], color=['b', 'm', 'r'][idx], alpha=.4, bins=100, density=True)
    injection = data.samples_dict[i]   # Loading the GW150914 Posterior distributions
    nmap = np.argmax(injection['log_likelihood'])
    plt.axvline(data.samples_dict[i]['mass_1'][nmap], color=['b', 'm', 'r'][idx])
    plt.axvline(np.median(data.samples_dict[i]['mass_1']), color=['b', 'm', 'r'][idx], linestyle='--')
plt.title('Posterior Distributions of GW150914')
plt.legend()
plt.show()'''

def remnant_prms_estimate(m1, m2, a1=0., a2=0., tilt1=0., tilt2=0., phi12=0.):
    """
    Returns the mass and spin of the final remnant based on initial binary configuration.

    Parameters
    ----------
    m1 : float or numpy.array
        The mass of the first component object in the binary (in solar masses).
    m2 : float or numpy.array
        The mass of the second component object in the binary (in solar masses).
    a1 : float, optional
        The dimensionless spin magnitude of the first binary component. Default = 0.
    a2 : float, optional
        The dimensionless spin magnitude of the second binary component. Default = 0.
    tilt1 : float, optional
        Zenith angle between S1 and LNhat (rad). Default = 0.
    tilt2 : float, optional
        Zenith angle between S2 and LNhat (rad). Default = 0.
    phi12 : float, optional
        Difference in azimuthal angle between S1 and S2 (rad). Default = 0.

    Returns
    -------
    dict: 
        Dictionary of:

        * m_f: float
            Final mass of the remnant (in solar masses).
        * a_f: float
            Final dimensionless spin of the remnant.    

    """        
            
    # Use the following final mass and spin fits to calculate fISCO
    Mf_fits = ["UIB2016", "HL2016"]
    af_fits = ["UIB2016", "HL2016", "HBR2016"]

    # Final mass computation does not use phi12, so we set it to zero
    Mf = nr.bbh_average_fits_precessing(
        m1,
        m2,
        a1,
        a2,
        tilt1,
        tilt2,
        phi12=np.array([0.0]),
        quantity="Mf",
        fits=Mf_fits,
    )
    af = nr.bbh_average_fits_precessing(
        m1, m2, a1, a2, tilt1, tilt2, phi12=phi12, quantity="af", fits=af_fits
    )
    return dict(m_f=float(Mf), a_f=float(af))

def r_ISCO(m_tot, a_f):
    """
    Returns the equatorial Innermost Stable Circular Orbit (ISCO), also known as radius of the marginally stable orbit. For Kerr metric, it depends on 
    whether the orbit is prograde (negative sign) or retrograde (positive sign).
    References: 
    Eq.2.21 of _Bardeen et al. <https://ui.adsabs.harvard.edu/abs/1972ApJ...178..347B/abstract>_,
    Eq.1 of _Chad Hanna et al. <https://arxiv.org/pdf/0801.4297.pdf>_
    
    Parameters
    ----------
    m_tot = m1+m2 : float
        Binary mass (in solar masses).
    a_f : float
        Dimensionless spin parameter of the remnant compact object.

    Returns
    -------
    dict: 
        Dictionary of:

        * R_ISCO_retrograde: float
            ISCO radius for a particle in retrogade motion (in solar masses).

    """

    fac = m_tot
    z1 = 1 + np.cbrt(1 - a_f**2)*(np.cbrt(1 + a_f) + np.cbrt(1 - a_f))
    z2 = np.sqrt(3*a_f**2 + z1**2)
    risco_n = fac*(3 + z2 - np.sqrt((3 - z1)*(3 + z1 + 2*z2)))
    risco_p = fac*(3 + z2 + np.sqrt((3 - z1)*(3 + z1 + 2*z2)))
    r_dict = dict(R_ISCO_retrograde=risco_p, R_ISCO_prograde=risco_n)
    return r_dict

def f_GW_Kerr_ISCO(m_tot, a_f):         
    """
    Returns GW frequency at ISCO for a spinning BH Binary.
    References: Eq.4 of `Chad Hanna et al. <https://arxiv.org/pdf/0801.4297.pdf>`

    Parameters
    ----------
    m_tot=m1+m2 : float
        Binary Mass (in solar masses).
    a_f : float
        Final dimensionless spin magnitude of the remnant.

    Returns
    -------
    dict: 
        Dictionary of:

        * f_ISCO_retrograde: float
            GW frequency at ISCO for binaries in retrogade motion (in solar masses).
        * f_ISCO_prograde: float
            ISCO radius for a particle in prograde motion (in solar masses).  
                
    """     

    fac = c_SI**3/(2*np.pi*G_SI*m_tot*Msun_SI)
    r_res = r_ISCO(m_tot, a_f)
    r_n = r_res['R_ISCO_prograde']
    r_p = r_res['R_ISCO_retrograde']
    f_orb_isco_n = fac*(a_f + pow(r_n/m_tot, 3/2))**(-1)
    f_orb_isco_p = fac*(a_f + pow(r_p/m_tot, 3/2))**(-1)
    f_n, f_p = 2*f_orb_isco_n, 2*f_orb_isco_p    # because of quadrupolar contributions, f_gw = 2 * f_orb
    f_dict = dict(f_ISCO_retrograde=f_p, f_ISCO_prograde=f_n)
    return f_dict

res = remnant_prms_estimate(injection_parameters['mass_1'], injection_parameters['mass_2'], a1=injection_parameters['a_1'], a2=injection_parameters['a_2'], tilt1=injection_parameters['tilt_1'], tilt2=injection_parameters['tilt_2'], phi12=injection_parameters['phi_12'])
#{'m_f': 65.68754493702306, 'a_f': 0.6891859534308202} 
fdict = f_GW_Kerr_ISCO(injection_parameters['mass_1']+injection_parameters['mass_2'], res['a_f'])
#{'f_ISCO_retrograde': 39.507152199864564, 'f_ISCO_prograde': 132.83935246242206}