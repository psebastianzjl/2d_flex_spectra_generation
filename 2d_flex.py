import os, sys
import numpy as np
import matplotlib.pyplot as plt


if len(sys.argv) <= 1:
    print("enter Tw as 1st argument")
    exit()
tw = int(sys.argv[1])

NIU = [0.1, 0.05] # Dephasing in eV
w_pu = 5.2 # Energy of the pump pulse in eV
w_pr = 5.2 # Energy of the probe pulse in eV
resolution = 400
W_T = np.linspace(2.0,
                   6.0,
                   resolution)
TAU_PROBE = 30 # Width of the envelope of the probe pulse in fs
TAU = [15] # Width of the envelope of the pump pulse in fs
TRAJ = [500] # Number of trajectories, can also be list of int
TSTEP = 401 # Number of time steps in dynamics simulation

num_states = 5 # total number of states (GS+excited states)


def envelope(w, tau):
    """
    Function to predict the intensity of the signal using a Gaussian envelope.
    Parameters
    ----------
    w : float
        Energy of the probe pulse in atomic units
    tau : float
        Width of the envelope in atomic units

    Returns
    -------
    Intensity of the signal
    """
    E = np.exp(-(w * tau) ** 2 / 4.) * tau
    return E

def envelope_nu(w, niu, tau):
    E = np.exp(-(w / niu) ** 2 / 4.) * tau
    return E

def envelope_flex(w, tau, niu):
    E = np.exp(-(w ** 2) / (2 / (tau ** 2) + 4 * niu ** 2)) * tau
    return E

def signal(eV, Ha, Ha_0, f, fs, niueV, delta=False):
    """
    Function that predicts the intensity of a signal with respect to an envelope or a delta function, Doorway
    Parameters
    ----------
    eV : float
        Energy of the probe pulse in eV
    Ha : float
        Energy of the populated state in atomic units
    Ha_0 : float
        Energy of the electronic ground state in atomic units
    f :  float
        Oscillator strength for the transition between the populated state and the electronic ground state
    fs : float
        Width of the laser pulse envelope
    delta :  bool, optional
        Switch to pulse width of 0 fs

    Returns
    -------
    Intensity of the signal
    """
    laser = eV2Ha(eV)
    niu = eV2Ha(niueV)      #broadening
    tau = fs2aut(fs)
    if f<0:
        f=0
    w_in = Ha - Ha_0
    dw = laser - w_in
    if w_in == 0:
        v2 = 0
    elif w_in != 0:
        v2 = f / 2. * 3 / w_in
    if not delta:
        return envelope_flex(dw, tau, niu) ** 2. * v2
    else:
        return 1. * v2


def dissignal_R(w_in, Ha, Ha_0, f, taufs, niueV, w_preV):
    """
    Function that predicts the intensity of a rephasing dissipation signal with respect to an envelope or a delta function, Doorway

    Parameters
    ----------
    w_in : float
        Energy of the pump pulse in eV
    Ha : float
        Energy of the populated state in atomic units
    Ha_0 : float
        Energy of the electronic ground state in atomic units
    f :  float
        Oscillator strength for the transition between the populated state and the electronic ground state
    taufs : float
        Width of the envelope in femto-seconds
    niueV : float
        Dephasing in eV
    w_preV : float
        Energy of the probe pulse in eV

    Returns
    -------
    Intensity of the rephasing dissipation signal
    """
    w_t = eV2Ha(w_in)       #parameter frequency
    w_pr = eV2Ha(w_preV)    #fixed frequency
    niu = eV2Ha(niueV)      #broadening
    tau = fs2aut(taufs)     #Tw
    w_ge = Ha - Ha_0        #Ueg
    dw = w_t - w_ge         #dispersion
    v2 = f / 2. * 3 / w_ge
    return envelope_nu(dw, niu, tau) * envelope(w_t - w_pr, tau) ** 2. * v2#/ complex(niu, dw)


def eV2Ha(eV):
    """
    Function that converts energy from electron Volt (eV) to atomic units (hartree).
    Parameters
    ----------
    eV : float
    Energy in eV

    Returns
    -------
    Energy in atomic units (hartree)
    """
    Ha = eV / 27.211386245
    return Ha

def fs2aut(fs):
    """
     Function that converts time from femto-seconds to atomic units.
     Parameters
     ----------
     fs : float
     Time in femto-seconds that is to be converted to atomic units of time

     Returns
     -------
     Time in atomic units
     """
    aut = fs * 41.341374575751
    return aut

def plot_signal_two_dim(energy, value, vmax = 0.3, vmin = -0.3,outname='spectrum.png'):
    """
    Function that plots the signal.
    Parameters
    ----------
    energy : list or array, float
    List or array containing the information about the energy
    value : list or array, float
    List or array containing information the signal intensities.
    outname : str, optional
    Name of the outputfile

    Returns
    -------
    PNG-file containing the spectrum.
    """
    value = np.reshape(value, (400, 400))
    value = np.nan_to_num(value)
    #print("Value post reshape: ", value)
    #energy = np.array(energy[0:121], dtype=float)
    fig, ax = plt.subplots(1,1)
    c = ax.pcolormesh(energy, energy, value, cmap='rainbow', vmin=vmin, vmax=vmax, shading='nearest')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel(r'$\hbar\omega_{\tau}$, [eV]', fontsize=20)
    ax.set_ylabel(r'$\hbar\omega_{t}$, [eV]', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlim(4.0, 6.0)
    #plt.show()
    plt.savefig(outname, bbox_inches='tight',dpi=400)

print(str(w_pu) + ' eV as pump frequency')
print(str(w_pr) + ' eV as probe frequency')
eaEmax = 100
eaEmin = 0
print(str(tw) + ' fs as Tw')
tws = tw * 2

for niu in NIU:
    for tau in TAU:
        for TRAJNO in TRAJ:
            exCount= np.zeros([TSTEP, 2], dtype=int)
            exTCount =  0
            for itraj in range(TRAJNO):
                trajNo = itraj + 1
                exfile = './ex/TRAJ' + str(trajNo) + '/dynamics.out' #excited state dynamics output
                ############################
                #  read excited dynamics   #
                ############################
                if os.path.isfile(exfile):
                    print(exfile)
                    exTCount += 1
                    exE = np.zeros([TSTEP, 7]) #t, S0, S1, S2, S3, S4, current
                    exf = np.zeros([TSTEP, 5]) #t, f1, f2, f3, f4
                    for t in range(TSTEP):
                        exCount[t, 0] = exE[t, 0] = exf[t, 0] = t * .5
                    with open(exfile,'r') as f:
                        tstep = 0
                        for line in f:
                            if line.startswith('1. Current electronic state:'):
                                exCount[tstep, 1] += 1
                                exE[tstep, -1] =  next(f).split()[-1]
                            if line.startswith('6. QM state energies:'):
                                exE[tstep, 1:6] = next(f).split()[0:5]
                            if line.startswith('11. QM oscillator strengths:'):
                                exf[tstep, 1:5] = next(f).split()
                                tstep += 1
                else: continue
                ############################
                #  calculating DW          #
                ############################
                for (iw_t, w_t) in enumerate(W_T):
                    SE = np.zeros([resolution, 3])  #w_tau, w_t, Intensity
                    for (iw_tau, w_tau) in enumerate(W_T):
                        SE[iw_tau, 0] = w_tau
                        Dee = dissignal_R(w_tau, exE[0, int(exE[0, -1]) + 1], exE[0, 1], exf[0, int(exE[0, -1])], tau, niu, w_pu) # R or NR does not matter here
                        if int(exE[tws, -1]) != 0:
                            Wee = signal(w_t, exE[tws, int(exE[tws, -1]) + 1], exE[tws, 1], exf[tws, int(exE[tws, -1])], TAU_PROBE, niu)
                        else:
                            Wee = 0
                        SE[iw_tau, 2] = np.real(Dee * Wee)

                    SE[:, 1] = w_t
                    if iw_t == 0:
                        SE_full = SE.copy()
                    else:
                        SE_full = np.vstack((SE_full, SE))

                #Accumulation of trajectories
                if trajNo == 1:
                    SE_full_sum = SE_full.copy()
                else:
                    SE_full_sum[:, -1] += SE_full.copy()[:, -1]

            print('Trajectory Count: ex ' + str(exTCount))
            SE = SE_full_sum.copy()
            SE[:, -1] /= exCount[tws, 1]
            Smax = SE[:, 2].max()
            SE[:, 2] /= Smax
            plot_signal_two_dim(W_T, SE[:, 2], vmax=1.0, vmin=0.0, outname='2d_tr_se_flex_tau' + str(tau) + '_tau_probe' + str(TAU_PROBE) + '_' + str(tw) + 'fs' + '_traj' + str(TRAJNO) + '_niu' + str(niu) + '.png')
            with open('new2D_adc_flex_tw' + str(tw) + '_tau' + str(tau) + '_tau_probe' + str(TAU_PROBE) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(TRAJNO) + '.dat', 'w') as output:
                np.savetxt(output, SE)
