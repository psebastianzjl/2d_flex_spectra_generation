import os, sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) <= 1:
    print("enter Tw as 1st argument")
    exit()
tw = int(sys.argv[1]) #[0, 20, 35, 80, 150]

NIU = [0.1,0.05]#0.01, 0.002, 0.05]
w_pu = 5.2 
w_pr = 5.2
resolution = 400
W_T = np.linspace(2.0,
                   6.0,
                   resolution)
TAU = [0.1, 5]#, 5]
TRAJ = [500] # trajectory number
TSTEP = 401 # time step of dynamics
GSB_TRIGGER = 'no' # get ground-state bleaching
SE_TRIGGER = 'yes' # get stimulated emission
EA_TRIGGER = 'no' # get excited state absorption

def fs2au(fs): #femtosecond to atomic unit of time
    aut = fs * 41.341374575751
    return aut

def eV2Ha(eV):
    Ha = eV / 27.211386245
    return Ha

def Ha2eV(Ha):
    eV = Ha * 27.211386245
    return eV

def envelope(w, tau):
    E = np.exp(-(w * tau) ** 2 / 4.) * tau
    return E

def envelope_nu(w, niu, tau):
    E = np.exp(-(w / niu) ** 2 / 4.) #* tau
    return E

def signal(eV, Ha, Ha_0, f, fs, delta=False): #Doorway-Function-wise
    laser = eV2Ha(eV)
    tau = fs2aut(fs)
    w_in = Ha - Ha_0
    dw = laser - w_in
    v2 = f / 2. * 3 / w_in
    if not delta:
        return envelope(dw, tau) ** 2. * v2
    else:
        return 1. * v2

def signal_sum(eV, Ha, Ha_0, f, fs, delta=False):
    laser = eV2Ha(eV)
    tau = fs2aut(fs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_in = val - Ha_0
        dw = laser - w_in
        v2 = f[i] / 2. * 3 / w_in
        if not delta:
            S += envelope(dw, tau) ** 2. * v2
        else:
            S += 1. * v2
    return S 

def dissignal(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2aut(taufs)
    w_ge = Ha - Ha_0
    dw = w_t - w_ge
    v2 = f / 2. * 3 / w_ge
    return envelope(w_t - w_pr, tau) ** 2. * v2 * niu / (niu ** 2 + dw ** 2)

def dissignal_sum(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2aut(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        v2 = f[i] / 2. * 3 / w_ge
        S += envelope(w_t - w_pr, tau) ** 2. * v2 * niu  / (niu ** 2 + dw ** 2)
    return S

def dissignal_sum_deV(w_in, deV, f, taufs, niueV, w_preV):
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2aut(taufs)
    S = 0
    for (i, val) in enumerate(deV):
        w_ge = -1. * eV2Ha(val) #given in negative
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 * niu / (niu ** 2 + dw ** 2)
    return S 

def dissignal_R(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)       #parameter frequency
    w_pr = eV2Ha(w_preV)    #fixed frequency
    niu = eV2Ha(niueV)      #broadening
    tau = fs2au(taufs)      #Tw
    w_ge = Ha - Ha_0        #Ueg
    dw = w_t - w_ge         #dispersion
    v2 = f / 2. * 3 / w_ge
    return  envelope_nu(dw, niu, tau) * envelope(w_t - w_pr, tau) ** 2. * v2 #/ complex(niu, dw)#  envelope_nu(dw, tau, niu) *

def dissignal_NR(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    w_ge = Ha - Ha_0
    dw = w_t - w_ge
    v2 = f / 2. * 3 / w_ge
    return envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, -dw)

def dissignal_sum_delta(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += 1 * v2 / complex(niu, dw)
    return S

def dissignal_sum_R(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, dw)
    return S

def dissignal_sum_NR(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, -dw)
    return S


def dissignal_sum_NR_ESA(w_in, dEeV, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    S = 0
    for (i, val) in enumerate(dEeV):
        w_ge = -1. * eV2Ha(val)
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, -dw)
    return S

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

print(str(w_pu) + 'eV as pump frequency')
print(str(w_pr) + 'eV as probe frequency')
eaEmax = 100
eaEmin = 0
print(str(tw) + ' fs as Tw')
tws = tw * 2

GSC = [1000]#1, 10, 25, 50, 100, 150, 200, 300, 1000]

for gsC in GSC:
  for niu in NIU:
    for tau in TAU:
        for TRAJNO in TRAJ:
            gsCount, exCount, eaCount = [ np.zeros([TSTEP, 2], dtype=int) for _ in range(3)]
            exTCount = gsTCount = eaTCount = 0
            for itraj in range(TRAJNO):
                trajNo = itraj + 1 
                gsfile = '/data/huang/exporttuunla/projects/Pyr/adc2/gs/TRAJ' + str(trajNo) + '/dynamics.out' #ground state dynamics output
                exfile = './ex/TRAJ' + str(trajNo) + '/dynamics.out' #excited state dynamics output
                eafile = '/data/huang/exporttuunla/projects/Pyr/adc2/exab/TRAJ' + str(trajNo) + '.exos.out' #excited absorption dynamics
                gscheck = False
                excheck = False
                eacheck = False
                if exCount[tws, 1] >= gsC:
                    break
                ############################
                #  read ground state dyn   #
                ############################
                if GSB_TRIGGER == 'yes':
                    if os.path.isfile(gsfile):
                        print(gsfile)
                        gscheck = True
                        gsTCount += 1
                        gsE = np.zeros([TSTEP,7]) #t, S0, S1, S2, S3, S4, current
                        gsf = np.zeros([TSTEP,5]) #t, f1, f2, f3, f4
                        for t in range(TSTEP):
                            gsCount[t, 0] = gsE[t, 0] = gsf[t, 0] = t * .5
                        with open(gsfile,'r') as f:
                            tstep = 0
                            for line in f:
                                if line.startswith('1. Current electronic state:'):
                                    gsCount[tstep, 1] += 1
                                    gsE[tstep, -1] =  next(f).split()[-1]
                                if line.startswith('6. QM state energies:'):
                                    gsE[tstep, 1:6] = next(f).split()[0:5]
                                if line.startswith('11. QM oscillator strengths:'):
                                    gsf[tstep, 1:5] = next(f).split()
                                    tstep += 1
                ############################
                #  read excited dynamics   #
                ############################
                if SE_TRIGGER == 'yes':
                    if os.path.isfile(exfile):
                        print(exfile)
                        excheck = True
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
                    ############################
                    #  read ESA dynamics       #
                    ############################
                    if EA_TRIGGER == 'yes':
                        if os.path.isfile(eafile):
                            print(eafile)
                            eacheck = True
                            eaTCount += 1
                            eaE = np.zeros([TSTEP,27 + 3]) #t, S5-S30, ignoring first 4
                            eaf = np.zeros([TSTEP,27 + 3]) #t, df5-df30
                            for t in range(TSTEP):
                                eaCount[t, 0] = eaE[t, 0] = eaf[t, 0] = t * .5
                            with open(eafile, 'r') as f:
                                eatstep = 0
                                eaS = 5
                                for line in f:
                                    if line.startswith(' t ='):
                                        eatstep += 1
                                        if exE[eatstep - 1, -1] <= 3 and exE[eatstep -1, -1] > 0:
                                            ealow = int(4 - exE[eatstep - 1, -1])
                                        else:
                                            ealow = 0
                                    if eaS == 31:
                                       eaS = 5
                                    if ealow != 0:
                                        if '|  ' + str(int(exE[eatstep - 1, -1])) + '^1a' in line:
                                            if float(line.replace(' ', '').split('|')[-4]) > 0:
                                                eaE[eatstep - 1, ealow - 4] = float(line.replace(' ', '').split('|')[-4]) * -1
                                                eaf[eatstep - 1, ealow - 4] = float(line.replace(' ', '').split('|')[-2])
                                            ealow += -1
                                    if str(int(exE[eatstep - 1, -1])) + '^1a   |  diplen  |' in line:
                                        if float(line.replace(' ', '').split('|')[-4]) < 0:
                                            eaE[eatstep - 1, eaS - 4] = line.replace(' ', '').split('|')[-4]
                                            eaf[eatstep - 1, eaS - 4] = line.replace(' ', '').split('|')[-2]
                                            eaS += 1
                            for (eai, eaival) in enumerate(eaE[:, 1]):
                                if eaival != 0:
                                    eaCount[eai, 1] += 1
                            for eai in range(len(eaE[:, 0])):
                                if eaEmin < np.amin(-1 * eaE[eai, :]) and np.amin(-1 * eaE[eai, :]) > 0:
                                    eaEmin = np.amin(-1 * eaE[eai, :])
                                if eaEmax > np.amax(-1 * eaE[eai, :]) and np.amax(-1 * eaE[eai, :]) > 0:
                                    eaEmax = np.amax(-1 * eaE[eai, :])
                            #print(eaEmax, eaEmin)
                            #print(eaE, eaf)
                            #print(eaE[0, 1:])
                            #print(eaE[0, 1:(27 + 4 - int(exE[0, -1]))])
                            #print(eaE[10, 1:])
                            #print(eaE[10, 1:(27 + 4 - int(exE[10, -1]))])
                            #print(eaE[0, 1:])
                            #print(eaE[0, 1:(27 + 4 - int(exE[0, -1]))])
                ############################
                #  calculating DW          #
                ############################
                for (iw_t, w_t) in enumerate(W_T):
                    GSB_R, GSB_NR = [np.zeros([resolution, 3]) for _ in range(2)] #w_tau, w_t, Intensity, Mean Intensity
                    SE_R, SE_NR = [np.zeros([resolution, 3]) for _ in range(2)] #w_tau, w_t, Intensity
                    EA_NR, EA_R = [np.zeros([resolution, 3]) for _ in range(2)] #w_tau, w_t, Intensity
                    for (iw_tau, w_tau) in enumerate(W_T):
                        if gscheck:
                            GSB_R[iw_tau, 0] = GSB_NR[iw_tau, 0] = w_tau
                            Dgg_R = dissignal_sum_R(w_tau, gsE[0, 2:6], gsE[0, 1], gsf[0, 1:5], tau, niu, w_pu)
                            Dgg_NR = dissignal_sum_NR(w_tau, gsE[0, 2:6], gsE[0, 1], gsf[0, 1:5], tau, niu, w_pu)
                            Wgg = dissignal_sum_NR(w_t, gsE[tws, 2:6], gsE[tws, 1], gsf[tws, 1:5], tau, niu, w_pr)
                            GSB_R[iw_tau, 2] = np.real(Dgg_R * Wgg)
                            GSB_NR[iw_tau, 2] = np.real(Dgg_NR * Wgg)
                        if excheck:
                            SE_R[iw_tau, 0] = SE_NR[iw_tau, 0] = w_tau
                            Dee_R = dissignal_R(w_tau, exE[0, int(exE[0, -1]) + 1], exE[0, 1], exf[0, int(exE[0, -1])], tau, niu, w_pu)
                            #Dee_NR = dissignal_NR(w_tau, exE[0, int(exE[0, -1]) + 1], exE[0, 1], exf[0, int(exE[0, -1])], tau, niu, w_pu)
                            if int(exE[tws, -1]) != 0:
                                Wee = dissignal_R(w_t, exE[tws, int(exE[tws, -1]) + 1], exE[tws, 1], exf[tws, int(exE[tws, -1])], tau, niu, w_pr)
                            else:
                                Wee = 0
                            SE_R[iw_tau, 2] = Dee_R * Wee
                            #SE_NR[iw_tau, 2] = np.real(Dee_NR * Wee)
                            if eacheck:
                                EA_NR[iw_tau, 0] = EA_R[iw_tau, 0] = w_tau
                                if int(exE[tws, -1]) != 0:
                                    if eaE[tws, 1] == 0:
                                        Wea = 0 
                                    else:
                                        Wea = dissignal_sum_NR_ESA(w_t, eaE[tws, 1:(27 + 4 - int(exE[tws, -1]))], eaf[tws, 1:(27 + 4 - int(exE[tws, -1]))], tau, niu, w_pr)
                                else:
                                    Wea = 0
                                EA_R[iw_tau, 2] = -1. * np.real(Dee_R * Wea)
                                EA_NR[iw_tau, 2] = -1. * np.real(Dee_NR * Wea)
                    if gscheck:
                        GSB_R[:, 1] = GSB_NR[:, 1] = w_t
                        if iw_t == 0:
                            GSB_R_full = GSB_R.copy()
                            GSB_NR_full = GSB_NR.copy()
                        else:
                            GSB_R_full = np.vstack((GSB_R_full, GSB_R))
                            GSB_NR_full = np.vstack((GSB_NR_full, GSB_NR))
                    if excheck:
                        SE_R[:, 1] = SE_NR[:, 1] = w_t
                        if iw_t == 0:
                            SE_R_full = SE_R.copy()
                            #SE_NR_full = SE_NR.copy()
                        else:
                            SE_R_full = np.vstack((SE_R_full, SE_R))
                            #SE_NR_full = np.vstack((SE_NR_full, SE_NR))
                    if eacheck:
                        EA_NR[:, 1] = EA_R[:, 1] = w_t
                        if iw_t == 0:
                            EA_R_full = EA_R.copy()
                            EA_NR_full = EA_NR.copy()
                        else:
                            EA_R_full = np.vstack((EA_R_full, EA_R))
                            EA_NR_full = np.vstack((EA_NR_full, EA_NR))
                #Accumulation of trajectories
                if gscheck:
                    if trajNo == 1:
                        GSB_R_full_sum = GSB_R_full.copy()
                        GSB_NR_full_sum = GSB_NR_full.copy()
                    else:
                        GSB_R_full_sum[:, -1] += GSB_R_full.copy()[:, -1]
                        GSB_NR_full_sum[:, -1] += GSB_NR_full.copy()[:, -1]
                if excheck:
                    if trajNo == 1:
                        SE_R_full_sum = SE_R_full.copy()
                        #SE_NR_full_sum = SE_NR_full.copy()
                    else:
                        SE_R_full_sum[:, -1] += SE_R_full.copy()[:, -1]
                        #SE_NR_full_sum[:, -1] += SE_NR_full.copy()[:, -1]
                if eacheck:
                    if trajNo == 1:
                        EA_R_full_sum = EA_R_full.copy()
                        EA_NR_full_sum = EA_NR_full.copy()
                    else:
                        EA_R_full_sum[:, -1] += EA_R_full.copy()[:, -1]
                        EA_NR_full_sum[:, -1] += EA_NR_full.copy()[:, -1]
            ###############
            #normalization#
            ###############
            #w_tau, w_t, SE/n, GSB/n, EA/n, S/n
            #    0,   1,    2,     3,    4,   5
            print('Trajectory Count: gs ' + str(gsTCount) + ' ; ex ' + str(exTCount) + ' ; ea ' + str(eaTCount))
            print('Valid points count: at ' + str(gsCount[tws, 0]) + ' fs, gs ' + str(gsCount[tws, 1]) + ' ; ex ' + str(exCount[tws, 1]) + ' ; ea ' + str(eaCount[tws, 1]))
            if SE_TRIGGER == 'yes':
                SE_R = SE_R_full_sum.copy()
                #SE_NR = SE_NR_full_sum.copy()
                SE_R[:, -1] /= exCount[tws, 1]
                #SE_NR[:, -1] /= exCount[tws, 1]
                #SE_tot = SE_R + SE_NR
                S_R = SE_R.copy()
                #S_NR = SE_NR.copy()
            if GSB_TRIGGER == 'yes':
                GSB_R = GSB_R_full_sum.copy()
                GSB_NR = GSB_NR_full_sum.copy()
                GSB_R[:, -1] /= gsCount[tws, 1]
                GSB_NR[:, -1] /= gsCount[tws, 1]
                if S_R.size != 0:
                    S_R = np.hstack((S_R, GSB_R[:, 2:3]))
                    S_NR = np.hstack((S_NR, GSB_NR[:, 2:3]))
                else:
                    S_R = GSB_R.copy()
                    S_NR = GSB_NR.copy()
            if EA_TRIGGER == 'yes':
                EA_R = EA_R_full_sum.copy()
                EA_NR = EA_NR_full_sum.copy()
                EA_R[:, -1] /= eaCount[tws, -1]
                EA_NR[:, -1] /= eaCount[tws, -1]
                if S_R.size != 0:
                    S_R = np.hstack((S_R, EA_R[:, 2:3]))
                    S_NR = np.hstack((S_NR, EA_NR[:, 2:3]))
                else:
                    S_R = EA_R.copy()
                    S_NR = EA_NR.copy()
            S_R = np.hstack((S_R, S_R[:, 2:3] + S_R[:, 3:4] + S_R[:, 4:5]))
            #S_NR = np.hstack((S_NR, S_NR[:, 2:3] + S_NR[:, 3:4] + S_NR[:, 4:5]))
            S = S_R.copy()
            #S[:, 2:6] += S_NR[:, 2:6]
            #Smax = np.maximum(abs(S_R[:, 2:6]).max(), abs(S_NR[:, 2:6]).max())
            #Smax = np.maximum(abs(S_R[:, 2:6]).max())
            #print(Smax)
            #S_R[:, 2:6] /= Smax 
            print('SE R: ', SE_R[:,2] )
            print('SE R: ', SE_R[:,2] / np.amax(SE_R[:,2]))
            plot_signal_two_dim(W_T, SE_R[:, 2] / np.amax(SE_R[:, 2]) , vmax=1.0, vmin=0.0, outname='2d_tr_se_tau' + str(tau) + '_' + str(tw) + 'fs' + '_traj' + str(TRAJNO) + '_niu' + str(niu) + '.png')
            with open('new2D_adc_R_tw' + str(tw) + '_tau' + str(tau) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(gsC) + '.dat', 'w') as output:
                np.savetxt(output, S_R)
            #S_NR[:, 2:6] /= Smax
            #with open('new2D_adc_NR_tw' + str(tw) + '_tau' + str(tau) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(gsC) + '.dat', 'w') as output:
            #    np.savetxt(output, S_NR)
            #S[:, 2:6] /= Smax
            #with open('new2D_adc_S_tw' + str(tw) + '_tau' + str(tau) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(gsC) + '.dat', 'w') as output:
            #    np.savetxt(output, S)
