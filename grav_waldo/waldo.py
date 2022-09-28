import os, h5py
import numpy as np
import matplotlib.pyplot as plt

from .evaltoolkit import Mismatch
from .wfdset import DataGen, DsetBuilder

import tensorflow as tf
from tensorflow.keras.models import load_model

class Waldo:
    
    '''
    The Waveform AnomaLy DetectOr (WALDO) uses the UNet architecture to learn the Numerical 
    Relativity (NR) catalog features. The waveforms (wfs) are timeseries hₗₘ(t) of modes 
    (l,m) from the spin-weighted spherical harmonics decomposition of the gravitational wave 
    strain h(t,x⃗),

            rh(t, x⃗)/M = Σₗₘ hₗₘ(t) ₛYₗₘ(θ, ϕ) .

    WALDO compute the mismatch between hₗₘ and its prediction h̄ₗₘ, and packs the results with
    a (l,m) label, together with the identification simulation number ID, the binary parameters
    (q, χ⃗1, χ⃗2, e), hₗₘ and h̄ₗₘ wfs. To save/load such packages, we use the "save_load" 
    initialization parameter.


    The class initializes with four parameters:
    · path_dset: the dataset path/name;
    · path_unet: the trained unet path/name;
    · path_waldo: the waldo results path/name for saving/loading;
    · save_load: set the integer for (0 → no saving, no loading), 
      (1 → saving), and (2 → loading).


    There are two methods:

    → plot_histogram(...): it plots the mismatch histograms according to the (l,m) mode,
      and provides the "quantile" parameter to isolate wfs with high mismatch value.  
      · l: mode number;
      · m: mode number;
      · quantile: float number to determine the best mismatch values, (defatult 0.99);
      · bins: integer number of histogram bins;
      · save: boolean key to allow saving;
      · save_path: path where the figure is saved;
      · param_space: the list of parameter-space boundary conditions. Use the parameters
        (q, chi11, chi12, chi13, chi21, chi22, chi23, e) for setup. Default: 

        param_space = ["q < 5.0", 
                       "abs(chi11) <= 1.0", 
                       "abs(chi12) <= 1.0", 
                       "abs(chi13) <= 1.0", 
                       "abs(chi21) <= 1.0", 
                       "abs(chi22) <= 1.0", 
                       "abs(chi23) <= 1.0",
                       "e < 1.0"]

    → plot_waveforms(...): it plots hₗₘ and h̄ₗₘ wfs chosen in plot_histogram().
      · x_limit: a list/tuple with the time plot limits (default [-200,100]);
      · save: boolean key to allow saving;
      · save_path: path where the figure is saved;
      · wf_label: y-axis label for ("h_lm" → $rh_{lm}/M$) and ("psi_lm" → $rM\psi_{lm}$). 
    '''
    
    def __init__(self, path_dset, path_unet, path_waldo='./waldo', save_load=0):
                        
        with h5py.File(path_dset + ".h5", "r") as f:
            Nf, _ = np.array(f['X']).shape

        self.name = path_waldo.split("/")[-1]
        
        unet = load_model(path_unet)
        
        IDs = np.arange(Nf)
            
        args = {'IDs':IDs, 'batch_size':32, 'shuffle':False, 'path_dset':path_dset}

        self.data = DataGen(**args)

        wf_IDs, Cij, modes, shift, norm, t = self.data.attached_data()

        IDs = {f'{l}{m}':[] for l, m in modes}
        params = {f'{l}{m}':[] for l, m in modes}
        mismatch = {f'{l}{m}':[] for l, m in modes}
        
        nr_wf = {f'{l}{m}':[] for l, m in modes}
        nn_wf = {f'{l}{m}':[] for l, m in modes}

        if save_load < 2:

            l_size = modes[-1][0] - 1
            m_size = modes[-1][0] - modes[-1][-1] + 1

            dset = DsetBuilder()

            for wf_ID, (X, NR_wf) in zip(wf_IDs, self.data):

                NN_wf = unet(NR_wf)
                
                for ID, x, wf1, wf2 in zip(wf_ID, X, NR_wf, NN_wf):

                    l = 2 + int(np.argmax(x[0:l_size]))
                    m = l - int(np.argmax(x[l_size:l_size+m_size])) 
                    p = dset.paramBack(x[l_size+m_size:], Cij)

                    wf2 = wf2.numpy()

                    h1 = (wf1.T[0] + 1j*wf1.T[1])
                    h2 = (wf2.T[0] + 1j*wf2.T[1])

                    nr_wf[f'{l}{m}'].append(h1)
                    nn_wf[f'{l}{m}'].append(h2)
                    
                    IDs[f'{l}{m}'].append(ID)
                    params[f'{l}{m}'].append(p)
                    mismatch[f'{l}{m}'].append(Mismatch(t, h1, h2))
                    
            if save_load == 1:
                with h5py.File(path_waldo + ".h5", "w") as f:
                    for lm, x in IDs.items():
                        f.create_dataset(f"IDs_{lm}", data=x)
                        
                    for lm, x in params.items():
                        f.create_dataset(f"params_{lm}", data=x)
                        
                    for lm, x in mismatch.items():
                        f.create_dataset(f"mismatch_{lm}", data=x)
                        
                    for lm, x in nr_wf.items():
                        f.create_dataset(f"nr_wf_{lm}", data=x)

                    for lm, x in nn_wf.items():
                        f.create_dataset(f"nn_wf_{lm}", data=x)
                        
                        
        elif save_load == 2:
            with h5py.File(path_waldo + ".h5", "r") as f:
                for l, m in modes:                    
                    IDs[f'{l}{m}'] = np.array(f[f"IDs_{l}{m}"])
                    params[f'{l}{m}'] = np.array(f[f"params_{l}{m}"])
                    mismatch[f'{l}{m}'] = np.array(f[f"mismatch_{l}{m}"])
                    nr_wf[f'{l}{m}'] = np.array(f[f"nr_wf_{l}{m}"])
                    nn_wf[f'{l}{m}'] = np.array(f[f"nn_wf_{l}{m}"])
        
        self.IDs = IDs
        self.params = params
        self.mismatch = mismatch
        self.nr_wf = nr_wf
        self.nn_wf = nn_wf
            
    
    def plot_histogram(self, l=2, m=2, quantile=0.99, bins=50, save=False, save_path="./", 
                       param_space=["q < 5.0", "abs(chi11) <= 1.0", "abs(chi12) <= 1.0", 
                                    "abs(chi13) <= 1.0", "abs(chi21) <= 1.0", "abs(chi22) <= 1.0", 
                                    "abs(chi23) <= 1.0","e < 1.0"]):
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        op = "(" + ") & (".join(param_space) + ")"
        
        self.lm = f'{l}{m}'
        p = self.params[self.lm]

        q = np.round(p.T[0], 3)
        chi11 = np.round(p.T[1], 3)
        chi12 = np.round(p.T[2], 3)
        chi13 = np.round(p.T[3], 3)
        chi21 = np.round(p.T[4], 3)
        chi22 = np.round(p.T[5], 3)
        chi23 = np.round(p.T[6], 3)
        e = np.round(p.T[7], 3)

        j = np.where(eval(op))[0]

        self._IDs = self.IDs[self.lm][j]
        self._params = self.params[self.lm][j]
        self._mismatch = self.mismatch[self.lm][j]
        self._nr_wf = self.nr_wf[self.lm][j]
        self._nn_wf = self.nn_wf[self.lm][j]

        mis = self._mismatch
        _quantile = np.quantile(mis, quantile)
        self.idx = np.where(mis >= _quantile)[0]

        num = np.arange(mis.size)

        print(f"Mismatch: average={np.mean(self._mismatch):.4e}, min={min(self._mismatch):.4e}, max={max(self._mismatch):.4e}")
        print(f"Number of isolated waveforms =", self.idx.size)
        print("Total number of wavefrms =", self._IDs.size)

        plt.rcParams.update({"text.usetex": True,
                     "font.family": "DejaVu Sans"})

        s = 24
        fig, ax = plt.subplots(1, 2, figsize=(20,6))

        ax[0].hist(mis, bins=bins, histtype="step", linewidth=2)

        ax[0].axvline(x=mis.mean(), color='green', lw=3, alpha=0.6, label=f'Mean = {mis.mean():.2e}')
        ax[0].axvline(x=_quantile, color='red', lw=3, alpha=0.3, label=f'Quantile {quantile} = {_quantile:.2e}')
        ax[0].set_ylabel("Number of waveforms", fontsize=s)
        ax[0].set_xlabel("Mismatch", fontsize=s)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].xaxis.set_tick_params(labelsize=s-7)
        ax[0].yaxis.set_tick_params(labelsize=s-7)
        ax[0].legend(fontsize=s-2)

        ax[1].plot(num, mis, 'o')
        ax[1].plot(num[self.idx], mis[self.idx], 'o')
        ax[1].set_ylabel("Mismatch", fontsize=s)
        ax[1].set_xlabel("Simulations", fontsize=s)
        ax[1].set_xticklabels([])
        ax[1].xaxis.set_tick_params(labelsize=s-7)
        ax[1].yaxis.set_tick_params(labelsize=s-7)
        ax[1].grid(True)

        if save: plt.savefig(f"{save_path}{self.name}_histogram_{l}{m}.pdf")
        else: plt.show()
            
            
    def plot_waveforms(self, x_limit=[-200, 100], save=False, save_path='./', wf_label="h_lm"):
    
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        wf_IDs, Cij, modes, shift, norm, t = self.data.attached_data()

        l_size = modes[-1][0] - 1
        m_size = modes[-1][0] - modes[-1][-1] + 1

        if wf_label == "h_lm":
            y_label = "$r\,\mathit{Re}\{h_{"+self.lm+"}\}/M$"
        
        elif wf_label == "psi_lm":
            y_label = "$rM\,\mathit{Re}\{\psi_{"+self.lm+"}\}$"

        else:
            y_label = f"waveform ({self.lm[0]},{self.lm[1]})"
            
        plt.rcParams.update({"text.usetex": True,
                             "font.family": "DejaVu Sans"})
        
        for i in self.idx:

            p = self._params

            print(f"Mismatch = {self._mismatch[i]:.4e}")
            print(f"NR metadata: [ID = {self._IDs[i]}] q = {p[i][0]:.2}", end="; ") 
            print(f"chi1 = ({p[i][1]:.2e}, {p[i][2]:.2e}, {p[i][3]:.2e})", end="; ") 
            print(f"chi2 = ({p[i][4]:.2e}, {p[i][5]:.2e}, {p[i][6]:.2e})", end="; ") 
            print(f"e = {p[i][7]:.1e}")

            h = norm*self._nr_wf[i]*np.exp(1j*shift)
            g = norm*self._nn_wf[i]*np.exp(1j*shift)

            s = 24
            fig, ax =plt.subplots(1, 2, figsize=(20,6))

            ax[0].plot(t, h.real)
            ax[0].plot(t, g.real, '--')
            ax[0].set_xlim(*x_limit)
            ax[0].set_ylabel(y_label, fontsize=s)
            ax[0].set_xlabel("$t/M$", fontsize=s)
            ax[0].xaxis.set_tick_params(labelsize=s-4)
            ax[0].yaxis.set_tick_params(labelsize=s-4)
            ax[0].grid()

            ax[1].plot(t, abs(h), label='$NR$')
            ax[1].plot(t, abs(g), '--', label="$NN$")
            ax[1].plot(t, 10*(abs(h)-abs(g)), alpha=0.5, label="$10\\times|NR-NN|$")
            ax[1].legend(fontsize=s-4)
            ax[1].set_xlim(*x_limit)
            ax[1].set_xlabel("$t/M$", fontsize=s)
            ax[1].xaxis.set_tick_params(labelsize=s-4)
            ax[1].yaxis.set_tick_params(labelsize=s-4)
            ax[1].grid()
            
            if save: plt.savefig(f"{save_path}{self.name}_wf_l{self.lm[0]}m{self.lm[1]}_ID{self._IDs[i]}.pdf")
            else: plt.show()
