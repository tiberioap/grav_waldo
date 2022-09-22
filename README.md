# **Waveform AnomaLy DetectOr (WALDO)**
___
WALDO is a *deep learning* data quality tool developed to flag possible anomalous Gravitational Waves (GW) from Numerical Relativity (NR) catalogs.
We use an U-Net architecture to learns the waveform features of a dataset. These waveforms are timeseries $h_{lm}(t)$ of modes $(l,\,m)$ from the spin-weighted spherical harmonics decomposition of the GW strain $h(t,\, \vec x)$,

$$h_{lm}(t) = \int d\Omega\, h(t,\, \vec x) _{-2}Y_{lm}(\theta,\, \phi) \, .$$ 

WALDO computes the mismatch between $h_{lm}(t)$ and its prediction $\bar h_{lm}(t)$ to compouse a histogram. Isolating 1% of the highest measurement values, we can identify anomalous waveforms. 

___
The project is compoused by three main codes:
* **wfdset:** for pre-processing NR dataset;
* **unet:** the neural network;
* **waldo:** for mismatch evaluation and anomaly search.

Check the tutorials.
