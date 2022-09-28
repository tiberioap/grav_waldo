# **Waveform AnomaLy DetectOr (WALDO)**

[![PyPI Version](https://img.shields.io/pypi/v/grav-waldo?color=)](https://pypi.org/project/grav-waldo/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tiberioap/grav_waldo/blob/main/LICENSE)

[![Orcid](https://img.shields.io/badge/orcid-A6CE39?style=for-the-badge&logo=orcid&logoColor=white)](https://orcid.org/0000-0003-1856-6881)

WALDO is a *deep learning* data quality tool developed to flag possible anomalous Gravitational Waves (GW) from Numerical Relativity (NR) catalogs.
We use a U-Net architecture to learn the waveform features of a dataset. These waveforms are timeseries $h_{lm}(t)$ of modes $(l,m)$ from the spin-weighted spherical harmonics decomposition of the GW strain $h(t,\vec x)$,

$$h_{lm}(t) = \int d\Omega h(t, \vec x)\_{-2}Y_{lm}(\theta, \phi) .$$ 

WALDO computes the mismatch between $h_{lm}(t)$ and its prediction $\bar h_{lm}(t)$ to compose a histogram. We can identify anomalous waveforms by isolating 1% of the highest measurement values. Below, the anomaly search associated with the radiation field $\psi_{32} = \ddot h_{32}$ from the [dataset](https://github.com/tiberioap/waldo/blob/main/simulations_ID.txt). Test.

<p float="central">
  <img src="figs/hist.png" width="400" />
  <img src="figs/wf.png" width="400" /> 
</p>

## Installation

To install *grav-waldo*, we can use the pip [command](https://pypi.org/project/grav-waldo/):

```pip install grav-waldo```

## Content

The project is composed of three main [codes](https://github.com/tiberioap/waldo/tree/main/waldo):
* **wfdset.py:** for pre-processing NR dataset;
* **unet.py:** the neural network;
* **waldo.py:** for mismatch evaluation and anomaly search.

Check the tutorials in [docs](https://github.com/tiberioap/waldo/tree/main/docs).
