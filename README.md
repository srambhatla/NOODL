![NOODL](https://github.com/srambhatla/NOODL/blob/master/extras/logo.png "NOODL")

Neurally plausible alternating Optimization-based Online Dictionary Learning 

![View Paper](https://openreview.net/forum?id=HJeu43ActQ)

## MATLAB
Use the `main.m` file as your gateway to all MATLAB-based implementations. It provides the data generation as well as the use-cases of different implementations (detailed below).

### Standalone NOODL
For implementation of NOODL, the following versions are available.

#### Vanilla NOODL
Non-distributed variant of NOODL is provided in `NOODL.m`

#### Distributed
Implementation of NOODL using MATLAB's `spmd` is provided in `NOODL_dist.m`.

### Comparing NOODL with Related Techniques
These variants can be used to compare the performance of NOODL with the techniques described in [Arora'15](http://proceedings.mlr.press/v40/Arora15.html) and [Mairal](https://dl.acm.org/citation.cfm?id=1553463). Analogous to the Standalone implementations, we provide the following two variants.

#### Vanilla
The non-distributed implementation can be accessed via `compare_algos_and_NOODL_dist.m`

#### Distributed
The distributed implementation via MATLAB's `spmd` in provided in `compare_algos_and_NOODL.m`.

## Neural Implementation via TensorFlow
In addition to the MATLAB implementations, we also provide an implementation of NOODL via `TensorFlow` to showcase the highly distributed implementation using neural architecture, as shown in the paper. This is presented in `noodl_via_tensorflow.py`.

# Copyright & License Notice
NOODL is copyrighted by the Regents of the University of Minnesota. It can be freely used for educational and research purposes by non-profit institutions and US government agencies only. Other organizations are allowed to use NOODL only for evaluation purposes, and any further uses will require prior approval. The software may not be sold or redistributed without prior approval. One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions.
As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice. 

The software is also available via a standard negotiated license agreement. Contact umotc@umn.edu for specific details.
