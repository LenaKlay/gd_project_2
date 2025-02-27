# Modelling daisy quorum drive: A short-term bridge across engineered fitness valleys

This code is associated with article <https://doi.org/10.1371/journal.pgen.1011262>. It simulates the spatial and temporal spread of the "daisy quorum drive", a construct that links a self-exhausting daisy-chain gene drive (see <https://doi.org/10.1073/pnas.1716358116>) with a fitness-valley construct, here a two-locus toxin-antidote system (see <https://doi.org/10.1006/jtbi.2001.2357>). The daisy quorum drive was proposed conceptually by Min et al. <https://doi.org/10.1101/115618> as a promising new approach to reduce the risks of spillovers while maintaining a low introduction threshold.

## Authors

This article has been written by Frederik J.H. de Haas, Léna Kläy, Florence Débarre and Sarah P. Otto. The code available in this repository has been written by Léna Kläy. It simulates the propagation of the daisy quorum drive in various homogeneous or heterogeneous environments, with variable spatial steps. It uses a Crank-Nicolson finite difference method and usually conserves a unique diffusion rate regardless of step size. The  relationship between the diffusion rate (D) and the migration rate (m) is given by D = (m * dx^2) / (2 * dt) where dx is the spatial step size and dt the time step size.

Another code has been written by Frederik J.H. de Haas, available here <https://github.com/freekdh/popgen-gene-drive>. It uses an individual based approach.  

## Contents

This Github repository is composed of several folders:

1) `Functions` contains the code to run the simulations (.py). It also contains a `README.rmd` file detailing each function.

2) `Outputs` stores the results of the simulations. It also contains a file `save` that contains the 'saved' outputs.

3) `Illustrations` contains important illustrations, usually improved with Inkscape.

4) `Migale` contains the code to run the heaviest simulations on the cluster Migale (INRAE, doi: 10.15454/1.5572390655343293E12) as well as some outputs of previous simulations.
