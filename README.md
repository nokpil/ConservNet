# ConservNet
This repository contains the official PyTorch implementation of **ConservNet** from:
**Discovering conserved quantities from grouped data by neural network**
Seungwoong Ha and Hawoong Jeong.

**Abstract** : Invariants and conservation laws convey critical information about the underlying dynamics, yet it is generally infeasible to find those without any prior knowledge. We propose ConservNet to achieve this goal, a neural network to extract a conserved quantity from a grouped data where the members of each group share their own invariant. By constructing neural networks with a novel and intuitive loss function called noise-variance loss, ConservNet learns hidden invariants in each group of multi-dimensional observables in a data-driven, end-to-end manner. We demonstrate the capability of our model with five simulated systems with invariants and a real-world double pendulum trajectory. ConservNet successfully discovers underlying invariants from the systems from a small number of data less than several thousand. Since the model is robust to noise and data conditions compared to baseline, our approach is directly applicable to experimental data for discovering hidden conservation law and relationships between variables. 

<img src="/ConservNet.png" width="400" height="600">

## Requirements
- Python 3.6+
- Pytorch 1.0+ (written for 1.6)

## Run experiments
To replicate the experiments by running
```
python ConservNet.py --system $1 --spreader $2 --iter $3 --epochs $4 --n $5 --m $6 --Q $7 --constant $8 --noise $9 --indicator ${10:-""} 

```
Similarly, we have implemented Siamese Neural Network (SNN) from  [S.  J.  Wetzel,  R.  G.  Melko,  J.  Scott,  M.  Panju,    and
6V. Ganesh, Physical Review Research2, 033499 (2020)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033499).

```
python Siam.py --system $1 --iter $3 --epochs $4 --n $5 --m $6 --noise $7 --indicator ${8:-""} 

```

### Argument descriptions
- system : Type of system. 'S1', 'S2', 'S3', 'P1', 'P2', 'P3'
- iter : iteration number. Perform same experiment multiple times (default: 1)
- spreader : Type of spreader. 'L1', 'L2', 'L8' (default : L2)
- epochs : total number of training epochs (default: 10000) 
- n : batch number (default: 20)
- m : batch size (default: 200)
- Q : spreading constant (default: 1.0)
- R : max norm of injected noise (default: 1.0)
- noise : scale of added noise (default : 0.0)
- indicator : string which will be concatenated to the name of the saved model. (default: '')

## System specifications
- S1, S2, S3 are the simulated system with invariants written on the paper.
- P1 : Simulated Lotka-Volterra equations with a prey and a pradator.
- P2 : Simulated Kepler problem with two bodies.
- P3 : Real double pendulum data from [M. Schmidt and H. Lipson, science 324, 81 (2009)](https://science.sciencemag.org/content/324/5923/81).
  - If you set the system as P3, batch_number and batch_size will be fixed to 1 and 818, ignoring arguments n and m.
  - Note that P3 for SNN is not available.
- The code automatically generates dataset if it isn't already exist at the 'data' folder.
- The trained model will be saved at 'result' folder.
