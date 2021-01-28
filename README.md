# ConservNet
Repository for the ConservNet architecture.

<img src="/ConservNet.png" width="400" height="600">

# Requirements
- Python 3.6+
- Pytorch 1.0+ (written for 1.6)

# Run experiments
To replicate the experiments by running
'''
python ConservNet.py --system $1 --spreader $2 --iter $3 --epochs $4 --n $5 --m $6 --Q $7 --constant $8 --beta $9 --noise ${10} --indicator ${11:-""} 

'''

- S1, S2, S3 are the simulated system with invariants written on the paper.
- P1 : Simulated Lotka-Volterra equations with a prey and a pradator.
- P2 : Simulated Kepler problem with two bodies.
- P3 : Real double pendulum data from [M. Schmidt and H. Lipson, science 324, 81 (2009)](https://science.sciencemag.org/content/324/5923/81).
- The code automatically generates dataset if it isn't already exist at the 'data' folder.
- The trained model will be saved at 'result' folder.
- The argument 'indicator' can be any string, which will be concatenated to the name of the saved model.
