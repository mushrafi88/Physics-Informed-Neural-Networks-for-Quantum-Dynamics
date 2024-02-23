# Physics-Informed Neural Networks (PINNs) for Solving Physical Systems

This repository contains implementations of Physics-Informed Neural Networks (PINNs) to solve various physical systems described by partial differential equations (PDEs) or ordinary differential equations (ODEs). PINNs leverage the power of deep learning to approximate solutions to complex physical and quantum mechanical problems by incorporating physical laws as part of their loss function.

## Table of Contents

- [1D Harmonic Oscillator](#1d-harmonic-oscillator)
- [1D Quantum Harmonic Oscillator](#1d-quantum-harmonic-oscillator)
- [2-D PDE Example](#2-d-pde-example)
- [Symmetric Infinite Potential Well](#symmetric-infinite-potential-well)
- [Non-linear Schrödinger Equation](#non-linear-schrödinger-equation)
- [Installation](#installation)
- [Acknowledgments](#acknowledgments)

## 1D Harmonic Oscillator

The 1D damped harmonic oscillator is described by the differential equation:

$$m \dfrac{d^2 x}{d t^2} + \mu \dfrac{d x}{d t} + kx = 0$$

with initial conditions $x(0) = 1$, $\dfrac{d x}{d t}(0) = 0$. We focus on solving for the under-damped state ($\delta < \omega_0$).

Original code inspired by [benmoseley](https://github.com/benmoseley/harmonic-oscillator-pinn).

## 1D Quantum Harmonic Oscillator

Solves the quantum harmonic oscillator problem, given by the Hamiltonian:

$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2$$

and the corresponding Schrödinger equation. The solution involves the eigenfunctions $\psi_n(x)$ expressed in terms of Hermite polynomials $H_n$.

## 2-D PDE Example

A simple 2-D PDE problem:

$$\frac{du}{dx} = 2 \frac{du}{dt} + u$$

with boundary condition $u(x,0) = 6e^{-3x}$. The analytical solution is $u(x,t) = 6e^{-3x-2t}$.

Original code by [nanditadoloi](https://github.com/nanditadoloi/PINN/tree/main).

## Symmetric Infinite Potential Well

Solves the time-dependent Schrödinger equation for a particle in a symmetric infinite potential well, focusing on the solutions within the well and the quantized energy levels.

## Non-linear Schrödinger Equation

Addresses the non-linear Schrödinger equation with periodic boundary conditions, showcasing the use of PINNs in solving complex, non-linear PDEs.

$$
i \frac{\partial \psi}{\partial t} + \frac{1}{2} \frac{\partial^2 \psi}{\partial x^2} + |\psi|^2 \psi = 0, \quad x \in [-5, 5], \quad t \in [0, \frac{\pi}{2}],
$$


## Installation

To set up your environment to run these examples, you will need Python 3 and the following libraries:
- NumPy
- PyTorch
- Matplotlib

Clone the repository:

```bash
git clone https://github.com/mushrafi88/Physics-Informed-Neural-Networks-for-Quantum-Dynamics.git
```

Install the required Python packages. For nixos users use the following directly to setup the environment with cuda enabled.

```
nix-shell cuda-pytorch-env.nix
```
For nixpkgs users use the following 
```bash
git clone https://github.com/mushrafi88/Physics-Informed-Neural-Networks-for-Quantum-Dynamics.git
cd Physics-Informed-Neural-Networks-for-Quantum-Dynamics 
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
```

## Acknowledgments

- Original PINN methodology by [Raissi et al.](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
- Example implementations inspired by contributions from [benmoseley](https://github.com/benmoseley/harmonic-oscillator-pinn) and [nanditadoloi](https://github.com/nanditadoloi/PINN/tree/main).


