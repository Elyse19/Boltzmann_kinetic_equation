# Quantum Boltzmann kinetic equation

The quantum Boltzmann kinetic equation (QBE) describes the dynamics of a Bose gas following a quench (abrupt change of a parameter of the Hamiltonian). This equation describes the evolution of state populations under the effect of inter-particle collisions. In this project, we solve the equation for a 3D Bose gas initially in the normal phase and quenched either in the normal phase or in the Bose Einstein condensate (BEC) phase. For now, the project is divided into two subprojects:

1. General QBE solver 
   -3D_kin_eq.py with two option modules: naive_integration.py, precise_integration.py
2. QBE solver for a degenerate Bose gas evolving in a weak spatial disordered potential under the influence of an external drive 
   -split_step.py

Cite

##Physics

1. General QBE

The dimensionless QBE describing the evolution of the energy distribution $n_\epsilon(t)$ solved in this subproject is :

$$
\partial_t n_{\epsilon} = \mathcal{I} \int_{\epsilon_1,\epsilon_2,\epsilon_3>0} d\epsilon_1 d\epsilon_2 W(\epsilon,\epsilon_1,\epsilon_2) [(n_{\epsilon} + n_{\epsilon_3})n_{\epsilon_1}n_{\epsilon_2}- (n_{\epsilon_1} + n_{\epsilon_2})n_{\epsilon}n_{\epsilon_3}],
$$

with $\epsilon$ is a dimensionless energy in units of $\epsilon_0 = \frac{\hbar^2}{2m}(4\pi^2\rho_0)^{2/3}$, the dimensionless time $t$ is in units of $\hbar/\epsilon_0$ and $\mathcal{I} = g^2m^3\epsilon_0/(2\pi^3\hbar^6)$ is a dimensionless parameter. This equation is the same in all dimensions, except for the collision kernel $W(\epsilon,\epsilon_1,\epsilon_2)$ which in 3D is :

$$
W(\epsilon,\epsilon_1,\epsilon_2) = \frac{\text{min}(\sqrt{\epsilon},\sqrt{\epsilon_1}, \sqrt{\epsilon_2}, \sqrt{\epsilon_3})}{\sqrt{\epsilon}}.
$$

Since the system is isolated, both the number of particles and the total energy are conserved : 
$$
\begin{pmatrix}
\int_0^\infty d\epsilon \nu_\epsilon\, n_\epsilon(t)&=\rho_0, \\
\int_0^\infty d\epsilon \epsilon \nu_\epsilon\, n_\epsilon(t)&=\rho_0 E.
\end{pmatrix}
$$

2. QBE including the effect of disorder/drive



##Installation

##Numerical method

##Benchmark

##Example plots

##Contact
