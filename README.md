# LPOs_in_CR3BP
For visualizing the Periodic Orbits in the Circular Restricted Three-Body Problem.
Uses Howell's method [see here](https://link.springer.com/article/10.1007/BF01358403), also known
as the single shooter method.

## Periodic_Orbits.py

This is the main file. Contains functions for Howell's method
for Halo, Horizontal Lyapunov and Vertical Lyapunov orbits.
Example usages are included in the file. User can specify their own
initial conditions. Files can be saved to csvs according to the Jacobi 
constant of the periodic solution. These are the contents of the folders:

- Horizontal Lyapunov Orbits - L1 - Sun and EMS - Integration Results
- Horizontal Lyapunov Orbits - L2 - Sun and EMS - Integration Results

## lyapfig.py

This is for visualizing a nice figure of the horizontal lyapunov periodic orbits

## vel_curves.py

For visualizing zero velocity surfaces in the CR3BP
