# Point Vortex Dynamics
This script uses Python to simulate and visualize point vortices in the plane. This version uses the scipy equivalent of 
Matlab's ODE45 solver, which works well unless the system of ODEs becomes stiff, which occurs when two point vortices undergo a
near collision.

Future to do:
  * Implement symplectic solver so Hamiltonian is conserved.
  * Implement bounded planar geometry (either simply-connected or mulitply-connected).
  * Implement spherical or other surface geometries.
  
