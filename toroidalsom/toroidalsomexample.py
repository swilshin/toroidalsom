'''
File toroidalsom.py

OVERVIEW
========

  This simple example takes some points scattered on a line on the 2-torus 
  and infers the line using a self organising map.

LICENSE
=======
  
  This file is part of Simon Wilshin's toroidalsom module.

  Simon Wilshin's toroidalsom module is free software: you can redistribute 
  it and/or modify it under the terms of the GNU General Public License as 
  published by the Free Software Foundation, either version 3 of the License, 
  or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.  

ACKNOWLEDGEMENTS
================

  Thanks are owed to the Royal Veterinary College where this software was 
  developed and who supported its release under the GPL.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: Jul 2018
'''

if __name__=="__main__":
  from toroidalsom import toroidalSOM

  from numpy import linspace,cos,sin,array,pi,meshgrid
  from numpy.random import randn,seed

  from os.path import join as osjoin

  seed(0) # Make sure we always generate the same example

  # Generate some test data
  theta0 = linspace(0,2*pi,1000)
  stheta = 0.1
  sx = 0.1
  x0 = array([
    pi+pi*sin(theta0+stheta*randn(*theta0.shape))+sx*randn(*theta0.shape),
    theta0+stheta*randn(*theta0.shape)+sx*randn(*theta0.shape)
  ]).T

  # Fit map
  tfac = 100
  tscale = 10
  alpha0 = 100.0/float(x0.shape[0])
  som = toroidalSOM(20,2)
  som.fit(x0,tfac,tscale,alpha0)

  from pylab import figure,scatter,savefig,axis
  from mpl_toolkits.mplot3d import Axes3D
  
  # Plot
  fig = figure(figsize=(12,8))
  ax = fig.add_subplot(121)
  scatter(*x0.T,color='b',marker='x',s=9,lw=1,alpha=0.5)
  scatter(*som.xmap,color='r',marker='o',s=36,lw=1)
  axis('off')
  
  # Lets make a nice 3D view of the map
  # Convert to a ring torus
  X = array([
    (2 + cos(x0[...,0]))*cos(x0[...,1]),
    (2 + cos(x0[...,0]))*sin(x0[...,1]),
    -sin(x0[...,0])
  ]).T # Training data
  Y = array([
    (2 + cos(som.xmap[0]))*cos(som.xmap[1]),
    (2 + cos(som.xmap[0]))*sin(som.xmap[1]),
    -sin(som.xmap[0])
  ]).T # SOM
  theta = linspace(0.,2.*pi,100)
  u,v = meshgrid(theta,theta)
  Z = array([
    (2 + cos(u))*cos(v),
    (2 + cos(u))*sin(v),
    -sin(u)
  ]).T # The torus
  
  ax = fig.add_subplot(122, projection='3d')
  ax.scatter(X[...,0],X[...,1],X[...,2],color='b',marker='x',s=9,lw=1,alpha=0.5)
  ax.scatter(Y[...,0],Y[...,1],Y[...,2],color='r',marker='o',s=36,lw=2)
  ax.plot_wireframe(Z[...,0],Z[...,1],Z[...,2],alpha=0.1)
  axis('off')
  savefig(osjoin("..","figure","toroidalsomexample.png"))