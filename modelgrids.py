#!/usr/bin/env python
#
"""modelgrids module.
Define classes that give acces to model grid metrics and operators
(e.g. gradients)

This submodule is still under development. See the project
[wiki page](https://github.com/lesommer/oocgcm/wiki/modelgrids_design)

"""

import numpy as np
import xarray as xr
import xarray.ufuncs as xu # ufuncs like np.sin for xarray

#=============================== Parameters ====================================
# Physical parameters
grav = 9.81                  # acceleration due to gravity (m.s-2)
omega = 7.292115083046061e-5 # earth rotation rate (s-1)
earthrad = 6371229            # mean earth radius (m)

# Maths parameters
deg2rad = np.pi / 180.
mod360 = lambda x: np.mod(x+180,360)-180

#======================== General Purpose Tools ================================
def return_xarray_dataset(filename,chunks=None):
    """Return an xarray dataset corresponding to filename.
    """
    return xr.open_dataset(filename,chunks=chunks,lock=False)

def get_xarray_dataarray(filename,varname,chunks=None):
    """Return a xarray dataarray corresponding to varname in filename.
    """
    ds = return_xarray_dataset(filename,chunks=chunks)
    return ds[varname]

#======================= NEMO Specific Material ================================
class variables_holder_for_2d_grid_from_nemo_ogcm:
    """This class create the variables used in generic_2d_grid.
    """
    def __init__(self,nemo_coordinate_file=None,\
                     nemo_byte_mask_file=None,\
                     chunks=None):
        self.coordinate_file = nemo_coordinate_file
        self.byte_mask_file  = nemo_byte_mask_file
        self.chunks = chunks
        self.variables = {}
        self._get = get_xarray_dataarray
        self.define_horizontal_metrics()
        self.define_masks()
        self.chunk(chunks=chunks)

    def define_horizontal_metrics(self):
        self.variables["cell_x_size_at_t_location"] = \
                       self._get(self.coordinate_file,"e1t",chunks=self.chunks)
        self.variables["cell_y_size_at_t_location"] = \
                       self._get(self.coordinate_file,"e2t",chunks=self.chunks)

    def define_masks(self):
        self.variables["sea_binary_mask"] = \
                      self._get(self.byte_mask_file,"tmask",\
                                           chunks=self.chunks)[0,0,...]

    def chunk(self,chunks=None):
        for data in self.variables:
            if isinstance(data, xr.DataArray):
                data = data.chunk(chunks)

def nemo_2d_grid(nemo_coordinate_file=None,nemo_byte_mask_file=None,\
                 chunks=None):
    """Return a generic 2d grid from nemo coord and mask files.
    """
    variables = variables_holder_for_2d_grid_from_nemo_ogcm(\
                     nemo_coordinate_file=nemo_coordinate_file,\
                     nemo_byte_mask_file=nemo_byte_mask_file,\
                     chunks=chunks)
    grid = generic_2d_grid(variables=variables.variables)
    return grid

#======================== Generic 2D Grid Class ================================
class generic_2d_grid:
    """Model agnostic grid object, two dimensional version.
    This class holds the arrays that describe the grid and
    implements grid related methods including :
     - vector calculus
     - differential operators (gradient, divergence, etc...)
     - spatial integration
    """
    # TODO : keep _variables and _methods upt to date.
    _variables = [\
        "grid_shape"\
       ,"sea_binary_mask"
       ,"cell_x_size"\
       ,"cell_y_size"\
       ]
    _methods = [\
        "rechunk"
       ,"d_i","d_j","m_i","m_j"\
       ]
    def __init__(self,variables=None,parameters=None):
        """Initialize grid from dictionary of variables.
        """
        self.variables = variables
        self.parameters = parameters

#---------------------------- Chunking -----------------------------------------
    def rechunk(self,chunks=None):
        """Rechunk all the variables defining the grid.
        """
        for data in self.variables.values():
            if isinstance(data, xr.DataArray):
                data = data.chunk(chunks)  ## TODO : does not work yet.


#---------------------------- Grid Swapping ------------------------------------
#- Core swapping utilities
    def _grid_i_to_ip1(self,var,weights=None):
        """
        """
        pass

#---------------------------- Vector Operators ---------------------------------
    def norm_of_vector_field(self,a):
        """Return the norm of a vector field a at t-point.
        """
        return self.scalar_product(a,a)

    def scalar_product(self,a,b):
        """Return the scalar product of two vector fields a and b at t-point.
        """
        pass

    def scalar_outer_product(self,q,a):
        """Return the outer product of a scalar q with the vector a at t-point.
        """
        pass

    def vertical_component_of_vector_cross_product(self,a,b):
        """Return the cross product of two vector fields a and b.
        """
        pass

#-------------------- Differences and Averages ---------------------------------
    def d_i(self,q):
        """Return the difference q(i+1) - q(i)
        """
        pass

    def d_j(self,q):
        """Return the difference q(j+1) - q(j)
        """
        pass

    def m_i(self,q):
        """Return the average of q(i+1) and q(i)
        """
        pass

    def m_j(self,q):
        """Return the average of q(j+1) and q(j)
        """
        pass

#-------------------- Differential Operators------------------------------------
    def horizontal_gradient(self,q):
        """
        Return the horizontal gradient of a scalar field.
        """
        pass

    def vertical_component_of_curl(self,a):
        """Return the vertical component of the curl of a vector field.
        """
        pass

    def horizontal_divergence(self,a,masked=False):
        """
        Return the horizontal divergence of a vector field.
        """
#----------------------- Specific operators ------------------------------------
    def geostrophic_current_from_sea_surface_height(self,ssh):
        """Return the geostrophic current on u,v-grids.
        """
        pass
