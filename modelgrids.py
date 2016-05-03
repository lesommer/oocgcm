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
mod360 = lambda x: xu.fmod(x+180,360) - 180

#======================== General Purpose Tools ================================
# TODO : this should probably move to the main module level.
#
def return_xarray_dataset(filename,chunks=None):
    """Return an xarray dataset corresponding to filename.
    """
    return xr.open_dataset(filename,chunks=chunks,lock=False)

def get_xarray_dataarray(filename,varname,chunks=None,**extra_kwargs):
    """Return a xarray dataarray corresponding to varname in filename.
    """
    ds = return_xarray_dataset(filename,chunks=chunks)
    dataarray = ds[varname]
    for kwargs in extra_kwargs:
        dataarray.attrs[kwargs] = extra_kwargs[kwargs]
    return dataarray

#======================== Working with DataArray ===============================
#

def finalize_dataarray_attributes(arr,**kwargs):
    """Update the disctionary of attibutes of a xarray dataarray.
    """
    if isinstance(arr, xr.DataArray):
        arr.attrs.update(kwargs)
    if arr.attrs.has_key('short_name'):
        arr.name = arr.attrs['short_name']
    return arr

def convert_dataarray_attributes_xderivative(attrs):
    """Return the dictionary of attributes corresponding to spatial derivative
    in the x-direction
    """
    new_attrs = attrs.copy()
    if attrs.has_key('long_name'):
        new_attrs['long_name'] = 'x-derivative of ' + attrs['long_name']
    if attrs.has_key('short_name'):
            new_attrs['short_name'] = 'd_' + attrs['short_name'] + '_dx'
    if attrs.has_key('unit'):
            new_attrs['unit'] = attrs['unit'] + '/m'
    return new_attrs

def convert_dataarray_attributes_yderivative(attrs):
    """Return the dictionary of attributes corresponding to spatial derivative
    in the y-direction
    """
    new_attrs = attrs.copy()
    if hasattr(attrs,'long_name'):
        new_attrs['long_name'] = 'y-derivative of ' + attrs['long_name']
    if hasattr(attrs,'short_name'):
            new_attrs['short_name'] = 'd_' + attrs['short_name'] + '_dy'
    if attrs.has_key('unit'):
            new_attrs['unit'] = attrs['unit'] + '/m'
    return new_attrs

#======================= NEMO-Specific Tools ===================================
# TODO : this section should move to a file dedicated to gcm specific features.
#
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
        self.define_projection_coordinate()
        self.define_horizontal_metrics()
        self.define_masks()
        self.chunk(chunks=chunks)

    def define_projection_coordinate(self):
        self.variables["projection_x_coordinate_at_t_location"] = \
                        self._get(self.coordinate_file,"nav_lon",\
                        chunks=self.chunks,grid_location='t')
        self.variables["projection_y_coordinate_at_t_location"] = \
                        self._get(self.coordinate_file,"nav_lat",\
                        chunks=self.chunks,grid_location='t')

    def define_horizontal_metrics(self):
        self.variables["cell_x_size_at_t_location"] = \
                        self._get(self.coordinate_file,"e1t",\
                        chunks=self.chunks,grid_location='t')
        self.variables["cell_y_size_at_t_location"] = \
                        self._get(self.coordinate_file,"e2t",\
                        chunks=self.chunks,grid_location='t')
        self.variables["cell_x_size_at_u_location"] = \
                        self._get(self.coordinate_file,"e1u",\
                        chunks=self.chunks,grid_location='u')
        self.variables["cell_y_size_at_u_location"] = \
                        self._get(self.coordinate_file,"e2u",\
                        chunks=self.chunks,grid_location='u')
        self.variables["cell_x_size_at_v_location"] = \
                        self._get(self.coordinate_file,"e1v",\
                        chunks=self.chunks,grid_location='v')
        self.variables["cell_y_size_at_v_location"] = \
                        self._get(self.coordinate_file,"e2v",\
                        chunks=self.chunks,grid_location='v')

    def define_masks(self):
        self.variables["sea_binary_mask_at_t_location"] = \
                      self._get(self.byte_mask_file,"tmask",\
                      chunks=self.chunks,grid_location='t')[0,0,...]
        self.variables["sea_binary_mask_at_u_location"] = \
                      self._get(self.byte_mask_file,"umask",\
                      chunks=self.chunks,grid_location='u')[0,0,...]
        self.variables["sea_binary_mask_at_v_location"] = \
                      self._get(self.byte_mask_file,"vmask",\
                      chunks=self.chunks,grid_location='v')[0,0,...]
        self.variables["sea_binary_mask_at_f_location"] = \
                      self._get(self.byte_mask_file,"vmask",\
                      chunks=self.chunks,grid_location='f')[0,0,...]

    def chunk(self,chunks=None):
        for dataname in self.variables:
            data = self.variables[dataname]
            if isinstance(data, xr.DataArray):
                self.variables[dataname] = data.chunk(chunks)

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

    Methods expect and return xarray dataarrays.

    Assume that dimension names ar 'x' and 'y'.
    """
    # TODO : keep _variables and _methods up to date.
    _required_variables = [\
       ]
    _available_methods = [\
         "rechunk", "__getitem__"\
        ,"d_i","d_j"\
       ]
    def __init__(self,variables=None,parameters=None):
        """Initialize grid from dictionary of variables.
        """
        self.variables = variables
        self.parameters = parameters
        self._define_aliases_for_variabless()
        self.shape = self.variables["sea_binary_mask_at_t_location"].shape

#--------------------- Aliases for DataArrays ----------------------------------
    def _define_aliases_for_variabless(self):
        """Define alias for frequently used variables.
        This mostly follows nemo name conventions and will evolve in the future.
        """
        # coordinates
        self._navlon = self.variables["projection_x_coordinate_at_t_location"]
        self._navlat = self.variables["projection_y_coordinate_at_t_location"]
        # metrics
        self._e1t = self.variables["cell_x_size_at_t_location"]
        self._e1u = self.variables["cell_x_size_at_u_location"]
        self._e1v = self.variables["cell_x_size_at_v_location"]
        self._e2t = self.variables["cell_y_size_at_t_location"]
        self._e2u = self.variables["cell_y_size_at_u_location"]
        self._e2v = self.variables["cell_y_size_at_v_location"]

        # masks
        self._tmask = self.variables["sea_binary_mask_at_t_location"]
        self._umask = self.variables["sea_binary_mask_at_u_location"]
        self._vmask = self.variables["sea_binary_mask_at_v_location"]
        self._fmask = self.variables["sea_binary_mask_at_f_location"]

#--------------------- Chunking and Slicing ------------------------------------
    def rechunk(self,chunks=None):
        """Rechunk all the variables defining the grid.
        """
        for dataname in self.variables:
            data = self.variables[dataname]
            if isinstance(data, xr.DataArray):
                self.variables[dataname] = data.chunk(chunks)

    def __getitem__(self,item):
        """Return a grid object restricted to a subdomain.
        """
        sliced_variables = {}
        for dataname in self.variables:
            sliced_variables[dataname] = self.variables[dataname][item]
        return generic_2d_grid(variables=sliced_variables,\
                             parameters=self.parameters)

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
        return xu.sqrt(self.scalar_product(a,a))

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
        di = q.shift(x=-1) - q
        return di

    def d_j(self,q):
        """Return the difference q(j+1) - q(j)
        """
        dj = q.shift(y=-1) - q
        return dj


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
        # TODO : test that the input is a xarray dataarray
        # TODO : test the input grid location
        # TODO : check that the chunks are aligned
        gx = self.d_i(q) / self._e1u
        gy = self.d_j(q) / self._e2v
        # finalize dataarray attributes
        gxatts = convert_dataarray_attributes_xderivative(q.attrs)
        gxatts['grid_location'] = 'u'
        gyatts = convert_dataarray_attributes_yderivative(q.attrs)
        gxatts['grid_location'] = 'v'
        gx = finalize_dataarray_attributes(gx,**gxatts)
        gy = finalize_dataarray_attributes(gy,**gyatts)
        return gx,gy

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
