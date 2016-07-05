#!/usr/bin/env python
#
"""oocgcm.core.vgrids
Define classes that give acces to vertical grid metrics and differential operators.

"""
from collections import namedtuple # for vector fields data structures

import numpy as np
import xarray as xr
import dask.array as da
import xarray.ufuncs as xu         # ufuncs like np.sin for xarray


from .utils import is_numpy,is_xarray,check_input_array
from .utils import _append_dataarray_extra_attrs
from .utils import _assert_and_set_grid_location_attribute
from .utils import _grid_location_equals

#from ..parameters.physicalparameters import coriolis_parameter, grav

#
#==================== Differences and Averages =================================
#

def _dk(scalararray):
    """Return the difference scalararray(k+1) - scalararray(k).

    A priori, for internal use only.

    Parameters
    ----------
    scalararray : xarray.DataArray
        xarray that should be differentiated.

    Returns
    -------
    di : xarray.DataArray
       xarray of difference, defined at point k+1/2
    """
    dk = scalararray.shift(depth=-1) - scalararray
    return dk

def _mk(scalararray):
    """Return the average of scalararray(k+1) and scalararray(k)

    A priori, for internal use only.

    Parameters
    ----------
    scalararray : xarray.DataArray
        xarray that should be averaged at k+1/2.

    Returns
    -------
    mi : xarray.DataArray
       averaged xarray, defined at point k+1/2
    """
    mk = ( scalararray.shift(depth=-1) + scalararray ) / 2.
    return mk

#
#==================== Methods for testing xarrays ==============================
#
def _finalize_dataarray_attributes(xarr,**kwargs):
    """Update the dictionary of attibutes of a xarray dataarray.

    Parameters
    ----------
    xarr : xarray.DataArray
       xarray dataarray which attributes will be updated.
    kwargs : dict-like object
       dictionnary of attributes

    Returns
    -------
    xarr : xarray.DataArray
    """
    if isinstance(xarr, xr.DataArray):
        xarr.attrs.update(kwargs)
    if xarr.attrs.has_key('short_name'):
        xarr.name = xarr.attrs['short_name']
    return xarr

def _convert_dataarray_attributes_zderivative(attrs,grid_location=None):
    """Return the dictionary of attributes corresponding to the spatial
    derivative of a scalar field in the z-direction

    Parameters
    ----------
    attrs : dict-like object
       dictionnary of attributes
    grid_location : str
        string describing the grid location : eg 'u','v','t','f'...

    Returns
    -------
    new_attrsarr : dict-like object
        dictionnary of attributes of the spatial derivative.
    """
    new_attrs = attrs.copy()
    if attrs.has_key('long_name'):
        new_attrs['long_name'] = 'z-derivative of ' + attrs['long_name']
    if attrs.has_key('short_name'):
        new_attrs['short_name'] = 'd_' + attrs['short_name'] + '_dz'
    if attrs.has_key('units'):
        new_attrs['units'] = attrs['units'] + '/m'
    if grid_location is not None:
        new_attrs['grid_location'] = grid_location
    print 'Vertical derivates at u/v points are labelled w but are in fact offset horizontally'
    return new_attrs

def _convert_dataarray_attributes_zderivative2(attrs,grid_location='t'):
    """Return the dictionary of attributes corresponding to the second order 
    vertical derivative of a scalar field.

    Parameters
    ----------
    attrs : dict-like object
       dictionary of attributes
    grid_location : str
        string describing the grid location : eg 'u','v','t','f'...

    Returns
    -------
    new_attrsarr : dict-like object
        dictionnary of attributes of the laplacian.
    """
    new_attrs = attrs.copy()
    if attrs.has_key('long_name'):
        new_attrs['long_name'] = 'second order vertical derivative of ' + attrs['long_name']
    if attrs.has_key('short_name'):
        new_attrs['short_name'] = 'hlap_' + attrs['short_name']
    if attrs.has_key('units'):
        new_attrs['units'] = attrs['units'] + '/m2'
    if grid_location is not None:
        new_attrs['grid_location'] = grid_location
    return new_attrs



#
#========================== Data structures ====================================
#

#
# data structures for vector fields and tensors.
#



#
#======================== Generic Vertical Grid Class ================================
#

class generic_vertical_grid:
    """Model agnostic grid object, vertical dimension version.

    This class holds the xarrays that describe the grid and
    implements grid related methods.

    This includes :
        - vector calculus (scalar product, norm, vector product,...)
        - interpolation between different grid locations (eg. 'u'-->'t')
        - differential operators (gradient, divergence, etc...)
        - spatial integration

    Most methods expect and return instance of xarray.DataArray.

    Assume that dimension names are 'x', 'y' and 'depth'.
    """
    # the following arrays should be defined for the grid to be functional.
    # attempts to build the gride withouth these arrays will raise an Exception.
    _required_arrays = [\
        "depth_at_t_location",\
        "depth_at_w_location",\
        "sea_binary_mask_at_t_location",\
        "sea_binary_mask_at_u_location",\
        "sea_binary_mask_at_v_location",\
        "sea_binary_mask_at_f_location",\
        "cell_z_size_at_t_location",\
        "cell_z_size_at_u_location",\
        "cell_z_size_at_v_location",\
        "cell_z_size_at_w_location",\
        ]

    # the following arrays may or may not be available depending on how the
    # grid was created.
    _coordinate_arrays = [\
        ]

    _accepted_arrays = _coordinate_arrays + _required_arrays

    def __init__(self,arrays=None,parameters=None):
        """Initialize a grid from a dictionary of xarrays and some parameters.

        Parameters
        ----------
        variables : dict-like object
            dictionnary of xarrays that describe the grid. Required variables
            for the method actually implemented are listed in _required_arrays.
            This naming convention follows a mixture of cf and comodo norms in
            order for this class to be model-agnostic.

        parameters : dict-object
            not used yet.
        """
        for arrayname in self._required_arrays:
            if not(arrays.has_key(arrayname)):
                raise Exception('Arrays are missing for building the grid.')

        # builds the list of keys in arrays that are also in _accepted_arrays
        used_arrays = list( set( arrays.keys() ) & set( self._accepted_arrays ))

        # stores only the arrays that are needed for building the grid.
        self._arrays = dict((key, arrays[key]) for key in used_arrays)

        self._extra_parameters = parameters
        self._define_aliases_for_arrays()
        #self._define_area_of_grid_cells()

#--------------------- Public attributes ---------------------------------------
    @property
    def dims(self):
        """Dimensions of the xarray dataarrays describing the grid.
        """
        dims = list(self._arrays["sea_binary_mask_at_t_location"].dims)
        if 't' in dims: dims.remove('t')
        return tuple(dims)

    @property
    def ndims(self):
        """Number of dimensions of the dataarrays describing the grid.
        """
        return len(self.dims)

    @property
    def shape(self):
        """Shape of the xarray dataarrays describing the grid.
        """
        return self._arrays["sea_binary_mask_at_t_location"].shape

    @property
    def chunks(self):
        """Chunks of the xarray dataarrays describing the grid.
        """
        # TODO : not clear whether to store the priori description of chunks
        #        (dictionnary) or the a posteriori value (tuple of tuples).

        return self._arrays["sea_binary_mask_at_t_location"].chunks

#--------------------- Extra xarrays for the grid ------------------------------

# Horizontal grid information would be required here

#     def _define_area_of_grid_cells(self):
#         """Define arrays of area at u,v,t,f grid_location.
# 
#         This is only a definition, the computation is performed only if used.
#         """
#         for gloc in ['t','u','v','f']:
#             self._arrays["cell_area_at_" + gloc + "_location"] = \
#                     self._arrays["cell_x_size_at_" + gloc + "_location"] \
#                   * self._arrays["cell_y_size_at_" + gloc + "_location"]




#--------------------- Aliases for DataArrays ----------------------------------
    def _define_aliases_for_arrays(self):
        """Define alias for frequently used variables.

        The following shortcuts mostly follows nemo name conventions.
        This could evolve in the future and should therefore not be used in
        external libraries.
        """
        # coordinates
        self._array_deptht = self._arrays["depth_at_t_location"]
        self._array_depthw = self._arrays["depth_at_w_location"]
        # metrics
        self._array_e3t = self._arrays["cell_z_size_at_t_location"]
        self._array_e3u = self._arrays["cell_z_size_at_u_location"]
        self._array_e3v = self._arrays["cell_z_size_at_v_location"]
        self._array_e3w = self._arrays["cell_z_size_at_w_location"]

        # masks
        self._array_tmask = self._arrays["sea_binary_mask_at_t_location"]
        self._array_umask = self._arrays["sea_binary_mask_at_u_location"]
        self._array_vmask = self._arrays["sea_binary_mask_at_v_location"]
        self._array_fmask = self._arrays["sea_binary_mask_at_f_location"]

#--------------------- Chunking and Slicing ------------------------------------
    def chunk(self,chunks=None):
        """Rechunk all the variables defining the grid.

        Parameters
        ----------
        chunks : dict-like
            dictionnary of sizes of chunk along xarray dimensions.

        Example
        -------
        >>> xr_chunk = {'x':200,'y':200}
        >>> grd = generic_2d_grid(...)
        >>> grd.chunk(xr_chunk)

        """
        for dataname in self._arrays:
            data = self._arrays[dataname]
            if isinstance(data, xr.DataArray):
                self._arrays[dataname] = data.chunk(chunks)

    def __getitem__(self,item):
        """The behavior of this function depends on the type of item.
            - if item is a string, return the array self._arrays[item]

        Parameters
        ----------
        item : str
            item can be a string identifying a key in self._arrays

        Returns
        -------
        out :  xarray.DataArray or generic_vertical_grid
            either a dataarray corresponding to self._arrays[item]

        Example
        -------
        >>> vgrd = generic_vertical_grid(...)

        for accessing a specific dataarray describing the grid
        >>> e1t = grd['cell_x_size_at_t_location']

        """
        returned = self._arrays[item]
        return returned

    def __contains__(self, key):
        """The 'in' operator will return true or false depending on whether
        'key' is an array in the dataset or not.
        """
        return key in self._arrays

    def __iter__(self):
        return iter(self._arrays)

#---------------------------- Misc utilities ------------------------------------
#-


#---------------------------- Grid Swapping ------------------------------------
#- Core swapping utilities
    def _to_upper_grid_location(self,scalararray,
                                weights_in=None,weights_out=None):
        """Return an average of scalararray at point k + 1/2
        """
        average = lambda xarr:( xarr.shift(depth=-1) + xarr ) / 2.
        if weights_in is None:
            out = average(scalararray)
        else:
            if weights_out is None:
                weights_out = average(weights_in)
            out = average(scalararray * weights_in) / weights_out
        return out

    def _to_lower_grid_location(self,scalararray,
                                weights_in=None,weights_out=None):
        """Return an average of scalararray at point k - 1/2
        """
        average = lambda xarr:( xarr.shift(depth=1) + xarr ) / 2.
        if weights_in is None:
            out = average(scalararray)
        else:
            if weights_out is None:
                weights_out = average(weights_in)
            out = average(scalararray * weights_in) / weights_out
        return out



#- User swapping utilities
    def _weights_for_change_grid_location(self,input=None,output=None,
                                          conserving=None):
        """Return the weights for changing grid location.

        Notes
        -----
        This function is used internally for change_grid_location_*_to_*
        """
        # shortcut weights for now:
        print 'Weighting is off in _weights_for_change_grid_location'
        weights_in  = 1.
        weights_out = 1.
#         if conserving is 'area':
#             weights_in  = self._arrays["cell_area_at_" + input  + "_location"]
#             weights_out = self._arrays["cell_area_at_" + output + "_location"]
#         elif conserving is 'x_flux':
#             weights_in  = self._arrays["cell_y_size_at_" + input  + "_location"]
#             weights_out = self._arrays["cell_y_size_at_" + output + "_location"]
#         elif conserving is 'y_flux':
#             weights_in  = self._arrays["cell_x_size_at_" + input  + "_location"]
#             weights_out = self._arrays["cell_x_size_at_" + output + "_location"]

        return weights_in, weights_out

    def change_grid_location_t_to_w(self,scalararray,conserving='area'):
        """Return a xarray corresponding to scalararray averaged at a new
        grid location.

        Parameters
        ----------
        scalararray : xarray.DataArray
            original array to be relocated
        conserving : str !!! turned of for now !!!
            any of 'area', 'x_flux' or 'y_flux'.
              - 'area' : conserves the area
              - 'x_flux' : conserves the flux in x-direction (eastward)
              - 'y_flux' : conserves the flux in y-direction (northward)
        """
        check_input_array(scalararray,\
                          chunks=self.chunks,grid_location='t',ndims=self.ndims)
        wi, wo = self._weights_for_change_grid_location(input='t',output='u',
                                                        conserving=conserving)
        out = self._to_upper_grid_location(scalararray,weights_in=wi,
                                                         weights_out=wo)
        return _append_dataarray_extra_attrs(out,grid_location='u')


        return out

#---------------------------- Vector Operators ---------------------------------


#-------------------- Differential Operators------------------------------------

    def vertical_derivative(self,scalararray):
        """
        Return the vertical derivative of the input datastructure.

        Parameters
        ----------
        datastructure : xarray.DataArray

        Returns
        -------
        result : xarray.DataArray

        Methods
        -------
        calls horizontal_gradient_vector or horizontal_gradient_tensor depending
        on th type of the input datastructure.

        See also
        --------
        self.horizontal_gradient_vector,  self.horizontal_gradient_tensor

        """
        
        #- check input arrays
        print 'checks missing in vertical_derivative'
        #if _grid_location_equals(scalararray, grid_location=None):
        #check_input_array(scalararray,\
        #                  chunks=self.chunks,grid_location='t',ndims=self.ndims)

        #- compute the vertical derivative
        gz = _dk(scalararray) / self._arrays["cell_z_size_at_t_location"]

        #- finalize attributes
        gzatts = _convert_dataarray_attributes_zderivative(scalararray.attrs,
                                                           grid_location='w')

        print 'Missing attribute finalization in vertical_derivative'
        #gx = _finalize_dataarray_attributes(gx,**gxatts)
        
        return gz


#
#-------------------- Spatial Integration and averages --------------------------
#

    def integrate_dz(self,array,where=None,grid_location=None,normalize=False):
        """Return the vertical integral of array in regions where
        where is True.

        Parameters
        ----------
        array : xarray.DataArray
            a dataarray with an additonal attribute specifying the grid_location.
            The dimension of array should include 'depth'.
            The shape of array should match the shape of the grid.
        where: boolean xarray.DataArray
            dataarray with value = True where the integration should be applied.
            The dimension of where should be a subset of the dimension of array.
            For each dimension, the size should be equal to the corresponding
            size of array dataarray.
            if where is None, the function return the integral in all the domain
            defined by the grid object.
        grid_location : str
            string describing the grid location : eg 'u','v','t','f'...
             - if grid_location is not None
                    check compatibility with array.attrs.grid_location
             - if grid_location is None
                    use array.attrs.grid_location by default
        normalize : boolean
            boolean stating whether of not the integral should be normalized
            by the area of the region over which the integration is performed.

        Returns
        -------
        integral: xarray.DataArray
            a dataarray with reduced dimension defining the integral of array in
            the region of interest.

        See also
        --------
        spatial_average_z : averaging over z
        """
        # check grid location
        if grid_location is None:
            if not(isinstance(array,xr.DataArray)):
                raise TypeError('input array should be a xarray.DataArray')
            elif array.attrs.has_key("grid_location"):
                grid_location = array.attrs["grid_location"]
            else:
                raise Exception('grid_location is not known.')
            #except:
            #    raise TypeError('input array should be a xarray.DataArray')

        # check arrays
        check_input_array(array,\
                            chunks=self.chunks,grid_location=grid_location,
                            ndims=self.ndims)
        if where is not None:
            check_input_array(where,\
                              chunks=self.chunks,grid_location=grid_location,
                              ndims=self.ndims)
        else:
            maskname = "sea_binary_mask_at_" + grid_location + "_location"
            where = self._arrays[maskname]

        # actual definition
        idims = ('depth')
        dz = self._arrays['cell_z_size_at_' + grid_location + '_location']
        array_dz = array.where(where.squeeze()) * dz # squeeze is not lazy
        integral = array_dxdy.sum(dim=idims)

        # normalize if required
        if normalize:
            integral /= dz.where(where.squeeze()).sum(dim=idims)
        #        Nota Bene : dataarray.squeeze() is not a lazy operation.
        return integral

    def spatial_average_z(self,array,where=None,grid_location=None):
        """Return the vertical average of array in regions where where is True.

        Parameters
        ----------
        array : xarray.DataArray
            a dataarray with an additonal attribute specifying the grid_location.
            The dimension of array should include 'depth'.
            The shape of array should match the shape of the grid.
        where: boolean xarray.DataArray
            dataarray with value = True where the integration should be applied
            The dimension of where should be a subset of the dimension of array.
            For each dimension, the size should be equal to the corresponding
            size of array dataarray.
            if where is None, the function return the integral in all the domain
            defined by the grid object.
        grid_location : str
            string describing the grid location : eg 'u','v','t','f'...
             - if grid_location is not None
                    check compatibility with array.attrs.grid_location
             - if grid_location is None
                    use array.attrs.grid_location by default

        Returns
        -------
        average: xarray.DataArray
            a dataarray with reduced dimension defining the average of array in
            the region of interest.

        See also
        --------
        integrate_z : vertical integral over a region
        """

        average = self.integrate_dz(array,where=where,
                                 grid_location=grid_location,normalize=True)
        return average


