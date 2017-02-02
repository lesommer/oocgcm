#!/usr/bin/env python
#
"""oocgcm.oceanmodels.mitgcm.grids
Define classes that give acces to MITGCM model grid metrics and operators.

"""
import xarray as xr

from ...core.grids import generic_2d_grid
from .io import return_xarray_dataarray # nemo version of io routine

#==================== Name of variables in MITGCM ==============================

# not used yet, only kept as an indication of the actual variables to be loaded
_MITGCM_KEYMAP_GEOGRAPHICAL_COORDINATES = {
    'XC': 'longitude_at_t_location',
    'YC': 'latitude_at_t_location',
    'XG': 'longitude_at_f_location',
    'YG': 'latitude_at_f_location',
}

_MITGCM_KEYMAP_HORIZONTAL_METRICS = {
    'dxC': 'cell_x_size_at_u_location',
    'dxG': 'cell_x_size_at_v_location',
    'dyC': 'cell_y_size_at_v_location',
    'dyG': 'cell_y_size_at_u_location',
    'rA': 'cell_area_at_t_location',
    'rAz': 'cell_area_at_f_location',
    'rAw': 'cell_area_at_u_location',
    'rAs': 'cell_area_at_v_location'
}

_MITGCM_KEYMAP_VERTICAL_METRICS = {
    'drC': 'cell_z_size_at_w_location',
    'drF': 'cell_z_size_at_t_location',
    'PHrefC': 'cell_reference_pressure_at_t_location',
    'PHrefF': 'cell_reference_pressure_at_w_location'
}

_MITGCM_KEYMAP_VOLUME_METRICS = {
    'hFacC': 'cell_vertical_fraction_at_t_location',
    'hFacW': 'cell_vertical_fraction_at_u_location',
    'hFacS': 'cell_vertical_fraction_at_v_location',
}


#==================== Variables holders for NEMO ===============================
#
class variables_holder_for_2d_grid_from_mitgcm:
    """This class create the dictionnary of variables used for creating a
    oocgcm.core.grids.generic_2d_grid from NEMO output files.
    """
    def __init__(self, mitgcm_coordinate_file=None, mitgcm_byte_mask_file=None,
                 chunks=None, byte_mask_level=0):
        """This holder uses the files meshhgr.nc and byte_mask.nc

        Parameters
        ----------
        mitgcm_coordinate_file : str
            path to MITGCM coordinate file associated to the model configuration.
        mitgcm_byte_mask_file : str
            path to MITGCM mask file associated to the model configuration.
        chunks : dict-like
            dictionnary of sizes of chunk for creating xarray.DataArray.
        byte_mask_level : int
            index of the level from which the masks should be loaded
        """
        self.coordinate_file = mitgcm_coordinate_file
        self.byte_mask_file  = mitgcm_byte_mask_file
        self.chunks = chunks
        self.byte_mask_level = byte_mask_level
        self.variables = {}
        #self._get = return_xarray_dataarray
        self._define_latitude_and_longitude()
        self._define_horizontal_metrics()
        self._define_masks()
        self.chunk(chunks=chunks)
        self.parameters = {}
        self.parameters['chunks'] = chunks


    def _get(self,*args,**kwargs):
        return return_xarray_dataarray(*args,**kwargs)

    def _define_latitude_and_longitude(self):
        self.variables["longitude_at_t_location"] = (
            self._get(self.coordinate_file, "XC",
                      chunks=self.chunks, grid_location='t'))
        self.variables["latitude_at_t_location"] = (
            self._get(self.coordinate_file, "YC",
                      chunks=self.chunks, grid_location='t'))
        self.variables["longitude_at_f_location"] = (
            self._get(self.coordinate_file, "XG",
                      chunks=self.chunks, grid_location='f'))
        self.variables["latitude_at_f_location"] = (
            self._get(self.coordinate_file, "YG",
                      chunks=self.chunks, grid_location='f'))

    def _define_horizontal_metrics(self):
        self.variables["cell_x_size_at_u_location"] = \
                        self._get(self.coordinate_file, "dxC",
                                  chunks=self.chunks, grid_location='u')
        self.variables["cell_y_size_at_u_location"] = \
                        self._get(self.coordinate_file, "dyG",
                                  chunks=self.chunks, grid_location='u')
        self.variables["cell_x_size_at_v_location"] = \
                        self._get(self.coordinate_file, "dxG",
                                  chunks=self.chunks, grid_location='v')
        self.variables["cell_y_size_at_v_location"] = \
                        self._get(self.coordinate_file, "dyC",
                                  chunks=self.chunks, grid_location='v')
        self.variables["cell_area_at_t_location"] = \
                        self._get(self.coordinate_file, "rA",
                                  chunks=self.chunks, grid_location='t')
        self.variables["cell_area_at_f_location"] = \
                        self._get(self.coordinate_file, "rAz",
                                  chunks=self.chunks, grid_location='f')
        self.variables["cell_area_at_u_location"] = \
                        self._get(self.coordinate_file,"rAw",
                                  chunks=self.chunks, grid_location='f')
        self.variables["cell_area_at_v_location"] = \
                        self._get(self.coordinate_file, "rAs",
                                  chunks=self.chunks, grid_location='f')

    def _define_masks(self):
        self.variables["sea_binary_mask_at_t_location"] = \
                        self._get(self.coordinate_file, "hFacC",
                                  chunks=self.chunks, grid_location='t')
        self.variables["sea_binary_mask_at_u_location"] = \
                        self._get(self.coordinate_file, "hFacW",
                                  chunks=self.chunks, grid_location='u')
        self.variables["sea_binary_mask_at_v_location"] = \
                        self._get(self.coordinate_file, "hFacS",
                                  chunks=self.chunks, grid_location='v')
    def chunk(self,chunks=None):
        """Chunk all the variables.

        Parameters
        ----------
        chunks : dict-like
            dictionnary of sizes of chunk along xarray dimensions.
        """
        for dataname in self.variables:
            data = self.variables[dataname]
            if isinstance(data, xr.DataArray):
                self.variables[dataname] = data.chunk(chunks)


def mitgcm_2d_grid(mitgcm_coordinate_file=None,
                   nemo_byte_mask_file=None,
                   chunks=None, byte_mask_level=0):
    """Return a generic 2d grid from mitgcm coordinate and mask files.

    Parameters
    ----------
    mitgcm_coordinate_file : str
        path to MIGCM coordinate file associated to the model configuration.
    nemo_byte_mask_file : str
        path to MIGCM mask file associated to the model configuration.
    chunks : dict-like
        dictionnary of sizes of chunk for creating xarray.DataArray.
    byte_mask_level : int
        index of the level from which the masks should be loaded

    Returns
    -------
    grid : oocgcm.core.grids.generic_2d_grid
        grid object corresponding to the model configuration.
    """
    variables = variables_holder_for_2d_grid_from_mitgcm(
                     mitgcm_coordinate_file=mitgcm_coordinate_file,
                     mitgcm_byte_mask_file=nemo_byte_mask_file,
                     chunks=chunks,byte_mask_level=byte_mask_level)
    grid = generic_2d_grid(arrays=variables.variables,
                           parameters= variables.parameters)
    return grid
