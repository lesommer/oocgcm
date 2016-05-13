#!/usr/bin/env python
#
"""oocgcm.oceanmodels.nemo.grids
Define classes that give acces to NEMO model grid metrics and operators.

"""
import xarray as xr

from ...core.grids import generic_2d_grid
from ...core.io import return_xarray_dataarray

#==================== Variables holders for NEMO ===============================
#
class variables_holder_for_2d_grid_from_nemo_ogcm:
    """This class create the dictionnary of variables used for creating a
    oocgcm.core.grids.generic_2d_grid from NEMO output files.
    """
    def __init__(self,nemo_coordinate_file=None,
                     nemo_byte_mask_file=None,
                     chunks=None):
        """This holder uses the files meshhgr.nc and byte_mask.nc

        Parameters
        ----------
        nemo_coordinate_file : str
            path to NEMO coordinate file associated to the model configuration.
        nemo_byte_mask_file : str
            path to NEMO mask file associated to the model configuration.
        chunks : dict-like
            dictionnary of sizes of chunk for creating xarray.DataArray.
        """
        self.coordinate_file = nemo_coordinate_file
        self.byte_mask_file  = nemo_byte_mask_file
        self.chunks = chunks
        self.variables = {}
        self._get = return_xarray_dataarray
        self._define_projection_coordinate()
        self._define_horizontal_metrics()
        self._define_masks()
        self.chunk(chunks=chunks)
        self.parameters = {}
        self.parameters['chunks'] = chunks

    def _define_projection_coordinate(self):
        self.variables["projection_x_coordinate_at_t_location"] = \
                        self._get(self.coordinate_file,"nav_lon",
                                  chunks=self.chunks,grid_location='t')
        self.variables["projection_y_coordinate_at_t_location"] = \
                        self._get(self.coordinate_file,"nav_lat",
                                  chunks=self.chunks,grid_location='t')                     

    def _define_horizontal_metrics(self):
        self.variables["cell_x_size_at_t_location"] = \
                        self._get(self.coordinate_file,"e1t",
                                  chunks=self.chunks,grid_location='t')
        self.variables["cell_y_size_at_t_location"] = \
                        self._get(self.coordinate_file,"e2t",
                                  chunks=self.chunks,grid_location='t')
        self.variables["cell_x_size_at_u_location"] = \
                        self._get(self.coordinate_file,"e1u",
                                  chunks=self.chunks,grid_location='u')
        self.variables["cell_y_size_at_u_location"] = \
                        self._get(self.coordinate_file,"e2u",
                                  chunks=self.chunks,grid_location='u')
        self.variables["cell_x_size_at_v_location"] = \
                        self._get(self.coordinate_file,"e1v",
                                  chunks=self.chunks,grid_location='v')
        self.variables["cell_y_size_at_v_location"] = \
                        self._get(self.coordinate_file,"e2v",
                                  chunks=self.chunks,grid_location='v')
        self.variables["cell_x_size_at_f_location"] = \
                        self._get(self.coordinate_file,"e1f",
                                  chunks=self.chunks,grid_location='f')
        self.variables["cell_y_size_at_f_location"] = \
                        self._get(self.coordinate_file,"e2f",
                                  chunks=self.chunks,grid_location='f')

    def _define_masks(self):
        self.variables["sea_binary_mask_at_t_location"] = \
                      self._get(self.byte_mask_file,"tmask",
                                chunks=self.chunks,grid_location='t')[0,0,...]
        self.variables["sea_binary_mask_at_u_location"] = \
                      self._get(self.byte_mask_file,"umask",
                                chunks=self.chunks,grid_location='u')[0,0,...]
        self.variables["sea_binary_mask_at_v_location"] = \
                      self._get(self.byte_mask_file,"vmask",
                                chunks=self.chunks,grid_location='v')[0,0,...]
        self.variables["sea_binary_mask_at_f_location"] = \
                      self._get(self.byte_mask_file,"vmask",
                                chunks=self.chunks,grid_location='f')[0,0,...]

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

#================== NEMO grids from generic grids ==============================
#
def nemo_2d_grid(nemo_coordinate_file=None,
                 nemo_byte_mask_file=None,
                 chunks=None):
    """Return a generic 2d grid from nemo coordinate and mask files.

    Parameters
    ----------
    nemo_coordinate_file : str
        path to NEMO coordinate file associated to the model configuration.
    nemo_byte_mask_file : str
        path to NEMO mask file associated to the model configuration.
    chunks : dict-like
        dictionnary of sizes of chunk for creating xarray.DataArray.

    Returns
    -------
    grid : oocgcm.core.grids.generic_2d_grid
        grid object corresponding to the model configuration.
    """
    variables = variables_holder_for_2d_grid_from_nemo_ogcm(
                     nemo_coordinate_file=nemo_coordinate_file,
                     nemo_byte_mask_file=nemo_byte_mask_file,
                     chunks=chunks)
    grid = generic_2d_grid(variables=variables.variables,
                           parameters= variables.parameters)
    return grid
