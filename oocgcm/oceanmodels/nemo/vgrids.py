#!/usr/bin/env python
#
"""oocgcm.oceanmodels.nemo.grids
Define classes that give acces to NEMO model grid metrics and operators.

"""
import xarray as xr

from ...core.vgrids import generic_vertical_grid
from .io import return_xarray_dataarray # nemo version of io routine

#==================== Name of variables in NEMO ================================
#

_nemo_keymap_vertical_metrics = {
    'e3t': 'cell_z_size_at_t_location',
    'e3u': 'cell_z_size_at_u_location',
    'e3v': 'cell_z_size_at_v_location',
    'e3w': 'cell_z_size_at_w_location',
}

_nemo_keymap_byte_mask = {
    'tmask': 'sea_binary_mask_at_t_location',
    'umask': 'sea_binary_mask_at_u_location',
    'vmask': 'sea_binary_mask_at_v_location',
    'fmask': 'sea_binary_mask_at_f_location',
}


#==================== Variables holders for NEMO ===============================
#
class variables_holder_for_vertical_grid_from_nemo_ogcm:
    """This class create the dictionnary of variables used for creating a
    oocgcm.core.vgrids.generic_vertical_grid from NEMO output files.
    """
    def __init__(self,nemo_coordinate_file=None,
                     nemo_byte_mask_file=None,
                     chunks=None):
        """This holder uses the files meshzgr.nc and byte_mask.nc

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
        #
        self.chunks3D = chunks
        self.chunks1D = {'depth': self.chunks3D['depth']}
        #
        self.variables = {}
        self._define_depth()
        self._define_vertical_metrics()
        self._define_masks()
        # turn off line below for now in order to avoid 1D/3D treatment
        #self.chunk(chunks=chunks)
        self.parameters = {}
        self.parameters['chunks'] = chunks


    def _get(self,*args,**kwargs):
        return return_xarray_dataarray(*args,**kwargs)

    def _define_depth(self):
        self.variables["depth_at_t_location"] = \
                        self._get(self.coordinate_file,"gdept_0",
                                  chunks=self.chunks1D,depth_location='t')
        self.variables["depth_at_w_location"] = \
                        self._get(self.coordinate_file,"gdepw_0",
                                  chunks=self.chunks1D,depth_location='w')

    def _define_vertical_metrics(self):
        self.variables["cell_z_size_at_t_location"] = \
                        self._get(self.coordinate_file,"e3t",
                                  chunks=self.chunks3D,depth_location='t')
        self.variables["cell_z_size_at_u_location"] = \
                        self._get(self.coordinate_file,"e3u",
                                  chunks=self.chunks3D,depth_location='u')
        self.variables["cell_z_size_at_v_location"] = \
                        self._get(self.coordinate_file,"e3v",
                                  chunks=self.chunks3D,depth_location='v')
        self.variables["cell_z_size_at_w_location"] = \
                        self._get(self.coordinate_file,"e3w",
                                  chunks=self.chunks3D,depth_location='w')

    def _define_masks(self):
        self.variables["sea_binary_mask_at_t_location"] = \
                     self._get(self.byte_mask_file,"tmask",
                            chunks=self.chunks3D,depth_location='t')[:]
        self.variables["sea_binary_mask_at_u_location"] = \
                     self._get(self.byte_mask_file,"umask",
                            chunks=self.chunks3D,depth_location='u')[:]
        self.variables["sea_binary_mask_at_v_location"] = \
                     self._get(self.byte_mask_file,"vmask",
                            chunks=self.chunks3D,depth_location='v')[:]
        self.variables["sea_binary_mask_at_f_location"] = \
                     self._get(self.byte_mask_file,"fmask",
                            chunks=self.chunks3D,depth_location='f')[:]

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
def nemo_vertical_grid(nemo_coordinate_file=None,
                       nemo_byte_mask_file=None,
                       chunks=None,byte_mask_level=0):
    """Return a generic vertical grid from nemo coordinate and mask files.

    Parameters
    ----------
    nemo_coordinate_file : str
        path to NEMO coordinate file associated to the model configuration.
    nemo_byte_mask_file : str
        path to NEMO mask file associated to the model configuration.
    chunks : dict-like
        dictionnary of sizes of chunk for creating xarray.DataArray.
    byte_mask_level : int
        index of the level from which the masks should be loaded

    Returns
    -------
    grid : oocgcm.core.grids.generic_vertical_grid
        grid object corresponding to the model configuration.
    """
    variables = variables_holder_for_vertical_grid_from_nemo_ogcm(
                     nemo_coordinate_file=nemo_coordinate_file,
                     nemo_byte_mask_file=nemo_byte_mask_file,
                     chunks=chunks)
    vgrid = generic_vertical_grid(arrays=variables.variables,
                                  parameters= variables.parameters)
    return vgrid
