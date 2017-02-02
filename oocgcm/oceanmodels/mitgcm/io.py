from ...core.io import return_xarray_dataset as _return_xarray_dataset
from ...core.io import return_xarray_mfdataset as _return_xarray_mfdataset
from ...core.io import return_xarray_dataarray as _return_xarray_dataarray


_MITGCM_KEYMAP_DIMS = {'time':'t', 'j':'x', 'j_g':'x', 'i':'y', 'i_g':'y'}

def return_xarray_dataset(*args, **kwargs):
    """Wrapper for core.io.return_xarray_dataset.

    Parameters
    ----------
    *args : arguments
    **kwargs : keyword arguments

    Returns
    ------
    ds : xarray.Dataset

    Methods
    ------
    change the name of dimension 'time_counter' in 't'
    """
    _ds = _return_xarray_dataset(*args, **kwargs)
    renamed_dims = {key: _MITGCM_KEYMAP_DIMS[key]
                    for key in _MITGCM_KEYMAP_DIMS if key in _ds.dims}
    ds = _ds.rename(renamed_dims)
    return ds

def return_xarray_mfdataset(*args,**kwargs):
    """Wrapper for core.io.return_xarray_mfdataset.

    Parameters
    ----------
    *args : arguments
    **kwargs : keyword arguments

    Returns
    ------
    ds : xarray.Dataset

    Methods
    ------
    change the name of dimension 'time_counter' in 't'
    """
    _ds = _return_xarray_mfdataset(*args,**kwargs)
    renamed_dims = {key: _MITGCM_KEYMAP_DIMS[key]
                    for key in _MITGCM_KEYMAP_DIMS if key in _ds.dims}
    ds = _ds.rename(renamed_dims)
    return ds

def return_xarray_dataarray(*args,**kwargs):
    """Wrapper for core.io.return_xarray_dataarray.

    Parameters
    ----------
    *args : arguments
    **kwargs : keyword arguments

    Returns
    -------
    da : xarray.DataArray

    Methods
    -------
    change the name of dimension 'time_counter' in 't'
    """
    _da = _return_xarray_dataarray(*args,**kwargs)
    renamed_dims = {key: _MITGCM_KEYMAP_DIMS[key]
                    for key in _MITGCM_KEYMAP_DIMS if key in _da.dims}
    da = _da.rename(renamed_dims)
    return da
