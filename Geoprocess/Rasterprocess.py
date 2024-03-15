#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   Rasterprocess.py
@Time    :   2024/03/15 15:34:59
@Author  :   Charles Keeling
@Version :   1.0
@Desc    :   Raster process functions.
"""


import rasterio as rio
from pathlib import PosixPath
import numpy as np
from geopandas.geoseries import GeoSeries
import pandas as pd


def tif_extent(meta: dict):

    return [
        meta["transform"][2],
        meta["transform"][5] + meta["height"] * meta["transform"][4],
        meta["transform"][2] + meta["width"] * meta["transform"][0],
        meta["transform"][5],
    ]


def readtif(
    path: str | PosixPath, rep_nodata: int | float = 0, is_meta: bool = False
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Reads a raster file and returns the data as a NumPy array.

    Args:
        path (str | PosixPath): The path to the raster file.
        rep_nodata (int | float, optional): The value to replace the nodata values with. Defaults to 0.
        is_meta (bool, optional): Whether to return the metadata along with the data. Defaults to False.

    Returns:
        np.ndarray | tuple[np.ndarray, dict]: The raster data as a NumPy array. If is_meta is True, returns a tuple containing the data and the metadata.
    """
    with rio.open(path) as dst:
        tifdata = dst.read(1)
        meta = dst.meta.copy()
        tifdata = np.where(tifdata == dst.nodata, rep_nodata, tifdata)
    if is_meta:
        return tifdata, meta
    return tifdata


def writetif(
    path: str | PosixPath,
    data: np.ndarray,
    meta: dict,
    is_rep_nd: bool = False,
    rep_nodata: int | float = 0,
) -> None:
    """Write a raster file with the given data and metadata.

    Args:
        path (str | PosixPath): The path to the output raster file.
        data (np.ndarray): The raster data as a NumPy array.
        meta (dict): The metadata of the raster file.
        is_rep_nd (bool, optional): Whether to replace nodata values in the data. Defaults to False.
        rep_nodata (int | float, optional): The value to replace nodata values with. Defaults to 0.
    """
    nodata = meta["nodata"]
    with rio.open(path, "w", **meta) as dst:
        if is_rep_nd:
            data = np.where(data == rep_nodata, nodata, data)
        dst.write(data, 1)


def aggregate_raster(
    input_raster_data_tuple: tuple[np.ndarray, dict],
    scale_factor: int,
    reference_file: str | PosixPath = None,
    method: str = "sum",
) -> tuple[np.ndarray, dict]:
    """Aggregate the input raster file and save the result as an output raster file.

    Args:
        - input_raster_data_tuple: A tuple containing the input raster data and metadata.
        - scale_factor: The aggregation factor (multiple of resolution side length).
        - reference_file: The path to the reference raster file used to obtain metadata information. Defaults to None.
        - method: The aggregation method, can be 'sum', 'min', 'max', 'mean', 'median'. Defaults to 'mean'.

    Returns:
        A tuple containing the aggregated raster data and metadata.
    """

    data, ref_meta = input_raster_data_tuple
    data = np.where(data == ref_meta["nodata"], np.nan, data)  # replace nodata with nan

    in_height = ref_meta["height"]
    in_width = ref_meta["width"]
    in_transform = ref_meta["transform"]

    out_height = int(np.ceil(in_height / scale_factor))
    out_width = int(np.ceil(in_width / scale_factor))

    if reference_file:
        with rio.open(reference_file) as ref_src:
            out_transform = ref_src.transform
    else:
        out_transform = in_transform * in_transform.scale(scale_factor)

    ref_meta.update(
        {"width": out_width, "height": out_height, "transform": out_transform}
    )

    extra_rows = int(out_height * scale_factor - in_height)
    extra_cols = int(out_width * scale_factor - in_width)
    data = np.pad(
        data,
        ((extra_rows, 0), (0, extra_cols)),
        mode="constant",
        constant_values=np.nan,
    )

    aggregation_funcs = {
        "sum": np.nansum,
        "min": np.nanmin,
        "max": np.nanmax,
        "mean": np.nanmean,
        "median": np.nanmedian,
    }
    if method in aggregation_funcs:
        aggregation_func = aggregation_funcs[method]
        reshaped_data = data.reshape(
            (ref_meta["height"], scale_factor, ref_meta["width"], scale_factor)
        )
        resampled_data = aggregation_func(reshaped_data, axis=(1, 3))
        resampled_data_mask = np.all(np.isnan(reshaped_data), axis=(1, 3))
    else:
        raise ValueError(
            "Invalid method. Supported methods: 'sum', 'min', 'max', 'mean', 'median'."
        )

    resampled_data = np.where(resampled_data_mask, ref_meta["nodata"], resampled_data)

    return resampled_data.astype(ref_meta["dtype"]), ref_meta


def downscale_raster_by_weight(
    input_raster_data_tuple: tuple[np.ndarray, dict],
    weight_raster_data_tuple: tuple[np.ndarray, dict],
) -> tuple[np.ndarray, dict]:
    """Downscale a raster by weight.

    This function takes an input raster and a weight raster and downscales the input raster based on the weight raster.
    The downscaled raster is calculated by multiplying the input raster with the weight raster, after normalizing the weight raster.
    The resulting downscaled raster has the same dimensions as the weight raster.

    Args:
        input_raster_data_tuple (tuple[np.ndarray, dict]): A tuple containing the input raster data as a NumPy array and its metadata as a dictionary.
        weight_raster_data_tuple (tuple[np.ndarray, dict]): A tuple containing the weight raster data as a NumPy array and its metadata as a dictionary.

    Returns:
        tuple[np.ndarray, dict]: A tuple containing the downscaled raster data as a NumPy array and its metadata as a dictionary.
    """

    data, ref_meta = input_raster_data_tuple
    data = np.where(data == ref_meta["nodata"], np.nan, data)  # replace nodata with nan

    in_height = ref_meta["height"]
    in_width = ref_meta["width"]
    in_transform = ref_meta["transform"]
    nodata = ref_meta["nodata"]
    dtype = ref_meta["dtype"]

    weight_data, weight_meta = weight_raster_data_tuple
    weight_data = np.where(weight_data == weight_meta["nodata"], np.nan, weight_data)

    weight_height = weight_meta["height"]
    weight_width = weight_meta["width"]
    weight_transform = weight_meta["transform"]

    scale_factor = int(in_transform.a / weight_transform.a)

    extra_rows = int(in_height * scale_factor - weight_height)
    extra_cols = int(in_width * scale_factor - weight_width)

    # normalize weight data
    sum_weight_data, _ = aggregate_raster(
        weight_raster_data_tuple, scale_factor, method="sum"
    )

    data = np.where(sum_weight_data != 0, data / sum_weight_data, np.nan)

    downscale_data = (
        np.repeat(np.repeat(data, scale_factor, axis=0), scale_factor, axis=1)[
            extra_rows:, :-extra_cols
        ]
        * weight_data
    )
    downscale_data = np.where(np.isnan(data), nodata, downscale_data)

    meta = weight_meta.copy()
    meta.update({"nodata": nodata, "dtype": dtype})

    return downscale_data, meta
