#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   Advancedprocess.py
@Time    :   2024/03/15 15:43:27
@Author  :   Charles Keeling
@Version :   1.0
@Desc    :   Raster and Shapefile advanced processing functions.
"""


import rasterio as rio
from pathlib import PosixPath
import numpy as np
import geopandas as gpd
import pandas as pd


def polygon2raster(
    geometry: gpd.GeoSeries,
    value: pd.Series,
    dtype: np.dtype,
    reference: str | PosixPath,
    nd: int | float = -99,
    is_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Rasterize a polygon geometry into a raster array.

    Args:
        geometry (gpd.GeoSeries): A GeoSeries containing polygon geometries.
        value (pd.Series): A pandas Series containing values corresponding to each polygon geometry.
        dtype (np.dtype): The data type of the output raster array.
        reference (str | PosixPath): The path to the reference raster file used to obtain metadata information.
        nd (int | float, optional): The value to fill the raster array with. Defaults to -99.
        is_meta (bool, optional): Whether to return the metadata along with the raster array. Defaults to False.

    Returns:
        np.ndarray | tuple[np.ndarray, dict]: The raster array representing the polygon geometries. If is_meta is True, returns a tuple containing the raster array and the metadata.
    """
    with rio.open(reference) as r:
        a = r.height
        b = r.width
        t = r.transform
        meta = r.meta.copy()
    shapes = zip(geometry, value)
    raster = rio.features.rasterize(
        shapes=shapes, out_shape=(a, b), fill=nd, transform=t, dtype=dtype
    )
    meta.update({"dtype": dtype, "nodata": nd})

    if is_meta:
        return raster, meta

    return raster


def zonal_stat(
    geometry: gpd.GeoSeries,
    id: pd.Series,
    rst_col_name: str,
    rst_data: np.ndarray,
    rst_reference: str | PosixPath,
    stat_method: str = "sum",
) -> gpd.GeoDataFrame:
    """Calculate zonal statistics for raster data based on polygon geometries.

    This function calculates zonal statistics for a given raster dataset based on the
    provided polygon geometries. The statistics can be calculated using different
    aggregation methods such as sum, min, max, mean, and median.

    Args:
        geometry (gpd.GeoSeries): A GeoSeries containing the polygon geometries.
        id (pd.Series): A Series containing the IDs for each polygon.
        rst_col_name (str): The name of the column in the output GeoDataFrame that will
            store the calculated statistics.
        rst_data (np.ndarray): A NumPy array containing the raster data.
        rst_reference (str | PosixPath): The reference path or string for the raster data.
        stat_method (str, optional): The aggregation method to be used for calculating
            the statistics. Defaults to "sum".

    Raises:
        ValueError: Raised if the shape of the rasterized polygon does not match the
            shape of the raster data.
        ValueError: Raised if an invalid aggregation method is provided.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the calculated zonal statistics for each polygon.
    """

    out_shp = gpd.GeoDataFrame({"id": id, "geometry": geometry})
    id_rst = polygon2raster(geometry, id, np.int32, rst_reference)
    if id_rst.shape != rst_data.shape:
        raise ValueError(
            "The shape of the rasterized polygon does not match the shape of the raster data."
        )

    data = pd.DataFrame({"id": id_rst.flatten(), rst_col_name: rst_data.flatten()})

    aggregation_funcs = {
        "sum": np.nansum,
        "min": np.nanmin,
        "max": np.nanmax,
        "mean": np.nanmean,
        "median": np.nanmedian,
    }
    if stat_method in aggregation_funcs:
        aggregation_func = aggregation_funcs[stat_method]
        sum_data = data.groupby("id")[rst_col_name].apply(aggregation_func)
    else:
        raise ValueError(
            "Invalid method. Supported methods: 'sum', 'min', 'max', 'mean', 'median'."
        )

    out_shp = out_shp.merge(sum_data, on="id", how="left")

    return out_shp
