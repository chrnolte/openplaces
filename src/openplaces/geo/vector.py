#!/usr/bin/env python
# coding: utf-8

"""
vector.py

Functions for processing vector data (Geodataframes, geometries)

"""


def fix_geometries(gdf):
    """Fix the geometries of a GeoDataFrame by adding a zero buffer.

    This fixes most invalid geometry issues found in parcel data.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe that is suspected to have invalid geometries
    """
    gdf['geometry'] = gdf['geometry'].buffer(0)
    return gdf
