#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   Geoplot.py
@Time    :   2024/03/15 16:39:38
@Author  :   Charles Keeling
@Version :   1.0
@Desc    :   
"""


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def add_northcompass(
    ax: plt.Axes,
    loc_x: float = 0.9,
    loc_y: float = 0.9,
    height_ratio: float = 0.02,
    pad: float = 0.02,
) -> None:

    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx

    height = ylen * height_ratio
    width = height * 0.55

    left = [minx + xlen * loc_x - width * 0.5, miny + ylen * loc_y - height]
    right = [minx + xlen * loc_x + width * 0.5, miny + ylen * loc_y - height]
    top = [minx + xlen * loc_x, miny + ylen * loc_y]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * 0.1]
    triangle = mpatches.Polygon([left, top, right, center], color="k")
    ax.text(
        s="N",
        x=minx + xlen * loc_x,
        y=miny + ylen * (loc_y + pad),
        fontsize=3,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.add_patch(triangle)


def add_scalebar(
    ax: plt.Axes,
    scalebar_length: int = 100000,
    major_step: int = 2,
    bar_height: int = 50000,
    fontsize: int = 2.5,
):
    x0, x1, y0, y1 = ax.get_extent()

    # bar offset is how far from bottom left corner scale bar is (x,y) and how far up is scale bar text
    bar_offset = [0.05, 0.05, 0.25]
    bar_x0 = x0 + (x1 - x0) * bar_offset[0]
    bar_y0 = y0 + (y1 - y0) * bar_offset[1]
    interval_length = scalebar_length / major_step
    for i in range(major_step + 1):
        ax.vlines(
            x=bar_x0 + interval_length * i,
            ymin=bar_y0,
            ymax=bar_y0 + bar_height,
            colors="black",
            ls="-",
            lw=0.2,
        )
    ax.hlines(
        y=bar_y0,
        xmin=bar_x0,
        xmax=bar_x0 + scalebar_length,
        colors="black",
        ls="-",
        lw=0.2,
    )
    ax.text(
        bar_x0,
        bar_y0 + bar_height + bar_height * bar_offset[2],
        "0",
        fontsize=fontsize,
        ha="center",
    )
    ax.text(
        bar_x0 + scalebar_length,
        bar_y0 + bar_height + bar_height * bar_offset[2],
        f"{scalebar_length/1000:.0f}",
        fontsize=fontsize,
        ha="center",
    )
    ax.text(
        bar_x0 + scalebar_length + interval_length / 2,
        bar_y0 + bar_height + bar_height * bar_offset[2],
        "km",
        fontsize=fontsize,
        ha="left",
    )
