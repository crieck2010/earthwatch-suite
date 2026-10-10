"""survey-3d: DSM/DEM differencing and cut/fill volumetrics.

Engine-first: the core is stdlib + numpy. GeoTIFF I/O needs rasterio
(``pip install "survey-3d[raster]"``); DEM acquisition from stereo pairs is
a user-supplied input — this engine starts where two elevation models
already exist.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
