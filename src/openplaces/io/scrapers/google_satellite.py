# Copyright (c) 2024 The Regents of the University of California
#
# This file is part of BRAILS++.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 10-14-2025

"""
This module defines GoogleSatellite class downloading Google satellite imagery.

.. autosummary::

    GoogleSatellite
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from .types import AssetInventory, ImageSet
from .types import Image as ScrapedImage

# Constants:
GOOGLE_TILE_URL = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
ZOOM_LEVEL = 20
TILE_SIZE = 256
RESIZED_IMAGE_SIZE = (640, 640)

FOOTPRINT_BUFFER_RATIO = 0.1
# Minimum absolute buffer in degrees (~5 m at mid-latitudes)
FOOTPRINT_BUFFER_MIN_DEG = 5 / 111_000

REQUESTS_RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504],
)


class GoogleSatellite:
    """
    A class for downloading satellite imagery from Google tilemaps.

    Provides functionality to obtain satellite images for assets defined in an
    AssetInventory. Images are retrieved based on the coordinates of the assets
    and saved to a specified directory.
    """

    def get_images(
        self,
        inventory: AssetInventory,
        entity_type: str = 'entity',
        download_year: int | None = None,
    ) -> ImageSet:
        """
        Fetch satellite images for the assets in an inventory, in memory.

        Nothing is written to disk. Google's Static API policy prohibits
        "pre-fetching, indexing, storing, or caching" of content, so each
        image is held as encoded bytes only until the caller has run
        inference over it.

        Parameters
        ----------
        inventory
            AssetInventory for which the images will be retrieved.
        entity_type
            Type of entity being photographed (e.g. 'footprint', 'parcel').
            Used only to label images for logging.
        download_year
            Year the images are fetched; recorded as image metadata.

        Returns
        -------
        ImageSet
            An ImageSet whose images carry in-memory payloads.

        Raises
        ------
        ValueError
            If the provided inventory is not an instance of AssetInventory.
        """
        if not isinstance(inventory, AssetInventory):
            raise ValueError('Invalid AssetInventory provided.')

        image_set = ImageSet()
        asset_keys = list(inventory.inventory)
        footprints = [inventory.inventory[key].coordinates for key in asset_keys]
        counts = {'fetched': 0, 'failed': 0}

        year_suffix = f'_{download_year}' if download_year else ''
        with tqdm(total=len(asset_keys), desc='Obtaining satellite imagery') as pbar:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.fetch_image_bytes, footprint): key
                    for footprint, key in zip(footprints, asset_keys)
                }
                for future in as_completed(futures):
                    key = futures[future]
                    pbar.update(n=1)
                    try:
                        payload = future.result()
                    except Exception as exc:
                        counts['failed'] += 1
                        tqdm.write(f'Error fetching image for {key}: {exc}')
                        continue
                    counts['fetched'] += 1
                    entity_id_safe = str(key).replace('+', '_')
                    image_set.add_image(
                        key,
                        ScrapedImage(
                            f'{entity_type}_{entity_id_safe}{year_suffix}.tif',
                            payload=payload,
                            metadata={'download_year': download_year},
                        ),
                    )

        image_set.counts = counts
        return image_set

    def fetch_image_bytes(self, footprint: list[tuple[float, float]]) -> bytes:
        """Fetch one footprint's satellite image as encoded bytes.

        The encoded GeoTIFF is built in a memory buffer and returned; it is
        never written to a durable location.
        """
        arr, transform = self._render_footprint(footprint)
        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=RESIZED_IMAGE_SIZE[1],
                width=RESIZED_IMAGE_SIZE[0],
                count=3,
                dtype='uint8',
                crs='EPSG:4326',
                transform=transform,
                compress='deflate',
            ) as dst:
                dst.write(arr.transpose(2, 0, 1))
            return memfile.read()

    def _render_footprint(self, footprint: list[tuple[float, float]]):
        """Fetch, mosaic, crop and resize tiles for one footprint.

        Returns the RGB array and its affine transform without touching the
        filesystem.
        """
        bbox_buffered = self._buffer_footprint(footprint)
        x_list, y_list = self._determine_tile_coords(bbox_buffered)
        tiles, offsets, imbnds = self._fetch_tiles(x_list, y_list)
        combined_image = self._combine_tiles(tiles, (len(x_list), len(y_list)), offsets)

        # Crop combined mosaic to the buffered bbox before resizing.
        # imbnds = [tile_west, tile_north, tile_east, tile_south].
        # Linear lon/lat interpolation is valid at building scale (<100 m):
        # Mercator distortion change over such extents is sub-pixel (<0.02 px).
        bbox_west = min(bbox_buffered[0])
        bbox_east = max(bbox_buffered[0])
        bbox_south = min(bbox_buffered[1])
        bbox_north = max(bbox_buffered[1])
        tile_west, tile_north, tile_east, tile_south = imbnds
        total_w = TILE_SIZE * len(x_list)
        total_h = TILE_SIZE * len(y_list)
        lon_scale = total_w / (tile_east - tile_west)
        lat_scale = total_h / (tile_north - tile_south)
        left = max(0, int((bbox_west - tile_west) * lon_scale))
        right = min(total_w, math.ceil((bbox_east - tile_west) * lon_scale))
        top = max(0, int((tile_north - bbox_north) * lat_scale))
        bottom = min(total_h, math.ceil((tile_north - bbox_south) * lat_scale))
        cropped_image = combined_image.crop((left, top, right, bottom))

        resized_image = cropped_image.resize(RESIZED_IMAGE_SIZE)
        arr = np.array(resized_image)
        transform = from_bounds(
            bbox_west,
            bbox_south,
            bbox_east,
            bbox_north,
            RESIZED_IMAGE_SIZE[0],
            RESIZED_IMAGE_SIZE[1],
        )
        return arr, transform

    def _buffer_footprint(
        self, footprint: list[tuple[float, float]]
    ) -> tuple[list[float], list[float]]:
        """
        Buffer the footprint to account for inaccuracies.

        Parameters
        ----------
        footprint
            Original footprint.

        Returns
        -------
        Tuple[List[float], List[float]]
            Buffered bounding box coordinates.
        """
        lon, lat = zip(*footprint)
        minlon, maxlon = min(lon), max(lon)
        minlat, maxlat = min(lat), max(lat)

        londiff = maxlon - minlon
        latdiff = maxlat - minlat

        lon_buf = max(londiff * FOOTPRINT_BUFFER_RATIO, FOOTPRINT_BUFFER_MIN_DEG)
        lat_buf = max(latdiff * FOOTPRINT_BUFFER_RATIO, FOOTPRINT_BUFFER_MIN_DEG)

        minlon_buff = minlon - lon_buf
        maxlon_buff = maxlon + lon_buf
        minlat_buff = minlat - lat_buf
        maxlat_buff = maxlat + lat_buf

        return (
            [minlon_buff, minlon_buff, maxlon_buff, maxlon_buff],
            [minlat_buff, maxlat_buff, maxlat_buff, minlat_buff],
        )

    def _determine_tile_coords(
        self, bbox_buffered: tuple[list[float], list[float]]
    ) -> tuple[list[int], list[int]]:
        """
        Determine tile x,y coordinates containing the buffered bounding box.

        Parameters
        ----------
        bbox_buffered
            Buffered bounding box.

        Returns
        -------
        Tuple[List[int], List[int]]
            Lists of x and y tile coordinates.
        """
        x_coords, y_coords = [], []
        for lon, lat in zip(bbox_buffered[0], bbox_buffered[1]):
            x, y = self._deg2num(lat, lon, ZOOM_LEVEL)
            x_coords.append(x)
            y_coords.append(y)

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return list(range(x_min, x_max + 1)), list(range(y_min, y_max + 1))

    @staticmethod
    def _deg2num(lat: float, lon: float, zoom: int) -> tuple[int, int]:
        """
        Convert latitude and longitude to tile numbers.

        Parameters
        ----------
        lat
            Latitude in degrees.
        lon
            Longitude in degrees.
        zoom
            Zoom level.

        Returns
        -------
        Tuple[int, int]
            Tile x and y numbers.
        """
        lat_rad = math.radians(lat)
        n = 2**zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile

    def _fetch_tiles(
        self, x_list: list[int], y_list: list[int]
    ) -> tuple[list[Image.Image], list[tuple[int, int]], list[float]]:
        """
        Fetch all tiles for the given x and y coordinates.

        Parameters
        ----------
        x_list
            List of x tile coordinates.
        y_list
            List of y tile coordinates.

        Returns
        -------
        Tuple[List[Image.Image], List[Tuple[int, int]], List[float]]
            List of tile images, their offsets, and image bounds.
        """
        tiles, offsets, imbnds = [], [], []
        ntiles = (len(x_list), len(y_list))

        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=REQUESTS_RETRY_STRATEGY))
        for y_idx, ycoord in enumerate(y_list):
            for x_idx, xcoord in enumerate(x_list):
                url = GOOGLE_TILE_URL.format(x=xcoord, y=ycoord, z=ZOOM_LEVEL)

                response = session.get(url)
                response.raise_for_status()

                tile_image = Image.open(BytesIO(response.content))
                tiles.append(tile_image)
                offsets.append((x_idx * TILE_SIZE, y_idx * TILE_SIZE))
                tile_bounds = self._tile_bbox(ZOOM_LEVEL, xcoord, ycoord)
                imbnds = self._update_image_bounds(
                    imbnds, tile_bounds, ntiles, x_idx, y_idx
                )

        return tiles, offsets, imbnds

    @staticmethod
    def _tile_bbox(zoom: int, x_coord: int, y_coord: int) -> list[float]:
        """
        Get the bounding box of a tile.

        Parameters
        ----------
        zoom
            Zoom level.
        x_coord
            Tile x number.
        y_coord
            Tile y number.

        Returns
        -------
        List[float]
            Bounding box coordinates in [south, north, west, east] order.
        """
        return [
            GoogleSatellite._tile_lat(y_coord, zoom),
            GoogleSatellite._tile_lat(y_coord + 1, zoom),
            GoogleSatellite._tile_lon(x_coord, zoom),
            GoogleSatellite._tile_lon(x_coord + 1, zoom),
        ]

    @staticmethod
    def _tile_lat(y_coord: int, z_coord: int) -> float:
        """
        Calculate latitude from tile y number.

        Parameters
        ----------
        y_coord
            Tile y number.
        z_coord
            Zoom level.

        Returns
        -------
        float
            Latitude in degrees.
        """
        n = math.pi - (2.0 * math.pi * y_coord) / (2**z_coord)
        return math.degrees(math.atan(math.sinh(n)))

    @staticmethod
    def _tile_lon(xcoord: int, zcoord: int) -> float:
        """
        Calculate longitude from tile x number.

        Parameters
        ----------
        xcoord
            Tile x number.
        zcoord
            Zoom level.

        Returns
        -------
        float
            Longitude in degrees.
        """
        return xcoord / (2**zcoord) * 360.0 - 180.0

    @staticmethod
    def _update_image_bounds(
        imbnds: list[float],
        tilebnds: list[float],
        ntiles: tuple[int, int],
        xind: int,
        yind: int,
    ) -> list[float]:
        """
        Update image bounds based on tile bounds.

        Parameters
        ----------
        imbnds
            Current image bounds.
        tilebnds
            Bounds of the current tile.
        ntiles
            Number of tiles in x and y directions.
        xind
            Current tile x index.
        yind
            Current tile y index.

        Returns
        -------
        List[float]
            Updated image bounds.
        """
        if ntiles[0] > 1 and ntiles[1] > 1:
            if xind == 0 and yind == 0:
                imbnds.append(tilebnds[2])
                imbnds.append(tilebnds[0])
            elif xind == ntiles[0] - 1 and yind == 0:
                imbnds.append(tilebnds[3])
            elif xind == 0 and yind == ntiles[1] - 1:
                imbnds.append(tilebnds[1])
        elif ntiles[0] == 1 and ntiles[1] == 1:
            imbnds = [tilebnds[2], tilebnds[0], tilebnds[3], tilebnds[1]]
        elif ntiles[0] == 1:
            if yind == 0:
                imbnds.append(tilebnds[2])
                imbnds.append(tilebnds[0])
                imbnds.append(tilebnds[3])
            elif yind == ntiles[1] - 1:
                imbnds.append(tilebnds[1])
        elif ntiles[1] == 1:
            if xind == 0:
                imbnds.append(tilebnds[2])
                imbnds.append(tilebnds[0])
            elif xind == ntiles[0] - 1:
                imbnds.append(tilebnds[3])
                imbnds.append(tilebnds[1])
        return imbnds

    @staticmethod
    def _combine_tiles(
        tiles: list[Image.Image],
        ntiles: tuple[int, int],
        offsets: list[tuple[int, int]],
    ) -> Image.Image:
        """
        Combine individual tiles into a single image.

        Parameters
        ----------
        tiles
            List of tile images.
        ntiles
            Number of tiles in x and y directions.
        offsets
            Offsets for pasting tiles.

        Returns
        -------
        Image.Image
            Combined image.
        """
        combined_image = Image.new(
            'RGB', (TILE_SIZE * ntiles[0], TILE_SIZE * ntiles[1])
        )
        for ind, image in enumerate(tiles):
            combined_image.paste(image, offsets[ind])
        return combined_image
