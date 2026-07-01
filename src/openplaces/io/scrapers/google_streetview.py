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
This module defines GoogleStreetview class downloading Google street imagery.

.. autosummary::

    GoogleStreetview
"""

import base64
import hashlib
import hmac
import json
import math
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import PIL
import requests
from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from shapely.geometry import Polygon
from tqdm import tqdm

from .types import AssetInventory, ImageSet
from .types import Image as ScrapedImage

# Constants:
BASE_API_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata'
PANORAMA_METADATA_URL = 'https://www.google.com/maps/photometa/v1'
TILE_URL_TEMPLATE = (
    'https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}'
)

STATIC_IMAGE_URL = 'https://maps.googleapis.com/maps/api/streetview'
STATIC_IMAGE_SIZE = '640x640'

ZOOM2FOV_MAPPER = {0: 360, 1: 180, 2: 90, 3: 45, 4: 22.5, 5: 11.25}
FOV_BUFFER_DEG = 10
FLOOR_HEIGHT_FT = 10.5
CAM_HEIGHT_FT = 8.2
PITCH_MIN_DEG = -40.0
PITCH_MAX_DEG = 60.0
TILE_SIZE = 512
CIRCUM_EARTH_FT = 131482560

REQUESTS_TIMEOUT_VAL = 30
REQUESTS_RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504],
)
DEFAULT_MAX_WORKERS = 8
DEFAULT_BATCH_SIZE = 32
API_REQUEST_MAX_ATTEMPTS = 5
API_RETRY_BACKOFF_SECONDS = 0.5
RETRYABLE_API_STATUSES = {'UNKNOWN_ERROR', 'OVER_QUERY_LIMIT'}
RETRYABLE_HTTP_STATUSES = {403, 429, 500, 502, 503, 504}

FILE_SUBDIRECTORIES = {'images': 'images'}


class GoogleStreetViewAPIError(RuntimeError):
    """Google rejected a Street View request for a non-coverage reason."""


class GoogleStreetViewPanoramaMetadataError(RuntimeError):
    """Google panorama metadata was unavailable or malformed."""


class GoogleStreetViewDownloadError(RuntimeError):
    """A Street View image could not be downloaded or decoded."""


class GoogleStreetViewImageUnavailable(RuntimeError):
    """Street View metadata exists, but no retrievable image is available."""


class GoogleStreetview:
    """
    A class that downloads street-level imagery and depth maps for buildings.

    Interfaces with the Google Street View API to obtain high-resolution
    street-level images and corresponding depth maps based on specified
    building footprints.

    Attributes
    ----------
    api_key
        API key for authenticating requests to the Google Street View API.
    """

    def __init__(self, input_data: dict[str, Any]):
        """
        Initialize the GoogleStreetview object.

        Parameters
        ----------
        input_data
            A dictionary containing ``'apiKey'`` (str): a valid Google API key
            with Street View Static API enabled.

        Raises
        ------
        ValueError
            If ``'apiKey'`` is missing or empty.
        ConnectionError
            If the API key validation request fails.
        """
        try:
            api_key = input_data['apiKey']
            if not api_key:
                raise ValueError('API key is empty. Please provide a valid API key.')
        except KeyError as exception:
            raise ValueError(
                'Please provide a Google API key to run theGoogleStreetview module'
            ) from exception
        self.api_key = api_key
        self.url_signing_secret = input_data.get('urlSigningSecret')
        self.pitch: float = float(input_data.get('pitch', 0))
        self.verbose: bool = bool(input_data.get('verbose', False))
        self.max_workers = int(input_data.get('max_workers', DEFAULT_MAX_WORKERS))
        self.batch_size = int(input_data.get('batch_size', DEFAULT_BATCH_SIZE))
        if self.max_workers < 1:
            raise ValueError('max_workers must be at least 1.')
        if self.batch_size < self.max_workers:
            raise ValueError('batch_size must be at least max_workers.')
        self._validate_api_key(api_key)
        self._validate_static_api()

    @staticmethod
    def _validate_api_key(api_key: str):
        """
        Validate the provided Google API Key for the Street View Static API.

        Parameters
        ----------
        api_key
            The Google API Key to validate.

        Raises
        ------
        ValueError
            If the API key does not have Street View Static API enabled.
        ConnectionError
            If the validation request fails.
        """
        try:
            params = {
                'location': '37.8725187407,-122.2596028649',
                'source': 'outdoor',
                'key': api_key,
            }
            response = requests.get(
                BASE_API_URL, params=params, timeout=REQUESTS_TIMEOUT_VAL
            )

            data = response.json()
            if data.get('status') not in ('OK', 'ZERO_RESULTS'):
                msg = data.get('error_message', data.get('status', response.text))
                raise ValueError(f'Google Street View API key validation failed: {msg}')
        except requests.RequestException as exception:
            raise ConnectionError(
                f'Failed to validate API key: {exception}'
            ) from exception

    def _validate_static_api(self) -> None:
        """Verify that the Static API accepts the configured credentials."""
        response = self._request_static_image(
            {
                'size': '64x64',
                'location': '37.8725187407,-122.2596028649',
                'source': 'outdoor',
                'key': self.api_key,
                'return_error_code': 'true',
            },
            max_attempts=1,
        )
        if response.status_code == 200:
            return

        if self.url_signing_secret:
            secret_hint = (
                'Google rejected the signature. Ensure url_signing_secret is '
                'the Current secret from the Secret Generator card in the '
                'same Google Cloud project as api_key.'
            )
        else:
            secret_hint = (
                'Add the Google Maps URL signing secret to credentials.yaml:\n\n'
                '  google_streetview:\n'
                '    api_key: YOUR_KEY_HERE\n'
                '    url_signing_secret: YOUR_URL_SIGNING_SECRET\n'
            )
        raise ValueError(
            f'Street View Static API credential validation failed '
            f'(HTTP {response.status_code}). {secret_hint}'
        )

    def get_images(
        self,
        inventory: AssetInventory,
        save_directory: str,
        entity_type: str = 'entity',
        download_year: int | None = None,
        redownload: bool = False,
    ) -> ImageSet:
        """
        Get street-level images of buildings from footprints in AssetInventory.

        Parameters
        ----------
        inventory
            AssetInventory for which the images will be retrieved.
        save_directory
            Path to the folder where the retrieved images will be saved.
        entity_type
            Type of entity being photographed (e.g. 'footprint', 'parcel').
            Used as a filename prefix.
        download_year
            Year the images are fetched; appended as a filename suffix.
        redownload
            Missing images are always fetched. When True, buildings whose
            image already exists on disk are re-fetched (overwritten) rather
            than reused.

        Returns
        -------
        ImageSet
            An ImageSet for the assets in the inventory.

        Raises
        ------
        ValueError
            If the provided inventory is not an instance of AssetInventory.
        """
        if not isinstance(inventory, AssetInventory):
            raise ValueError('Invalid AssetInventory provided.')

        base_dir_path = Path(save_directory)
        base_dir_path.mkdir(parents=True, exist_ok=True)

        image_set = ImageSet()
        image_set.dir_path = str(base_dir_path)

        asset_footprints = []
        asset_keys = []
        asset_n_stories = []
        for key, asset in inventory.inventory.items():
            asset_footprints.append(asset.coordinates)
            asset_keys.append(key)
            asset_n_stories.append(getattr(asset, 'n_stories', None))

        street_images, metadata = self._download_images(
            asset_footprints,
            asset_keys,
            base_dir_path,
            False,
            True,
            entity_type=entity_type,
            download_year=download_year,
            n_stories_list=asset_n_stories,
            redownload=redownload,
        )

        image_set.dir_path = str(base_dir_path / FILE_SUBDIRECTORIES['images'])

        for index, image_path in enumerate(street_images):
            if image_path.exists():
                img = ScrapedImage(image_path.name, metadata[image_path])
                image_set.add_image(asset_keys[index], img)

        return image_set

    def _download_images(
        self,
        footprints: list[list[tuple[float, float]]],
        keys: list,
        save_dir: Path,
        save_interim_images: bool,
        save_all_cam_metadata: bool,
        entity_type: str = 'entity',
        download_year: int | None = None,
        n_stories_list: list[float | None] | None = None,
        redownload: bool = False,
    ) -> tuple[list[Path], dict[str, list[Any]]]:
        """
        Download street-level imagery and depthmap for building footprints.

        Parameters
        ----------
        footprints
            List of building footprints.
        keys
            Entity IDs corresponding to each footprint; used as filenames.
        save_dir
            Directory to save the images.
        save_interim_images
            Whether to save interim panorama images.
        save_all_cam_metadata
            Whether to save all camera metadata.
        entity_type
            Entity type prefix for filenames (e.g. 'footprint').
        download_year
            Year suffix for filenames.
        redownload
            Missing images are always fetched. When True, buildings whose
            image already exists on disk are re-fetched rather than reused.

        Returns
        -------
        tuple[list[Path], dict]
            List of image paths and a metadata dictionary keyed by image path.
        """
        image_dir = save_dir / FILE_SUBDIRECTORIES['images']
        image_dir.mkdir(parents=True, exist_ok=True)

        street_image_paths = []
        bldg_centroids = []
        inps = []
        year_suffix = f'_{download_year}' if download_year else ''
        for i, (footprint_raw, key) in enumerate(zip(footprints, keys)):
            footprint = np.fliplr(np.squeeze(np.array(footprint_raw))).tolist()
            footprint_cent = Polygon(footprint_raw).centroid
            entity_id_safe = str(key).replace('+', '_')
            stem = f'{entity_type}_{entity_id_safe}{year_suffix}'
            image_path = image_dir / f'{stem}.jpg'
            street_image_paths.append(image_path)
            bldg_centroids.append((footprint_cent.y, footprint_cent.x))
            n_stories = n_stories_list[i] if n_stories_list is not None else None
            inps.append(
                (
                    footprint,
                    (footprint_cent.y, footprint_cent.x),
                    image_path,
                    n_stories,
                )
            )

        results = {}
        counts = {'found': 0, 'missing': 0, 'cached': 0}
        with (
            tqdm(
                total=len(footprints),
                desc='Obtaining street-level imagery',
                unit='building',
            ) as pbar,
            ThreadPoolExecutor(max_workers=self.max_workers) as executor,
        ):
            for batch_start in range(0, len(inps), self.batch_size):
                batch = inps[batch_start : batch_start + self.batch_size]
                futures = {}
                for footprint, footprint_cent, image_path, n_stories in batch:
                    if image_path.exists() and not redownload:
                        # Reuse the cached image; redownload would re-fetch it.
                        results[image_path] = None
                        counts['cached'] += 1
                        pbar.update()
                        continue
                    future = executor.submit(
                        self._download_streetlev_image,
                        footprint,
                        footprint_cent,
                        image_path,
                        save_intermediate_imagery=save_interim_images,
                        save_all_cam_meta=save_all_cam_metadata,
                        n_stories=n_stories,
                    )
                    futures[future] = image_path

                for future in as_completed(futures):
                    image_path = futures[future]
                    try:
                        result = future.result()
                    except GoogleStreetViewAPIError:
                        for pending in futures:
                            pending.cancel()
                        raise
                    except (
                        KeyError,
                        ConnectionError,
                        PIL.UnidentifiedImageError,
                        requests.RequestException,
                        ValueError,
                    ) as exc:
                        for pending in futures:
                            pending.cancel()
                        raise GoogleStreetViewDownloadError(
                            f'Failed to download Street View image for '
                            f'{image_path.stem}: {exc}'
                        ) from exc
                    else:
                        results[image_path] = result
                        counts['missing' if result is None else 'found'] += 1
                    finally:
                        pbar.update()

                pbar.set_postfix(counts, refresh=True)

        metadata = self._process_meta_for_images(
            street_image_paths, bldg_centroids, results, save_all_cam_metadata
        )

        return street_image_paths, metadata

    def _download_streetlev_image(
        self,
        footprint: list[list[float, float]],
        fpcent: tuple[float, float],
        im_path: Path,
        save_intermediate_imagery: bool = False,
        save_all_cam_meta: bool = False,
        n_stories: float | None = None,
    ) -> tuple | None:
        """
        Download a street-level panoramic image and process it.

        Parameters
        ----------
        footprint
            Coordinates of the building footprint to crop the image.
        fpcent
            Latitude and longitude of the query point.
        im_path
            Path object for the output image file.
        save_intermediate_imagery
            Whether to save intermediate panorama images. Defaults to False.
        save_all_cam_meta
            Whether to return all camera metadata. Defaults to False.
        n_stories
            Number of building stories; used to compute dynamic pitch.

        Returns
        -------
        tuple or None
            Camera elevation, camera location, bounding angles, and pitch
            (plus additional metadata if save_all_cam_meta is True).
            Returns None if no street-level imagery is found.
        """
        pano_name, comp_im_name = '', ''

        im_name = str(im_path.as_posix())

        if save_intermediate_imagery:
            im_name_base = im_name.rsplit('.', 1)[0]
            pano_name = f'{im_name_base}_pano.jpg'
            comp_im_name = f'{im_name_base}_composite.jpg'

        pano = {
            'queryLatLon': fpcent,
            'camLatLon': fpcent,
            'id': '',
            'panoSize': (),
            'camHeading': 0,
            'camElev': None,
            'panoTilt': None,
            'panoRoll': None,
            'depthMap': 0,
            'depthMapString': '',
            'panoImFile': pano_name,
            'compositeImFile': comp_im_name,
        }

        pano_metadata = self._get_pano_metadata(fpcent, self.api_key)
        if pano_metadata is None:
            return None
        pano['id'] = pano_metadata['pano_id']
        location = pano_metadata.get('location') or {}
        pano['camLatLon'] = (
            location.get('lat', fpcent[0]),
            location.get('lng', fpcent[1]),
        )

        face = self._find_near_face(pano, footprint)
        face_mid = face[0] if face is not None else fpcent

        pitch = (
            self._compute_building_pitch(pano, face_mid, n_stories)
            if n_stories is not None
            else self.pitch
        )

        try:
            pano = self._download_bldg_image(
                pano, footprint, im_name, self.api_key, pitch=pitch
            )
        except GoogleStreetViewImageUnavailable:
            return None

        if save_all_cam_meta:
            return (
                pano['camElev'],
                pano['camLatLon'],
                pano['panoBndAngles'],
                pano['panoSize'],
                pano['camHeading'],
                pano['panoTilt'],
                pano['panoFOV'],
                pano['panoRoll'],
                pitch,
            )
        return (
            pano['camElev'],
            pano['camLatLon'],
            pano['panoBndAngles'],
            pitch,
        )

    @staticmethod
    def _process_meta_for_images(
        street_image_paths: list[Path],
        bldg_centroids: list[tuple[float, float]],
        results: dict[Path, dict[str, float | str | None]],
        save_all_cam_metadata: bool,
    ) -> dict[Path, dict[str, tuple[float, float] | float | str]]:
        """
        Process the downloaded street image data and extract relevant metadata.

        Parameters
        ----------
        street_image_paths
            List of file paths for the downloaded street-level imagery.
        bldg_centroids
            List of centroids of buildings.
        results
            Dictionary mapping image file paths to raw result tuples.
        save_all_cam_metadata
            Whether to save all available camera metadata.

        Returns
        -------
        dict
            Dictionary mapping each image path to a metadata dict.
        """
        metadata = {}
        additional_keys = [
            'panoSize',
            'camHeading',
            'panoTilt',
            'panoFOV',
            'panoRoll',
        ]

        for image_ind, image_path in enumerate(street_image_paths):
            properties = {
                'bdlgLatLon': (),
                'camElev': None,
                'camLatLon': None,
                'panoBndAngles': None,
                'camPitch': None,
            }
            if save_all_cam_metadata:
                properties.update({key: None for key in additional_keys})
            properties['bdlgLatLon'] = bldg_centroids[image_ind]
            if results[image_path] is not None:
                properties['camElev'] = results[image_path][0]
                properties['camLatLon'] = results[image_path][1]
                properties['panoBndAngles'] = results[image_path][2]

                if save_all_cam_metadata:
                    properties['panoSize'] = results[image_path][3]
                    properties['camHeading'] = results[image_path][4]
                    properties['panoTilt'] = results[image_path][5]
                    properties['panoFOV'] = results[image_path][6]
                    properties['panoRoll'] = results[image_path][7]
                    properties['camPitch'] = results[image_path][8]
                else:
                    properties['camPitch'] = results[image_path][3]

            metadata[image_path] = dict(properties)
        return metadata

    @staticmethod
    def _get_pano_metadata(
        latlon: tuple[float, float], api_key: str
    ) -> dict[str, Any] | None:
        """Return official Street View metadata, or None when coverage is absent."""
        params = {
            'location': f'{latlon[0]}, {latlon[1]}',
            'key': api_key,
            'source': 'outdoor',
        }
        for attempt in range(1, API_REQUEST_MAX_ATTEMPTS + 1):
            try:
                response = requests.get(
                    BASE_API_URL, params=params, timeout=REQUESTS_TIMEOUT_VAL
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException:
                if attempt == API_REQUEST_MAX_ATTEMPTS:
                    raise
                GoogleStreetview._sleep_before_retry(attempt)
                continue

            status = data.get('status')
            if status == 'ZERO_RESULTS':
                return None
            if status == 'OK':
                if not data.get('pano_id'):
                    raise GoogleStreetViewAPIError(
                        'Google Street View returned OK without a pano_id.'
                    )
                return data
            if status in RETRYABLE_API_STATUSES:
                if attempt < API_REQUEST_MAX_ATTEMPTS:
                    GoogleStreetview._sleep_before_retry(attempt)
                    continue
                message = data.get('error_message', status)
                raise GoogleStreetViewAPIError(
                    f'Google Street View metadata request failed after '
                    f'{attempt} attempts: {message}'
                )

            message = data.get('error_message', status or 'unknown response')
            raise GoogleStreetViewAPIError(
                f'Google Street View metadata request failed: {message}'
            )

        raise AssertionError('Street View metadata retry loop exited unexpectedly.')

    @staticmethod
    def _sleep_before_retry(attempt: int) -> None:
        """Sleep with exponential backoff after a failed API attempt."""
        time.sleep(API_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))

    @classmethod
    def _get_pano_id(cls, latlon: tuple[float, float], api_key: str) -> str | None:
        """
        Obtain the pano ID for the given latitude and longitude.

        Parameters
        ----------
        latlon
            Latitude and longitude.
        api_key
            Google API key.

        Returns
        -------
        str or None
            Pano ID, or None when Google reports no outdoor imagery.
        """
        metadata = cls._get_pano_metadata(latlon, api_key)
        return metadata['pano_id'] if metadata is not None else None

    @staticmethod
    def _get_pano_meta(pano: dict[str, Any], dmap_outname: str = '') -> dict[str, Any]:
        """
        Retrieve metadata for the pano.

        Parameters
        ----------
        pano
            Pano dictionary.
        dmap_outname
            Filename to save the depth map string.

        Returns
        -------
        dict
            Updated pano dictionary.
        """
        params = {
            'authuser': '0',
            'hl': 'en',
            'gl': 'us',
            'pb': (
                '!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2suk!3m3!1m2!'
                '1e2!2s' + pano['id'] + '!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!'
                '1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!'
                '9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!'
                '1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!'
                '3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3'
            ),
        }

        try:
            response = requests.get(
                PANORAMA_METADATA_URL,
                params=params,
                proxies=None,
                timeout=REQUESTS_TIMEOUT_VAL,
            )
            response.raise_for_status()

            response_json = json.loads(response.content[4:])
            pano_zoom = 3
            depth_map_string = response_json[1][0][5][0][5][1][2]
            cam_latlon = (
                response_json[1][0][5][0][1][0][2],
                response_json[1][0][5][0][1][0][3],
            )
            pano_size = tuple(response_json[1][0][2][3][0][pano_zoom][0])[::-1]
            cam_heading = response_json[1][0][5][0][1][2][0]
            pano_tilt = response_json[1][0][5][0][1][2][1]
            pano_roll = response_json[1][0][5][0][1][2][2]
            cam_elev = response_json[1][0][5][0][1][1][0]
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            requests.RequestException,
        ) as exc:
            raise GoogleStreetViewPanoramaMetadataError(
                f'invalid photometa response for pano {pano["id"]}'
            ) from exc

        pano.update(
            {
                'panoZoom': pano_zoom,
                'panoFOV': ZOOM2FOV_MAPPER[pano_zoom],
                'depthMapString': depth_map_string,
                'camLatLon': cam_latlon,
                'panoSize': pano_size,
                'camHeading': cam_heading,
                'panoTilt': pano_tilt,
                'panoRoll': pano_roll,
                'camElev': cam_elev,
            }
        )

        if dmap_outname:
            with open(dmap_outname, 'w', encoding='utf-8') as dmapfile:
                dmapfile.write(pano['depthMapString'])

        return pano

    def _download_pano_image(
        self, pano: dict[str, Any], im_name: str
    ) -> dict[str, Any]:
        """
        Download the pano image composed of tiles.

        Parameters
        ----------
        pano
            Pano dictionary.
        im_name
            Filename to save the pano image.

        Returns
        -------
        dict
            Updated pano dictionary.
        """
        pano_id = pano['id']
        image_size = pano['panoSize']
        zoom = pano['panoZoom']

        baseurl = TILE_URL_TEMPLATE.format(pano_id=pano_id, zoom=zoom, x='{x}', y='{y}')

        urls = []
        offsets = []
        for x_coord in range(int(image_size[0] / TILE_SIZE)):
            for y_coord in range(int(image_size[1] / TILE_SIZE)):
                urls.append(baseurl.format(x=f'{x_coord}', y=f'{y_coord}'))
                offsets.append((x_coord * TILE_SIZE, y_coord * TILE_SIZE))

        tiles = self._download_tiles(urls)

        combined_im = PIL.Image.new('RGB', image_size)

        for ind, image in enumerate(tiles):
            combined_im.paste(image, offsets[ind])

        pano['panoImage'] = combined_im.copy()
        if im_name:
            combined_im.save(im_name)
        pano['panoImFile'] = im_name
        return pano

    def _get_bldg_image(
        self,
        pano: dict[str, Any],
        footprint: list[tuple[float, float]],
        im_name: str = 'imstreet.jpg',
        save_depthmap: bool = False,
    ) -> dict[str, Any]:
        """
        Generate an image and depthmap cropped around a building from a pano.

        Parameters
        ----------
        pano
            Panorama dictionary.
        footprint
            Building footprint vertices as (lat, lon) pairs.
        im_name
            Output image filename.
        save_depthmap
            Whether to save the building depth map.

        Returns
        -------
        dict
            Updated panorama dictionary with ``panoBndAngles`` and
            ``depthMapBldg`` added.
        """
        camera_angles = self._get_view_angles(pano, footprint)
        bnd_angles = np.rint(
            (
                np.array(
                    [
                        round(min(camera_angles), -1) - FOV_BUFFER_DEG,
                        round(max(camera_angles), -1) + FOV_BUFFER_DEG,
                    ]
                )
                + 180
            )
            / 360
            * pano['panoSize'][0]
        )

        bldg_image = pano['panoImage']
        bldg_im_cropped = bldg_image.crop(
            (bnd_angles[0], 0, bnd_angles[1], pano['panoSize'][1])
        )
        bldg_im_cropped.save(im_name)
        pano['panoBndAngles'] = np.copy(bnd_angles)

        if save_depthmap:
            pano_dmap_name = (
                im_name.replace('.' + im_name.split('.')[-1], '') + '_pano_depthmap.jpg'
            )
            pano = self._get_depth_map(pano, dmap_imname=pano_dmap_name)
            mask = pano['depthMap']

            dmap_name = (
                im_name.replace('.' + im_name.split('.')[-1], '') + '_depthmap.jpg'
            )
            mask_cropped = mask.crop(
                (bnd_angles[0], 0, bnd_angles[1], pano['panoSize'][1])
            )
            pano['depthMapBldg'] = mask_cropped.copy()
            mask_cropped.convert('RGB').save(dmap_name)

        return pano

    def _download_bldg_image(
        self,
        pano: dict[str, Any],
        footprint: list[tuple[float, float]],
        im_name: str,
        api_key: str,
        pitch: float | None = None,
    ) -> dict[str, Any]:
        """Download a building-facing street-view image via the Static API.

        Replaces the old tile-based panorama download + crop approach.
        Computes heading and FOV from the building footprint angles, then
        fetches a single JPEG from the Street View Static API.

        Parameters
        ----------
        pano
            Panorama dictionary; must already contain ``id``, ``camHeading``.
        footprint
            Building footprint vertices as (lat, lon) pairs.
        im_name
            Output JPEG path.
        api_key
            Google API key with Street View Static API enabled.
        pitch
            Vertical camera angle in degrees (positive = up). When None,
            falls back to ``self.pitch``.

        Returns
        -------
        dict
            Updated pano dictionary with ``panoFOV``, ``panoSize``,
            ``panoBndAngles`` set.
        """
        if pitch is None:
            pitch = self.pitch
        face = self._find_near_face(pano, footprint)
        all_angles = self._get_view_angles(pano, footprint)
        if face is not None:
            _, face_angles = face
            center_offset = sum(face_angles) / 2
        else:
            center_offset = (min(all_angles) + max(all_angles)) / 2
        heading = (pano['camHeading'] + center_offset) % 360
        half_fov = (
            max(max(all_angles) - center_offset, center_offset - min(all_angles))
            + FOV_BUFFER_DEG
        )
        fov = min(max(2 * half_fov, 10), 120)

        view_params = {
            'size': STATIC_IMAGE_SIZE,
            'heading': heading,
            'fov': fov,
            'pitch': pitch,
            'key': api_key,
            'return_error_code': 'true',
        }
        response = self._request_static_image({**view_params, 'pano': pano['id']})
        if response.status_code in {403, 404}:
            # Panorama IDs can become inaccessible even when a fresh metadata
            # lookup returns them. Refresh the image by camera location before
            # deciding whether this is a true miss or an API-level failure.
            cam_lat, cam_lon = pano['camLatLon']
            response = self._request_static_image(
                {
                    **view_params,
                    'location': f'{cam_lat},{cam_lon}',
                    'source': 'outdoor',
                    'radius': 50,
                }
            )

        if response.status_code == 404:
            raise GoogleStreetViewImageUnavailable(
                f'No retrievable Street View image for pano {pano["id"]}.'
            )
        if response.status_code != 200:
            content_type = response.headers.get('content-type', 'unknown')
            raise ConnectionError(
                f'Street View Static API failed '
                f'(HTTP {response.status_code}, content-type={content_type}).'
            )

        img = PIL.Image.open(BytesIO(response.content))
        if im_name:
            img.save(im_name)

        width, height = img.size
        pano['panoFOV'] = fov
        pano['panoSize'] = (width, height)
        pano['panoBndAngles'] = np.array([0, width])
        return pano

    def _request_static_image(
        self,
        params: dict[str, Any],
        max_attempts: int = API_REQUEST_MAX_ATTEMPTS,
    ) -> requests.Response:
        """Request a Static API image, retrying transient delivery failures."""
        prepared = requests.Request('GET', STATIC_IMAGE_URL, params=params).prepare()
        request_url = prepared.url
        if request_url is None:
            raise ValueError('Could not prepare Street View Static API URL.')
        if self.url_signing_secret:
            request_url = self._sign_url(request_url, self.url_signing_secret)

        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(request_url, timeout=REQUESTS_TIMEOUT_VAL)
            except requests.RequestException:
                if attempt == max_attempts:
                    raise
                self._sleep_before_retry(attempt)
                continue

            if response.status_code == 200:
                return response
            if response.status_code == 404:
                return response
            if (
                response.status_code in RETRYABLE_HTTP_STATUSES
                and attempt < max_attempts
            ):
                self._sleep_before_retry(attempt)
                continue
            return response

        if response is None:
            raise AssertionError(
                'Street View Static API retry loop exited unexpectedly.'
            )
        return response

    @staticmethod
    def _sign_url(url: str, signing_secret: str) -> str:
        """Add a Google Maps digital signature to a request URL."""
        prepared = requests.PreparedRequest()
        prepared.prepare_url(url, None)
        if prepared.path_url is None or prepared.url is None:
            raise ValueError('Could not prepare URL for signing.')

        padded_secret = signing_secret + '=' * (-len(signing_secret) % 4)
        try:
            decoded_secret = base64.urlsafe_b64decode(padded_secret)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                'google_streetview.url_signing_secret is not valid URL-safe base64.'
            ) from exc
        signature = hmac.new(
            decoded_secret,
            prepared.path_url.encode('utf-8'),
            hashlib.sha1,
        )
        encoded_signature = base64.urlsafe_b64encode(signature.digest()).decode('ascii')
        separator = '&' if '?' in prepared.url else '?'
        return f'{prepared.url}{separator}signature={encoded_signature}'

    @staticmethod
    def _get_composite_pano(
        pano: dict[str, Any], compim_name: str = 'panoOverlaid.jpg'
    ) -> None:
        """
        Create a composite pano image by overlaying a heat map of the depth map.

        Parameters
        ----------
        pano
            Panorama dictionary containing ``depthMap`` and ``panoImage``.
        compim_name
            Filename for saving the overlaid image.
        """
        image = np.array(pano['depthMap'].convert('L'))
        cm_jet = mpl.colormaps['jet']

        im_colored = cm_jet(image)
        im_colored = np.uint8(im_colored * 255)
        im_mask = Image.fromarray(im_colored).convert('RGB')

        im_pano = pano['panoImage']
        im_overlaid = Image.blend(im_mask, im_pano, 0.5)
        im_overlaid.save(compim_name)

    @staticmethod
    def _compute_building_pitch(
        pano: dict[str, Any],
        bldg_latlon: tuple[float, float],
        n_stories: float,
    ) -> float:
        """Compute camera pitch to aim at mid-height of a building facade.

        Uses the actual camera position from *pano* and the building centroid
        to derive horizontal distance, then applies the formula
        ``atan2(mid_height − cam_height, horiz_dist)`` in the same foot-based
        flat-earth coordinates used by `_get_view_angles`.

        Parameters
        ----------
        pano
            Panorama dict with ``camLatLon`` already populated.
        bldg_latlon
            Building centroid as ``(lat, lon)`` in decimal degrees.
        n_stories
            Number of building stories.

        Returns
        -------
        float
            Pitch angle in degrees, clamped to
            [``PITCH_MIN_DEG``, ``PITCH_MAX_DEG``].
        """
        lat0, lon0 = pano['camLatLon']
        lat1, lon1 = bldg_latlon
        dx = (lon1 - lon0) * (
            CIRCUM_EARTH_FT * math.cos(math.radians((lat0 + lat1) / 2)) / 360
        )
        dy = (lat1 - lat0) * CIRCUM_EARTH_FT / 360
        horiz_dist = math.sqrt(dx**2 + dy**2)
        if horiz_dist < 1:
            return 0.0
        target_ht = (n_stories * FLOOR_HEIGHT_FT) / 2 - CAM_HEIGHT_FT
        pitch = math.degrees(math.atan2(target_ht, horiz_dist))
        return max(PITCH_MIN_DEG, min(PITCH_MAX_DEG, pitch))

    def _get_view_angles(
        self, pano: dict[str, Any], footprint: list[tuple[float, float]]
    ) -> list[float]:
        """
        Calculate viewing angles of each footprint vertex from camera location.

        Parameters
        ----------
        pano
            Panorama dictionary with ``camLatLon`` and ``camHeading``.
        footprint
            Building footprint vertices as (lat, lon) pairs.

        Returns
        -------
        list[float]
            Viewing angles for each vertex relative to the camera's heading.
        """
        (lat0, lon0) = pano['camLatLon']
        xy_footprint = []
        for lat1, lon1 in footprint:
            xcoord = (lon1 - lon0) * (
                CIRCUM_EARTH_FT * math.cos((lat0 + lat1) * math.pi / 360) / 360
            )
            ycoord = (lat1 - lat0) * CIRCUM_EARTH_FT / 360
            xy_footprint.append((xcoord, ycoord))

        return [
            self._get_angle_from_heading(coord, pano['camHeading'])
            for coord in xy_footprint
        ]

    def _find_near_face(
        self,
        pano: dict[str, Any],
        footprint: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], list[float]] | None:
        """Return the footprint edge whose midpoint is closest to the camera.

        Parameters
        ----------
        pano
            Panorama dict with ``camLatLon`` and ``camHeading`` set.
        footprint
            Building footprint vertices as (lat, lon) pairs (closed ring).

        Returns
        -------
        tuple or None
            ``((mid_lat, mid_lon), [angle_v0, angle_v1])`` for the nearest
            edge, or ``None`` for degenerate footprints (fewer than 2 vertices).
        """
        if len(footprint) < 2:
            return None
        cam_lat, cam_lon = pano['camLatLon']
        best_dist = math.inf
        best_edge = 0
        for i in range(len(footprint) - 1):
            lat0, lon0 = footprint[i]
            lat1, lon1 = footprint[i + 1]
            mid_lat = (lat0 + lat1) / 2
            mid_lon = (lon0 + lon1) / 2
            cos_lat = math.cos((cam_lat + mid_lat) * math.pi / 360)
            dx = (mid_lon - cam_lon) * CIRCUM_EARTH_FT * cos_lat / 360
            dy = (mid_lat - cam_lat) * CIRCUM_EARTH_FT / 360
            dist = math.sqrt(dx**2 + dy**2)
            if dist < best_dist:
                best_dist = dist
                best_edge = i
        v0 = footprint[best_edge]
        v1 = footprint[best_edge + 1]
        mid = ((v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2)
        return mid, self._get_view_angles(pano, [v0, v1])

    @staticmethod
    def _download_tiles(urls: list[str]) -> list[Image.Image]:
        """
        Download image tiles from the provided URLs with retry strategy.

        Parameters
        ----------
        urls
            List of tile URLs.

        Returns
        -------
        list[PIL.Image.Image]
            List of downloaded tile images.
        """
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=REQUESTS_RETRY_STRATEGY))
        tiles = []
        for url in urls:
            response = session.get(url)
            if not response.ok:
                raise ConnectionError(
                    f'Tile download failed (HTTP {response.status_code}): {url}'
                )
            tiles.append(PIL.Image.open(BytesIO(response.content)))

        return tiles

    def _get_depth_map(
        self, pano: dict[str, Any], dmap_imname: str = ''
    ) -> dict[str, Any]:
        """
        Compute and process the depth map for a panoramic image.

        Parameters
        ----------
        pano
            Panoramic image data dictionary.
        dmap_imname
            Filename to save the depth map image (empty = do not save).

        Returns
        -------
        dict
            Updated pano dictionary with ``depthMap`` and ``depthImFile``.
        """
        depth_map_data = self._parse_dmap_str(pano['depthMapString'])
        header = self._parse_dmap_header(depth_map_data)
        data = self._parse_dmap_planes(header, depth_map_data)
        depth_map = self._compute_dmap(header, data['indices'], data['planes'])

        dmap_array = depth_map['depthMap']
        dmap_array[np.where(dmap_array == np.max(dmap_array))] = 255
        if np.min(dmap_array) < 0:
            dmap_array[np.where(dmap_array < 0)] = 0
        dmap_array = dmap_array.reshape((depth_map['height'], depth_map['width']))

        dmap_array = np.fliplr(dmap_array)

        im_dmap = Image.fromarray(np.uint8(dmap_array))
        im_dmap = im_dmap.resize(pano['panoSize'])
        pano['depthMap'] = im_dmap.copy()

        if dmap_imname:
            im_dmap_save = im_dmap.convert('L')
            im_dmap_save.save(dmap_imname)

        pano['depthImFile'] = dmap_imname

        return pano

    def _get_angle_from_heading(
        self, coord: tuple[float, float], heading: float
    ) -> float:
        """
        Calculate the viewing angle of a coordinate relative to camera heading.

        Parameters
        ----------
        coord
            Cartesian coordinates.
        heading
            Camera heading angle in degrees.

        Returns
        -------
        float
            Calculated viewing angle.
        """
        x_0 = 100 * math.sin(math.radians(heading))
        y_0 = 100 * math.cos(math.radians(heading))

        ang = 360 - self._get_3pt_angle((x_0, y_0), (0, 0), coord)

        return ang if ang <= 180 else ang - 360

    @staticmethod
    def _parse_dmap_str(b64_string: str) -> np.ndarray:
        """
        Parse a base64-encoded depth map string into a numpy byte array.

        Parameters
        ----------
        b64_string
            Base64-encoded depth map string.

        Returns
        -------
        np.ndarray
            Decoded byte data.
        """
        b64_string += '=' * ((4 - len(b64_string) % 4) % 4)
        data = b64_string.replace('-', '+').replace('_', '/')
        data = base64.b64decode(data)
        return np.array(data)

    def _parse_dmap_header(self, depth_map: np.ndarray) -> dict[str, int]:
        """
        Parse the header information from the depth map.

        Parameters
        ----------
        depth_map
            Numpy array containing the depth map data.

        Returns
        -------
        dict
            Parsed header fields: headerSize, numberOfPlanes, width, height,
            offset.
        """
        return {
            'headerSize': depth_map[0],
            'numberOfPlanes': self._get_uint16(depth_map, 1),
            'width': self._get_uint16(depth_map, 3),
            'height': self._get_uint16(depth_map, 5),
            'offset': self._get_uint16(depth_map, 7),
        }

    def _parse_dmap_planes(
        self, header: dict[str, int], depth_map: np.ndarray
    ) -> dict[str, list]:
        """
        Parse the plane information and indices from the depth map.

        Parameters
        ----------
        header
            Parsed header information.
        depth_map
            Numpy array containing the depth map data.

        Returns
        -------
        dict
            Dictionary with ``'planes'`` and ``'indices'`` keys.
        """
        indices = []
        for i in range(header['width'] * header['height']):
            indices.append(depth_map[header['offset'] + i])

        planes = []
        normal = [0.0, 0.0, 0.0]
        for i in range(header['numberOfPlanes']):
            byte_offset = (
                header['offset'] + header['width'] * header['height'] + i * 4 * 4
            )
            normal[0] = self._get_float32(depth_map, byte_offset)
            normal[1] = self._get_float32(depth_map, byte_offset + 4)
            normal[2] = self._get_float32(depth_map, byte_offset + 8)
            dist = self._get_float32(depth_map, byte_offset + 12)
            planes.append({'n': normal.copy(), 'd': dist})

        return {'planes': planes, 'indices': indices}

    @staticmethod
    def _compute_dmap(
        header: dict[str, int], indices: list[int], planes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Compute depth map using provided planes and indices.

        Parameters
        ----------
        header
            Parsed header containing image dimensions.
        indices
            Flat list of plane indices for each pixel.
        planes
            List of plane dicts with ``'n'`` (normal) and ``'d'`` (distance).

        Returns
        -------
        dict
            Dictionary with ``'width'``, ``'height'``, and ``'depthMap'``.
        """
        ray_dir = [0, 0, 0]
        width = header['width']
        height = header['height']

        depth_map = np.empty(width * height)

        for y_coord in range(height):
            theta = (height - y_coord - 0.5) / height * np.pi
            for x_coord in range(width):
                plane_idx = indices[y_coord * width + x_coord]

                phi = (width - x_coord - 0.5) / width * 2 * np.pi + np.pi / 2
                ray_dir[0] = np.sin(theta) * np.cos(phi)
                ray_dir[1] = np.sin(theta) * np.sin(phi)
                ray_dir[2] = np.cos(theta)

                if plane_idx > 0:
                    plane = planes[plane_idx]
                    depth = np.abs(
                        plane['d'](
                            ray_dir[0] * plane['n'][0]
                            + ray_dir[1] * plane['n'][1]
                            + ray_dir[2] * plane['n'][2]
                        )
                    )
                    depth_map[y_coord * width + (width - x_coord - 1)] = depth
                else:
                    depth_map[y_coord * width + (width - x_coord - 1)] = (
                        9999999999999999999.0
                    )
        return {'width': width, 'height': height, 'depthMap': depth_map}

    @staticmethod
    def _get_3pt_angle(
        pt1: tuple[float, float], pt2: tuple[float, float], pt3: tuple[float, float]
    ) -> float:
        """
        Calculate the angle formed by three points.

        Parameters
        ----------
        pt1
            First point.
        pt2
            Vertex point.
        pt3
            Third point.

        Returns
        -------
        float
            Angle in degrees.
        """
        ang = math.degrees(
            math.atan2(pt3[1] - pt2[1], pt3[0] - pt2[0])
            - math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
        )
        return ang + 360 if ang < 0 else ang

    def _get_uint16(self, arr: list[int], ind: int) -> int:
        """
        Combine two bytes into a 16-bit unsigned integer.

        Parameters
        ----------
        arr
            Array of byte values.
        ind
            Starting index of the two bytes.

        Returns
        -------
        int
            16-bit unsigned integer.
        """
        return int(self._get_bin(arr[ind + 1]) + self._get_bin(arr[ind]), 2)

    @staticmethod
    def _get_bin(int_inp: int) -> str:
        """
        Convert an integer to an 8-bit binary string.

        Parameters
        ----------
        int_inp
            The integer to convert.

        Returns
        -------
        str
            8-bit binary string representation.
        """
        binary_str_int = bin(int_inp)[2:]
        return '0' * (8 - len(binary_str_int)) + binary_str_int

    def _get_float32(self, arr: list[int], ind: int) -> float:
        """
        Convert 4 bytes at a given index into a 32-bit float.

        Parameters
        ----------
        arr
            Array of byte values.
        ind
            Starting index of the 4 bytes.

        Returns
        -------
        float
            32-bit floating-point number.
        """
        return self._bin_to_float(
            ''.join(self._get_bin(i) for i in arr[ind : ind + 4][::-1])
        )

    @staticmethod
    def _bin_to_float(binary: str) -> float:
        """
        Convert a 32-bit binary string to a float.

        Parameters
        ----------
        binary
            32-bit binary string.

        Returns
        -------
        float
            Corresponding 32-bit float.
        """
        return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]
