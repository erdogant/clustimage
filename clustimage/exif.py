"""Python package clustimage is for unsupervised clustering of images."""
# --------------------------------------------------
# Name        : plot_map.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/clustimage
# Licence     : See licences
# --------------------------------------------------

from tqdm import tqdm

from PIL import Image
from io import BytesIO
import base64

import numpy as np
import os
import time
from datetime import datetime as dt


# %% Extract Metadata using Exif information from the photos
def extract_from_image(pathnames, ext=["jpg", "jpeg", "png", "tiff", "bmp", "gif", "webp", "psd", "raw", "cr2", "nef", "heic", "sr2"], logger=None):
    """Extract latitude, longitude, and other metadata from a list of image files.

    This function processes image files to extract GPS coordinates, device information,
    and other metadata from their EXIF data. Supported file formats can be customized
    via the `ext` parameter.

    Parameters
    ----------
    pathnames : list of str
        A list of file paths to the images.
    ext : list of str, optional
        A list of file extensions to filter the images to be processed.
        common image formats: `["jpg", "jpeg", "png", "tiff", "bmp", "gif", "webp", "psd", "raw", "cr2", "nef", "heic", "sr2"]`.

    Returns
    -------
    metadata_df : pandas.DataFrame
        A DataFrame containing the extracted metadata for each image. Columns include:
        - `filenames` : str, the name of the image file.
        - `pathnames` : str, the full path to the image file.
        - `lat` : float, the latitude in decimal degrees.
        - `lon` : float, the longitude in decimal degrees.
        - `altitude` : float, the altitude in meters.
        - `gps_altitude_ref` : int, reference for altitude (0 for above sea level, 1 for below).
        - `make` : str, the camera manufacturer.
        - `device` : str, the camera model.
        - `software` : str, the software used to process the image.
        - `datetime` : str, the date and time when the image was last modified.
        - `date_time_original` : str, the original date and time the image was taken.
        - `location` : str, a placeholder for location name (default is empty).

    Raises
    ------
    ImportError
        If the `piexif` library is not installed.
    Exception
        For any issues during the processing of individual image files.

    Notes
    -----
    - Requires the `piexif` library for extracting EXIF metadata.
    - Images without GPS or EXIF data will be skipped.
    - The `gps_to_decimal` helper function is used to convert GPS coordinates to decimal degrees.

    Examples
    --------
    Extract metadata from a list of image files:

    >>> pathnames = ["image1.jpg", "image2.jpeg", "image3.png"]
    >>> metadata_df = extract_metadata(pathnames)
    >>> print(metadata_df)
          filename      lat       lon  altitude  ...
    0  image1.jpg  52.3676   4.9041     15.0    ...
    1  image2.jpeg 51.9244   4.4777     10.0    ...
    2  image3.png  48.8566   2.3522     35.0    ...

    """
    try:
        import piexif
    except ImportError:
        raise ImportError(
            "The 'piexif' library is not installed. Please install it using the following command:\n"
            "pip install piexif")

    metadata_list = []
    for pathname in tqdm(pathnames, disable=disable_tqdm(logger), desc='[clustimage]'):
        # Check extension
        file_ext = pathname.lower().split('.')[-1]
        if np.isin(file_ext, ext):
            # Extract filename
            filename = os.path.split(pathname)[-1]
            # Open the image
            img = Image.open(pathname)

            # Get exif data from image
            try:
                exif_data = piexif.load(img.info["exif"])
            except:
                exif_data = {}

            # Extract GPS latitude, longitude, and altitude data
            try:
                gps_latitude = exif_data['GPS'][piexif.GPSIFD.GPSLatitude]
                gps_latitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLatitudeRef]
                gps_longitude = exif_data['GPS'][piexif.GPSIFD.GPSLongitude]
                gps_longitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLongitudeRef]
                gps_altitude = exif_data['GPS'][piexif.GPSIFD.GPSAltitude]
                gps_altitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSAltitudeRef]
                # Convert GPS latitude and longitude data to decimal degrees
                gps_latitude_decimal = gps_to_decimal(gps_latitude, gps_latitude_ref)
                gps_longitude_decimal = gps_to_decimal(gps_longitude, gps_longitude_ref)
            except:
                gps_altitude_ref, gps_altitude, gps_latitude_decimal, gps_longitude_decimal = None, None, None, None

            try:
                make = exif_data['0th'][piexif.ImageIFD.Make].decode('utf-8')
                device = exif_data['0th'][piexif.ImageIFD.Model].decode('utf-8')
                software = exif_data['0th'][piexif.ImageIFD.Software].decode('utf-8')
            except:
                make, device, software = None, None, None

            try:
                datetime = exif_data['0th'][piexif.ImageIFD.DateTime].decode('utf-8')
                date_time_original = exif_data['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
            except:
                # Get the file's last modification time (or creation time)
                timestamp = os.path.getmtime(pathname)  # or use getctime() for creation time
                # Convert the timestamp to a readable datetime format
                datetime = dt.fromtimestamp(timestamp).strftime('%Y:%m:%d %H:%M:%S')
                date_time_original = None

            # Create metadata for the photo
            metadata = {
                'filenames': filename,
                'pathnames': pathname,
                'datetime': datetime,
                'date_time_original': date_time_original,
                'exif_location': '',
                'lat': gps_latitude_decimal,
                'lon': gps_longitude_decimal,
                'altitude': gps_altitude,
                'gps_altitude_ref': gps_altitude_ref,
                'make': make,
                'device': device,
                'software': software,

                # 'exposure_time': exif_data['Exif'][piexif.ExifIFD.ExposureTime],
                # 'f_number': exif_data['Exif'][piexif.ExifIFD.FNumber],
                # 'iso_speed_ratings': exif_data['Exif'][piexif.ExifIFD.ISOSpeedRatings],
                # 'focal_length': exif_data['Exif'][piexif.ExifIFD.FocalLength],
                # 'focal_length_in_35mm_film': exif_data['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm],
                # 'exposure_mode': exif_data['Exif'][piexif.ExifIFD.ExposureMode],
                # 'white_balance': exif_data['Exif'][piexif.ExifIFD.WhiteBalance],
                # 'metering_mode': exif_data['Exif'][piexif.ExifIFD.MeteringMode],
                # 'flash': exif_data['Exif'][piexif.ExifIFD.Flash],
                # 'exposure_program': exif_data['Exif'][piexif.ExifIFD.ExposureProgram],
                # 'exif_version': exif_data['Exif'][piexif.ExifIFD.ExifVersion],
                # 'date_time_digitized': exif_data['Exif'][piexif.ExifIFD.DateTimeDigitized].decode('utf-8'),
                # 'components_configuration': exif_data['Exif'][piexif.ExifIFD.ComponentsConfiguration],
                # 'compressed_bits_per_pixel': exif_data['Exif'][piexif.ExifIFD.CompressedBitsPerPixel],
                # 'shutter_speed_value': exif_data['Exif'][piexif.ExifIFD.ShutterSpeedValue],
                # 'aperture_value': exif_data['Exif'][piexif.ExifIFD.ApertureValue],
                # 'brightness_value': exif_data['Exif'][piexif.ExifIFD.BrightnessValue],
                # 'exposure_bias_value': exif_data['Exif'][piexif.ExifIFD.ExposureBiasValue],
                # 'max_aperture_value': exif_data['Exif'][piexif.ExifIFD.MaxApertureValue],
                # 'subject_distance': exif_data['Exif'][piexif.ExifIFD.SubjectDistance],
                # 'metering_mode': exif_data['Exif'][piexif.ExifIFD.MeteringMode],
                # 'light_source': exif_data['Exif'][piexif.ExifIFD.LightSource],
                # 'flash': exif_data['Exif'][piexif.ExifIFD.Flash],
                # 'focal_length': exif_data['Exif'][piexif.ExifIFD.FocalLength],
                # 'subject_area': exif_data['Exif'][piexif.ExifIFD.SubjectArea],
            }

            # Append to list
            metadata_list.append(metadata)

    # Convert the metadata list to a pandas dataframe
    logger.info(f'exif information is retrieved from {len(metadata)} out of {len(pathnames)} files')
    # Return
    return metadata_list


def gps_to_decimal(coord, ref):
    """Convert GPS coordinates to decimal degrees.

    Args:
        coord (tuple): A tuple containing the GPS coordinates.
        ref (str): The reference direction (e.g., N, S, E, W).

    Returns:
        float: The GPS coordinates in decimal degrees.

    """
    decimal = coord[0][0] / coord[0][1] + coord[1][0] / \
        (60 * coord[1][1]) + coord[2][0] / (3600 * coord[2][1])
    if ref in ['S', 'W']:
        decimal *= -1
    return decimal


def location(Xfeat, logger):
    """Get location information from lat-lon coordinates.

    Nominatim imposes rate limits on the number of requests that can be made.
    For Nominatim, the rate limit is usually 1 request per second.

    """
    # Get location information
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        raise ImportError(
            "The 'geopy' library is not installed. Please install it using the following command:\n"
            "pip install geopy")

    # Get the exif location
    Xfeat_new = []
    logger.info('Extraction location information from lat/lon coordinates.')
    # Initialize
    geolocator = Nominatim(user_agent="exif_location")

    # Loop over all photos
    for row in tqdm(Xfeat, disable=disable_tqdm(logger), desc='[clustimage]'):
        # Sleep to prevent time-out requests
        time.sleep(1.1)
        # Get location using lat/lon
        location = geolocator.reverse(f"{row['lat']}, {row['lon']}", exactly_one=True)
        # Store
        if location is None:
            row['exif_location'] = ''
        else:
            row['exif_location'] = location.address
        # Append to new list
        Xfeat_new.append(row)

    # Return
    return Xfeat_new


#%%
def plot_map(metadata_df, clusterlabels, metric, cluster_icons=True, polygon=True, thumbnail_size=400, blacklist=[0]):
    """Plots a map using Folium to visualize clusters, with options to add markers and polygons for each cluster.

    Parameters
    ----------
    metadata_df : pandas.DataFrame
        A DataFrame containing metadata for the images. Must include the following columns:
        - 'lat': Latitude of each point.
        - 'lon': Longitude of each point.
        - 'datetime': Timestamp in the format '%Y:%m:%d %H:%M:%S'.
        - 'filenames': Image filenames.
        - 'pathnames': Full file paths for generating thumbnails.
        - 'clusterlabels': Cluster labels for each point.
        - 'cluster_color': Colors assigned to each cluster.
    clusterlabels : array-like
        Cluster labels for the points in `metadata_df`.
    metric : str
        - 'location'
        - 'datetime'
    cluster_icons : bool, optional
        Whether to add individual markers to a MarkerCluster. Default is True.
    polygon : bool, optional
        Whether to draw polygons around clusters. Default is True.
    thumbnail_size : int, optional
        Size of the thumbnail images to display in the marker popups. Set to None or <=10 to disable thumbnails. Default is 300.
    blacklist : list, optional
        List of cluster IDs for which polygons should not be drawn. Default is [0].

    Returns
    -------
    folium.Map
        A Folium map object displaying the clusters with markers and optional polygons.

    Example
    -------
    >>> import pandas as pd
    >>> from your_module import plot_map

    # Sample metadata DataFrame
    data = {
        'lat': [52.3676, 52.3680, 52.3690],
        'lon': [4.9041, 4.9050, 4.9060],
        'datetime': ['2023:09:30 10:12:37', '2023:09:30 11:15:20', '2023:09:30 12:45:00'],
        'filenames': ['IMG_8749.JPG', 'IMG_3750.JPG', 'IMG_3751.JPG'],
        'pathnames': ['/path/to/IMG_8749.JPG', '/path/to/IMG_3750.JPG', '/path/to/IMG_3751.JPG'],
        'clusterlabels': [1, 1, 2],
        'cluster_color': ['blue', 'blue', 'red']
    }
    metadata_df = pd.DataFrame(data)

    # Cluster labels
    clusterlabels = metadata_df['clusterlabels'].values

    # Plot the map
    map_display = plot_map(metadata_df, clusterlabels, metric='euclidean')
    map_display.save("map.html")  # Save the map to an HTML file

    Notes
    -----
    - Ensure the `metadata_df` includes all required columns.
    - The function relies on external helper functions, such as `get_colors` for assigning cluster colors,
      `create_thumbnail` for generating image thumbnails, and `set_params` for setting default parameters.
    - The map includes a full-screen button and layer control for better visualization.

    """
    # Get location information
    try:
        from geopy.distance import geodesic
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        raise ImportError(
            "The 'folium' library is not installed. Please install it using the following command:\n"
            "pip install folium")

    # Set parameters is None
    cluster_icons, polygon = set_params(cluster_icons, polygon, metric)

    # Add cluster labels
    metadata_df['clusterlabels'] = clusterlabels

    # Remove all images without lat/lon
    metadata_df = metadata_df.dropna(subset=['lat', 'lon'], how='all')

    # Assign cluster colors and labels
    metadata_df['cluster_color'] = get_colors(metadata_df['clusterlabels'])

    # Create a new folium map
    map_display = folium.Map(location=[metadata_df['lat'].mean(), metadata_df['lon'].mean()], tiles='OpenStreetMap', zoom_start=10)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     name='ESRI Satellite',
                     attr='Tiles © Esri — GIS User Community').add_to(map_display)

    # Create MarkerCluster which will automatically group points that are close together
    if cluster_icons:
        combined_group = MarkerCluster().add_to(map_display)

    cluster_groups = {}
    uicluster_id = np.array(sorted(metadata_df['clusterlabels'].unique()))

    # Iterate through clusters to create separate feature groups
    for cluster_id in tqdm(uicluster_id):
        # Get subset of data
        cluster_data = metadata_df[metadata_df['clusterlabels'] == cluster_id]
        # Sort on datetime for the correct polygons
        cluster_data = cluster_data.sort_values(by='datetime').reset_index(drop=True)

        # Get datetime range from the photos.
        dt_range = create_datetime_string(cluster_data['datetime'])

        # Create a combined feature group for markers and polygons
        if not cluster_icons: combined_group = folium.FeatureGroup(name=f"Cluster {int(cluster_id)} - {dt_range}")
        cluster_groups[cluster_id] = combined_group

        # Add markers to the feature group
        for _, row in cluster_data.iterrows():
            dt_obj = dt.strptime(row['datetime'], '%Y:%m:%d %H:%M:%S')
            month_year = str(dt_obj.year) + ' ' + dt_obj.strftime('%B') + ' - ' + str(dt_obj.hour) + ':' + str(dt_obj.minute)
            dirname = os.path.basename(os.path.split(row['pathnames'])[0])

            # Generate thumbnail HTML
            thumbnail_base64 = ''
            if thumbnail_size is not None and thumbnail_size > 10:
                thumbnail_base64 = create_thumbnail(row['pathnames'], max_size=thumbnail_size)

            # Compute the total polygon length in km
            # total_distance = 0.0
            # points = cluster_data[['lat', 'lon']].values.tolist()
            # for i in range(len(points) - 1):
            #     total_distance += geodesic(points[i], points[i + 1]).km

            popup_content = f"""
                <div>
                    <p>
                        Cluster: {int(row['clusterlabels'])}, n={cluster_data.shape[0]} <br>
                        Filename: {row['filenames']} <br>
                        Dirname: {dirname} <br>
                        Date/Time: {month_year} <br>
                        Location: {row['exif_location']} <br>
                    </p>
                    {thumbnail_base64}
                </div>
            """
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=popup_content,
                tooltip=popup_content,
                icon=folium.Icon(color=row['cluster_color']),
            ).add_to(combined_group)

        # Add Polygon
        if polygon and (cluster_data.shape[0] > 2) and (not np.isin(cluster_id, blacklist)):
            # Add polygons for the cluster (if applicable)
            points = cluster_data[['lat', 'lon']].values.tolist()

            # Calculate polygon length
            total_distance = 0.0
            for i in range(len(points) - 1):
                total_distance += geodesic(points[i], points[i + 1]).km

            # Format distance for tooltip
            distance_tooltip = f"Total Distance: {total_distance:.1f} km"

            folium.Polygon(
                locations=points,
                color=row['cluster_color'],
                fill=False,         # Fill the polygon to make it visible
                fill_opacity=0.3,   # Adjust fill opacity
                weight=3,           # Polygon border thickness
                tooltip=distance_tooltip,
            ).add_to(combined_group)

        # Add the combined group (both markers and polygons) to the map
        combined_group.add_to(map_display)

    # Add layer control to toggle the combined clusters (markers + polygons)
    folium.LayerControl(collapsed=False).add_to(map_display)
    # Add the full screen button.
    folium.plugins.Fullscreen(
        position="topright",
        title="Open full-screen map",
        title_cancel="Close full-screen map",
        force_separate_button=True).add_to(map_display)

    return map_display


# %%
def set_params(cluster_icons, polygon, metric):
    # Automatically set the best parameters
    if cluster_icons is None and metric=='datetime':
        cluster_icons=False
    elif cluster_icons is None and metric=='location':
        cluster_icons=True

    if polygon is None and metric=='datetime':
        polygon=True
    elif polygon is None and metric=='location':
        polygon=False

    return cluster_icons, polygon


# %%
def create_datetime_string(dt_strings):
    """Creates a string representing the range of dates based on a list of datetime strings.

    Parameters
    ----------
    dt_strings : list of str
        A list of datetime strings in the format '%Y:%m:%d %H:%M:%S'.

    Returns
    -------
    str
        A string representing the range of dates. If the minimum and maximum dates fall within 
        the same month and year, the output will be formatted as "Month Year". Otherwise, it 
        will be formatted as "StartMonth StartYear - EndMonth EndYear".

    Example
    -------
    >>> from your_module import create_datetime_string
    >>> dt_strings = [
    ...     '2023:09:30 10:12:37',
    ...     '2023:09:30 11:15:20',
    ...     '2023:10:01 12:45:00'
    ... ]
    >>> result = create_datetime_string(dt_strings)
    >>> print(result)
    'September 2023 - October 2023'

    Notes
    -----
    - The function assumes that all input strings are valid and follow the '%Y:%m:%d %H:%M:%S' format.

    - It uses Python's `datetime` module to parse and handle datetime objects.
    """
    # Convert to datetime objects
    dt_obj = [dt.strptime(x, '%Y:%m:%d %H:%M:%S') for x in dt_strings]

    # Get the minimum and maximum datetime
    min_date = min(dt_obj)
    max_date = max(dt_obj)

    # Extract month names and years
    start_month = min_date.strftime('%B')
    start_year = min_date.year
    end_month = max_date.strftime('%B')
    end_year = max_date.year

    # Format the output
    if start_month==end_month and start_year==end_year:
        date_range_string = f"{start_month} {start_year}"
    else:
        date_range_string = f"{start_month} {start_year} - {end_month} {end_year}"

    return date_range_string


# %% Function to create a thumbnail and encode it in base64
def create_thumbnail(image_path, max_size=300):
    """Creates a thumbnail image from the provided image path and returns an HTML string containing the base64-encoded image.

    Parameters
    ----------
    image_path : str
        Path to the image file to create a thumbnail for.
    max_size : int or tuple of int, optional
        The maximum size of the thumbnail. If an integer is provided, it is treated as 
        the width, and the height is scaled proportionally. If a tuple (width, height) 
        is provided, it specifies the exact dimensions. Defaults to 300.

    Returns
    -------
    str
        An HTML string containing the base64-encoded thumbnail image. If the thumbnail 
        cannot be created, the string "Thumbnail not available" is returned.

    Example
    -------
    >>> from your_module import create_thumbnail
    >>> image_path = "example_image.jpg"
    >>> html_thumbnail = create_thumbnail(image_path, max_size=150)
    >>> print(html_thumbnail)
    '<img src="data:image/jpeg;base64,...encoded_image..." width="150" height="100">'

    Notes
    -----
    - The function maintains the aspect ratio of the original image when resizing.
    - If `max_size` is `None`, the default size is set to 300 pixels wide, with the 
      height automatically calculated based on the aspect ratio.
    - If an error occurs while creating the thumbnail (e.g., invalid image path), 
      "Thumbnail not available" is returned.
    - The function uses `Pillow` for image processing and `base64` for encoding.

    Dependencies
    ------------
    - Pillow (PIL)
    - base64
    - BytesIO from io
    """

    try:
        with Image.open(image_path) as img:
            # Automatically set the thumbnail size if max_size is None or an int
            if max_size is None:
                max_size = [300, np.round(300 * img.size[1] / img.size[0], 0)]
            if isinstance(max_size, int):
                max_size = [max_size, np.round(max_size * (img.size[1] / img.size[0]), 0)]

            # Resize the image while maintaining the aspect ratio
            img = img.resize((int(max_size[0]), int(max_size[1])), Image.Resampling.LANCZOS)

            # Set the thumbnail size
            img.thumbnail(max_size)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Return
            return f'<img src="data:image/jpeg;base64,{encoded}" width="{max_size[0]}" height="{max_size[1]}">'

    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return "Thumbnail not available"


# %%
def get_colors(clusterlabels):
    # Predefined list of colors for the clusters
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'gray',
        'pink', 'darkgreen', 'lightgray', 'darkblue', 'lightblue', 'darkred',
        'cadetblue', 'black'] # 'lightgreen', 'beige'

    # Get the unique cluster labels
    cluster_labels = np.unique(clusterlabels)

    # If there are more cluster labels than available colors, cycle through colors
    if len(cluster_labels) > len(colors):
        color_mapping = {label: colors[i % len(colors)] for i, label in enumerate(cluster_labels)}
    else:
        # Otherwise, assign a unique color to each cluster label
        color_mapping = {label: colors[i] for i, label in enumerate(cluster_labels)}

    color_mapping = list(map(lambda x: color_mapping.get(x), clusterlabels))
    # Return
    return color_mapping


# %%
def disable_tqdm(logger):
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)