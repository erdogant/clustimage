""" Cluster on lat lon on the metadata:

    This script extracts metadata from image files and returns it as a pandas dataframe.
    It uses the piexif library to extract metadata from the images, and the geopy library to convert GPS coordinates to place names.

"""

import os
import time
from tqdm import tqdm

import pandas as pd
import piexif
from geopy.geocoders import Nominatim
from PIL import Image

import numpy as np
from sklearn.cluster import DBSCAN

import folium
from folium.plugins import MarkerCluster

from PIL import Image
from io import BytesIO
import base64

from folium import FeatureGroup
import webbrowser

from datetime import datetime as dt

tqdm.pandas()

# allows Pillow to open and manipulate images in the HEIF (i.e. HEIC) format
from pillow_heif import register_heif_opener
register_heif_opener()


# %%
def extract_metadata(dir_path, black_list=None, ext=["jpg", "jpeg", "png", "tiff", "bmp", "gif", "webp", "psd", "raw", "cr2", "nef", "heic", "sr2"]):
    """
    Extracts the lat/lon and other metadata from images.

    """
    from clustimage.clustimage import listdir
    metadata_list = []
    filepaths = listdir(dir_path, ext=ext, black_list=black_list)

    for file in filepaths:
        if np.isin(file.lower().split('.')[-1], ext):
            print("[+] Extracting metadata from {}".format(file))

            try:
                # Open the image
                img = Image.open(file)

                # Get exif data from image
                exif_data = piexif.load(img.info["exif"])
                print("[+] Extract metadata: {}".format(file))

                # Extract GPS latitude, longitude, and altitude data
                gps_latitude = exif_data['GPS'][piexif.GPSIFD.GPSLatitude]
                gps_latitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLatitudeRef]
                gps_longitude = exif_data['GPS'][piexif.GPSIFD.GPSLongitude]
                gps_longitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLongitudeRef]
                gps_altitude = exif_data['GPS'][piexif.GPSIFD.GPSAltitude]
                gps_altitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSAltitudeRef]

                # Convert GPS latitude and longitude data to decimal degrees
                gps_latitude_decimal = gps_to_decimal(gps_latitude, gps_latitude_ref)
                gps_longitude_decimal = gps_to_decimal(gps_longitude, gps_longitude_ref)

                metadata = {
                    'filename': os.path.split(file)[-1],
                    'image_path': file,
                    'lat': gps_latitude_decimal,
                    'lon': gps_longitude_decimal,
                    'altitude': gps_altitude,
                    'gps_altitude_ref': gps_altitude_ref,
                    'make': exif_data['0th'][piexif.ImageIFD.Make].decode('utf-8'),
                    'device': exif_data['0th'][piexif.ImageIFD.Model].decode('utf-8'),
                    'software': exif_data['0th'][piexif.ImageIFD.Software].decode('utf-8'),
                    'datetime': exif_data['0th'][piexif.ImageIFD.DateTime].decode('utf-8'),
                    'exif_location': '',
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
                    # 'date_time_original': exif_data['Exif'][piexif.ExifIFD.DateTimeOriginal],
                    # 'date_time_digitized': exif_data['Exif'][piexif.ExifIFD.DateTimeDigitized],
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

                metadata_list.append(metadata)
            except Exception as e:
                print("[!] No exif information could be retrieved from{}: {}".format(file, str(e)))

    # Convert the metadata list to a pandas dataframe
    metadata_df = pd.DataFrame(metadata_list)

    return metadata_df

# %%
def gps_to_decimal(coord, ref):
    """Converts GPS coordinates to decimal degrees.

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


def get_location(lat, lon):
    """Get location information from lat-lon coordinates.
    Nominatim imposes rate limits on the number of requests that can be made. 
    For Nominatim, the rate limit is usually 1 request per second.

    """
    time.sleep(1.1)
    location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
    if location is None:
        return None
    else:
        return location.address


def cluster_by_location(metadata_df, radius_meters=500):
    """Clusters images based on GPS coordinates within a given radius.

    Parameters:
    - metadata_df: DataFrame containing at least 'filename', 'lat', and 'lon' columns.
    - radius_meters: The clustering radius in meters.

    Returns:
    - cluster_labels: A vector of cluster labels corresponding to each image.
    - map_display: A folium map displaying the clusters.
    """
    # Extract coordinates
    coordinates = metadata_df[['lat', 'lon']].values

    # Convert radius from meters to kilometers (DBSCAN uses km for Haversine distances)
    radius_km = radius_meters / 1000

    # Perform DBSCAN clustering using haversine distance. DBSCAN requires the input coordinates in radians for haversine metric
    coords_radians = np.radians(coordinates)
    db = DBSCAN(eps=radius_km / 6371, min_samples=1, metric='haversine')
    cluster_labels = db.fit_predict(coords_radians)

    # Return
    return cluster_labels


# Find the indices of consecutive 1s separated by 0s
def cluster_by_time(df_datetime, timeframe='4H', min_clust=2, window_length=5):
    # datetime=metadata_df['datetime'].copy()
    from scipy.signal import savgol_filter

    # Step 1: Convert datetime column from string to datetime
    df_datetime = pd.DataFrame(pd.to_datetime(df_datetime, format='%Y:%m:%d %H:%M:%S'))
    df_datetime = df_datetime.sort_values(by='datetime')#.reset_index(drop=True)

    # Create a new column for the hour
    df_datetime['hour'] = df_datetime['datetime'].dt.floor(timeframe)  # Rounds down to the nearest hour

    # Count the number of events per hour
    events_per_hour = df_datetime.groupby('hour').size().reset_index(name='event_count')

    # Step 2: Set datetime as index and resample the data by hour, counting the photos
    datetime_proc = df_datetime.set_index('hour')
    # Resample on time
    metadata_df_hourly = datetime_proc.resample(timeframe).size()  # Count number of photos in each hour

    # Step 3: Apply a 2nd-degree polynomial fit for smoothing (Savitzky-Golay filter)
    window_length = np.minimum(metadata_df_hourly.shape[0], window_length)
    metadata_df_hourly_smoothed = savgol_filter(metadata_df_hourly, window_length=window_length, polyorder=2)
    metadata_df_hourly_smoothed[metadata_df_hourly_smoothed < 0] = 0

    groups = []
    current_group = []
    for i, val in enumerate(metadata_df_hourly_smoothed):
        if val > 0:
            current_group.append(i)
        elif current_group:  # If we encounter a 0 and the current cluster is not empty
            groups.append(current_group)
            current_group = []
    # Add the last cluster if it exists
    if current_group:
        groups.append(current_group)

    clusters = np.zeros(len(df_datetime.values))
    df_datetime = df_datetime.sort_index()
    counter = 1

    for group in groups:
        timestart = metadata_df_hourly.index[group].min()
        timestop = metadata_df_hourly.index[group].max()
        # datetime_in_range = metadata_df_hourly.loc[timestart:timestop].index
        loc = (df_datetime['datetime'] >= timestart) & (df_datetime['datetime'] <= timestop)

        if sum(loc) >= min_clust:
            clusters[loc] = counter
            counter = counter + 1

    return clusters.astype(int)


def plot_map(metadata_df, clusterlabels, cluster_icons=True, thumbnail_size=300, polygon=True, blacklist=[0]):
    # Assign cluster colors and labels
    metadata_df['cluster_color'] = get_colors(clusterlabels)
    metadata_df['cluster'] = clusterlabels

    # Create a new folium map
    map_display = folium.Map(location=[metadata_df['lat'].mean(), metadata_df['lon'].mean()], tiles='OpenStreetMap', zoom_start=10)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     name='ESRI Satellite',
                     attr='Tiles © Esri — GIS User Community').add_to(map_display)

    # Create MarkerCluster which will automatically group points that are close together
    if cluster_icons:
        combined_group = MarkerCluster().add_to(map_display)

    cluster_groups = {}
    uicluster_id = np.array(sorted(metadata_df['cluster'].unique()))

    # Iterate through clusters to create separate feature groups
    for cluster_id in tqdm(uicluster_id):
        cluster_data = metadata_df[metadata_df['cluster'] == cluster_id]
        cluster_data = cluster_data.sort_values(by='datetime').reset_index(drop=True)

        # Get datetime range from the photos.
        dt_range = create_datetime_string(cluster_data['datetime'])

        # Create a combined feature group for markers and polygons
        if not cluster_icons:
            combined_group = FeatureGroup(name=f"Cluster {cluster_id} - {dt_range}")
        cluster_groups[cluster_id] = combined_group

        # Add markers to the feature group
        for _, row in cluster_data.iterrows():
            dt_obj = dt.strptime(row['datetime'], '%Y:%m:%d %H:%M:%S')
            month_year = dt_obj.strftime('%B') + ' ' + str(dt_obj.year)

            # Generate thumbnail HTML
            thumbnail_base64 = ''
            if thumbnail_size is not None and thumbnail_size > 10:
                thumbnail_base64 = create_thumbnail(row['image_path'], max_size=thumbnail_size)

            popup_content = f"""
                <div>
                    <p>
                        Cluster: {row['cluster']} <br>
                        Filename: {row['filename']} <br>
                        Location: {row['exif_location']} <br>
                        Date/Time: {month_year} <br>
                    </p>
                    {thumbnail_base64}
                </div>
            """
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=popup_content,
                icon=folium.Icon(color=row['cluster_color']),
            ).add_to(combined_group)

        # Add Polygon
        if polygon and (len(cluster_data) > 2) and (not np.isin(cluster_id, blacklist)):
            # Add polygons for the cluster (if applicable)
            points = cluster_data[['lat', 'lon']].values.tolist()
            folium.Polygon(
                locations=points,
                color=row['cluster_color'],
                fill=False,         # Fill the polygon to make it visible
                fill_opacity=0.3,   # Adjust fill opacity
                weight=2,           # Polygon border thickness
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


def create_datetime_string(dt_strings):
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


# Function to create a thumbnail and encode it in base64
def create_thumbnail(image_path, max_size=300):
    # max_size=(300, 170)
    # None: Automatically scale based on existing size
    # 300: scale heigth automatically

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


def get_colors(labels):
    # Predefined list of colors for the clusters
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'gray',
        'pink', 'darkgreen', 'lightgray', 'darkblue', 'lightblue', 'darkred',
        'darkpurple', 'lightred', 'cadetblue', 'black',
        ] # 'lightgreen', 'beige'

    # Get the unique cluster labels
    cluster_labels = labels.unique()

    # If there are more cluster labels than available colors, cycle through colors
    if len(cluster_labels) > len(colors):
        color_mapping = {label: colors[i % len(colors)] for i, label in enumerate(cluster_labels)}
    else:
        # Otherwise, assign a unique color to each cluster label
        color_mapping = {label: colors[i] for i, label in enumerate(cluster_labels)}

    color_mapping = list(map(lambda x: color_mapping.get(x), labels.values))
    return color_mapping


#%%
if __name__ == "__main__":
    # Define the directory containing the image files
    dir_path = r"\\NAS_SYNOLOGY\Photo\2023\vliegclub"

    # Extract the metadata from the image files
    metadata_df = extract_metadata(dir_path)

    # Get location information
    geolocator = Nominatim(user_agent="exif_location")
    metadata_df['exif_location'] = metadata_df.progress_apply(lambda row: get_location(row['lat'], row['lon']), axis=1)

    # Get clusters
    metadata_df['cluster_location'] = cluster_by_location(metadata_df, radius_meters=1000)
    metadata_df['cluster_datetime'] = cluster_by_time(metadata_df['datetime'], timeframe='4H', min_clust=3)

    # Plot on map
    map_display = plot_map(metadata_df, metadata_df['cluster_location'], thumbnail_size=300, polygon=False, cluster_icons=True)
    map_display.save(os.path.join(dir_path, "map_location.html"))
    webbrowser.open(os.path.join(dir_path, "map_location.html"))

    map_display = plot_map(metadata_df, metadata_df['cluster_datetime'], thumbnail_size=300, polygon=True, cluster_icons=False)
    map_display.save(os.path.join(dir_path, "map_time.html"))
    webbrowser.open(os.path.join(dir_path, "map_time.html"))
