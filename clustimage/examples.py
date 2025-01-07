# %%
# import clustimage
# print(dir(clustimage))
# print(clustimage.__version__)

# %%
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
# from scatterd import scatterd

# # Load dataset
# # data = pd.read_csv("your_dataset.csv")
# data = cl.results['feat'][['lat','lon','datetime']]

# # Drop rows with NaN in lat and lon
# data = data.dropna(subset=['lat', 'lon'])

# # Transform datetime to useful features
# data['datetime'] = pd.to_datetime(data['datetime'], format='%Y:%m:%d %H:%M:%S')
# data['hour'] = data['datetime'].dt.hour
# data['day_of_week'] = data['datetime'].dt.dayofweek
# data['year'] = data['datetime'].dt.year
# data['timestamp'] = data['datetime'].astype('int64') / 1e9  # Convert to seconds since epoch

# # Select features for clustering
# features = data[['lat', 'lon', 'timestamp']]  # Use 'hour' or 'timestamp' depending on granularity
# # features = data[['lat', 'lon', 'hour', 'day_of_week', 'year']]  # Use 'hour' or 'timestamp' depending on granularity
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# # DBSCAN clustering
# db = DBSCAN(eps=0.5, min_samples=5).fit(scaled_features)
# data['cluster'] = db.labels_
# cl.results['labels'] = db.labels_

# # Visualize clusters (latitude vs longitude)
# scatterd(x=scaled_features[:,0], y=scaled_features[:,2], c=data['cluster'], cmap='viridis')
# # plt.scatter(data['lat'], data['lon'], c=data['cluster'], cmap='viridis')
# # plt.xlabel('Latitude')
# # plt.ylabel('Longitude')
# # plt.title('Clusters')
# # plt.colorbar(label='Cluster ID')
# # plt.show()


#%% Workflow to clean your [personal] photo files
# Suppose you have photos downloaded from whatsapp, your iphone and combined with the screenshots and selfies you have.
# In addition, friends also took pictures, burst and shard them with your.
# All imges are now in one folder, many are very similar and it has become a time consuming task to sort, remove and figure out which photos belong to the same event.
# Luckily We can automate many of these tasks with the clustimage and undouble library.
#
# With the clustimage library we can group photos on date/time or on location. In my personal experience, clustering photos on date/time works the best because
# when traveling, the location changes all the time. In addition, it also occurs that I come to the same place multiple times a year but these photos belong to different events.
#
# -------
# Step 1: use clustimage and set the timeframe to 6 hours to make capture all events of one day but prevent grouping with the other day.
# -------

# Import library
from clustimage import Clustimage
import os

# Working directory
# dir_path = r'\\NAS_SYNOLOGY\Photo\2024\Various'
dir_path = r'd://temp/'
# When using method is EXIF and metric is datetime, extentions such as .mp4, .txt etc can also be clustered.
allowed_ext = ["mov", "mp4", "jpg", "jpeg", "png", "tiff", "bmp", "gif", "webp", "psd", "raw", "cr2", "nef", "heic", "sr2", "tif"]

# Initialize for datetime.
cl = Clustimage(method='exif',
                params_exif = {'timeframe': 6, 'radius_meters': 1000, 'min_samples': 2, 'exif_location': False},
                ext=allowed_ext,
                verbose='info')

# Run the model to find the clusters based on datetime method. Use metric='latlon' in case location is more important than time.
results = cl.fit_transform(dir_path, metric='datetime', black_list=['undouble'], recursive=True)

# Show the cluster labels
print(cl.results['labels'])

# Show filenames from cluster 1
cl.results['pathnames'][cl.results['labels']==5]
# Plot only files in cluster 1
cl.plot(labels=1, blacklist=[-2, -1], min_samples=3, invert_colors=True)

# -------
# Step 2: Use the plot function to determine what event the cluster of photos represents.
# -------
# Make plot but exclude cluster 0, and only show when there are 4 or more photos in the group.
cl.plot(blacklist=[-2, -1], min_samples=3, invert_colors=True)

# -------
# Step 3: Visualize photos on on map
# -------
# Now we have map where the photos are grouped together in clustere that we can visually inspect.
cl.plot_map(cluster_icons=False, open_in_browser=True, thumbnail_size=400, polygon=True, save_path=os.path.join(dir_path, 'map_latlon.html'))

# -------
# Step 4: We can now easily re-organize our disk using the move functionality.
# -------
# We need to create a dictionary where we can define for each cluster number a subfolder name like this:
# The first column is the cluster label and the second string is the destinated subfolder name. All files in the cluster will be moved to the subfolder.

target_labels = {
    0: 'group 1',
    -1: 'Rest groep',
}

# Run the script to physically move the photos to the specified directories using the cluster labels.
cl.move_to_dir(target_labels=target_labels, savedir=dir_path, user_input=False)

# -------
# Step 6: Undouble.
# -------
# At this point we still may have many photos that are similar or are part of a photo-burst.
# We can undouble the photos using the undouble library.

# Load library
from undouble import Undouble

# Init with default settings
model = Undouble(grayscale=False, method='phash', hash_size=8, ext=allowed_ext)

# Import the re-structured data-folder
model.import_data(dir_path)

# Compute image-hash to determine which photos are very alike.
model.compute_hash()

# Find images with image-hash <= threshold. When using threshold=0, only exactly the same photos are detected.
model.group(threshold=10)

# Plot the images
model.plot()

# Move the images
model.move(gui=True)


# %% import from disk
from clustimage import Clustimage

cl = Clustimage(method='pca',
                embedding='tsne',
                grayscale=False,
                dim=(128, 128),
                params_pca={'n_components':0.95},
                store_to_disk=True,
                ext=['png', 'tiff', 'jpg', 'heic', 'jpeg'],
                verbose='info')

path = cl.import_data('D://temp//Various//')

# Run the model to find the optimal clusters
results = cl.fit_transform(path, min_clust=3)

# Scatter
cl.scatter()
cl.scatter(zoom=1, img_mean=False)

# Scatter
cl.scatter(plt_all=True)


cl.plot();
cl.dendrogram();


# Predict which image is closests to input image.
# predict = cl.find(path[0])
# cl.plot_find()
# cl.scatter()


# %%
from clustimage import Clustimage

# Initialize
cl = Clustimage(method='pca', verbose='info')
cl = Clustimage(method='phash', verbose='info')

# Import data
Xraw = cl.import_example(data='flowers')
# Xraw, y = cl.import_example(data='mnist')
# Xraw, y = cl.import_example(data='faces')

# Check whether in is dir, list of files or array-like
X = cl.import_data(Xraw)

# Extract features using method
Xfeat = cl.extract_feat(X)

# Embedding using tSNE
xycoord = cl.embedding(Xfeat)

# Cluster
labels = cl.cluster(min_clust=7, metric='hamming', linkage='complete')
# labels = cl.cluster(min_clust=7)

# Return
results = cl.results

# Or all in one run
# results = cl.fit_transform(X)

# Plots
# cl.clusteval.plot();
cl.scatter(dotsize=75, plt_all=True, zoom=0.1)
# cl.plot_unique();
# cl.plot();
cl.dendrogram();

# Find
results_find = cl.find(Xraw[0], k=0, alpha=0.05)
cl.plot_find()


# %% Dendrogram merge clusters
import numpy as np
import matplotlib.pyplot as plt
from clustimage import Clustimage

# Initialize
cl = Clustimage()

# Import data
X = cl.import_example(data='flowers')
# Fit transform
cl.fit_transform(X)
# Check number of clusters
len(np.unique(cl.results['labels']))

# Scatter
cl.scatter(dotsize=75)
# Create dendrogram
cl.dendrogram();


# Set to 5 clusters
labels = cl.cluster(min_clust=5, max_clust=5)
# Check number of clusters
len(np.unique(cl.results['labels']))
# Scatter
cl.scatter(dotsize=75)
# Create dendrogram
cl.dendrogram();



# Look at the dendrogram y-axis and specify the height to merge clusters
dendro_results = cl.dendrogram(max_d=60000)
# Check number of clusters
len(np.unique(cl.results['labels']))
# Scatter
cl.scatter(dotsize=75)


# Specify to expand clusters
dendro_results = cl.dendrogram(max_d=30000)
# Scatter
cl.scatter(dotsize=75)
# Check number of clusters
len(np.unique(cl.results['labels']))


# # Return desired number of clusters
# labels = cl.cluster(min_clust=7, max_clust=7)
# dendro_results = cl.dendrogram()
# # Check number of clusters
# np.unique(dendro_results['labels'])



# %% issue 26
from clustimage import Clustimage

cl = Clustimage()

# load example with flowers
pathnames = cl.import_example(data='flowers')

results = cl.fit_transform(pathnames)
cl.scatter(args_scatter={'title':'test title' })

# %%
# Init with default settings
import clustimage as cl
# Importing the files files from disk, cleaning and pre-processing
url_to_images = ['https://erdogant.github.io/datasets/images/flower_images/flower_orange.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_2.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_2.png']

# Import into model
out = cl.url2disk(url_to_images, r'c:/temp/out/')


# %% load examples
import os
import numpy as np
from clustimage import Clustimage
cl = Clustimage()
some_files = cl.import_example(data='flowers')
some_files = cl.import_example(data='scenes')
X, y = cl.import_example(data='mnist')
X, y = cl.import_example(data='faces')

cl.fit_transform(X)
np.all(np.array(list(map(os.path.basename, cl.results['pathnames'])))==cl.results['filenames'])


# %%
from clustimage import Clustimage
cl = Clustimage(method='pca', grayscale=True)
face_results = cl.extract_faces(r'D://magweg//Various//')
results = cl.fit_transform(face_results['pathnames_face'])
cl.plot()
cl.scatter(img_mean=False, plt_all=True)
cl.plot_faces()

# %%
import sys
import os
import numpy as np
from clustimage import Clustimage
import matplotlib.pyplot as plt

# cl = Clustimage(method='pca',dirpath=None, embedding='tsne',grayscale=False, dim=(128,128),params_pca={'n_components':0.95})
cl = Clustimage(method='hog', grayscale=True, dim=(128,128))

# some_files = cl.import_example(data='flowers')[0:10]
some_files, y = cl.import_example(data='faces')[0:10]
results = cl.fit_transform(some_files, min_clust=4)

# cl.clusteval.plot()
# cl.clusteval.scatter()
# cl.pca.plot()
cl.plot_unique(img_mean=False)
cl.plot()
cl.plot(cmap='binary')
cl.scatter(zoom=None, dotsize=200, figsize=(25, 15), args_scatter={'fontsize':24, 'gradient':'#FFFFFF', 'cmap':'Set2', 'legend':True})
cl.scatter(zoom=0.15, img_mean=False, plt_all=True)

# %%
cl.results['labels']
cl.results_unique['labels']

# %%
from clustimage import Clustimage
import matplotlib.pyplot as plt
import pandas as pd

# Init with default settings
cl = Clustimage(method='pca')


from sklearn.datasets import load_digits
X = load_digits(n_class=10, return_X_y=True)
df = pd.DataFrame(data=X[0], index=X[1])

# Cluster digits
results = cl.fit_transform(df.values)

from scatterd import scatterd
scatterd(x=results['feat'][:, 0], y=results['feat'][:, 1], labels=df.index.values)
scatterd(x=results['xycoord'][:, 0], y=results['xycoord'][:, 1], labels=df.index.values)

import numpy as np
dffin = pd.DataFrame(data=np.c_[df.index.values, results['labels'], results['feat'][:, 0:2], results['xycoord'][:, 0:2]], columns=['y', 'cluster_labels', 'PC1', 'PC2', 'tsne_1', 'tsne_2'])
dffin['y']=dffin['y'].astype(int)
dffin['cluster_labels']=dffin['cluster_labels'].astype(int)


# %%
import sys
import os
import glob
import numpy as np
from clustimage import Clustimage
import matplotlib.pyplot as plt

cl = Clustimage(method='pca',dirpath=None,embedding='tsne',grayscale=False,dim=(128,128),params_pca={'n_components':0.95})
# in_files = input("""Give the absolute path to a directory with your files: \n""")
# some_files = glob.glob(in_files)

some_files =  cl.import_example(data='flowers')

results = cl.fit_transform(some_files,
                           cluster='agglomerative',
                           evaluate='silhouette',
                           metric='euclidean',
                           linkage='ward',
                           min_clust=3,
                           max_clust=6,
                           cluster_space='high')

cl.clusteval.plot()
cl.clusteval.scatter(cl.results['xycoord'])
cl.pca.plot()
cl.plot_unique(img_mean=False)
cl.plot(cmap='binary')
cl.scatter(zoom=1, img_mean=False)
cl.scatter(zoom=None, dotsize=200, figsize=(25, 15), args_scatter={'fontsize':24, 'gradient':'#FFFFFF', 'cmap':'Set2', 'legend':True})

# %% Iterative learning
from clustimage import Clustimage
import numpy as np

# Init with default settings
cl = Clustimage(method='pca')
# load example with digits
X, y = cl.import_example(data='mnist')

# Make 1st subset
idx = np.unique(np.random.randint(0,X.shape[0], 25))
X1 = X[idx, :]
X = X[np.setdiff1d(range(0, X.shape[0]), idx), :]

# Cluster dataset X
results = cl.fit_transform(X)
# Results are also stored in object results
cl.results.keys()
# Scatter results
cl.scatter(zoom=3, dotsize=50, figsize=(25, 15), legend=False, text=False)

# Find images for 1st subset of images
X1_results = cl.find(X1, alpha=0.05)
# Make scatter
cl.scatter(zoom=5, dotsize=100, text=False, figsize=(35, 20))

# Print first key
keys = list(X1_results.keys())[1:]
print(X1_results.get(keys[0]).columns)
# ['y_idx', 'distance', 'y_proba', 'labels', 'y_filenames', 'y_pathnames', 'x_pathnames']
print(X1_results.get(keys[0])[['labels', 'distance','y_proba']])

#     labels    distance   y_proba
# 0        8  189.373756  0.000035
# 1        8  305.290164  0.000822
# 2        8  338.588849  0.001812
# 3        8  341.933366  0.001956
# 4        8  351.231864  0.002412
# ..     ...         ...       ...
# 57       8  506.617886  0.044141
# 58       8  507.522983  0.044750
# 59       8  508.852247  0.045657
# 60       8  509.517201  0.046116
# 61       8  511.862011  0.047765

# Get most often seen class label for key
for key in keys:
    uiy, ycounts = np.unique(X1_results.get(key)['labels'], return_counts=True)
    y_predict = uiy[np.argmax(ycounts)]
    print('class:[%s] - %s' %(y_predict, key))

# class:[9] - b2ea44d9-de55-421b-8bd1-6d13509533f5.png
# class:[4] - 60dabaf0-7e1a-4c57-bb57-5d464fd2d8fb.png
# class:[6] - dbc30522-5c83-4563-9c9f-40a86f24a091.png
# class:[1] - deb11282-a992-4282-8212-ba62ef1e26fc.png
# class:[3] - 29104a7c-775b-462f-84ba-e824c7c4b9c7.png
# class:[9] - 45e6f5fd-4423-4743-850e-921914bfb9c9.png
# class:[9] - 300a866e-5440-444a-b284-d2bfb1b1178b.png
# class:[8] - 2dd0defc-d72a-4189-abae-bd33a94ee044.png
# class:[7] - a308d13e-fa3e-4428-8b5e-edb26554d723.png
# class:[2] - 6dc9c2b5-e1eb-4b6e-891a-db657c663013.png
# class:[9] - 30477a7a-e7a0-44f2-8bb2-222719ebe12b.png
# class:[5] - 50261737-f812-4665-b46f-cf8afb3cc88c.png
# class:[0] - c83ab55a-2983-44a6-9e2e-4964dc13c1b0.png
# class:[9] - 3eafd007-b6ed-4d17-a9db-3477854525df.png
# class:[8] - 411c207e-4804-4e30-a1c5-546876e36c51.png
# class:[4] - 661ac0f2-6f41-493a-bb3a-2c068c632d86.png
# class:[9] - 9fc65ef1-ff93-4ac6-9cdb-f4605ee9661a.png
# class:[7] - 6d9e017e-fc6a-424f-8f40-52995db771dc.png
# class:[4] - dfc5cf51-2157-437f-b1ce-920100b74119.png
# class:[4] - 507c365d-d35d-4623-985a-e9512c511d11.png
# class:[0] - 058f3759-2608-490e-ac79-e5fde8d10f7e.png
# class:[2] - a10dbae8-81f1-4613-8fd9-0bfa4815f1ad.png
# class:[2] - a889af11-961c-42f6-8b34-3d0f7050c34c.png
# class:[0] - 2e92d26c-3bf4-4c25-9d7e-4ceaa2e06d69.png
# class:[4] - b7706e78-c653-4e26-9d7a-bcb512751526.png

# %% 
from clustimage import Clustimage
import matplotlib.pyplot as plt
import pandas as pd

# Init with default settings
cl = Clustimage(method='pca')

# load example with digits
X, y = cl.import_example(data='mnist')

# Cluster digits
results = cl.fit_transform(X)

# Lets search for the following image:
plt.figure(); plt.imshow(X[0,:].reshape(cl.params['dim']), cmap='binary')

# Find images
results_find = cl.find(X[0:3,:], k=None, alpha=0.05)

# Show whatever is found. This looks pretty good.
cl.plot_find()
cl.scatter(zoom=3)
cl.plot()

# Extract the first input image name
filename = [*results_find.keys()][1]

# Plot the probabilities
plt.figure(figsize=(8,6))
plt.plot(results_find[filename]['y_proba'],'.')
plt.grid(True)
plt.xlabel('samples')
plt.ylabel('Pvalue')

# Extract the cluster labels for the input image
results_find[filename]['labels']

# The majority (=171) of labels is for class [0]
print(pd.value_counts(results_find[filename]['labels']))
# 0    171
# 7      8
# Name: labels, dtype: int64


# %% SAVE AND LOAD
from clustimage import Clustimage

# Init
cl = Clustimage(method='pca')

# load example with flowers
pathnames = cl.import_example(data='flowers')

# Cluster flowers
cl.fit_transform(pathnames)

cl.save(filepath=None, overwrite=True)
cl.load()

results_find = cl.find(pathnames[0:5], k=10, alpha=0.05)
cl.plot_find()

cl.save(overwrite=True)
cl.load()

results_find = cl.find(pathnames[0:5], k=10, alpha=0.05)
cl.plot_find()


# %% SAVE AND LOAD
from clustimage import Clustimage

cl = Clustimage(method='pca',dirpath=None, grayscale=False,dim=(128,128),params_pca={'n_components':0.5})
cl = Clustimage()

# load example with flowers
pathnames = cl.import_example(data='flowers')

# Cluster flowers
cl.fit_transform(pathnames);

# Make plot
cl.clusteval.plot()
cl.clusteval.scatter(cl.results['xycoord'], s=100)
cl.clusteval.scatter(embedding='pca', density=True, s=100, params_scatterd={'edgecolor': 'black'})


cl.results['labels']
cl.results_unique['labels']


# %%
from clustimage import Clustimage
import pandas as pd
import numpy as np

# Initialize
cl = Clustimage()

# Import data
Xraw = cl.import_example(data='mnist')

print(Xraw)
# array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 10.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 16.,  9.,  0.],
#        ...,
#        [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#        [ 0.,  0.,  2., ..., 12.,  0.,  0.],
#        [ 0.,  0., 10., ..., 12.,  1.,  0.]])

filenames = list(map(lambda x: str(x) + '.png', np.arange(0, Xraw.shape[0])))
Xraw = pd.DataFrame(Xraw, index=filenames)

print(Xraw)
#            0    1     2     3     4     5   ...   58    59    60    61   62   63
# 0.png     0.0  0.0   5.0  13.0   9.0   1.0  ...  6.0  13.0  10.0   0.0  0.0  0.0
# 1.png     0.0  0.0   0.0  12.0  13.0   5.0  ...  0.0  11.0  16.0  10.0  0.0  0.0
# 2.png     0.0  0.0   0.0   4.0  15.0  12.0  ...  0.0   3.0  11.0  16.0  9.0  0.0
# 3.png     0.0  0.0   7.0  15.0  13.0   1.0  ...  7.0  13.0  13.0   9.0  0.0  0.0
# 4.png     0.0  0.0   0.0   1.0  11.0   0.0  ...  0.0   2.0  16.0   4.0  0.0  0.0
#       ...  ...   ...   ...   ...   ...  ...  ...   ...   ...   ...  ...  ...
# 1792.png  0.0  0.0   4.0  10.0  13.0   6.0  ...  2.0  14.0  15.0   9.0  0.0  0.0
# 1793.png  0.0  0.0   6.0  16.0  13.0  11.0  ...  6.0  16.0  14.0   6.0  0.0  0.0
# 1794.png  0.0  0.0   1.0  11.0  15.0   1.0  ...  2.0   9.0  13.0   6.0  0.0  0.0
# 1795.png  0.0  0.0   2.0  10.0   7.0   0.0  ...  5.0  12.0  16.0  12.0  0.0  0.0
# 1796.png  0.0  0.0  10.0  14.0   8.0   1.0  ...  8.0  12.0  14.0  12.0  1.0  0.0

# Or all in one run
results = cl.fit_transform(Xraw)

print(results['filenames'])
# array(['0.png', '1.png', '2.png', ..., '1794.png', '1795.png', '1796.png'],

    # Plots
    # cl.clusteval.plot()
    # cl.scatter(zoom=None, dotsize=200, figsize=(25, 15), args_scatter={'fontsize':24, 'gradient':'#FFFFFF', 'cmap':'Set2', 'legend':True})
    # cl.plot_unique()
    # cl.plot()
    # cl.dendrogram()

    # Find
    # results_find = cl.find(Xraw[0], k=0, alpha=0.05)
    # cl.plot_find()


# %% Import list of images from url adresses
from clustimage import Clustimage

# Initialize
cl = Clustimage(method='pca', embedding='umap', dim=(128, 128), verbose=20)

# Importing the files files from disk, cleaning and pre-processing
url_to_images = ['https://erdogant.github.io/datasets/images/flower_images/flower_orange.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_2.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_2.png',
                 'https://erdogant.github.io/datasets/images/LARGE_elevation.jpg']

# Import into model
X = cl.import_data(url_to_images)

# Extract features using method
Xfeat = cl.extract_feat(X)

# Embedding
xycoord = cl.embedding(Xfeat)

# Cluster
labels = cl.cluster()

# Return
results = cl.results

# Make plots
cl.plot()
cl.dendrogram()
cl.scatter()

results['url']
cl.results['url']
# cl.clean_files(clean_tempdir=True)
# cl.plot()

# %%
from clustimage import Clustimage

# Initialize
cl = Clustimage(method='pca', embedding='tsne')

# Import data
Xraw = cl.import_example(data='flowers')
# Xraw, y = cl.import_example(data='mnist')
# Xraw, y = cl.import_example(data='faces')

# Import data in a standardized manner
X = cl.import_data(Xraw)

# Extract features using method
Xfeat = cl.extract_feat(X)

# Embedding
xycoord = cl.embedding(Xfeat)

# Cluster
labels = cl.cluster()

# Or all in one run
# results = cl.fit_transform(X)

# Plots
cl.clusteval.plot()
cl.scatter(zoom=None, dotsize=200, figsize=(25, 15), args_scatter={'fontsize':24, 'gradient':'#FFFFFF', 'cmap':'Set2', 'legend':True})
cl.scatter(zoom=0.5, dotsize=200, figsize=(25, 15), args_scatter={'fontsize':24, 'gradient':'#FFFFFF', 'cmap':'Set2', 'legend':True})
# cl.plot_unique()
# cl.plot()
# cl.dendrogram()

# Find
# results_find = cl.find(Xraw[0], k=0, alpha=0.05)
# cl.plot_find()


# %% Match flowers with with internet images
url_to_images = ['https://erdogant.github.io/datasets/images/flower_images/flower_orange.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_white_2.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_1.png',
                 'https://erdogant.github.io/datasets/images/flower_images/flower_yellow_2.png',
                 'https://erdogant.github.io/datasets/images/LARGE_elevation.jpg']

# # Import into model
# imgs = cl.import_data(url_to_images)

results_find = cl.find(url_to_images, k=0, alpha=0.05)
cl.plot_find()

# Scatter with new unseen images
cl.scatter()






# %% HASHES
import matplotlib.pyplot as plt
from clustimage import Clustimage

# Cluster on image-hash
cl = Clustimage(method='phash', params_hash={'threshold': 0, 'hash_size': 32})

# Example data
X, y = cl.import_example(data='mnist')
# Preprocessing, feature extraction and cluster evaluation
results = cl.fit_transform(X, min_clust=4, max_clust=15, metric='euclidean', linkage='ward')

# Scatter
cl.scatter(zoom=3, img_mean=False, text=False)
cl.scatter(zoom=None, img_mean=False, dotsize=20, text=False)
cl.scatter(zoom=3, img_mean=False, text=True, plt_all=True, figsize=(35, 25))

# cl.clusteval.plot()
# cl.plot_unique(img_mean=False)
# cl.plot(min_clust=5)


# %% Run clustimage
from clustimage import Clustimage

cl = Clustimage(method='pca',
                embedding='tsne',
                grayscale=False,
                dim=(128, 128),
                params_pca={'n_components':0.95},
                store_to_disk=True,
                verbose=50)

path = cl.import_example(data='flowers')

# Run the model to find the optimal clusters
results = cl.fit_transform(path, min_clust=3)
cl.scatter()

predict = cl.find(path[0])
cl.plot_find()
cl.scatter()

# %% Run clustimage in seperate steps
from clustimage import Clustimage

# Initialize
cl = Clustimage(method='pca', params_pca={'n_components':0.95}) 
# Import data
X = cl.import_example(data='flowers')
# X, y = cl.import_example(data='mnist')
# X, y = cl.import_example(data='faces')

# Check whether in is dir, list of files or array-like
X = cl.import_data(X)
# Extract features using method
Xfeat = cl.extract_feat(X)
# Embedding using tSNE
xycoord = cl.embedding(Xfeat)
# Cluster
labels = cl.cluster()
# Return
cl.results

# Or all in one run
# results = cl.fit_transform(X)

# Plots
cl.clusteval.plot()
cl.scatter()
cl.plot_unique()
cl.plot()
cl.dendrogram()


# %%
# Import library
from clustimage import Clustimage

# Init with settings such as PCA
# cl = Clustimage(method='hog', params_pca={'n_components':0.95}) 
cl = Clustimage(method='pca', params_pca={'n_components':0.95}) 
# cl = Clustimage(method='pca', params_pca={'n_components':50}) 

# load example with flowers
pathnames = cl.import_example(data='flowers')

# Cluster flowers
results = cl.fit_transform(pathnames)

print(cl.results['feat'].shape)

# Read the unseen image. Note that it is import to use the cl.imread functionality as these will perform exactly the same preprocessing steps as for the clustering.
# img = cl.imread(unseen_image)
# plt.figure(); plt.imshow(img.reshape((128,128,3)));plt.axis('off')

# Find images using the path location.
results_find = cl.find(pathnames[2], k=0, alpha=0.05)

# Show whatever is found. This looks pretty good.
cl.plot_find()
cl.scatter()




# %% FACES
from clustimage import Clustimage
# Init
cl = Clustimage(method='pca', grayscale=False, dim=(64,64))
# Load example with faces
X, y = cl.import_example(data='faces')
# Preproceesing, cluster detection
results = cl.fit_transform(X, min_clust=4, max_clust=20)

cl.clusteval.plot(figsize=(10,6))

# Scatter
cl.scatter(zoom=0.2, img_mean=False)
cl.scatter(zoom=None)


cl.plot_unique(img_mean=True)
cl.plot_unique(img_mean=False, show_hog=True)

# Plot dendrogram
cl.dendrogram()

# Make plot
cl.plot()
cl.plot(labels=2, show_hog=True)
# Cleaning files
cl.clean_files()


cl.clusteval.plot()
cl.clusteval.scatter(cl.results['feat'])
cl.clusteval.scatter(cl.results['xycoord'])
cl.pca.plot()
cl.pca.scatter(legend=False, label=False)

cl.save(filepath='clustimage.pkl', overwrite=True)
cl.load(filepath='clustimage.pkl')



# %% FLOWERS
# Load library
from clustimage import Clustimage

# Init with default settings
cl = Clustimage(method='hog', params_hog={'orientations':8, 'pixels_per_cell':(8,8)})
# cl = Clustimage(method='pca')
# load example with flowers
pathnames = cl.import_example(data='flowers')
# Detect cluster
results = cl.fit_transform(pathnames, min_clust=3, max_clust=13)

# cl.cluster(min_clust=3, max_clust=13)
# Plot the evaluation of the number of clusters
cl.clusteval.plot()
# cl.pca.plot()

# uiimgs = cl.unique()
cl.plot_unique(img_mean=False, show_hog=True)
cl.plot_unique(img_mean=True, show_hog=True)

# Scatter
cl.scatter(dotsize=50)
cl.scatter(dotsize=50, img_mean=False, zoom=0.5)
# Plot clustered images
cl.plot()
cl.plot(labels=[3], show_hog=True)
cl.plot(show_hog=False)
# Plot dendrogram
cl.dendrogram()

# Make prediction
results_find = cl.find(pathnames[0:5], k=10, alpha=0.05)
cl.plot_find()

# Plot the explained variance
if cl.params['method']=='pca': cl.pca.plot()
# Make scatter plot of PC1 vs PC2
if cl.params['method']=='pca': cl.pca.scatter(legend=False, label=True)
# Plot the evaluation of the number of clusters
cl.clusteval.plot()
# Make silhouette plot
cl.clusteval.scatter(cl.results['xycoord'])


# %% MNIST DATAST
import matplotlib.pyplot as plt
from clustimage import Clustimage
# init
# cl = Clustimage(method='pca')
cl = Clustimage(method='hog', grayscale=False, params_hog={'pixels_per_cell':(2,2)})
# cl = Clustimage(method='hog', embedding='tsne', grayscale=False, dim=(8,8), params_pca={'n_components':50})
# Example data
X, y = cl.import_example(data='mnist')
# Preprocessing, feature extraction and cluster evaluation
results = cl.fit_transform(X)

cl.clusteval.plot()

# Scatter
cl.scatter(zoom=3, img_mean=False)
cl.scatter(zoom=None, img_mean=False)

cl.clusteval.plot()
cl.clusteval.scatter(X)

# cl.pca.plot()

cl.plot_unique(img_mean=True, show_hog=True)
cl.plot(cmap='binary', labels=[4,5])

results_find = cl.find(X[0:2,:])
cl.plot_find()

cl.plot_unique(img_mean=False, show_hog=True)
cl.plot_unique(img_mean=True, show_hog=True)

# labx = cl.cluster()
# labx = cl.cluster(cluster_space='high', cluster='agglomerative', evaluate='silhouette', metric='euclidean', linkage='ward', min_clust=2, max_clust=25)

# Plot the clustered images
cl.plot(cmap='binary', labels=[1,2], show_hog=True)
cl.plot(cmap='binary')
# Plotting
cl.dendrogram()


# %% FLOWERS HOG FEATURES

import matplotlib.pyplot as plt
from clustimage import Clustimage
# init
cl = Clustimage()
# cl = Clustimage(method='hog')
# Load example data
path_to_imgs = cl.import_example(data='flowers')
# Read image according the preprocessing steps
dim=(128,128)
X = cl.imread(path_to_imgs[0], dim=dim, flatten=True, colorscale=0)
# Extract HOG features
img_hog = cl.extract_hog(X)

plt.figure();
fig,axs=plt.subplots(1,2)
if len(X.shape)==1:
    axs[0].imshow(X.reshape(dim))
else:
    axs[0].imshow(X)
axs[0].axis('off')
axs[0].set_title('Preprocessed image', fontsize=10)
axs[1].imshow(img_hog.reshape(dim), cmap='binary')
axs[1].axis('off')
axs[1].set_title('HOG', fontsize=10)



# %% 101_ObjectCategories
# import clustimage as cl
# pathnames = cl.listdir('D://magweg//101_ObjectCategories//')
# idx=[]
# for i in range(0,500):
#     idx.append(np.random.randint(9000))
# idx = np.unique(np.array(idx))
# pathnames = list(np.array(pathnames)[idx])
# print(len(pathnames))

from clustimage import Clustimage
# init
cl = Clustimage(method='pca', params_pca={'n_components':0.95}, dim=(60,60), grayscale=True)
# cl = Clustimage(method='pca-hog', grayscale=True, params_hog={'pixels_per_cell':(4,4)})
# Collect samples

# Preprocessing and feature extraction
# 
pathnames='D://magweg//101_ObjectCategories//'
pathnames='D://magweg//archive//images//'

# Check whether in is dir, list of files or array-like
X = cl.import_data(pathnames)
# Extract features using method
Xfeat = cl.extract_feat(X)
# Embedding using tSNE
xycoord = cl.embedding(Xfeat)
cl.scatter(zoom=None)
# Cluster
labels = cl.cluster(min_clust=5, max_clust=50)


# results = cl.fit_transform(pathnames, min_clust=60, max_clust=110)
results = cl.fit_transform(pathnames, min_clust=15, max_clust=50)
# Cluster
# cl.cluster(evaluate='silhouette', min_clust=15, max_clust=60)

cl.clusteval.plot()
cl.pca.plot()

# Scatter
cl.scatter(dotsize=8, zoom=None)
cl.scatter(dotsize=8, zoom=1, img_mean=False)

# uiimgs = cl.unique()
cl.plot_unique(img_mean=True, show_hog=True)
cl.plot_unique(img_mean=False)

# Plot the clustered images
cl.plot()
# Plotting
cl.dendrogram()

import os
y_pred = results['labx']
y_true=[]
for path in results['pathnames']:
    getpath=os.path.split(results['pathnames'][0])[0]
    y_true.append(os.path.split(getpath)[1])

#
cl.model.plot()
cl.model.scatter(legend=False)
cl.model.results.keys()

#
from clusteval import clusteval
ce = clusteval()
ce.fit()

# %%