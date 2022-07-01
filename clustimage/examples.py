# %%
import clustimage
# print(dir(clustimage))
# print(clustimage.__version__)

# %% 
from clustimage import Clustimage
import matplotlib.pyplot as plt
import pandas as pd

# Init with default settings
cl = Clustimage(method='pca')

# load example with digits
X = cl.import_example(data='mnist')

# Cluster digits
results = cl.fit_transform(X)

# Lets search for the following image:
plt.figure(); plt.imshow(X[0,:].reshape(cl.params['dim']), cmap='binary')

# Find images
results_find = cl.find(X[0:3,:], k=None, alpha=0.05)

# Show whatever is found. This looks pretty good.
cl.plot_find()
cl.scatter(zoom=3)

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

cl = Clustimage(method='pca',dirpath=None,embedding='tsne',grayscale=False,dim=(128,128),params_pca={'n_components':0.5})


# load example with flowers
pathnames = cl.import_example(data='flowers')

# Cluster flowers
cl.fit_transform(pathnames)

# Make plot
cl.clusteval.plot()
cl.clusteval.scatter(cl.results['xycoord'])




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
# Xraw = cl.import_example(data='mnist')
# Xraw = cl.import_example(data='faces')

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
X = cl.import_example(data='mnist')
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
# X = cl.import_example(data='mnist')
# X = cl.import_example(data='faces')

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
X = cl.import_example(data='faces')
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
X = cl.import_example(data='mnist')
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