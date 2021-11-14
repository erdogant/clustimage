# %%
# import clustimage
# print(dir(clustimage))
# print(clustimage.__version__)
# 
# Read image according the preprocessing steps


# %%
# Load library
from clustimage import Clustimage

# Init with default settings
cl = Clustimage(method='hog')
# load example with flowers
pathnames = cl.import_example(data='flowers')
# Detect cluster
results = cl.fit_transform(pathnames, min_clust=7)

# uiimgs = cl.unique()
cl.plot_unique()
cl.plot_unique(img_mean=False)

# Scatter
cl.scatter(dotsize=50)
cl.scatter(dotsize=50, img_mean=False)
# Plot clustered images
cl.plot(labels=[12], show_hog=True)
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

# %% Detect faces

from clustimage import Clustimage
# Init
cl = Clustimage(method='hog', grayscale=True)
# Load example with faces
pathnames = cl.import_example(data='faces')
# Detect faces
face_results = cl.detect_faces(pathnames)
# Preproceesing, cluster detection
results = cl.fit_transform(face_results['pathnames_face'])


out = cl.unique()
cl.plot_unique(img_mean=True)

out = cl.find(face_results['pathnames_face'][20], k=None, alpha=0.05, metric='euclidean')
cl.plot_find()

out = cl.find(face_results['pathnames_face'][20], k=5, alpha=None, metric='euclidean')
cl.plot_find()

cl.plot_faces(eyes=False)

# cluster labels
labels = results['labels']

# Plot dendrogram
cl.dendrogram()
# Scatter
cl.scatter()
cl.scatter(zoom=None)
# Plot faces
cl.plot_faces(eyes=False)
# Make plot
cl.plot(labels=17, show_hog=True)
# Cleaning files
cl.clean_files()


cl.clusteval.plot()
cl.clusteval.scatter(cl.results['feat'])
cl.clusteval.scatter(cl.results['xycoord'])
cl.pca.plot()
cl.pca.scatter(legend=False, label=False)

cl.save(overwrite=True)
cl.load()


# %%
from clustimage import Clustimage
# init
cl = Clustimage(store_to_disk=True)
cl = Clustimage(method='pca', embedding='tsne', grayscale=True, params_pca={'n_components':50}, store_to_disk=True)
# cl = Clustimage(method='hog', embedding='tsne',grayscale=False, params_pca={'n_components':50})
# cl = Clustimage(method='hog', embedding='tsne', grayscale=False, dim=(8,8), params_pca={'n_components':50})
# Example data
X = cl.import_example(data='digits')
# Preprocessing, feature extraction and cluster evaluation
results = cl.fit_transform(X)

results_find = cl.find(X[0:2,:])
cl.plot_find()

# labx = cl.cluster()
# labx = cl.cluster(cluster_space='high', cluster='agglomerative', evaluate='silhouette', metric='euclidean', linkage='ward', min_clust=2, max_clust=25)

# Scatter
cl.scatter()
# Plot the clustered images
cl.plot(cmap='binary', labels=[1,2])
# Plotting
cl.dendrogram()


# %%

import matplotlib.pyplot as plt
from clustimage import Clustimage
# init
cl = Clustimage()
# cl = Clustimage(method='hog')
# Load example data
path_to_imgs = cl.import_example(data='flowers')
# Read image according the preprocessing steps
img = cl.imread(path_to_imgs[0], dim=(128,128))
# Extract HOG features
img_hog = cl.extract_hog(img)

plt.figure();
fig,axs=plt.subplots(1,2)
axs[0].imshow(img.reshape(128,128,3))
axs[0].axis('off')
axs[0].set_title('Preprocessed image', fontsize=10)
axs[1].imshow(img_hog.reshape(128,128), cmap='binary')
axs[1].axis('off')
axs[1].set_title('HOG', fontsize=10)


# %%
from clustimage import Clustimage
# init
cl = Clustimage(method='pca', params_pca={'n_components':250, 'detect_outliers':None})
# cl = Clustimage(method='pca', embedding='tsne', grayscale=False)
# Collect samples
# path_to_imgs = cl.get_images_from_path('D://magweg//101_ObjectCategories//')
# Preprocessing and feature extraction
results = cl.fit_transform('D://magweg//101_ObjectCategories//', min_clust=30, max_clust=60)
# Cluster
# cl.cluster(method='silhouette', min_clust=30, max_clust=60)

uiimgs = cl.unique()
cl.plot_unique()

# Scatter
cl.scatter(dotsize=10, zoom=0.2)
# Plot the clustered images
cl.plot(labels=10)
# Plotting
cl.dendrogram()

import os
y_pred = results['labx']
y_true=[]
for path in results['pathnames']:
    getpath=os.path.split(results['pathnames'][0])[0]
    y_true.append(os.path.split(getpath)[1])

# %%
cl.model.plot()
cl.model.scatter(legend=False)
cl.model.results.keys()

# %%
from clusteval import clusteval
ce = clusteval()
ce.fit()

# %%
# cv2 = stats._import_cv2()
# X = cv2.imread(PATH_TO_DATA)

