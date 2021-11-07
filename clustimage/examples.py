# %%
# import clustimage
# print(dir(clustimage))
# print(clustimage.__version__)
# 
# %%
# from clustimage import Clustimage
# cl = Clustimage()
# cl.fit_transform()

# %%
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)
X, y = digits.data, digits.target

from clustimage import Clustimage
# init
cl = Clustimage(method=None, embedding='tsne', grayscale=True, params_pca={'n_components':50}, dim=(8,8), store_to_disk=True)
# Preprocessing and feature extraction
results = cl.fit_transform(X)
# Scatter
cl.scatter()
# Plotting
cl.dendrogram()
# Plot the clustered images
cl.plot(cmap='binary')


# %%
from clustimage import Clustimage
# init
cl = Clustimage(method='pca', embedding='tsne', grayscale=True)
# load example with flowers
path_to_imgs = cl.import_example(data='flowers')
# Extract images and the accompanying features
# X, feat = cl.extract_feat(path_to_imgs)
# Extract features (raw images are not stored and handled per-image to save memory)
results = cl.fit_transform(path_to_imgs, min_clust=10)
# Cluster
# labx = cl.cluster(min_clust=10)
# Plot dendrogram
cl.dendrogram()
# Scatter
cl.scatter(dot_size=50)
# Plot clustered images
cl.plot()

# Predict
results_predict = cl.predict(path_to_imgs[0:5], k=None, alpha=0.05)
cl.plot_predict()
cl.scatter()

# %%
from clustimage import Clustimage
# init
cl = Clustimage(method='hog', embedding='tsne', grayscale=False)
# Collect samples
# path_to_imgs = cl.get_images_from_path('D://magweg//101_ObjectCategories//')
# Preprocessing and feature extraction
results = cl.fit_transform('D://magweg//101_ObjectCategories//')
# Cluster
labx = cl.cluster(method='silhouette', min_clust=30, max_clust=50)
# Scatter
cl.scatter()
# Plot the clustered images
cl.plot()
# Plotting
cl.dendrogram()

# %% Detect faces
from clustimage import Clustimage
# Init
cl = Clustimage(method='pca', image_type='faces', grayscale=False, params_pca={'n_components':3})
# Load example with faces
pathnames = cl.import_example(data='faces')
# Detect faces
face_results = cl.detect_faces(pathnames)
# Cluster
results = cl.fit_transform(face_results['facepath'])

cl.scatter()
# Plot faces
cl.plot_faces()

# Cluster
labx = cl.cluster()
# Plot dendrogram
cl.dendrogram()
cl.plot()

cl.clean()

# plt.imshow(results['feat'][0,:].reshape(cl.dim), cmap=plt.cm.gray)
# plt.imshow(face_results['img'][0][0,:].reshape(cl.dim_face), cmap=plt.cm.gray)
results['feat'].shape



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

