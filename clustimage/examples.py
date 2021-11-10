# %%
# import clustimage
# print(dir(clustimage))
# print(clustimage.__version__)
# 
# %%
# from clustimage import Clustimage
# cl = Clustimage()
# cl.fit_transform()

# import clustimage.clustimage as cl
# img=cl.img_read(path_to_imgs[0])
# img = cv2.imread(path_to_imgs[0])
# plt.figure();plt.imshow(img);plt.axis('off')


# %%
from clustimage import Clustimage
# init
cl = Clustimage(store_to_disk=True)
cl = Clustimage(method='pca', embedding='tsne', grayscale=True, params_pca={'n_components':50}, store_to_disk=True)
# cl = Clustimage(method='hog', embedding='tsne',grayscale=False, params_pca={'n_components':50})
# cl = Clustimage(method='hog', embedding='tsne', grayscale=False, dim=(8,8), params_pca={'n_components':50})
# Example data
X = cl.import_example(data='digits')
# Preprocessing and feature extraction
results = cl.fit_transform(X)

# Scatter
cl.scatter()
# Plot the clustered images
cl.plot(cmap='binary', labx=0)

# Cluster differently
cl.cluster(cluster_space='low')
# Scatter
cl.scatter()
# Plotting
cl.dendrogram()

# %%

import matplotlib.pyplot as plt
from clustimage import Clustimage
# init
cl = Clustimage(method='hog')
# Load example data
path_to_imgs = cl.import_example(data='flowers')
# Read image according the preprocessing steps
img = cl.img_read_pipeline(path_to_imgs[0], dim=(128,128))
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


import matplotlib.pyplot as plt
import clustimage as cl
path_to_imgs = cl.import_example(data='flowers')
img = cl.img_read_pipeline(path_to_imgs[0], dim=(128,128))

plt.figure()
plt.imshow(img.reshape(128,128,3))


# %%
from clustimage import Clustimage
# init
cl = Clustimage(method='hog', embedding='tsne')
# load example with flowers
path_to_imgs = cl.import_example(data='flowers')
# Extract images and the accompanying features
# X, feat = cl.extract_feat(path_to_imgs)
# Extract features (raw images are not stored and handled per-image to save memory)
results = cl.fit_transform(path_to_imgs, min_clust=10, max_clust=30)
# Cluster
# labx = cl.cluster(min_clust=10)
# Scatter
cl.scatter(dot_size=50)
# Plot dendrogram
cl.dendrogram()
# Plot clustered images
cl.plot(show_hog=True)

# Predict
results_predict = cl.predict(path_to_imgs[0:5], k=None, alpha=0.05)
cl.plot_predict()
cl.scatter()








# %% Detect faces
from clustimage import Clustimage
# Init
cl = Clustimage(method='pca', grayscale=True, params_pca={'n_components':14})
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
# Make plot
cl.plot(ncols=2, show_hog=True)
# Cleaning files
cl.clean_files()
# plt.imshow(face_results['img'][0][0,:].reshape(cl.dim_face), cmap=plt.cm.gray)



# %%
from clustimage import Clustimage
# init
cl = Clustimage(method='pca', params_pca={'n_components':250, 'detect_outliers':None})
cl = Clustimage(method='pca', embedding='tsne', grayscale=False)
# Collect samples
# path_to_imgs = cl.get_images_from_path('D://magweg//101_ObjectCategories//')
# Preprocessing and feature extraction
results = cl.fit_transform('D://magweg//101_ObjectCategories//', min_clust=30, max_clust=60)
# Cluster
# cl.cluster(method='silhouette', min_clust=30, max_clust=60)
# Scatter
cl.scatter(dotsize=10)
# Plot the clustered images
cl.plot(labx=9)
# Plotting
cl.dendrogram()


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

