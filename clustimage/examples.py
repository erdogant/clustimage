# %%
# import clustimage
# print(dir(clustimage))
# print(clustimage.__version__)
# 
# %%
# from clustimage import Clustimage
# cl = Clustimage()
# cl.fit_transform()

# %% Detect faces
from clustimage import Clustimage
# Init
cl = Clustimage(image_type='faces')
# Load examples
pathnames = cl.import_example(data='faces')
# Detect faces
faces = cl.detect_faces(pathnames)
# Plot faces
cl.plot_faces()

# %%
from clustimage import Clustimage
cl = Clustimage(grayscale=False)
path_to_imgs = cl.import_example()
feat = cl.fit_transform(path_to_imgs)

cl.plot()
results = cl.predict(path_to_imgs[0:5], k=None, alpha=0.05)
cl.plot()

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

