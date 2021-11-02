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
from clustimage import Clustimage
cl = Clustimage(grayscale=False)
path_to_imgs = cl.import_example()
cl.fit_transform(path_to_imgs)

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

