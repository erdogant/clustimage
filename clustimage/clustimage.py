"""Clustimage"""
# --------------------------------------------------
# Name        : clustimage.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/clustimage
# Licence     : See licences
# --------------------------------------------------

from pca import pca
from distfit import distfit
from clusteval import clusteval
import pandas as pd
import colourmap
from tqdm import tqdm
from sklearn.manifold import TSNE
import os
import logging
from urllib.parse import urlparse
import fnmatch
import zipfile
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from skimage.feature import hog
import tempfile
import uuid
import shutil

logger = logging.getLogger('')
for handler in logger.handlers[:]: #get rid of existing old handlers
    logger.removeHandler(handler)
console = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [clustimage]> %(levelname)s> %(message)s', datefmt='%H:%M:%S')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger()


class Clustimage():
    """clustimage."""

    def __init__(self, method='pca', embedding='tsne', image_type='object', grayscale=False, dim=(128,128), dirpath=None, ext=['png','tiff','jpg'], params_pca={'n_components':50, 'detect_outliers':None}, verbose=20):
        """Initialize distfit with user-defined parameters."""
        if not np.any(np.isin(image_type, ['object', 'faces'])): raise Exception(logger.error('image_type: "%s" is unknown', image_type))
        if not np.any(np.isin(method, [None, 'pca','hog'])): raise Exception(logger.error('method: "%s" is unknown', method))
        if dirpath is None: dirpath = tempfile.mkdtemp()
        if not os.path.isdir(dirpath): raise Exception(logger.error('[%s] does not exists.', dirpath))

        # If face detection, grayscale should be True.
        # if image_type=='faces': grayscale=True
        if (image_type=='faces') and (not grayscale): logger.warning('It is advisable to set "grayscale=True" when "image_type=faces".')

        # Find path of xml file containing haarcascade file and load in the cascade classifier
        # self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # Reads the image as grayscale and results in a 2D-array. In case of RGB, no transparency channel is included.
        self.method = method
        self.embedding = embedding
        self.image_type = image_type
        self.grayscale = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        # self.grayscale = cv2.COLOR_BGR2GRAY if grayscale else cv2.COLOR_BGR2RGB
        self.dim = dim
        self.dim_face = (64,64)
        self.params_pca = params_pca
        self.dirpath = dirpath
        self.ext = ext
        set_logger(verbose=verbose)

    def fit_transform(self, pathnames=None, X=None):
        """Import example dataset from github source."""
        if X is None:
            if (self.image_type=='faces') and (pathnames is None):
                pathnames = list(np.hstack(self.results['facepath']))
            # Extract features using method
            raw, X = self.extract_feat(pathnames)

        # Embedding using tSNE
        xycoord = self.compute_embedding(X)

        # Store results
        self.results = {}
        self.results['feat'] = X
        self.results['xycoord'] = xycoord
        self.results['pathnames'] = raw['pathnames']
        self.results['filenames'] = raw['filenames']
        # Return
        return self.results
    
    def compute_embedding(self, X):
        # Embedding using tSNE
        if self.embedding=='tsne':
            logger.info('Computing embedding using %s..', self.embedding)
            init='random'
            # if self.method is None: init='pca'
            xycoord = TSNE(n_components=2, learning_rate='auto', init=init).fit_transform(X)
        else:
            xycoord = X[:,0:2]
        # Return
        return xycoord

    def extract_feat(self, pathnames):
        # If the input is a directory, first collect the images from path
        logger.info('Extracting features using %s..', self.method)
        if isinstance(pathnames, str) and os.path.isdir(pathnames):
            pathnames = self.get_images_from_path(pathnames, ext=self.ext)
        # Extract features
        if self.method=='pca':
            raw, X = self.extract_pca(pathnames)
        elif self.method=='hog':
            raw, X = self.extract_hog(pathnames)
        else:
            # Read images and preprocessing and flattening of images
            raw = self.preprocessing(pathnames, grayscale=self.grayscale, dim=self.dim, flatten=True)
            X = raw['img']
        return raw, X

    def cluster(self, cluster='agglomerative', method='silhouette', metric='euclidean', linkage='ward', min_clust=2, max_clust=25, savemem=False, verbose=3):
        if self.results.get('feat', None) is None: raise Exception(logger.error('First run the "fit_transform(pathnames)" function.'))
        logger.info('Cluster evaluation using dataset %s.' %(self.method))
        # Init
        ce = clusteval(cluster=cluster, method=method, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust, savemem=False, verbose=3)
        # Fit
        ce.fit(self.results['feat'])
        # Store
        logger.info('Cluster labels "labx" is added to the object.')
        self.results['labx'] = ce.results['labx']
        self.clusteval = ce
        # Return
        return ce.results['labx']

    def predict(self, pathnames, metric='euclidean', k=1, alpha=0.05):
        """Import example dataset from github source."""
        if (k is None) and (alpha is None):
            raise Exception(logger.error('Nothing to collect! input parameter "k" and "alpha" can not be None at the same time.'))
        out = None
        # Read images and preprocessing. This is indepdent on the method type but should be in similar manner.
        X = self.preprocessing(pathnames, grayscale=self.grayscale, dim=self.dim, flatten=True)

        # Predict according PCA method
        if self.method=='pca':
            Y, dist, feat = self.compute_distances_pca(X, metric=metric, alpha=alpha)
            out = self.collect_pca(X, Y, dist, k, alpha, feat, todf=True)

        # Store
        self.results['predict'] = out
        # Return
        return out

    def preprocessing(self, pathnames, grayscale, dim, flatten=True):
        """Import example dataset from github source."""
        logger.info("Reading images..")
        img, filenames = None, None
        if isinstance(pathnames, str):
            pathnames=[pathnames]
        if isinstance(pathnames, list):
            filenames = list(map(basename, pathnames))
            img = list(map(lambda x: img_read_pipeline(x, grayscale=grayscale, dim=dim, flatten=flatten), tqdm(pathnames)))
            if flatten: img = np.vstack(img)

        out = {}
        out['img'] = img
        out['pathnames'] = pathnames
        out['filenames'] = filenames
        return out

    def extract_hog(self, pathnames):
        # Read images and preprocessing
        X = self.preprocessing(pathnames, grayscale=self.grayscale, dim=self.dim, flatten=False)
        # Extract hog features per image
        feat = []
        for img in X['img']:
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            feat.append(hog_image.flatten())
        # Stack all hog features
        feat = np.vstack(feat)
        # Message
        logger.info("Extracted features with HOG: %s", str(feat.shape))
        # Stack all hog features into one array and return
        return X, feat

    def extract_pca(self, pathnames):
        # Read images and preprocessing and flattening of images
        X = self.preprocessing(pathnames, grayscale=self.grayscale, dim=self.dim, flatten=True)
        # Fit model using PCA
        self.model = pca(**self.params_pca)
        self.model.fit_transform(X['img'], row_labels=X['filenames'])
        logger.info("Extracted features with PCA: %s", str(self.model.results['PC'].shape))
        # Return Principal Components (features)
        return X, self.model.results['PC'].values

    # Compute distances and probabilities after transforming the data using PCA.
    def compute_distances_pca(self, X, metric, alpha):
        dist = None
        # Transform new "unseen" data. Note that these datapoints are not really unseen as they are readily fitted above.
        PCnew = self.model.transform(X['img'], row_labels=X['filenames'])
        # Compute distance to all samples
        Y = distance.cdist(self.results['feat'], PCnew, metric=metric)
        Ytot = distance.cdist(self.results['feat'], self.results['feat'], metric=metric)
        # Fit distribution to emperical data and compute probability of the distances of interest
        if alpha is not None:
            dist = distfit(bound='down', multtest=None)
            dist.fit_transform(Ytot)
            # model.plot()

        # Sanity check
        if len(X['filenames'])!=Y.shape[1]: raise Exception(logger.error('Number of input files does not match number of computed distances.'))
        # Return
        return Y, dist, PCnew

    # Collect data
    def collect_pca(self, X, Y, dist, k, alpha, feat, todf=True):
        # Collect the samples that are closest in according the metric
        filenames = X['filenames']
        out = {}
        out['feat'] = feat

        # Collect nearest neighor and sample with highes probability per input sample
        for i, filename in enumerate(filenames):
            logger.info('Predict: %s', filename)
            store_key = {}
            idx_dist, idx_k = None, None
            # Collect bes samples based on k-nearest neighbor
            if k is not None:
                idx_k = np.argsort(Y[:,i])[0:k]
            # Collect samples based on probability
            if alpha is not None:
                dist_results = dist.predict(Y[:,i], verbose=0)
                idx_dist = np.where(dist_results['y_proba']<=alpha)[0]
            else:
                # If alpha is not used, set all to nan
                dist_results={}
                dist_results['y_proba'] = np.array([np.nan]*Y.shape[0])

            # Combine the unique k-nearest samples and probabilities.
            idx = unique_no_sort(np.append(idx_dist, idx_k))

            # Store in dict
            store_key = {**store_key, 'x_path': X['pathnames'][i], 'y_idx': idx, 'distance': Y[idx, i], 'y_proba': dist_results['y_proba'][idx], 'y_label': np.array(self.results['filenames'])[idx].tolist(), 'y_path': np.array(self.results['pathnames'])[idx].tolist()}
            if todf: pd.DataFrame(store_key)
            out[filename] = store_key

        # Return
        return out

    def detect_faces(self, pathnames):
        # Read and pre-proces the input images
        logger.info("Reading images..")
        # Create empty list
        faces = {'img':[], 'pathnames':[], 'filenames':[], 'facepath':[], 'coord_faces':[], 'coord_eyes':[]}
        # Extract faces and eyes from image
        for pathname in pathnames:
            # Extract faces
            facepath, imgfaces, coord_faces, coord_eyes, filename, path_to_image = self.extract_faces(pathname)
            # Store
            faces['pathnames'].append(path_to_image)
            faces['filenames'].append(filename)
            faces['facepath'].append(facepath)
            faces['img'].append(imgfaces)
            faces['coord_faces'].append(coord_faces)
            faces['coord_eyes'].append(coord_eyes)

        # Return
        self.results = faces
        return self.results
    
    def extract_faces(self, pathname):
        # Set defaults
        coord_eyes, facepath, imgstore = [], [], []
        # Get image
        # X = self.preprocessing(pathname, grayscale=cv2.COLOR_BGR2RGB, dim=None, flatten=False)
        X = self.preprocessing(pathname, grayscale=self.grayscale, dim=None, flatten=False)
        # Get the image and Convert into grayscale if required
        img = X['img'][0]
        # img = to_gray(X['img'][0])
        # Detect faces using the face_cascade
        coord_faces = self.face_cascade.detectMultiScale(img, 1.3, 5)

        # Collect the faces from the image
        for (x,y,w,h) in coord_faces:
            # Create filename for face
            filename = os.path.join(self.dirpath, str(uuid.uuid4()))+'.png'
            facepath.append(filename)
            # Store faces seperately
            imgface = img_resize(img[y:y+h, x:x+w], dim=self.dim_face)
            # Write to disk
            cv2.imwrite(filename, imgface)
            # Store face image
            # imgstore.append(imgface.flatten())
            imgstore.append(img_flatten(imgface))
            # Detect eyes
            roi_gray = img[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if eyes==(): eyes=None
            coord_eyes.append(eyes)
        # Return
        return facepath, np.array(imgstore), coord_faces, coord_eyes, X['filenames'][0], X['pathnames'][0]

    def plot_faces(self, faces=True, eyes=True):
        # Walk over all detected faces
        if hasattr(self, 'results'):
            for i, pathname in enumerate(self.results['pathnames']):
                # Import image
                img = self.preprocessing(pathname, grayscale=cv2.COLOR_BGR2RGB, dim=None, flatten=False)['img'][0].copy()

                # Plot the faces
                if faces:
                    coord_faces = self.results['coord_faces'][i]
                    plt.figure()
                    for (x,y,w,h) in coord_faces:
                        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                    if len(img.shape)==3:
                        plt.imshow(img[:,:,::-1]) # RGB-> BGR
                    else:
                        plt.imshow(img)

                # Plot the eyes
                if eyes:
                    coord_eyes = self.results['coord_eyes'][i]
                    for k in np.arange(0, len(self.results['facepath'][i])):
                        # face = self.results['img'][i][k].copy()
                        facepath = self.results['facepath'][i][k]
                        if os.path.isfile(facepath):
                            face = self.preprocessing(facepath, grayscale=cv2.COLOR_BGR2RGB, dim=None, flatten=False)['img'][0].copy()
                            if coord_eyes[k] is not None:
                                plt.figure()
                                for (ex,ey,ew,eh) in coord_eyes[k]:
                                    cv2.rectangle(face, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                                if len(face.shape)==3:
                                    plt.imshow(face[:,:,::-1]) # RGB-> BGR
                                else:
                                    plt.imshow(face)
                        else:
                            logger.warning('File is removed: %s', facepath)
        else:
            logger.warning('Nothing to plot. First detect faces with ".detect_faces(pathnames)"')

    def dendrogram(self, max_d=None, figsize=(15,10)):
        if hasattr(self, 'clusteval'):
            self.clusteval.plot()
            results = self.clusteval.dendrogram(max_d=max_d, figsize=figsize)
            # if feat is None: feat=self.results['feat']
            # results = self.clusteval.dendrogram(X=feat, max_d=max_d, figsize=figsize)
        return results

    def scatter(self, legend=False, figsize=(15,10)):
        # Set default settings
        labx = self.results.get('labx', None)
        if labx is None: labx=np.zeros_like(self.results['xycoord'][:,0]).astype(int)

        # Scatter cluster evaluation
        if hasattr(self, 'clusteval'):
            self.clusteval.scatter(self.results['feat'])

        if self.embedding=='tsne':
            from scatterd import scatterd
            colours=np.vstack(colourmap.fromlist(labx)[0])
            fig, ax = scatterd(self.results['xycoord'][:,0], self.results['xycoord'][:,1], s=10, c=colours, label=labx, figsize=figsize, title='tSNE plot')

        # Scatter all points
        if self.method=='pca':
            fig, ax = self.model.plot(figsize=figsize)
            fig, ax = self.model.scatter(y=labx, legend=legend, label=False, figsize=figsize)
            # fig, ax = self.model.scatter3d(y=1, legend=legend, label=False, figsize=figsize)

        # Scatter the predicted cases
        if self.results.get('predict', None) is not None:
            fig, ax = self.model.scatter(y=labx, legend=legend, label=False, figsize=figsize)
            # Create unique colors
            colours = colourmap.fromlist(self.results['predict']['feat'].index)[1]
            for key in self.results['predict'].keys():
                if self.results['predict'].get(key).get('y_idx', None) is not None:
                    x,y,z = self.results['predict']['feat'].iloc[:,0:3].loc[key]
                    idx = self.results['predict'][key]['y_idx']
                    # Scatter
                    ax.scatter(x, y, color=colours[key], edgecolors=[0,0,0])
                    ax.text(x,y, key, color=colours[key])
                    ax.scatter(self.results['feat'][idx][:,0], self.results['feat'][idx][:,1], edgecolors=[0,0,0])

    def plot_predict(self):
        # Plot the images that are similar to each other.
        if self.results.get('predict', None) is not None:
            for key in self.results['predict'].keys():
                if self.results['predict'].get(key).get('y_idx', None) is not None:
                    # Collect images
                    input_img = self.results['predict'][key]['x_path']
                    predict_img = self.results['predict'][key]['y_path']
                    # Input label
                    if isinstance(input_img, str): input_img=[input_img]
                    # Input images
                    I_input = list(map(lambda x: img_read_pipeline(x, grayscale=self.grayscale, dim=self.dim, flatten=False), input_img))
                    # Predicted label
                    I_predict = list(map(lambda x: img_read_pipeline(x, grayscale=self.grayscale, dim=self.dim, flatten=False), predict_img))
                    # Make the real plot
                    fig, axes = plt.subplots(len(I_predict)+1,1,sharex=True,sharey=True,figsize=(8,10))
                    axes[0].set_title('Input image')
                    axes[0].imshow(I_input[0])
                    for i, I in enumerate(I_predict):
                        axes[i+1].set_title('Predicted: %s' %(i+1))
                        axes[i+1].imshow(I)

    def plot(self, nrcols=None, legend=False):
        # Plot the clustered images
        if self.results.get('labx', None) is not None:
            uilabx = np.unique(self.results['labx'])
            for labx in tqdm(uilabx):
                idx = self.results['labx']==labx
                getfiles = np.array(self.results['pathnames'])[idx]
                # Get the images that cluster together
                imgs = list(map(lambda x: img_read_pipeline(x, grayscale=self.grayscale, dim=self.dim, flatten=False), getfiles))

                if nrcols is None:
                    nrcols = 4
                    if len(getfiles)>=25: nrcols=5
                    if len(getfiles)>=50: nrcols=10
                    if len(getfiles)>=100: nrcols=15
                    if len(getfiles)>=150: nrcols=20

                # Setup rows and columns
                nrows = int(np.ceil(len(imgs)/nrcols))
                fig, axes = plt.subplots(nrows, nrcols, figsize=(15,10))
                if len(axes.shape)==1: axes=[axes]
                colnr=0
                for i, img in enumerate(imgs):
                    rownr = np.mod(i, nrows)
                    if rownr==0: colnr = colnr+1
                    axes[rownr][colnr-1].imshow(img)
                    axes[rownr][colnr-1].set_axis_off()
                plt.axis('off')
                plt.pause(0.1)

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        # ax1.axis('off')
        # ax1.imshow(img, cmap=plt.cm.gray)
        # ax1.set_title('Input image')
        # # Rescale histogram for better display
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        # ax2.axis('off')
        # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        # ax2.set_title('Histogram of Oriented Gradients')
        # plt.show()

    def get_images_from_path(self, dirpath, ext):
        return _get_images_from_path(dirpath, ext=ext)
    
    def clean(self):
        # Cleaning
        from pathlib import Path
        out = []
        for sublist in self.results['facepath']:
            out.extend(sublist)

        p = Path(out[0])
        dirpath = str(p.parent)

        if os.path.isdir(dirpath):
            logger.info('Removing directory with all content: %s', dirpath)
            shutil.rmtree(dirpath)

    def import_example(self, data='flowers', url=None):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            'flowers', 'faces', 'scenes'

        Returns
        -------
        list of str
            list of str containing filepath to images.

        """
        return import_example(data=data, url=url)


# %% Unique without sort
def unique_no_sort(x):
    x = x[x!=None]
    indexes = np.unique(x, return_index=True)[1]
    return [x[index] for index in sorted(indexes)]

# %% Resize image
def basename(label):
    return os.path.basename(label)

# %% Resize image
def img_flatten(img):
    return img.flatten()

# %% Resize image
def img_resize(img, dim=(128, 128)):
    if dim is not None:
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # assert img.shape[0:2]==dim
    return img


# %% Convert into grayscale
# def to_gray(img):
#     try:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     except:
#         gray = img
#     return gray

# %% Read image
def img_read(filepath, grayscale=1):
    """Read image from filepath using colour-scaling.

    Parameters
    ----------
    filepath : str
        path to file.
    grayscale : int, default: 1 (gray)
        colour-scaling from opencv.
        * cv2.COLOR_GRAY2RGB

    Returns
    -------
    img : numpy array
        raw rgb or gray image.

    """
    img=None
    if os.path.isfile(filepath):
        # Read the image
        img = cv2.imread(filepath, grayscale)
    else:
        logger.warning('File does not exists: %s', filepath)
    
    # In case of rgb images: make gray images compatible with RGB
    if ((grayscale!=0) and (grayscale!=6)) and (len(img.shape)<3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


# %% Read image
def img_read_pipeline(filepath, grayscale=1, dim=(128, 128), flatten=True):
    # Read the image
    img = img_read(filepath, grayscale=grayscale)
    # Resize the image
    img = img_resize(img, dim=dim)
    # Flatten the image
    if flatten: img = img_flatten(img)
    return img

# %%
def set_logger(verbose=20):
    logger.setLevel(verbose)


# %% Import example dataset from github.
def import_example(data='flowers', url=None):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'flowers', 'faces'
    url : str
        url link to to dataset.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='flowers':
            url='https://erdogant.github.io/datasets/flower_images.zip'
        elif data=='faces':
            url='https://erdogant.github.io/datasets/faces_images.zip'
        elif data=='scenes':
            url='https://erdogant.github.io/datasets/scenes.zip'
    else:
        logger.warning('Lets try your dataset from url: %s.', url)

    if url is None:
        logger.warning('Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    filename = basename(urlparse(url).path)
    path_to_data = os.path.join(curpath, filename)
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(path_to_data):
        logger.info('Downloading [%s] dataset from github source..', data)
        wget(url, path_to_data)

    # Unzip
    dirpath = _unzip(path_to_data)
    # Import local dataset
    image_files = _get_images_from_path(dirpath)
    # Return
    return image_files


# %% Recursively list files from directory
def _get_images_from_path(dirpath, ext=['png','tiff','jpg']):
    """ Recursively collect images from path.

    Parameters
    ----------
    dirpath : str
        Path to directory; "/tmp" or "c://temp/" 
    ext : list, default: ['png','tiff','jpg']
        extentions to collect form directories.

    Returns
    -------
    getfiles : list of str.
        Full pathnames to images.

    """
    if not isinstance('dirpath', str): raise Exception(print('Error: "dirpath" should be of type string.'))
    if not os.path.isdir(dirpath): raise Exception(print('Error: The directory can not be found: %s.' %dirpath))
    
    getfiles = []
    for iext in ext:
        for root, _, filenames in os.walk(dirpath):
            for filename in fnmatch.filter(filenames, '*.'+iext):
                getfiles.append(os.path.join(root, filename))
    logger.info('[%s] files are collected recursively from path: [%s]' %(len(getfiles), dirpath))
    return getfiles


# %% unzip
def _unzip(path_to_zip):
    getpath = None
    if path_to_zip[-4:]=='.zip':
        if not os.path.isdir(path_to_zip):
            logger.info('Extracting files..')
            pathname, _ = os.path.split(path_to_zip)
            # Unzip
            zip_ref = zipfile.ZipFile(path_to_zip, 'r')
            zip_ref.extractall(pathname)
            zip_ref.close()
            getpath = path_to_zip.replace('.zip', '')
            if not os.path.isdir(getpath):
                logger.error('Extraction failed.')
                getpath = None
    else:
        logger.warning('Input is not a zip file: [%s]', path_to_zip)
    # Return
    return getpath


# %% Download files from github source
def wget(url, writepath):
    r = requests.get(url, stream=True)
    with open(writepath, "wb") as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)

# %% Main
# if __name__ == "__main__":
#     import clustimage as clustimage
#     df = clustimage.import_example()
#     out = clustimage.fit(df)
#     fig,ax = clustimage.plot(out)
