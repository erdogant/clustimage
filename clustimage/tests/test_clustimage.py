from clustimage import Clustimage
import itertools as it
import numpy as np
import unittest

class TestCLUSTIMAGE(unittest.TestCase):

    def test_import_data(self):
        cl = Clustimage()
        # Check initialization results
        assert cl.results=={'img': None, 'feat': None, 'xycoord': None, 'pathnames': None, 'labels': None, 'url': None}
        # Import flowers example
        X = cl.import_example(data='flowers')

        # Check numpy array imports
        assert cl.import_data(np.array(X))
        assert cl.import_data(X[0])
        assert cl.import_data([X[0]])

        # Check output
        cl = Clustimage(dim=(128, 128), grayscale=False)
        _ =  cl.import_data(X)
        assert np.all(np.isin([*cl.results.keys()], ['img', 'feat', 'xycoord', 'pathnames', 'labels', 'filenames', 'url']))
        assert cl.results['img'].shape==(214, 49152)
        # Check grayscale parameter with imports
        cl = Clustimage(dim=(128, 128), grayscale=True)
        _ = cl.import_data(X)
        assert cl.results['img'].shape==(214, 16384)

        # Import mnist example
        X = cl.import_example(data='mnist')
        cl = Clustimage()
        _ = cl.import_data(X)
        assert np.all(np.isin([*cl.results.keys()], ['img', 'feat', 'xycoord', 'pathnames', 'labels', 'filenames', 'url']))
        assert cl.results['img'].shape==(1797, 64)
        assert len(cl.results['pathnames'])==X.shape[0]
        assert len(cl.results['filenames'])==X.shape[0]

    def test_extract_feat(self):
        cl = Clustimage(method='pca')
        # Import flowers example
        X = cl.import_example(data='flowers')
        X = cl.import_data(X)
        _ = cl.extract_feat(X)
        assert cl.results['feat'].shape==(X['img'].shape[0], 153)

        # Init with settings such as PCA
        cl = Clustimage(method='hog', verbose=50)
        # load example with flowers
        pathnames = cl.import_example(data='flowers')
        # Cluster flowers
        results = cl.fit_transform(pathnames)
        # Read the unseen image. Note that the find functionality also performs exactly the same preprocessing steps as for the clustering.
        results_find = cl.find(pathnames[0:2], k=0, alpha=0.05)


    def test_embedding(self):
        cl = Clustimage(method='pca')
        # Import flowers example
        X = cl.import_example(data='flowers')
        X = cl.import_data(X)
        Xfeat = cl.extract_feat(X)
        _ = cl.embedding(Xfeat)
        assert cl.results['xycoord'].shape==(X['img'].shape[0], 2)

    def test_embedding(self):
        cl = Clustimage(method='pca')
        # Import flowers example
        X = cl.import_example(data='flowers')
        X = cl.import_data(X)
        Xfeat = cl.extract_feat(X)
        xycoord = cl.embedding(Xfeat)
        labels = cl.cluster()
        assert len(cl.results['labels'])==X['img'].shape[0]

    def test_cluster(self):
        cl = Clustimage()
        X = cl.import_example(data='flowers')
        results = cl.fit_transform(X)
        assert np.all(np.isin([*cl.results.keys()], ['img', 'feat', 'xycoord', 'pathnames', 'filenames', 'labels', 'url']))
        assert len(cl.cluster())==len(X)

        # Parameters combinations to check
        param_grid = {
        	'cluster_space':['high','low'],
        	'cluster':['agglomerative'],
        	'evaluate' : ['silhouette', 'dbindex'],
            'min_clust' : [2, 4, 6],
            'max_clust' : [10, 20, 30],
        	}
        # Make the combinatinos
        allNames = param_grid.keys()
        combinations = list(it.product(*(param_grid[Name] for Name in allNames)))
        for combination in combinations:
            labx = cl.cluster(cluster_space=combination[0], cluster=combination[1], evaluate=combination[2], metric='euclidean', linkage='ward', min_clust=combination[3], max_clust=combination[4])
            assert len(labx)==len(X)

    def save_and_load(self):
        methods = ['pca', 'hog', 'pca-hog']
        for method in methods:
            cl = Clustimage(method=method)
            # Init
            pathnames = cl.import_example(data='mnist')
            # Cluster flowers
            cl.fit_transform(pathnames)
            cl.save(overwrite=True)
            cl.load()
            assert cl.find(pathnames[0:5], k=10, alpha=0.05)

    def test_find(self):
        cl = Clustimage(method='pca', grayscale=False)
        # load example with flowers
        path_to_imgs = cl.import_example(data='flowers')
        # Extract features (raw images are not stored and handled per-image to save memory)
        results = cl.fit_transform(path_to_imgs, min_clust=10)
        # Check nr. of features
        featshape = cl.results['feat'].shape
        # Predict
        results_find = cl.find(path_to_imgs[0:5], k=None, alpha=0.05)
        assert  cl.results['feat'].shape==featshape

    def test_predict(self):
        # Init
        cl = Clustimage(method='pca', grayscale=True, params_pca={'n_components':14})
        # Load example with faces
        X = cl.import_example(data='flowers')
        # Cluster
        results = cl.fit_transform(X)
        assert np.all(np.isin([*cl.results.keys()], ['img', 'feat', 'xycoord', 'pathnames', 'filenames', 'labels', 'url']))
        
    def test_fit_transform(self):
        # Example data
        cl = Clustimage()
        Xflowers = cl.import_example(data='flowers')
        Xflowers=Xflowers[0:50]
        Xdigits = cl.import_example(data='mnist')
        Xdigits=Xdigits[0:50,:]
        Xfaces = cl.import_example(data='faces')
        Xfaces=Xfaces[0:50,:]

        # Parameters combinations to check
        param_grid = {
        	'method':['ahash', 'pca', 'hog', None],
        	'embedding':['tsne', 'umap', None],
        	'cluster_space' : ['high', 'low'],
        	'grayscale' : [True, False],
            'dim' : [(8,8), (128,128), (256,256)],
            'data' : [Xflowers, Xdigits]
        	}
        # Make the combinatinos
        allNames = param_grid.keys()
        combinations = list(it.product(*(param_grid[Name] for Name in allNames)))
        # Iterate over all combinations
        for i, combination in enumerate(combinations):
            # init
            cl = Clustimage(method=combination[0], embedding=combination[1], grayscale=combination[3], dim=combination[4], verbose=30, params_pca={'n_components':50})
            # Preprocessing and feature extraction
            assert cl.fit_transform(combination[5], cluster_space=combination[2])
