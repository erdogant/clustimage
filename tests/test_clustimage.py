from clustimage import Clustimage
import itertools as it

def test_fit_transform():
    # Example data
    cl = Clustimage()
    X = cl.import_example(data='flowers')
    Xdigits = cl.import_example(data='digits')
    data = [X, Xdigits]
    # Parameters combinations to check
    param_grid = {
    	'method':['pca', 'hog', None],
    	'embedding':['tsne', None],
    	'cluster_space' : ['high', 'low'],
    	'grayscale' : [True, False],
        'dim' : [(8,8), (128,128), (256,256)],
        'data' : [X, Xdigits]
    	}
    # Make the combinatinos
    allNames = param_grid.keys()
    combinations = list(it.product(*(param_grid[Name] for Name in allNames)))
    # Iterate over all combinations
    for combination in combinations:
        # init
        cl = Clustimage(method=combination[0], embedding=combination[1], cluster_space=combination[2], grayscale=combination[3], dim=combination[4], params_pca={'n_components':50, 'detect_outliers':None})
        # Preprocessing and feature extraction
        assert cl.fit_transform(combination[5])

def test_cluster():
    cl = Clustimage()
    X = cl.import_example(data='flowers')
