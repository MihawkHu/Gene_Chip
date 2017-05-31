from sklearn import decomposition
from sklearn import datasets

# usage: pca(n*d np_arr, target dimension)


def pca(arr, ndim):
    if ndim > arr.shape[1]:
        raise Exception('target dimension too high')
    pca = decomposition.PCA(n_components=ndim)
    return pca.fit_transform(arr)


def kernel_pca(arr, ndim):
    if ndim > arr.shape[1]:
        raise Exception('target dimension too high')
    kpca = decomposition.KernelPCA(n_components=ndim)
    return kpca.fit_transform(arr)


def sparse_pca(arr, ndim):
    if ndim > arr.shape[1]:
        raise Exception('target dimension too high')
    spca = decomposition.SparsePCA(n_components=ndim)
    return spca.fit_transform(arr)
