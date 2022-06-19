import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale


def vipy(X, y, ncomponents):
    pls = PLSRegression(n_components=ncomponents)
    (xscrs, yscrs) = pls.fit_transform(scale(X,axis=0,with_mean=True,with_std=True), y)
    xw = pls.x_weights_
    b = np.linalg.pinv(xscrs) @ y
    xw = (xw / np.linalg.norm(xw, axis=0)) ** 2
    sz = np.size(X, 1)
    vipscrs = []

    for vrbl in range(sz):
        nmtr = np.sum((b**2) @ xscrs.T @ xscrs @ xw[vrbl, :])
        dmtr = np.sum((b**2) @ xscrs.T @ xscrs)
        vipscrs.append(np.sqrt((sz * nmtr) / dmtr))

    return vipscrs
