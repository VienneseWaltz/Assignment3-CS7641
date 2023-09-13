################################
# Implementation of the 4 dimensionality reduction techniques on the wine_quality dataset and the
# Star3642_balanced dataset. The 4 dimensionality reduction techniques featured here are:
# (1) Principal Component Analysis (PCA)
# (2) Independent Component Analysis (ICA)
# (3) Random Component Analysis (RCA)
# (4) Random Forest Classifier (RFC)
#################################
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.random_projection import SparseRandomProjection as RCA



def pairwiseDistCorr(X1, X2):
    """
    Compute the distance matrix between vector X1 and X2
    :param X1: features from dataset1
    :param X2: features from dataset2
    :return: a distance matrix
    """

    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    # Get a contiguous flattened-out array for both d1 and d2. Then find the correlation of
    # the first flattened-out array of d1 and the second flattened-out array of d2.
    # Refer to https://stackoverflow.com/questions/61557443
    # /numpy-corrcoef-doubts-about-return-value
    return np.corrcoef(d1.ravel(), d2.ravel())[0,1]



def run_PCA(X, y, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    pca = PCA(random_state=5).fit(X)  # for all components
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(pca.explained_variance_ratio_))), cum_var, 'b-')
    ax1.set_xlabel('Principal Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'm-')
    ax2.set_ylabel('Eigenvalues', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("PCA Explained Variance and Eigenvalues: " + title)
    fig.tight_layout()
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/pca_results.png' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()



def run_ICA(X, y, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    dims = list(np.arange(2, (X.shape[1] - 1), 3))
    dims.append(X.shape[1])
    ica = ICA(random_state=5)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: " + title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/ica_results.png' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()




def run_RCA(X, y, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    dims = list(np.arange(2, (X.shape[1] - 1), 3))
    dims.append(X.shape[1])
    tmp = defaultdict(dict)

    for i, dim in product(range(5), dims):
        rp = RCA(random_state=i, n_components=dim)
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    tmp = pd.DataFrame(tmp).T
    mean_recon = tmp.mean(axis=1).tolist()
    std_recon = tmp.std(axis=1).tolist()

    fig, ax1 = plt.subplots()
    ax1.plot(dims, mean_recon, 'b-')
    ax1.set_xlabel('Random Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean Reconstruction Correlation', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(dims, std_recon, 'm-')
    ax2.set_ylabel('STD Reconstruction Correlation', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("Random Components for 5 Restarts: " + title)
    fig.tight_layout()
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/rca_results.png' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


def run_RFC(X,y, df_original):
    # Adapted from https://github.com/kylewest520
    # /CS-7641---Machine-Learning/
    rfc = RFC(n_estimators=500,min_samples_leaf=round(len(X)*.01),random_state=5)
    imp = rfc.fit(X,y).feature_importances_
    imp = pd.DataFrame(imp,columns=['Feature Importance'],index=df_original.columns)
    imp.sort_values(by=['Feature Importance'],inplace=True,ascending=False)
    imp['Cum Sum'] = imp['Feature Importance'].cumsum()
    imp = imp[imp['Cum Sum']<=0.95]
    top_cols = imp.index.tolist()
    return imp, top_cols


###########################################
# Functions that will plot
###########################################

def compare_fit_time(n, full_fit, pca_fit, ica_fit, rca_fit, rfc_fit, title):
    plt.figure()
    plt.title("Model Training Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Training Time (s)")
    plt.plot(n, full_fit, '-', color="m", label="Full Dataset")
    plt.plot(n, pca_fit, '-', color="g", label="PCA")
    plt.plot(n, ica_fit, '-', color="r", label="ICA")
    plt.plot(n, rca_fit, '-', color="b", label="RCA")
    plt.plot(n, rfc_fit, '-', color="k", label="RFC")
    plt.legend(loc="best")
    plt.show()


def compare_pred_time(n, full_pred, pca_pred, ica_pred, rca_pred, rfc_pred, title):
    plt.figure()
    plt.title("Model Prediction Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Prediction Time (s)")
    plt.plot(n, full_pred, '-', color="m", label="Full Dataset")
    plt.plot(n, pca_pred, '-', color="g", label="PCA")
    plt.plot(n, ica_pred, '-', color="r", label="ICA")
    plt.plot(n, rca_pred, '-', color="b", label="RCA")
    plt.plot(n, rfc_pred, '-', color="k", label="RFC")
    plt.legend(loc="best")
    plt.show()


def compare_learn_time(n, full_learn, pca_learn, ica_learn, rca_learn, rfc_learn, title):
    plt.figure()
    plt.title("Model's Accuracy: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.plot(n, full_learn, '-', color="m", label="Full Dataset")
    plt.plot(n, pca_learn, '-', color="g", label="PCA")
    plt.plot(n, ica_learn, '-', color="r", label="ICA")
    plt.plot(n, rca_learn, '-', color="b", label="RCA")
    plt.plot(n, rfc_learn, '-', color="k", label="RFC")
    plt.legend(loc="best")
    plt.show()

