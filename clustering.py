import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as EM
from sklearn.metrics import silhouette_score as sil_score, f1_score, homogeneity_score
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix
import time
from collections import Counter
import seaborn as sns


# Random seed for reproducibility
random_seed = 2021
np.random.seed(random_seed)


def cluster_predictions(Y, cluster_labels):
    assert (Y.shape == cluster_labels.shape)
    pred = np.empty_like(Y)

    # Refer to https://stackoverflow.com/questions
    # /47216388/efficiently-creating-masks-numpy-python
    for label in set(cluster_labels):
        mask = cluster_labels == label
        sub = Y[mask]

        # find all the elements with equal counts
        # Refer to https://stackoverflow.com/questions/57595431
        # /find-all-the-elements-with-equal-counts-with-counter-most-common
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return pred


def run_kmeans(X, y, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    kclusters = list(np.arange(2, 100, 2))

    # Initializaing sil_scores, f1_scores, and  homogeneity_scores and training_times
    sil_scores = []
    f1_scores = []
    homogeneity_scores = []
    training_times = []

    for k in kclusters:
        start_time = time.time()
        km = KMeans(n_clusters=k, n_init=15, random_state=random_seed).fit(X)
        end_time = time.time()
        elapsed_time = end_time - start_time
        training_times.append(elapsed_time)
        sil_scores.append(sil_score(X, km.labels_))
        y_mode_vote = cluster_predictions(y, km.labels_)
        # Refer to https://scikit-learn.org/stable/modules
        # /generated/sklearn.metrics.precision_recall_fscore_support.html
        f1_scores.append(f1_score(y, y_mode_vote, average='micro'))
        homogeneity_scores.append(homogeneity_score(y, km.labels_))

    # Plot silhouette scores
    fig = plt.figure()
    # Refer to https://stackoverflow.com/questions
    # /3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111
    ax = fig.add_subplot(111)
    ax.plot(kclusters, sil_scores)
    plt.grid(True)
    plt.title('Silhouette Scores for KMeans: ' + title, fontweight="bold")
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/silhouette_scores_for_kmeans' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


    # Plot Homogeneity scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, homogeneity_scores)
    plt.grid(True)
    plt.title('Homogeneity Scores for kMeans: ' + title, fontweight="bold")
    plt.xlabel('Number of Clusters')
    plt.ylabel('Homogeneity Score')
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/homogeneity_scores_for_kmeans' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


    # Plot f1-scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, f1_scores)
    plt.grid(True)
    plt.title('F1-scores for kMeans: ' + title)
    plt.xlabel('Number of Clusters')
    plt.ylabel('F1-Score')
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/f1-scores_for_kmeans' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


    # Plot model's training time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, training_times)
    plt.grid(True)
    plt.title('Training Time for KMeans: ' + title)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Training Time(secs)')
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/f1-scores_for_kmeans' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()

def find_inertia(X, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    k_grid = np.arange(2, 100)
    loss = np.zeros(k_grid.size)
    for idx, k in enumerate(k_grid):
        kmeans = KMeans(n_clusters=k, random_state=random_seed)
        kmeans.fit(X)
        loss[idx] = kmeans.inertia_

    # Plot loss vs. k curve
    plt.figure()
    plt.plot(k_grid, loss)
    plt.title('Dataset ' + name)
    plt.xlabel('k')
    plt.ylabel('Loss')
    plt.grid()
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/inertia_plot for kmeans' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Greens, saveFig=False):
    name = title.lower().replace(' ', '_')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/confusion_matrix_plot' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()




########################################
# Evaluate KMeans
########################################
def evaluate_kmeans(km, X, y, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    start_time = time.time()
    km.fit(X, y)
    end_time = time.time()
    elapsed_time = end_time - start_time

    y_mode_vote = cluster_predictions(y, km.labels_)
    auc_y = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)

    print("Model Evaluation Metrics Using Mode Cluster Vote")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.4f}".format(elapsed_time))
    print("No. Iterations to Converge: {}".format(km.n_iter_))
    print("F1 Score:  " + "{:.4f}".format(f1))
    print("Accuracy:  " + "{:.4f}".format(accuracy) + "     AUC:       " + "{:.4f}".format(auc_y))
    print("Precision: " + "{:.4f}".format(precision) + "     Recall:    " + "{:.4f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix', saveFig=saveFig)
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/confusion_matrix' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


################################################################
# Run Expectiation-Maximization algorithm
#################################################################
def run_EM(X, y, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    kdist = list(np.arange(2, 100, 5))

    # Initializaing sil_scores, f1_scores, and  homogeneity_scores and training_times
    sil_scores = []
    f1_scores = []
    homogeneity_scores = []
    training_times = []
    aic_scores = []
    bic_scores = []

    for k in kdist:
        start_time = time.time()
        em = EM(n_components=k, covariance_type='diag',n_init=1,warm_start=True,random_state=100).fit(X)
        end_time = time.time()
        elapsed_time = end_time - start_time
        training_times.append(elapsed_time)

        labels = em.predict(X)
        sil_scores.append(sil_score(X, labels))
        y_mode_vote = cluster_predictions(y, labels)
        f1_scores.append(f1_score(y, y_mode_vote, average='micro'))
        homogeneity_scores.append(homogeneity_score(y, labels))
        # Refer to https://machinelearningmastery.com/probabilistic-model-selection-measures/
        aic_scores.append(em.aic(X))
        bic_scores.append(em.bic(X))

    # Using silhouette scores, we could find the optimal value of k
    fig = plt.figure()
    # Refer to https://stackoverflow.com/questions
    # /3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111
    ax = fig.add_subplot(111)
    ax.plot(kdist, sil_scores)
    plt.grid(True)
    plt.title('Silhouette Scores for EM: ' + title)
    plt.xlabel('Number of Distributions')
    plt.ylabel('Average Silhouette Score')
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/silhouette_scores_for_EM' + '_' + name + '.png', format='png', dpi=120)
        plt.close()
    else:
        plt.show()


    # Plot Homogeneity scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, homogeneity_scores)
    plt.grid(True)
    plt.title('Homogeneity Scores for EM: ' + title)
    plt.xlabel('Number of Distributions')
    plt.ylabel('Homogeneity Score')
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/homogeneity_scores_for_EM' + '_' + name + '.png', format='png', dpi=120)
        plt.close()
    else:
        plt.show()



    # Plot f1-scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, f1_scores)
    plt.grid(True)
    plt.title('F1-scores for EM: ' + title)
    plt.xlabel('Number of Distributions')
    plt.ylabel('F1-Score')
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/f1-scores_for_EM' + '_'  + name + '.png', format='png', dpi=120)
        plt.close()
    else:
        plt.show()


    # Plot model's AIC and BIC
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, aic_scores, label='AIC')
    ax.plot(kdist, bic_scores, label='BIC')
    plt.grid(True)
    plt.title('Model Complexity of EM: ' + title)
    plt.xlabel('Number of Distributions')
    plt.ylabel('Model Complexity Score ')
    plt.legend(loc="best")
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/model_complexity_for_EM' + '_' + name + '.png', format='png', dpi=120)
        plt.close()
    else:
        plt.show()



################################################
# Evaluate outcomes of Expectation-Maximization algorithm
###################################################
def evaluate_EM(em, X, y, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    start_time = time.time()
    em.fit(X, y)
    end_time = time.time()
    elapsed_time = end_time - start_time

    labels = em.predict(X)
    y_mode_vote = cluster_predictions(y, labels)
    auc_y = roc_auc_score(y, y_mode_vote)
    # f1 = f1_score(y, y_mode_vote, average='micro')
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)

    # Adapted from https://github.com/kylewest520/CS-7641---Machine-Learning/blob/master
    # /Assignment%203%20Unsupervised%20Learning/CS%207641%20HW3%20Code.ipynb

    print("Model Evaluation Metrics Using Mode Cluster Vote")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.4f}".format(elapsed_time))
    print("No. of iterations to Converge: {}".format(em.n_iter_))
    print("Log-likelihood Lower Bound: {:.2f}".format(em.lower_bound_))
    print("F1 Score:  " + "{:.4f}".format(f1))
    print("Accuracy:  " + "{:.4f}".format(accuracy) + "     AUC:       " + "{:.4f}".format(auc_y))
    print("Precision: " + "{:.4f}".format(precision) + "     Recall:    " + "{:.4f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix', saveFig=saveFig)
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/model_complexity_for_EM' + '_' + name + '.png', format='png', dpi=120)
        plt.close()
    else:
        plt.show()


def add_clusters(X, km_labels, em_labels):
    df = pd.DataFrame(X)
    df['KMeans Cluster'] = km_labels
    df['EM Cluster'] = em_labels
    col_1hot = ['KMeans Cluster', 'EM Cluster']
    df_1hot = df[col_1hot]
    df_1hot = pd.get_dummies(df_1hot).astype('category')
    df_others = df.drop(col_1hot, axis=1)
    df = pd.concat([df_others, df_1hot], axis=1)
    new_X = np.array(df.values, dtype='int64')

    return new_X

def plot_snspairplot(X, y, km_labels, em_labels, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    df = pd.DataFrame(X)
    df['KMeans Cluster'] = km_labels
    df['EM Cluster'] = em_labels
    df['target'] = y
    sns.pairplot(df, vars=['KMeans Cluster', 'EM Cluster'], hue='target', markers=["o", "s"])
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/snspairplot' + '_' + name + '.png', format='png', dpi=120)
        plt.close()
    else:
        plt.show()









