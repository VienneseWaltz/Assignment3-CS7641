from clustering import run_kmeans, evaluate_kmeans, run_EM, evaluate_EM, add_clusters, find_inertia, plot_snspairplot
from dimensionality_reduction import run_PCA, run_ICA, run_RCA, run_RFC, \
    compare_fit_time, compare_pred_time, compare_learn_time
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture as EM
from util import final_classifier_evaluation, plot_learning_curve, load_wine_quality_data, load_Star3642_balanced_data
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import SparseRandomProjection as RCA

import numpy as np
from sklearn import preprocessing


def main(saveFig = False):
    # Load the wine quality dataset
    X1, y1 = load_wine_quality_data('data/wine_quality.csv')

    #######
    # Standardize the features by removing the mean and scaling to unit
    # variance. fit_transform() used on std_scaler scales the training data
    # and learns the scaling parameters of that data.
    ######
    X1_orig = X1.copy()
    std_scaler = preprocessing.StandardScaler()
    X1 = std_scaler.fit_transform(X1)


    # Load the Star3642 balanced dataset
    X2, y2 = load_Star3642_balanced_data('data/Star3642_balanced.csv')
    
    # Standardize the features of Star3642 balanced dataset
    # std_scaler = preprocessing.StandardScaler()
    X2_orig = X2.copy()
    X2 = std_scaler.fit_transform(X2)


    ###################################################################
    # Step 1a): Run Clustering algorithms (k-Means and EM) on wine quality dataset
    ###################################################################
    run_kmeans(X1, y1, 'Wine Quality Data', saveFig=saveFig)
    find_inertia(X1, 'Wine Quality Data', saveFig=saveFig)
    # Found the optimal k to be 20 from above. Using k=20 for evaluating k-Means
    evaluate_kmeans(KMeans(n_clusters=20, n_init=10, random_state=100), X1, y1, 'Wine Quality Data', saveFig=saveFig)
    km = KMeans(n_clusters=20, n_init=12, random_state=100).fit(X1)
    km_labels = km.labels_

    run_EM(X1, y1, 'Wine Quality Data', saveFig=saveFig)
    # plot_BIC(X1, 'Wine Quality Data', saveFig=saveFig)
    evaluate_EM(EM(n_components=98, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), X1, y1, 'Wine Quality Data', saveFig=saveFig)
    em = EM(n_components=98, covariance_type='diag', n_init=1, random_state=100, warm_start=True).fit(X1)
    em_labels = em.predict(X1)

    plot_snspairplot(X1, y1, km_labels, em_labels, 'Wine Quality Data',saveFig=saveFig)

    ###################################################################
    # Step 1b): Run Clustering algorithms (k-Means and EM) on Star3642 balanced dataset
    ###################################################################
    #X_train, X_test, y_train, y_test = train_test_split(np.array(X2), np.array(y2), test_size=0.2)
    run_kmeans(X2, y2, 'Star3642 Data', saveFig=saveFig)
    find_inertia(X2, 'Star3642 Data', saveFig=saveFig)
    # Found the optimal value of k to be 18 from above. Using that to evalutae k-Means
    evaluate_kmeans(KMeans(n_clusters=18, n_init=10, random_state=100), X2, y2, 'Star3642 Data', saveFig=saveFig)
    km = KMeans(n_clusters=18, n_init=12, random_state=100).fit(X2)
    km_labels = km.labels_
    run_EM(X2, y2, 'Star3642 Data', saveFig=saveFig)
    # plot_BIC(X2, 'Star3642 Data', saveFig=saveFig)
    evaluate_EM(EM(n_components=82, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), X2, y2, 'Star3642 Data', saveFig=saveFig)
    em = EM(n_components=82, covariance_type='diag', n_init=1, random_state=100, warm_start=True).fit(X2)
    em_labels = em.predict(X2)
    plot_snspairplot(X2, y2, km_labels, em_labels, 'Star3642 Data', saveFig=saveFig)


    ####################################################################
    # Step 2a): Run dimensionality reduction algorithms on wine quality dataset
    ####################################################################

    run_PCA(X1, y1, "Wine Quality Data", saveFig=saveFig)
    run_ICA(X1, y1, "Wine Quality Data", saveFig=saveFig)
    run_RCA(X1, y1, "Wine Quality Data", saveFig=saveFig)

    # Obtaining the values of the most important features and their corresponding columns from running RFC
    imp_winequality, topcols_winequality = run_RFC(X1, y1, X1_orig)



    #####################################################################
    # Step 2b): Run dimensionality reduction algorithms on the Star3642 dataset
    #####################################################################
    #X_train, X_test, y_train, y_test = train_test_split(np.array(X2), np.array(y2), test_size=0.2)
    #df_star3642 = pd.read_csv('data/Star3642_balanced.csv')
    run_PCA(X2, y2, "Star3642 Balanced Data", saveFig=saveFig)
    run_ICA(X2, y2, "Star3642 Balanced Data", saveFig=saveFig)
    run_RCA(X2, y2, "Star3642 Balanced Data", saveFig=saveFig)

    # Obtaining the values of the most important features and their corresponding columns from running RFC
    imp_star3642, topcols_star3642 = run_RFC(X2, y2, X2_orig)

    #####################################################################################
    # Step 3a): Run dimensionality reduction algorithms (PCA, ICA, RCA and RFC) on the
    #           wine quality dataset. Then re-run clustering experiments (k-Means and EM)
    #           on the dimensionality-reduced dataset.
    ######################################################################################

    # Explained variance is highest, and eigenvalues=10 when principal components=10 (Refer to Fig.  )
    pca_X1 = PCA(n_components=10, random_state=5).fit_transform(X1)

    # Maximum average kurtosis of 12 occurs when independent components=8 (Refer to Fig.  )
    ica_X1 = ICA(n_components=8 , random_state=5).fit_transform(X1)

    # Mean reconstruction correlation is highest at 0.90 when random components=10 (Refer to Fig. )
    rca_X1 = RCA(n_components=10, random_state=5).fit_transform(X1)

    # From Step 2a), we already obtained the most important features. rfc_X1 is simply X1[topcols_winequality]
    rfc_X1 = X1_orig[topcols_winequality]


    run_kmeans(pca_X1, y1, 'PCA Wine Quality Data', saveFig=saveFig)
    run_kmeans(ica_X1, y1, 'ICA Wine Quality Data', saveFig=saveFig)
    run_kmeans(rca_X1, y1, 'RCA Wine Quality Data', saveFig=saveFig)
    run_kmeans(rfc_X1, y1, 'RFC Wine Quality Data', saveFig=saveFig)

    evaluate_kmeans(KMeans(n_clusters=20, n_init=10, random_state=100), pca_X1, y1, 'PCA Wine Quality', saveFig=saveFig)
    evaluate_kmeans(KMeans(n_clusters=20, n_init=10, random_state=100), ica_X1, y1, 'ICA Wine Quality', saveFig=saveFig)
    evaluate_kmeans(KMeans(n_clusters=20, n_init=10, random_state=100), rca_X1, y1, 'RCA Wine Quality', saveFig=saveFig)
    evaluate_kmeans(KMeans(n_clusters=20, n_init=10, random_state=100), rfc_X1, y1, 'RFC Wine Quality', saveFig=saveFig)

    run_EM(pca_X1, y1, 'PCA Wine Quality Data', saveFig=saveFig)
    run_EM(ica_X1, y1, 'ICA Wine Quality Data', saveFig=saveFig)
    run_EM(rca_X1, y1, 'RCA Wine Quality Data', saveFig=saveFig)
    run_EM(rfc_X1, y1, 'RFC Wine Quality Data', saveFig=saveFig)

    evaluate_EM(EM(n_components=10, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), pca_X1, y1, 'PCA Wine Quality', saveFig=saveFig)

    evaluate_EM(EM(n_components=8, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), ica_X1, y1, 'ICA Wine Quality', saveFig=saveFig)

    evaluate_EM(EM(n_components=10, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), rca_X1, y1, 'RCA Wine Quality', saveFig=saveFig)

    evaluate_EM(EM(n_components=82, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), rfc_X1, y1, 'RFC Wine Quality', saveFig=saveFig)


    ###########################################################################################
    # Step 3b): Run dimensionality reduction algorithms on Star3642 Balanced dataset.
    #           Then re-run clustering experiments (k-Means and EM) on this dimensionality-reduced
    #           dataset.
    ############################################################################################
    #X_train, X_test, y_train, y_test = train_test_split(np.array(X2), np.array(y2), test_size=0.2)
    pca_X2 = PCA(n_components=5, random_state=5).fit_transform(X2)
    ica_X2 = ICA(n_components=4, random_state=5).fit_transform(X2)
    rca_X2 = RCA(n_components=5, random_state=5).fit_transform(X2)

    # From Step 2b) above, we already obtained the most important features/corresponding columns. rfc_X2 is simply
    # X2[topcols_star3642]
    rfc_X2 = X2_orig[topcols_star3642]


    run_kmeans(pca_X2, y2, 'PCA Star3642 Data', saveFig=saveFig)
    run_kmeans(ica_X2, y2, 'ICA Star3642 Data', saveFig=saveFig)
    run_kmeans(rca_X2, y2, 'RCA Star3642 Data', saveFig=saveFig)
    run_kmeans(rfc_X2, y2, 'RFC Star3642 Data', saveFig=saveFig)

    evaluate_kmeans(KMeans(n_clusters=18, n_init=10, random_state=100), pca_X2, y2, 'Star3642 Data', saveFig=saveFig)
    evaluate_kmeans(KMeans(n_clusters=18, n_init=10, random_state=100), ica_X2, y2, 'Star3642 Data', saveFig=saveFig)
    evaluate_kmeans(KMeans(n_clusters=18, n_init=10, random_state=100), rca_X2, y2, 'Star3642 Data', saveFig=saveFig)
    evaluate_kmeans(KMeans(n_clusters=98, n_init=10, random_state=100), rfc_X2, y2, 'Star3642 Data', saveFig=saveFig)

    run_EM(pca_X2, y2, 'PCA Star3642 Data', saveFig=saveFig)
    run_EM(ica_X2, y2, 'ICA Star3642 Data', saveFig=saveFig)
    run_EM(rca_X2, y2, 'RCA Star3642 Data', saveFig=saveFig)
    run_EM(rfc_X2, y2, 'RFC Star3642 Data', saveFig=saveFig)

    evaluate_EM(EM(n_components=5, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), pca_X2, y2, 'Star3642 Data', saveFig=saveFig)

    evaluate_EM(EM(n_components=5, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), ica_X2, y2, 'Star3642 Data', saveFig=saveFig)

    evaluate_EM(EM(n_components=5, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), rca_X2, y2, 'Star3642 Data', saveFig=saveFig)

    evaluate_EM(EM(n_components=82, covariance_type='diag', n_init=1,
                   warm_start=True, random_state=100), rfc_X2, y2, 'Star3642 Data', saveFig=saveFig)


    ###############################################################
    # Step 4: Training Neural Network on the newly projected data
    ###############################################################

    # Neural Network is sensitive to stadardization
    std_scaler = preprocessing.StandardScaler()
    X1 = std_scaler.fit_transform(X1)


    # Original dataset before dimensionality reduction
    X_train, X_test, y_train, y_test = train_test_split(np.array(X1), np.array(y1), test_size=0.20)

    clf_nn = MLPClassifier(hidden_layer_sizes=(5,2), random_state=7, max_iter=1000)
    clf_nn.fit(X_train, y_train)
    training_samp, NN_training_score, NN_fit_time, NN_pred_time = plot_learning_curve(clf_nn,
                                                                                      X_train,
                                                                                      y_train,
                                                                                      title='Neural Net Wine')
    final_classifier_evaluation(clf_nn, X_train, X_test, y_train, y_test, saveFig=saveFig)

    #################
    # Neural Net: PCA
    ##################
    X_train, X_test, y_train, y_test = train_test_split(np.array(pca_X1), np.array(y1), test_size=0.20)

    pca_est = MLPClassifier(hidden_layer_sizes=(5,2), random_state=7, max_iter=1000)
    pca_est.fit(X_train, y_train)
    training_samp_pca, NN_training_score_pca, NN_fit_time_pca, NN_pred_time_pca = plot_learning_curve(pca_est,
                                                                                                      X_train,
                                                                                                      y_train,
                                                                                                      title='Neural Net Wine')

    final_classifier_evaluation(pca_est, X_train, X_test, y_train, y_test, saveFig=saveFig)

    #################
    # Neural Net: ICA
    ##################
    X_train, X_test, y_train, y_test = train_test_split(np.array(ica_X1), np.array(y1), test_size=0.20)

    ica_est = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    ica_est.fit(X_train, y_train)
    training_samp_ica, NN_training_score_ica, NN_fit_time_ica, NN_pred_time_ica = plot_learning_curve(ica_est,
                                                                                                      X_train,
                                                                                                      y_train,
                                                                                                      title='Neural Net Wine')
    final_classifier_evaluation(ica_est, X_train, X_test, y_train, y_test, saveFig=saveFig)

    #################
    # Neural Net: RCA
    ##################
    X_train, X_test, y_train, y_test = train_test_split(np.array(rca_X1), np.array(y1), test_size=0.20)

    rca_est = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    rca_est.fit(X_train, y_train)
    training_samp_rca, NN_training_score_rca, NN_fit_time_rca, NN_pred_time_rca = plot_learning_curve(rca_est,
                                                                                                      X_train,
                                                                                                      y_train,
                                                                                                      title='Neural Net Wine')
    final_classifier_evaluation(rca_est, X_train, X_test, y_train, y_test, saveFig=saveFig)


    #################
    # Neural Net: RFC
    ##################
    X_train, X_test, y_train, y_test = train_test_split(np.array(rfc_X1), np.array(y1), test_size=0.20)

    rfc_est = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    rfc_est.fit(X_train, y_train)
    training_samp_rfc, NN_training_score_rfc, NN_fit_time_rfc, NN_pred_time_rfc = plot_learning_curve(rfc_est,
                                                                                                      X_train,
                                                                                                      y_train,
                                                                                                      title='Neural Net Wine')
    final_classifier_evaluation(rfc_est, X_train, X_test, y_train, y_test, saveFig=saveFig)


    ##############
    # Plotting the training times and learning rates of the 4 different NN models above
    ##############
    compare_fit_time(training_samp, NN_fit_time, NN_fit_time_pca, NN_fit_time_ica,
                     NN_fit_time_rca, NN_fit_time_rfc, 'Wine Quality Dataset')
    compare_pred_time(training_samp, NN_pred_time, NN_pred_time_pca, NN_pred_time_ica,
                      NN_pred_time_rca, NN_pred_time_rfc, 'Wine Quality Dataset')
    compare_learn_time(training_samp, NN_training_score, NN_training_score_pca, NN_training_score_ica,
                       NN_training_score_rca, NN_training_score_rfc, 'Wine Quality Dataset')


    #################
    # Step 5: Training Neural Network on the 4 projected datasets with cluster labels
    #################
    km = KMeans(n_clusters=20, n_init=12, random_state=100).fit(X1)
    km_labels = km.labels_
    em = EM(n_components=98, covariance_type='diag', n_init=1, random_state=100, warm_start=True).fit(X1)
    em_labels = em.predict(X1)
    cluster_orig = add_clusters(X1, km_labels, em_labels)
    
    km = KMeans(n_clusters=10, n_init=12, random_state=100).fit(pca_X1)
    km_labels = km.labels_
    em = EM(n_components=25, covariance_type='diag', n_init=1, random_state=100, warm_start=True).fit(pca_X1)
    em_labels = em.predict(pca_X1)
    cluster_pca = add_clusters(pca_X1, km_labels, em_labels)
    
    km = KMeans(n_clusters=10, n_init=12, random_state=100).fit(ica_X1)
    km_labels = km.labels_
    em = EM(n_components=25, covariance_type='diag', n_init=1, random_state=100, warm_start=True).fit(ica_X1)
    em_labels = em.predict(ica_X1)    
    cluster_ica = add_clusters(ica_X1, km_labels, em_labels)
    
    km = KMeans(n_clusters=10, n_init=12, random_state=100).fit(rca_X1)
    km_labels = km.labels_
    em = EM(n_components=25, covariance_type='diag', n_init=1, random_state=100, warm_start=True).fit(rca_X1)
    em_labels = em.predict(rca_X1)    
    cluster_rca = add_clusters(rca_X1, km_labels, em_labels)

    km = KMeans(n_clusters=10, n_init=12, random_state=100).fit(rfc_X1)
    km_labels = km.labels_
    em = EM(n_components=25, covariance_type='diag', n_init=1, random_state=100, warm_start=True).fit(rfc_X1)
    em_labels = em.predict(rfc_X1)    
    cluster_rfc = add_clusters(rfc_X1, km_labels, em_labels)


    ################
    # Original Dataset
    ################
    X_train, X_test, y_train, y_test = train_test_split(cluster_orig, y1, test_size=0.20)

    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    clf_nn.fit(X_train, y_train)
    training_samp, NN_training_score, NN_fit_time, NN_pred_time = plot_learning_curve(clf_nn,
                                                                                      X_train,
                                                                                      y_train,
                                                                                      title="Neural Net Wine Quality with Clusters: Original")
    final_classifier_evaluation(clf_nn, X_train, X_test, y_train, y_test, saveFig=saveFig)


    ################
    # Neural Net with Clusters: PCA
    ################
    X_train, X_test, y_train, y_test = train_test_split(cluster_pca, y1, test_size=0.20)

    pca_est = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    pca_est.fit(X_train, y_train)
    training_samp_pca, NN_training_score_pca, NN_fit_time_pca, NN_pred_time_pca = plot_learning_curve(clf_nn,
                                                                                                      X_train,
                                                                                                      y_train,
                                                                                                      title="Neural Net Wine Quality with Clusters: PCA")

    final_classifier_evaluation(pca_est, X_train, X_test, y_train, y_test, saveFig=saveFig)

    ################
    # Neural Net with Clusters: ICA
    ################
    X_train, X_test, y_train, y_test = train_test_split(cluster_ica, y1, test_size=0.20)

    ica_est = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    ica_est.fit(X_train, y_train)
    training_samp_ica, NN_training_score_ica, NN_fit_time_ica, NN_pred_time_ica = plot_learning_curve(ica_est,
                                                                                                      X_train,
                                                                                                      y_train,
                                                                                                      title="Neural Net Wine Quality with Clusters: ICA")
    final_classifier_evaluation(ica_est, X_train, X_test, y_train, y_test, saveFig=saveFig)



    ################
    # Neural Net with Clusters: RCA
    ################
    X_train, X_test, y_train, y_test = train_test_split(cluster_rca, y1, test_size=0.20)

    rca_est = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    rca_est.fit(X_train, y_train)
    training_samp_rca, NN_training_score_rca, NN_fit_time_rca, NN_pred_time_rca = plot_learning_curve(rca_est,
                                                                                                      X_train,
                                                                                                      y_train,
                                                                                                      title="Neural Net Wine Quality with Clusters: RCA")
    final_classifier_evaluation(rca_est, X_train, X_test, y_train, y_test, saveFig=saveFig)


    ################
    # Neural Net with Clusters: RFC
    ################
    X_train, X_test, y_train, y_test = train_test_split(cluster_rfc, y1, test_size=0.20)

    rfc_est = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    rfc_est.fit(X_train, y_train)
    training_samp_rfc, NN_training_score_rfc, NN_fit_time_rfc, NN_pred_time_rfc = plot_learning_curve(rfc_est,
                                                                                                      X_train,
                                                                                                      y_train,
                                                                                                      title="Neural Net Wine Quality with Clusters: RFC")
    final_classifier_evaluation(rfc_est, X_train, X_test, y_train, y_test, saveFig=saveFig)


    ###########################
    # Evaluating the new datasets with cluster labels added
    ###########################
    compare_fit_time(training_samp, NN_fit_time, NN_fit_time_pca, NN_fit_time_ica,
                     NN_fit_time_rca, NN_fit_time_rfc, 'Wine Quality Dataset')
    compare_pred_time(training_samp, NN_pred_time, NN_pred_time_pca, NN_pred_time_ica,
                      NN_pred_time_rca, NN_pred_time_rfc, 'Wine Quality Dataset')
    compare_learn_time(training_samp, NN_training_score, NN_training_score_pca, NN_training_score_ica,
                       NN_training_score_rca, NN_training_score_rfc, 'Wine Quality Dataset')


if __name__ == '__main__':
    main(saveFig=True)