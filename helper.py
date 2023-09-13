import matplotlib.pyplot as plt
import itertools
import time
import numpy as np

from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


def plot_learning_curve(classifier, X, Y, title="Insert Title"):

    n = len(y)
    # Training mean and standard deviation
    training_mean = []
    training_std = []

    # Cross-validation mean and standard deviation
    cv_mean = []
    cv_std = []

    # Model fit/training time
    fit_mean = []
    fit_std = []

    # Model testing and prediction times
    pred_mean = []
    pred_std = []  # model test/prediction times
    training_sizes = (np.linspace(.05, 1.0, 20) * n).astype('int')

    for i in training_sizes:
        index = np.random.randint(X.shape[0], size=i)
        X_subset = X[index, :]
        y_subset = y[index]
        scores = cross_validate(classifier, X_subset, y_subset, cv=8, scoring='f1',
                                n_jobs=-1, return_train_score=True)
        training_mean.append(np.mean(scores['train_score']))
        training_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score']))
        cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time']))
        fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time']))
        pred_std.append(np.std(scores['score_time']))

    training_mean = np.array(training_mean)
    training_std = np.array(training_std)
    cv_mean = np.array(cv_mean)
    cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean)
    fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean)
    pred_std = np.array(pred_std)

    # Invoke plot_LC()
    plot_LC(training_sizes, training_mean, training_std, cv_mean, cv_std, title)
    plot_times(training_sizes, fit_mean, fit_std, pred_mean, pred_std)

    return training_sizes, training_mean, fit_mean, pred_mean


def plot_LC(training_sizes, training_mean, training_std, cv_mean, cv_std, title):
    plt.figure()
    plt.title("Learning Curve: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(training_sizes, training_mean - 2 * training_std, train_mean + 2 * training_std, alpha=0.1, color="b")
    plt.fill_between(training_sizes, cv_mean - 2 * cv_std, cv_mean + 2 * cv_std, alpha=0.1, color="r")
    plt.plot(training_sizes, training_mean, 'o-', color="b", label="Training Score")
    plt.plot(training_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()


def plot_times(training_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    plt.figure()
    plt.title("Modeling Time: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(training_sizes, fit_mean - 2 * fit_std, fit_mean + 2 * fit_std, alpha=0.1, color="b")
    plt.fill_between(training_sizes, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.1, color="r")
    plt.plot(training_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(training_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.BuPu):

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


def final_classifier_evaluation(classifier, X_train, X_test, y_train, y_test):

    start_time = time.time()
    classifier.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    start_time = time.time()
    y_pred = classifier.predict(X_test)
    end_time = time.time()
    pred_time = end_time - start_time

    # Evaluating accuracy of the classification model
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.5f}".format(training_time))
    print("Model Prediction Time (s): " + "{:.5f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy) + "     AUC:       " + "{:.2f}".format(auc))
    print("Precision: " + "{:.2f}".format(precision) + "     Recall:    " + "{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    plt.show()


def cluster_predictions(Y, clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)

    # Refer to https://stackoverflow.com/questions
    # /47216388/efficiently-creating-masks-numpy-python
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]

        # find all the elements with equal counts
        # Refer to https://stackoverflow.com/questions/57595431
        # /find-all-the-elements-with-equal-counts-with-counter-most-common
        target = Counter(sub).most_common(1)[0][0]
        return pred


def pairwiseDistCorr(X1, X2):
    '''
    Compute the distance matrix between vector X1 and X2
    :param X1: features from dataset1
    :param X2: features from dataset2
    :return: a distance matrix
    '''

    d1 = pairwise_distanes(X1)
    d2 = pairwise_distances(X2)
    # Get a contiguous flattened-out array for both d1 and d2. Then find the correlation of
    # the first flattened-out array of d1 and the second flattened-out array of d2.
    # Refer to https://stackoverflow.com/questions/61557443
    # /numpy-corrcoef-doubts-about-return-value
    return np.corrcoef(d1.ravel(), d2.ravel())[0,1]


#####################################################
# Building a feedforward Neural Network Classifier or a Multi-layered
# Network of Neurons (MLN)
######################################################
def ffNN(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    hlist = np.linspace(1, 150, 30).astype('int')
    for i in hlist:
        classifier = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic',
                                   learning_rate_init=0.05, random_state=100)
        classifier.fit(X_train, y_train)
        y_pred_test = classifier.predict(X_test)
        y_pred_train = classifier.predict(X_train)
        f1_test.append(accuracy_score(y_test, y_pred_test))
        f1_train.append(accuracy_score(y_train, y_pred_train))

    plt.plot(hlist, f1_train, 'o-', color = 'b', label='Train Accuracy')
    plt.plot(hlist, f1_test, 'o-', color = 'g', label='Test Accuracy')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Accuracy of the Model')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def NNGridSearchCV(X_train, y_train):
    num_hidden_units = [5, 10, 30, 50, 70, 75, 80, 100]
    params_to_tune = {'hidden_layer_sizes': num_hidden_units}
    grid_nn = GridSearchCV(estimator=MLPClassifier(solver='adam',activation='logistic',learning_rate_init=0.05,random_state=100),
                           param_grid=params_to_tune, cv=8)
    grid_nn.fit(X_train, y_train)

    best_nn_params = grid_nn.best_params_
    print(f'The best neural network parameters for {dataset_name} dataset found are: {best_nn_params}')
    print()
    return best_nn_params['hidden_layer_size']










