import pandas as pd
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, pairwise_distances
from sklearn.neural_network import MLPClassifier
from clustering import plot_confusion_matrix


def load_wine_quality_data(filename):
    """
    Loading the wine_quality.csv file
    :param filename: path to the csv file
    :return: X (data) and y (labels)
    """
    data = pd.read_csv(filename)
    data.loc[(data.quality == 'good'), 'quality'] = 1
    data.loc[(data.quality == 'bad'), 'quality'] = 0

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].values.astype(int)

    return X, y

def load_Star3642_balanced_data(filename):
    """
    Loading the Star3642_balanced.csv file
    :param filename: path to the csv file
    :return: X (data) and y (labels)
    """

    # For this dataset, if the target column shows '1', it is a Giant star
    # and a '0' indicates a Dwarf star
    data = pd.read_csv(filename)

    # Map column 'SpType' from string to integer.
    mapping = {k: v for v, k in enumerate(data.SpType.unique())}
    data['SpType'] = data.SpType.map(mapping)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].values.astype(int)

    return X, y

def plot_learning_curve(classifier, X, y, title="Insert Title"):

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
    plot_LC(training_sizes, training_mean, training_std, cv_mean, cv_std, title, saveFig=False)
    plot_times(training_sizes, fit_mean, fit_std, pred_mean, pred_std, title, saveFig=False)

    return training_sizes, training_mean, fit_mean, pred_mean


def plot_LC(training_sizes, training_mean, training_std, cv_mean, cv_std, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    plt.figure()
    plt.title("Learning Curve: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(training_sizes, training_mean - 2 * training_std, training_mean + 2 * training_std, alpha=0.1, color="b")
    plt.fill_between(training_sizes, cv_mean - 2 * cv_std, cv_mean + 2 * cv_std, alpha=0.1, color="r")
    plt.plot(training_sizes, training_mean, 'o-', color="b", label="Training Score")
    plt.plot(training_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/learning_curve' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()



def plot_times(training_sizes, fit_mean, fit_std, pred_mean, pred_std, title, saveFig=False):
    name = title.lower().replace(' ', '_')
    plt.figure()
    plt.title("Modeling Time: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(training_sizes, fit_mean - 2 * fit_std, fit_mean + 2 * fit_std, alpha=0.1, color="b")
    plt.fill_between(training_sizes, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.1, color="r")
    plt.plot(training_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(training_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/modeling_time' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()



def final_classifier_evaluation(classifier, X_train, X_test, y_train, y_test, saveFig=False):
    start_time = time.time()
    classifier.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    start_time = time.time()
    y_pred = classifier.predict(X_test)
    end_time = time.time()
    pred_time = end_time - start_time

    # Evaluating accuracy of the classification model
    # Adapted from https://github.com/kylewest520/CS-7641---Machine-Learning/blob/master
    # /Assignment%203%20Unsupervised%20Learning/CS%207641%20HW3%20Code.ipynb
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("              Model Evaluation Metrics               ")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.5f}".format(training_time))
    print("Model Prediction Time (s): " + "{:.5f}\n".format(pred_time))
    print("F1 Score:  " + "{:.4f}".format(f1))
    print("Accuracy:  " + "{:.4f}".format(accuracy) + "     AUC:       " + "{:.4f}".format(auc))
    print("Precision: " + "{:.4f}".format(precision) + "     Recall:    " + "{:.4f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/final_classifier_evaluation.png', fomrat='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()



#####################################################
# Building a feedforward Neural Network Classifier or a Multi-layered
# Network of Neurons (MLN)
######################################################
def ffNN(X_train, y_train, X_test, y_test, title, saveFig=False):
    name = title.lower().replace(' ', '_')
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

    plt.plot(hlist, f1_train, 'o-', color='b', label='Train Accuracy')
    plt.plot(hlist, f1_test, 'o-', color='g', label='Test Accuracy')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Accuracy of the Model')

    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/feedforward_neural_network' + '_' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


def NNGridSearchCV(X_train, y_train):
    num_hidden_units = [5, 10, 30, 50, 70, 75, 80, 100]
    params_to_tune = {'hidden_layer_sizes': num_hidden_units}
    grid_nn = GridSearchCV(estimator=MLPClassifier(solver='adam',activation='logistic',learning_rate_init=0.05,random_state=100),
                           param_grid=params_to_tune, cv=8)
    grid_nn.fit(X_train, y_train)

    best_nn_params = grid_nn.best_params_
    print(f'The best neural network parameters found are: {best_nn_params}')
    print()
    return best_nn_params['hidden_layer_size']


