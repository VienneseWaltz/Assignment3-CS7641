import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as EM
from sklearn.metrics import silhouette_score as sil_score, f1_score, homogeneity_score
import time
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix

# Random seed for reproducibility
random_seed = 2021
np.random.seed(random_seed)

def run_EM(X, y, title):

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
        f1_scores.append(f1_score(y, y_mode_vote))
        homogeneity_scores.append(homogeneity_score(y, labels))
        # Refer to https://machinelearningmastery.com/probabilistic-model-selection-measures/
        aic_scores.append(em.aic(X))
        bic_scores.append(em.bic(X))

    # Using elbow plot, we could find the optimal value of k
    fig = plt.figure()
    # Refer to https://stackoverflow.com/questions
    # /3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111
    ax = fig.add_suplot(111)
    ax.plot(kdist, sil_scores)
    plt.grid(True)
    plt.title('Elbow Plot for EM: ' + title)
    plt.xlabel('Number of Distributions')
    plt.ylabel('Average Silhouette Score')
    plt.show()

    # Plot Homogeneity scores
    fig = plt.figure()
    ax = fig.add_suplot(111)
    ax.plot(kclusters, homogeneity_scores)
    plt.grid(True)
    plt.title('Homogeneity Scores for EM: ' + title)
    plt.xlabel('Number of Distributions')
    plt.ylabel('Homogeneity Score')
    plt.show()

    # Plot f1-scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, f1_scores)
    plt.grid(True)
    plt.title('F1-scores for EM: ' + title)
    plt.xlabel('Number of Distributions')
    plt.ylabel('F1-Score')
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
    plt.show()


def evaluate_EM(em, X, Y):
    start_time = time.time()
    em.fit(X, y)
    end_time = time.time()
    elapsed_time = end_time - start_time

    labels = em.predict(X)
    y_mode_vote = cluster_predictions(y, labels)
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
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    plt.show()


# Load the dataset
X1, y1 = load_wine_quality_data('data/wine_quality.csv')
X2, y2 = load_Star3642_balanced_data('data/Star3642_balanced.csv')

# Preprocess the data
std_scaler = preprocessing.StandardScaler()
X1 = std_scaler.fit_transform(X1)
X2 = std.scaler.fit_transform(X2)

# Obtaining the cluster centers for wine_quality dataset
run_EM(X1,y1, 'Wine Quality Data')
em = EM(n_clusters=22, covariance_type='diag', n_init=1, random_state=100, warm_start=True)
evaluate_EM(em, X1, y1)
df = pd.DataFrame(em.means_)
df.to_csv("Wine Quality EM Component Means.csv")

# Obtaining the cluster centers for Star3642 balanced dataset
run_EM(X2, y2, 'Star3642 Balanced Data')
em = EM(n_clusters=42, covariance_type='diag', n_init=1, random_state=100, warm_start=True)
evaluate_EM(em, X2, y2)
df = pd.DataFrame(em.means_)
df.to_csv("Star3642 Balanced EM Component Means.csv")







