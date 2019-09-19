import random
import numpy as np
import pandas as pd
import seaborn as sns
import krippendorff as ka
import matplotlib.pyplot as plt

from scipy import stats
from nltk import agreement
from sklearn import metrics
from sklearn.cluster import DBSCAN

np.random.seed(0)
sns.set()

write_distances = False
blockchain = 'ARK'  # ARK|Lisk
timeunit = 'month'  # block|day|week|month
randomness = ''  # Real|Rand|RandLess|RandLessHigh|RandLessMed|EMPTY_STRING


# '' (EMPTY_STRING), Real data set
# 'Rand', Random validation list (51) sampled from the list of unique validators in the ARK dataset (162)
# 'RandLess', Random validation list (51) sampled from a part of the list of unique validators in the ARK dataset (70)
# 'RandLessMed', Random validation list (51) sampled from the list of all unique validators in the ARK dataset (100)
# 'RandLessHigh', Random validation list (51) sampled from the 2x list of all unique validators in the ARK dataset (324)

# epss = np.arange(5, 100, 2) / 100
# mins = range(5, 200, 2)

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)


def clustering_metric_study(case_, epss, mins):
    print(case_)

    # read the data
    data_raw = pd.read_csv("data/" + case_ + ".csv")
    data_vectors = {}
    count = 0

    # flatten the data per timeunit
    for timeunit_ in data_raw['timeunit'].unique():
        data_vectors[count] = list(data_raw[data_raw['timeunit'] == timeunit_]['name'])
        count += 1

    # precompute the distance matrix between the vectors
    distances_str = ''
    distances = []
    random.seed = 10
    for vector1 in data_vectors:
        dist_temp = []
        for vector2 in data_vectors:
            dist = 1 - jaccard_similarity(data_vectors[vector1], data_vectors[vector2])
            dist_temp.append(dist)
            if write_distances:
                distances_str += str(vector1) + ',' + str(vector2) + ',' + str(dist) + '\n'

        distances.append(dist_temp)

    if write_distances:
        f = open('output/distances.csv', 'w')
        f.write(distances_str)
        f.close()

    # create dataframes to host the output
    df_silho = pd.DataFrame(columns=epss, index=mins)
    df_clust = pd.DataFrame(columns=epss, index=mins)
    df_noise = pd.DataFrame(columns=epss, index=mins)

    # loop over the dbscan configuration combinations
    for eps_r in epss:
        for min_r in mins:
            # cluster using the precomputed Jaccard distance matrix
            db = DBSCAN(eps=eps_r, min_samples=min_r, metric="precomputed").fit(distances)

            # get the label map (which point belongs to which cluster)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # compute the silhouette of the clustering
            try:
                sil = metrics.silhouette_score(distances, labels)
            except:
                sil = -1

            # number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            df_silho.loc[min_r][eps_r] = np.round(sil, 2)
            df_clust.loc[min_r][eps_r] = n_clusters_
            df_noise.loc[min_r][eps_r] = n_noise_

    # plot the heatmaps
    df_sillo = df_silho[df_silho.columns].astype(float)
    df_clust = df_clust[df_clust.columns].astype(float)
    df_noise = df_noise[df_noise.columns].astype(float)

    fig, (ax, ax1, ax2) = plt.subplots(ncols=3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.2)

    sns.heatmap(df_sillo, cmap='Greens', ax=ax, vmin=-1, vmax=1)
    sns.heatmap(df_clust, cmap='Reds', ax=ax1, vmin=0, vmax=8)
    sns.heatmap(df_noise, cmap='Greys', ax=ax2, vmin=0, vmax=800)

    ax.set_title('Silhouette')
    ax.set_ylabel('Min points')
    ax.set_xlabel('eps')

    ax1.set_title('# of clusters')
    ax1.set_ylabel('Min points')
    ax1.set_xlabel('eps')

    ax2.set_title('# of outliers')
    ax2.set_ylabel('Min points')
    ax2.set_xlabel('eps')

    plt.savefig('res/' + case_ + 'N1.png')
    return


def correlation_metric_study(case_, ):
    print(case_)

    # read the data
    data_raw = pd.read_csv("data/" + case_ + ".csv")
    data_vectors = {}

    # get unique rounds and unique delegates
    rounds_ = data_raw['timeunit'].unique()
    delegates = data_raw['name'].unique()

    print("Loaded {rounds} sample rounds".format(rounds=len(rounds_)))
    print("Rounds contain {delegates} delegates".format(delegates=len(delegates)))

    # flatten the data per timeunit
    for round_ in rounds_:
        data_vectors[round_] = list(data_raw[data_raw['timeunit'] == round_]['name'])

    # create the co-occurrence matrix
    data_new = pd.DataFrame(0, index=rounds_, columns=delegates)
    for round_ in rounds_:
        for name_ in data_vectors[round_]:
            data_new[name_][round_] = 1

    # manual p-value and corr matrix computation
    corr_matrix = pd.DataFrame()  # Correlation matrix
    p_matrix = pd.DataFrame()  # Matrix of p-values
    for x in data_new.columns:
        for y in data_new.columns:
            corr = stats.pearsonr(data_new[x], data_new[y])
            corr_matrix.loc[x, y] = corr[0]
            p_matrix.loc[x, y] = corr[1]

    # Pandas .corr() function computation
    # corr_matrix = data_new.corr()

    # plot the heatmaps
    f, (ax, ax1) = plt.subplots(ncols=2, figsize=(24, 10))
    f.subplots_adjust(wspace=0.2)

    sns.heatmap(corr_matrix,
                # mask = mask,
                ax=ax,
                square=True,
                linewidths=.0,
                cbar_kws={'shrink': 0.8, 'ticks': [-1, -.5, 0, 0.5, 1]},
                vmin=-1,
                vmax=1,
                cmap='YlGnBu_r',
                annot=False,
                annot_kws={"size": 12}
                )

    sns.heatmap(p_matrix,
                # mask = mask,
                ax=ax1,
                square=True,
                linewidths=.0,
                cbar_kws={'shrink': 0.8, 'ticks': [0, 0.05, 0.1, 0.15, 0.2]},
                vmin=0,
                vmax=.2,
                cmap='YlGnBu_r',
                annot=False,
                annot_kws={"size": 12}
                )

    ax.set_title('Correlation')
    ax1.set_title('p-values')
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

    plt.savefig('res/' + case_ + '.png')
    return


def krippendorff_metric_study(case_, additional_measures):
    data_raw = pd.read_csv("data/" + case_ + ".csv")
    data_vectors = {}

    rounds_ = data_raw['timeunit'].unique()
    delegates = data_raw['name'].unique()

    print("Loaded {rounds} sample rounds".format(rounds=len(rounds_)))
    print("Rounds contain {delegates} delegates".format(delegates=len(delegates)))

    data_new = pd.DataFrame(0, index=rounds_, columns=delegates)
    for round_ in rounds_:
        data_vectors[round_] = list(data_raw[data_raw['year'] == round_]['name'])

    for round_ in rounds_:
        for name_ in data_vectors[round_]:
            data_new[name_][round_] = 1

    alpha = ka.alpha(data_new)
    print("k'aplpha " + str(alpha))

    # TODO: test this
    if additional_measures:
        data_new_nltk = []
        for round_ in rounds_:
            for name_ in delegates:
                data_new_nltk.append([round_, name_, data_new[name_][round_]])

        ratingtask = agreement.AnnotationTask(data=data_new_nltk)
        print("kappa " + str(ratingtask.kappa()))
        print("fleiss " + str(ratingtask.multi_kappa()))
        print("scotts " + str(ratingtask.pi()))

    return alpha
