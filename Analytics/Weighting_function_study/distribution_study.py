import heapq
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# generate random reputation profiles following a Beta distribution
def generate_profiles(num_, weights_):
    data = pd.DataFrame(columns=['s', 'b', 'c'] + ['rep' + str(i) for i in range(len(weights_))], index=range(num_))
    for index, row in data.iterrows():
        # stake
        s = np.random.beta(0.2, 1, 1)[0]
        # behavior
        b = np.random.beta(5, 2, 1)[0]
        # community
        c = np.random.beta(1, 0.2, 1)[0]
        reps = []

        for weight in weights_:
            reps.append(compute_reps([s, b, c], weight))

        data.loc[index] = [s, b, c] + reps

    return data


# compute the aggregated reputation score
def compute_reps(p, w):
    res_reps = np.dot(w, p)
    return res_reps


# Plot the distribution of reputation scores w.r.t. the component weights
def plot_weight_impact(data, weights_):
    colors = sns.color_palette("Paired", 9)

    for i in range(len(weights_)):
        plt.subplot(3, 3, i + 1)
        sns.distplot(list(data['rep' + str(i)]), hist=False, rug=True, color=colors[i], label=round(weights_[i][0], 1))

        sns.distplot(data['s'], hist=False, rug=False, color='grey', label='s', kde_kws={'linestyle': ':'})
        sns.distplot(data['b'], hist=False, rug=False, color='grey', label='b', kde_kws={'linestyle': '--'})
        sns.distplot(data['c'], hist=False, rug=False, color='grey', label='c', kde_kws={'linestyle': '-.'})

    # plt.show()


# generate data file containing round validator list based on an input reputation profile distribution
def generate_rounds(data, rounds_, weights_, eligibility_threshold_, eligible_pool_size_, round_list_size_,
                    write=False):
    selected_validators_per_round = dict(zip(list(range(len(weights_))), [[] for i in range(len(weights_))]))

    for i in range(len(weights_)):
        # eligibility reputation threshold
        data_th = data[data['rep' + str(i)] > eligibility_threshold_]
        validators_dict_threshold = dict(zip(data_th.index, data_th.loc[:, 'rep' + str(i)]))

        # eligible pool size threshold
        validators_dict_keys = heapq.nlargest(eligible_pool_size_, validators_dict_threshold,
                                              key=validators_dict_threshold.get)

        validators_dict = dict(
            zip(validators_dict_keys, [validators_dict_threshold[key] for key in validators_dict_keys]))

        # normalize the reputation to find selection probabilities
        validators_dict_normalized = {}
        total = sum(list(validators_dict.values()))
        for k in validators_dict:
            validators_dict_normalized[k] = validators_dict[k] / total

        for n in range(rounds_):
            # sample the list of validator for each each round
            temp = np.random.choice(list(validators_dict_normalized.keys()), size=round_list_size_, replace=False,
                                    # p=list(validators_dict_normalized.values())
                                    )

            selected_validators_per_round[i].append(list(temp))

        # write the generated data into files
        if write:
            f = open('case' + str(i) + '.csv', 'w')
            f.write("name,value,year,lastValue,rank,dist,diff\n")
            round_counter = 0
            for temp in selected_validators_per_round:
                # formatted to fit the requirements of the decentralisation script
                for j in range(round_list_size_):
                    f.write('v_' + str(temp[j]) + ',0,' + str(round_counter) + ',0,' + str(j) + ',0,' + '0\n')
                round_counter += 1

            f.close()

    return selected_validators_per_round


# select the colluding nodes
def pick_ghettos(pool, colluding_ghettos_, colluding_ghettos_sizes_, key_):
    # Generate colluding ghettos C : lists of colluding validators IDs
    # eligibility reputation threshold
    pool_th = pool[pool['rep' + str(key_)] > 0.6]
    print(key_, pool_th.shape)
    ghettos_ = {}
    for C in range(colluding_ghettos_):
        ghettos_['ghetto_' + str(C)] = list(
            np.random.choice(list(pool_th.index), min(colluding_ghettos_sizes_[C], len(list(pool_th.index))),
                             replace=False))

    return ghettos_


# Study the collusion factor for different pool size and eligibility threshold configuration
def study_collusion(rounds_, round_list_size_, ghettos_):
    ghettos_collusion = dict(zip(ghettos_.keys(), [[] for i in ghettos_.keys()]))

    for round_ in rounds_:
        # Count colluding nodes in each round
        for C in ghettos_.keys():
            ghettos_collusion[C].append(len(set(round_).intersection(set(ghettos_[C]))) / round_list_size_)

    results = pd.DataFrame.from_dict(ghettos_collusion)
    means = results.mean(axis=0)

    return means


# Plot collusion study results
def plot_collusion_study(res_means_res_, rounds_, pool_size_, weights_, eligible_pool_size_, eligibility_threshold_,
                         colluding_ghettos_sizes_):
    # Prepare the plot
    plt.figure()
    colors = sns.color_palette("Paired", 11)
    plt.title('P:' + str(pool_size_) + ' R:' + str(rounds_) + ' E:' + str(eligible_pool_size_) + ' T:' + str(
        eligibility_threshold_))

    plt.xlabel('size of the collusion ghetto')
    plt.ylabel('% of colluding nodes')

    # plot the results of the collusion study
    for key_ in range(len(res_means_res_)):
        sns.lineplot(colluding_ghettos_sizes_, list(res_means_res_[key_].values),
                     label='ws' + str(round(weights_[key_][0], 1)),
                     color=colors[key_], marker="o")


pool_size = 30000
rounds = 2000
round_list_size = 51
weights = [[ws, 1 - ws, 0] for ws in np.arange(0.0, 1.01, 0.2)]
eligibility_thresholds = np.arange(0.0, 1., 0.1)
eligible_pool_size = 500
colluding_ghettos_sizes = range(0, 3501, 500)
colluding_ghettos = len(colluding_ghettos_sizes)

d = generate_profiles(pool_size, weights)

for eligibility_threshold in eligibility_thresholds[:1]:
    val_rounds = generate_rounds(d, rounds, weights, eligibility_threshold, eligible_pool_size, round_list_size)
    res_means = []
    for key in val_rounds:
        ghettos = pick_ghettos(d, colluding_ghettos, colluding_ghettos_sizes, key)
        res_means.append(study_collusion(val_rounds[key], round_list_size, ghettos))

    # plot_weight_impact(d, weights)
    plot_collusion_study(res_means, rounds, pool_size, weights, eligible_pool_size, eligibility_threshold, colluding_ghettos_sizes)

    plt.show()


_sum = 80000000
n = 50000

a = np.random.beta(0.5, 3, n)
sns.distplot([k * _sum for k in a], hist=False, rug=False, color='b', label='a', kde_kws={'linestyle': '--'})

plt.show()
