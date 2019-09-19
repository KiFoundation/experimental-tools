import csv
import pprint
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

k = 7.69230769 * 10 ** -7

# total_ = 'sum'
total_ = 'max'
# total_ = 'sup'

sep_plots = 1
stake = 1
age = 0
hist = 1

# create a color palette
palette = plt.get_cmap('Set1')

# epochs = (i * 10 ** exp for exp in range(1, 4) for i in range(1, 2))
epochs = range(1000000, 5100000, 500000)


def plot_voter_balances():
    p = 1
    total = [0, 0]

    for i in epochs:
        data = {}

        with open('output/' + str(int(i)) + '.csv', mode='r') as csv_file:
            reader = csv.reader(csv_file)
            line_count = 0
            for row in reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    data[row[0]] = []
                    data[row[0]].append(float(row[1]) * (1 - np.exp(-k * (i - float(row[2])))))
                    data[row[0]].append(float(row[1]))
                    line_count += 1

        if total_ == 'max':
            total[0] = max([bal[0] for bal in data.values()])
            total[1] = max([bal[1] for bal in data.values()])

        if total_ == 'sum':
            total[0] = sum([bal[0] for bal in data.values()])
            total[1] = sum([bal[1] for bal in data.values()])

        if total_ == 'sup':
            total = 125000000

        if not sep_plots:
            if stake:
                sns.distplot([bal[0] / total[0] for bal in data.values()], kde=True, hist=False,
                             label=str(i) + " - Stake")
            if age:
                sns.distplot([bal[1] / total[1] for bal in data.values()], kde=True, hist=False,
                             label=str(i) + " - Age")

        if sep_plots:
            if not hist:
                ax = plt.subplot(4, 4, p)
                if stake:
                    sns.distplot([bal[0] / total[0] for bal in data.values()], kde=True, hist=False, label="Stake")
                if age:
                    sns.distplot([bal[1] / total[1] for bal in data.values()], kde=True, hist=False, label="Age")

                plt.title(i, fontsize=10, y=0.8, fontweight=0)
                p += 1

            if hist:
                ax = plt.subplot(4, 4, p)
                if stake:
                    plt.hist([bal[0] / total[0] for bal in data.values()], label="Stake", bins=100)
                if age:
                    plt.hist([bal[1] / total[1] for bal in data.values()], label="Age", bins=100)

                plt.title(i, fontsize=10, y=0.8, fontweight=0)
                p += 1


def plot_delegate_stakes():
    p = 1
    total = 0
    top = 200

    for i in epochs:
        data = {}

        with open('output_del/' + str(int(i)) + '.csv', mode='r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)
            for row in reader:
                data[row[0]] = float(row[1])

        if total_ == 'max':
            total = max([bal for bal in data.values()])

        if total_ == 'sum':
            total = sum([bal for bal in data.values()])

        if total_ == 'sup':
            total = 125000000

        data_sorted = sorted(data.values(), reverse=True)
        print(len(data_sorted) - data_sorted.count(0))

        if not sep_plots:
            sns.distplot([bal / total for bal in data_sorted[:top]], kde=True, hist=False, label=str(i) + " - Stake")

        if sep_plots:
            if not hist:
                ax = plt.subplot(3, 3, p)
                sns.distplot([bal / total for bal in data_sorted[:top]], kde=True, hist=False, label="Stake")
                plt.title(i, fontsize=10, y=0.8, fontweight=0)
                p += 1

            if hist:
                ax = plt.subplot(3, 3, p)
                plt.hist([bal / total for bal in data_sorted[:top]], label="Stake", bins=50)
                plt.title(i, fontsize=10, y=0.8, fontweight=0)
                p += 1


plot_delegate_stakes()
plt.show()
