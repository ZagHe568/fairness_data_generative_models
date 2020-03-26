from scipy import stats
from info_gain import info_gain


def calc_entropy(data):
    entropies = []
    for column in data.columns:
        p_data = data[column].value_counts()
        entropies.append(stats.entropy(p_data))
    return entropies


def calc_info_gain(data, target_column):
    info_gains = []
    for column in data.columns:
        if column == target_column:
            continue
        info_gains.append(info_gain(data[column], data[target_column]))
    return info_gains

