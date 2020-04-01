from scipy import stats
from info_gain import info_gain


def calc_entropy(data):
    entropies = []
    for column in data.columns:
        p_data = data[column].value_counts()
        entropies.append(round(stats.entropy(p_data), 2))
    entropies.insert(0, round(sum(entropies), 2))
    return entropies


def calc_info_gain(data, target_column):
    info_gains = []
    for column in data.columns:
        if column == target_column:
            continue
        info_gains.append(round(info_gain(data[column], data[target_column]), 2))
    info_gains.insert(0, round(sum(info_gains), 2))
    return info_gains

