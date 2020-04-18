from scipy import stats
from info_gain import info_gain
import pandas as pd
import os

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


def evaluate(decoded_A, decoded_B, original_path='data'):
    original_A_path = os.path.join(original_path, 'male_adult_dataset')
    original_B_path = os.path.join(original_path, 'female_adult_dataset')

    original_A = pd.read_csv(original_A_path, sep=', ', engine='python')
    original_B = pd.read_csv(original_B_path, sep=', ', engine='python')

    decoded_A = decoded_A[:100]
    decoded_B = decoded_B[:100]

    augmented_A = pd.concat([original_A, decoded_A])
    augmented_B = pd.concat([original_B, decoded_B])

    results_entropy = []
    results_entropy.append(calc_entropy(original_A))
    results_entropy.append(calc_entropy(augmented_A))
    results_entropy.append(calc_entropy(original_B))
    results_entropy.append(calc_entropy(augmented_B))
    pd.DataFrame(results_entropy, index=['original_A', 'augmented_A', 'original_B', 'augmented_B']).to_csv(f'results_entropy.csv')


    results_info_gain = []
    results_info_gain.append(calc_info_gain(original_A, 'Y'))
    results_info_gain.append(calc_info_gain(augmented_A, 'Y'))
    results_info_gain.append(calc_info_gain(original_B, 'Y'))
    results_info_gain.append(calc_info_gain(augmented_B, 'Y'))
    pd.DataFrame(results_info_gain, index=['original_A', 'augmented_A', 'original_B', 'augmented_B']).to_csv(f'results_info_gain.csv')
