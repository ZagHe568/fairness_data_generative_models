import pandas as pd
import sys
sys.path.append('..')
from utils import evaluate

if __name__ == '__main__':
    generated_data = pd.read_csv('generated_data.csv')
    generated_data_male = generated_data[generated_data['sex'] == 'Male']
    generated_data_female = generated_data[generated_data['sex'] == 'Female']
    generated_data_male.drop(columns=['sex'])
    generated_data_female.drop(columns=['sex'])
    evaluate(generated_data_male, generated_data_female, '../data')
