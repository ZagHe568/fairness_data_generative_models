from tgan.data import load_demo_data
from tgan.model import TGANModel
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    data = pd.read_csv('../data/adult_dataset.csv', index_col=None)
    data = data.reindex(np.random.permutation(data.index))
    # data = data[:1000]
    columns = data.columns
    continuous_columns = [0, 2, 4, 10, 11, 12]
    continuous_columns_names = [columns[i] for i in continuous_columns]

    print(data.head(5).T)

    tgan = TGANModel(
        continuous_columns,
        output='output',
        max_epoch=5,
        steps_per_epoch=10000,
        save_checkpoints=True,
        restore_session=False,
        batch_size=200,
        z_dim=200,
        noise=0.2,
        l2norm=0.00001,
        learning_rate=0.001,
        num_gen_rnn=100,
        num_gen_feature=100,
        num_dis_layers=1,
        num_dis_hidden=100,
        optimizer='AdamOptimizer'
    )

    tgan.fit(data)

    num_samples = data.shape[0]
    generated_data = tgan.sample(num_samples)
    generated_data.columns = columns
    for column in continuous_columns_names:
        generated_data[column] = generated_data[column].astype(int)
    print(generated_data.head(5).T)
    generated_data.to_csv('generated_data.csv', index=None)
    tgan.save('./tgan.pkl', force=True)
