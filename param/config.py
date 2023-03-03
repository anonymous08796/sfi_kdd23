from pathlib import Path

__all__ = ['project_path', 'dataset_config']

project_path = Path(__file__).parent

dataset_config = {
    'pokec_n':
        {
            'dataset': 'region_job_2',
            'sens_attr': "region",
            'predict_attr': "I_am_working_in_field",
            'label_number': 1000,
            'sens_number': 200,
            'seed': 20,
            'path': "./dataset/pokec/",
            'test_idx': False,
            'random_range': 5,
            'y_labels': 2,
            's_labels': 2,
            'prior_dis': [0.7111, 0.2889]
        },

    'pokec_z':
        {
            'dataset': 'region_job',
            'sens_attr': "region",
            'predict_attr': "I_am_working_in_field",
            'label_number': 1000,
            'sens_number': 200,
            'seed': 20,
            'path': "./dataset/pokec/",
            'test_idx': False,
            'random_range': 5,
            'y_labels': 2,
            's_labels': 2,
            'prior_dis': [0.6484, 0.3516],
        },
    'adult':
        {
            'dataset': 'adult',
            'sens_attr': "marital",
            'predict_attr': "income",
            'path': "./datasets/Adult/Adult_10000_reform.csv",
            'y_dim': 2,
            's_dim': 2,
            'prior_dis': [0.4746, 0.5254],
            'num_trans': 11,
            'trans_type': 'mul',
            'enc_nlayers': 5,
            'enc_bias': False,
            'trans_nlayers': 2,
            'batch_norm': False,
        },
    'gss':
        {
            'dataset': 'gss',
            'sens_attr': "sex",
            'predict_attr': "hapmar",
            'path': "./datasets/GSS/GSS_5079_reform.csv",
            'y_dim': 2,
            's_dim': 2,
            'prior_dis': [0.5369, 0.4631],
            'num_trans': 11,
            'trans_type': 'mul',
            'enc_nlayers': 5,
            'enc_bias': False,
            'trans_nlayers': 2,
            'batch_norm': False,
        },
}
