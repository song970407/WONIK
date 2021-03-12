import torch
from box import Box

def get_config(model_type: str):
    """
    :param model_type: one of ['IC', 'NIC', 'IC_residual', 'NIC_residual']
    :return:
    """
    info = {
        'data': {
            'ACTION_WS': True,
            'PREPROCESS': True,
            'train_data_path': ['docs/new_data/expert/data_2.csv',
                                'docs/new_data/icgrnn/data_1.csv'],
            'test_data_path': ['docs/new_data/expert/data_3.csv', 'docs/new_data/linear/data_2.csv'],
            'scaler': None
        },
        'graph': {
            'glass_tc_pos_path': 'docs/new_location/glass_TC.csv',
            'control_tc_pos_path': 'docs/new_location/control_TC.csv',
            'DIST_THRESHOLD': 850,
            'DIST_WEIGHT': (1., 1., 1.),
            'ONLY_CONTROL': True
        },
        'model': {
            'ENC_TCN': False,
            'HIST_WINDOW': 50,
            'HIST_X': 20,
            'HIST_U': 20
        },
        'train': {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'ROLLOUT_WINDOWS': [20, 30, 40],
            'CONSISTENT_LOSS': True,
            'CONSISTENT_GAMMA': 100,
            'LR': 5e-4,
            'BS': 64,
            'EPOCHS': 50,
            'PERTURB_X_PARAM': 0.0,
            'TEST_EVERY': 100
        },
        'MPC': {
            'state_dim': 1,
            'action_dim': 1,
            'H': 5,
            'DEVICE': 'cpu',
            'action_min': -10.0,
            'action_max': 5.0,
            'Q': 10000,
            'R': 1,
            'r': 0,
            'SIMULATION_PERTURBATION': 1e-4,
            'test_data_path': ['docs/new_data/test/data_6.csv']
        },
        'reference': {
            'initial_temp': 150,
            'heatup_times': [545],
            'anneal_times': [363],
            'target_temps': [375]
        }
    }
    info = Box(info)
    if model_type == 'IC':
        info.data['SCALING'] = True
        info.data['ACTION_DIF'] = False
        info.graph['ONLY_CONTROL'] = True
        info.model['RESIDUAL'] = False
        info.model['hidden_reparam_method'] = ['Softmax']
        info.model['dec_reparam_method'] = ['ReLU']
        info.model['control_reparam_method'] = ['ReLU']
        info.train['save_model_path'] = 'model_icgrnn.pt'
        info.MPC['load_model_path'] = 'model/model_icgrnn.pt'
        info.MPC['is_convex'] = True
    elif model_type == 'IC_TCN':
        info.data['SCALING'] = True
        info.data['ACTION_DIF'] = False
        info.graph['ONLY_CONTROL'] = False
        info.model['ENC_TCN'] = True
        info.model['HIST_CONTROL'] = 10
        info.model['RESIDUAL'] = False
        info.model['hidden_reparam_method'] = ['Softmax']
        info.model['dec_reparam_method'] = ['ReLU']
        info.train['save_model_path'] = 'model_icgrnn_tcn.pt'
        info.MPC['load_model_path'] = 'model/model_icgrnn_tcn.pt'
        info.MPC['is_convex'] = True
    elif model_type == 'IC_MULTI_U':
        info.data['SCALING'] = True
        info.data['ACTION_DIF'] = False
        info.graph['ONLY_CONTROL'] = False
        info.model['ENC_TCN'] = True
        info.model['RESIDUAL'] = False
        info.model['hidden_reparam_method'] = ['Softmax']
        info.model['dec_reparam_method'] = ['ReLU']
        info.train['save_model_path'] = 'model_icgrnn_tcn.pt'
        info.MPC['load_model_path'] = 'model/model_icgrnn_tcn.pt'
        info.MPC['is_convex'] = True
    elif model_type == 'NIC':
        info.data['SCALING'] = True
        info.data['ACTION_DIF'] = False
        info.graph['ONLY_CONTROL'] = True
        info.model['RESIDUAL'] = False
        info.model['hidden_reparam_method'] = [None]
        info.model['dec_reparam_method'] = [None]
        info.train['save_model_path'] = 'model_grnn.pt'
        info.MPC['load_model_path'] = 'model/model_grnn.pt'
        info.MPC['is_convex'] = False
    elif model_type == 'IC_residual':
        info.data['SCALING'] = False
        info.data['ACTION_DIF'] = True
        info.model['RESIDUAL'] = True
        info.model['hidden_reparam_method'] = ['Softmax']
        info.model['dec_reparam_method'] = ['ReLU']
        info.train['save_model_path'] = 'model_icgrnn_residual.pt'
        info.MPC['SIMULATION_PERTURBATION'] = 0.025
        info.MPC['load_model_path'] = 'model/model_icgrnn_residual.pt'
        info.MPC['is_convex'] = True
    elif model_type == 'NIC_residual':
        info.data['SCALING'] = False
        info.data['ACTION_DIF'] = True
        info.model['RESIDUAL'] = True
        info.model['hidden_reparam_method'] = [None]
        info.model['dec_reparam_method'] = [None]
        info.train['save_model_path'] = 'model_grnn_residual.pt'
        info.MPC['SIMULATION_PERTURBATION'] = 0.025
        info.MPC['load_model_path'] = 'model/model_grnn_residual.pt'
        info.MPC['is_convex'] = False
    else:
        assert False, "Model_type is not appropriate!"
    return info