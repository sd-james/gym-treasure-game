import numpy as np

CONFIG = {

    1: {
        'linking_threshold': 0.1,
        'specify_rewards': True,
        'effect_epsilon': 3.7,
        'init_epsilon': 4,
        'generate_positive_samples': True,
        'low_threshold': 0.45,
        'high_threshold': 0.9,
        'augment_negative': True,
        'max_precondition_samples': 2000,
        'precondition_c_range': np.logspace(0.01, 0.5, 10),
        'precondition_gamma_range': np.logspace(0.1, 1, 10),
    },

    2: {
        'linking_threshold': 0.1,
        'specify_rewards': True,
        'effect_epsilon': 3.7,
        'init_epsilon': 4,
        'generate_positive_samples': True,
        'low_threshold': 0.45,
        'high_threshold': 0.9,
        'augment_negative': True,
        'max_precondition_samples': 2000,
        'precondition_c_range': np.logspace(0.01, 0.5, 10),
        'precondition_gamma_range': np.logspace(0.1, 1, 10),
    },

}
