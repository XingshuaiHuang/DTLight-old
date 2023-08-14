# Neighborhood information

mdp_configs = {
    'FMA2CFull': {
        'grid4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['A0', 'B0', 'B1', 'A1'],
                'bot_right_mgr': ['C0', 'D0', 'D1', 'C1'],
                'top_right_mgr': ['C2', 'D2', 'D3', 'C3'],
                'top_left_mgr': ['A2', 'B2', 'B3', 'A3']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'arterial4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['nt1', 'nt2', 'nt6', 'nt5'],
                'bot_right_mgr': ['nt3', 'nt4', 'nt8', 'nt7'],
                'top_right_mgr': ['nt11', 'nt12', 'nt16', 'nt15'],
                'top_left_mgr': ['nt9', 'nt10', 'nt14', 'nt13']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'cologne8': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['247379907', '256201389', '26110729', '280120513', '62426694'],
                'bot_mgr': ['32319828', '252017285', 'cluster_1098574052_1098574061_247379905']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
    },
    'FMA2C': {
        'grid4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['A0', 'B0', 'B1', 'A1'],
                'bot_right_mgr': ['C0', 'D0', 'D1', 'C1'],
                'top_right_mgr': ['C2', 'D2', 'D3', 'C3'],
                'top_left_mgr': ['A2', 'B2', 'B3', 'A3']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'arterial4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['nt1', 'nt2', 'nt6', 'nt5'],
                'bot_right_mgr': ['nt3', 'nt4', 'nt8', 'nt7'],
                'top_right_mgr': ['nt11', 'nt12', 'nt16', 'nt15'],
                'top_left_mgr': ['nt9', 'nt10', 'nt14', 'nt13']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'ingolstadt1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['gneJ207']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        'cologne3': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['360082', '360086'],
                'bot_mgr': ['GS_cluster_2415878664_254486231_359566_359576']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne8': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['247379907', '256201389', '26110729', '280120513', '62426694'],
                'bot_mgr': ['32319828', '252017285', 'cluster_1098574052_1098574061_247379905']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['GS_cluster_357187_359543']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },

        '3lane': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['A0']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        '2lane': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['J3']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
    },
    'MA2C': {
        'grid4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['A0', 'B0', 'B1', 'A1'],
                'bot_right_mgr': ['C0', 'D0', 'D1', 'C1'],
                'top_right_mgr': ['C2', 'D2', 'D3', 'C3'],
                'top_left_mgr': ['A2', 'B2', 'B3', 'A3']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'arterial4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['nt1', 'nt2', 'nt6', 'nt5'],
                'bot_right_mgr': ['nt3', 'nt4', 'nt8', 'nt7'],
                'top_right_mgr': ['nt11', 'nt12', 'nt16', 'nt15'],
                'top_left_mgr': ['nt9', 'nt10', 'nt14', 'nt13']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'ingolstadt1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['gneJ207']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        'cologne3': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['360082', '360086'],
                'bot_mgr': ['GS_cluster_2415878664_254486231_359566_359576']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne8': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['247379907', '256201389', '26110729', '280120513', '62426694'],
                'bot_mgr': ['32319828', '252017285', 'cluster_1098574052_1098574061_247379905']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['GS_cluster_357187_359543']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        '3lane': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['A0']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        '2lane': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['J3']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
    },
    'FMA2CVAL': {
        'grid4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['A0', 'B0', 'B1', 'A1'],
                'bot_right_mgr': ['C0', 'D0', 'D1', 'C1'],
                'top_right_mgr': ['C2', 'D2', 'D3', 'C3'],
                'top_left_mgr': ['A2', 'B2', 'B3', 'A3']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'arterial4x4': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'bot_left_mgr': ['nt1', 'nt2', 'nt6', 'nt5'],
                'bot_right_mgr': ['nt3', 'nt4', 'nt8', 'nt7'],
                'top_right_mgr': ['nt11', 'nt12', 'nt16', 'nt15'],
                'top_left_mgr': ['nt9', 'nt10', 'nt14', 'nt13']
            },
            'management_neighbors': {
                'bot_left_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'bot_right_mgr': ['bot_left_mgr', 'top_right_mgr'],
                'top_right_mgr': ['bot_right_mgr', 'top_left_mgr'],
                'top_left_mgr': ['top_right_mgr', 'bot_left_mgr']
            }
        },
        'ingolstadt1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['gneJ207']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
        'cologne3': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['360082', '360086'],
                'bot_mgr': ['GS_cluster_2415878664_254486231_359566_359576']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne8': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['247379907', '256201389', '26110729', '280120513', '62426694'],
                'bot_mgr': ['32319828', '252017285', 'cluster_1098574052_1098574061_247379905']
            },
            'management_neighbors': {
                'top_mgr': ['bot_mgr'],
                'bot_mgr': ['top_mgr']
            }
        },
        'cologne1': {
            'coef': 0.4,
            'coop_gamma': 0.9,
            'clip_wave': 4.0,
            'clip_wait': 4.0,
            'norm_wave': 5.0,
            'norm_wait': 100.0,
            'alpha': 0.75,
            'management': {
                'top_mgr': ['GS_cluster_357187_359543']
            },
            'management_neighbors': {
                'top_mgr': []
            }
        },
    }
}