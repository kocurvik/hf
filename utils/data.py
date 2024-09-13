import seaborn as sns

experiments = ['4pH + 4pH + 3vHf + scale', '4pH + 4pH + 3vHf + p3p', '6pf + p3p', '6pf + 5p', '4pH + 4pH + 3vHf', '6pf']

iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
colors = {exp: sns.color_palette("tab10")[i] for i, exp in enumerate(experiments)}
# styles = {exp: 'dashed' if 'O' in exp else 'solid' for exp in experiments}

basenames_pt = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior', 'grand_place_brussels',
                'notre_dame_front_facade', 'palace_of_westminster', 'pantheon_exterior', 'reichstag', #'st_peters_square',
                'sacre_coeur', 'taj_mahal', 'temple_nara_japan', 'trevi_fountain']

basenames_eth = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'list.py', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
basenames_eth_test = ['botanical_garden', 'boulders', 'bridge', 'door', 'exhibition_hall', 'lecture_room', 'living_room', 'lounge', 'observatory', 'old_computer', 'statue', 'terrace_2']
basenames_cambridge = ['GreatCourt', 'KingsCollege', 'ShopFacade', 'StMarysChurch', 'OldHospital']
basenames_aachen = ['aachen_v1.1']

def pose_err_max(out):
    # return max([out['R_12_err'], out['R_13_err'], out['R_23_err'], out['t_12_err'], out['t_13_err'], out['t_23_err']])
    return max([out['R_12_err'], out['R_13_err'], out['t_12_err'], out['t_13_err']])

def get_basenames(dataset):
    if dataset == 'pt':
        basenames = basenames_pt
        name = '\\Phototourism'
    elif dataset == 'eth3d':
        basenames = basenames_eth
        name = '\\ETH'
    elif dataset == 'aachen':
        basenames = basenames_aachen
    elif dataset == 'urban':
        name = 'Urban'
        basenames = ['kyiv-puppet-theater']
    elif dataset == 'cambridge':
        basenames = basenames_cambridge
    elif dataset == 'eth3d_test':
        basenames = basenames_eth_test
    elif dataset == 'phone_planar':
        basenames = [f'a{i}' for i in range(2, 7)]
    else:
        raise ValueError
    return basenames
