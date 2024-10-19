import seaborn as sns

experiments = ['4pH + 4pH + 3vHf + scale', '4pH + 4pH + 3vHf + p3p', '6pf + p3p', '6pf + 5p', '4pH + 4pH + 3vHf', '6pf']

experiments_case_1 = ['4pH + 4pH + 3vHfc1 + p3p', '6p fEf + p3p', '6p fEf + p3p + degensac', '6p fEf (pairs)',
                      '6p fEf (pairs) + degensac']

experiments_case_2 = ['4pH + 4pH + 3vHfc2 + p3p', '6p fEf + p3p', '6p fEf + p3p + degensac', '6p Ef + p3p',
                       '6p fEf (pairs)', '6p fEf (pairs) + degensac', '6p Ef (pairs)']

all_experiments = ['4pH + 4pH + 3vHfc1 + p3p', '4pH + 4pH + 3vHfc2 + p3p', '6p fEf + p3p', '6p fEf + p3p + degensac',
                   '6p fEf (pairs)', '6p fEf (pairs) + degensac', '6p Ef + p3p',  '6p Ef (pairs)']


iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

basenames_pt = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior', 'grand_place_brussels',
                'notre_dame_front_facade', 'palace_of_westminster', 'pantheon_exterior', 'reichstag', #'st_peters_square',
                'sacre_coeur', 'taj_mahal', 'temple_nara_japan', 'trevi_fountain']

basenames_eth = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'list.py', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
basenames_eth_test = ['botanical_garden', 'boulders', 'bridge', 'door', 'exhibition_hall', 'lecture_room', 'living_room', 'lounge', 'observatory', 'old_computer', 'statue', 'terrace_2']
basenames_cambridge = ['GreatCourt', 'KingsCollege', 'ShopFacade', 'StMarysChurch', 'OldHospital']
basenames_aachen = ['aachen_v1.1']
basenames_custom = ['Asphalt', 'Boats', 'Book', 'Calib', 'Facade', 'Floor', 'Papers']

colors = {exp: sns.color_palette("tab10")[i] for i, exp in enumerate(all_experiments)}

def get_experiments(case):
    if case == 1:
        experiments = experiments_case_1
    elif case == 2:
        experiments = experiments_case_2
    else:
        raise ValueError("Wrong case number")



    return experiments, colors


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
    elif dataset == 'custom_planar':
        basenames = basenames_custom
    else:
        raise ValueError
    return basenames


def is_image(x):
    return '.jpg' in x.lower() or '.png' in x.lower() or '.jpeg' in x.lower()
