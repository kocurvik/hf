import numpy as np
import seaborn as sns

experiments = ['4pH + 4pH + 3vHf + scale', '4pH + 4pH + 3vHf + p3p', '6pf + p3p', '6pf + 5p', '4pH + 4pH + 3vHf', '6pf']

experiments_case_1 = ['4pH + 4pH + 3vHfc1 + p3p', '4pH + 4pH + 3vHfc1 + p3p + Haikkila', '6p fEf + p3p', '6p fEf + p3p + degensac', '6p fEf (pairs)',
                      '6p fEf (pairs) + degensac']

experiments_case_2 = ['4pH + 4pH + 3vHfc2 + p3p', '6p fEf + p3p', '6p fEf + p3p + degensac', '6p Ef + p3p',
                       '6p fEf (pairs)', '6p fEf (pairs) + degensac', '6p Ef (pairs)']

experiments_case_3 = ['4pH + 4pH + 3vHfc3 + p3p', '6p fEf + p4pf', '6p fEf + p4pf + degensac']

experiments_case_4 = ['4pH + 4pH + 3vHfc4 + p3p', '6p Ef + p4pf']

all_experiments = ['4pH + 4pH + 3vHfc1 + p3p', '4pH + 4pH + 3vHfc2 + p3p', '6p fEf + p3p', '6p fEf + p3p + degensac',
                   '6p Ef + p3p', '6p fEf (pairs)', '6p fEf (pairs) + degensac', '6p Ef (pairs)']

experiments_case34 = ['4pH + 4pH + 3vHfc3 + p3p', '4pH + 4pH + 3vHfc4 + p3p', '6p fEf + p4pf', '6p fEf + p4pf + degensac', '6p Ef + p4pf']
experiments_case34_fo = ['4pH + 4pH + 3vHfc3 + p3p + LO(0.3)', '4pH + 4pH + 3vHfc4 + p3p + FO(0.3)', '6p fEf + p4pf + FO(0.3)', '6p fEf + p4pf + degensac + FO(0.3)', '6p Ef + p4pf + FO(0.3)']
experiments_case34_lo = ['4pH + 4pH + 3vHfc3 + p3p + LO(0)', '4pH + 4pH + 3vHfc4 + p3p + LO(0)', '6p fEf + p4pf + LO(0)', '6p fEf + p4pf + degensac + LO(0)', '6p Ef + p4pf + LO(0)']



iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

basenames_pt = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior', 'grand_place_brussels',
                'notre_dame_front_facade', 'palace_of_westminster', 'pantheon_exterior', 'reichstag', #'st_peters_square',
                'sacre_coeur', 'taj_mahal', 'temple_nara_japan', 'trevi_fountain']

basenames_eth = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'list.py', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
basenames_eth_test = ['botanical_garden', 'boulders', 'bridge', 'door', 'exhibition_hall', 'lecture_room', 'living_room', 'lounge', 'observatory', 'old_computer', 'statue', 'terrace_2']
basenames_cambridge = ['GreatCourt', 'KingsCollege', 'ShopFacade', 'StMarysChurch', 'OldHospital']
basenames_aachen = ['aachen_v1.1']
# basenames_custom = ['Asphalt', 'Boats', 'Book', 'Calib', 'Facade', 'Floor', 'Papers']
basenames_custom = ['Asphalt', 'Boats', 'Book', 'Facade', 'Floor', 'Papers']

colors = {exp: sns.color_palette("tab10")[i] for i, exp in enumerate(all_experiments)}
colors.update({exp: sns.color_palette("tab10")[i] for i, exp in enumerate(experiments_case34)})
colors.update({f'{exp} + LO(0)': sns.color_palette("tab10")[i] for i, exp in enumerate(experiments_case34)})
colors.update({f'{exp} + FO(0.3)': sns.color_palette("tab10")[i] for i, exp in enumerate(experiments_case34)})
colors.update({f'{exp} + FR': sns.color_palette("tab10")[i] for i, exp in enumerate(experiments_case34)})

def get_experiments(case, include_pairs=False):
    if case == 1:
        experiments = experiments_case_1
    elif case == 2:
        experiments = experiments_case_2
    elif case == 3:
        experiments = experiments_case_3
    elif case == 4:
        experiments = experiments_case_4
    else:
        raise ValueError("Wrong case number")

    if not include_pairs:
        experiments = [x for x in experiments if 'pairs' not in x]

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


def err_f1(r):
    return min(r['f1_err'], 1.0)



def err_f1f2(r):
    return min(np.sqrt(r['f1_err'] * r['f2_err']), 1.0)

def err_f1f3(r):
    return min(np.sqrt(r['f1_err'] * r['f3_err']), 1.0)
