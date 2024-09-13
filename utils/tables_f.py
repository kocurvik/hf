import json
import os

import numpy as np

from eval import print_results_summary
from utils.data import basenames_pt, basenames_eth, basenames_cambridge, get_basenames

incdec = [1, 1, -1, -1, -1, 1]

experiments = ['6p3v',
               '4p3v(M)', '4p3v(M) + R', '4p3v(M) + R + C',
               '4p3v(M-D)', '4p3v(M-D) + R', '4p3v(M-D) + R + C',
               '4p3v(L)', '4p3v(L) + R', '4p3v(L-ID) + R',
               '4p3v(O)', '4p3v(O) + R']

names = {
        '4p(HC)' : '\\sshc',
        '6p3v': '\\sst',
        '4p3v(M)': '\\midrule\\sstm',
        '4p3v(M) + R': '\\sstmR',
        '4p3v(M) + R + C': '\\sstmRC',
        '4p3v(M-D)': '\\sstmd',
        '4p3v(M-D) + R': '\\sstmdR',
        '4p3v(M-D) + R + C': '\\sstmdRC',
        '4p3v(L)': '\\sstl',
        '4p3v(L) + R': '\\sstlR',
        '4p3v(L) + R + C': '\\sstlRC',
        '4p3v(L-D)': '\\sstld',
        '4p3v(L-D) + R': '\\sstldR',
        '4p3v(L-D) + R + C': '\\sstldRC',
        '4p3v(L-ID)': '\\sstlid',
        '4p3v(L-ID) + R': '\\sstlidR',
        '4p3v(L-ID) + R + C': '\\sstlidRC',
        '4p3v(O)': '\\midrule\\ssto',
        '4p3v(O) + R': '\\sstoR',
        '4p3v(O) + R + C': '\\sstoRC',
    }

def print_table_text(experiments, rows):

    print(
        f'\\begin{{tabular}}{{ l | c c | c c c | c}}\n'
        f'    \\toprule\n'
        f'    Estimator & AVG $(^\\circ)$ $\\downarrow$ & MED $(^\\circ)$ $\\downarrow$ & AUC@5 $\\uparrow$ & @10 $\\uparrow$ & @20 $\\uparrow$ & Runtime (ms) $\\downarrow$\\\\\n'
        f'    \\midrule\n')

    for experiment, row in zip(experiments, rows):
        print(f'{names[experiment]} & {row} \\\\')

    print('\\midrule')

def get_rows(results, order):
    num_rows = []

    l = np.array(['(O)' in exp for exp in order])

    for experiment in order:
        exp_results = [x for x in results if x['experiment'] == experiment]

        # p_errs = np.array([max(0.5 *(out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err'])) for out in exp_results])
        p_errs = np.array([out['f_err'] for out in exp_results])
        p_errs[np.isnan(p_errs)] = 180
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 21)])
        p_auc_5 = np.mean(p_res[:5])
        p_auc_10 = np.mean(p_res[:10])
        p_auc_20 = np.mean(p_res)
        p_avg = np.mean(p_errs)
        p_med = np.median(p_errs)

        times = [r['info']['runtime'] for r in exp_results]
        time_avg = np.mean(times)

        num_rows.append([p_avg, p_med, 100 * p_auc_5, 100 * p_auc_10, 100 * p_auc_20, time_avg])

    text_rows = [[f'{x:0.2f}' for x in y] for y in num_rows]
    lens = np.array([[len(x) for x in y] for y in text_rows])
    arr = np.array(num_rows)
    arr[l, :] = np.inf * np.array([incdec])
    for j in range(len(text_rows[0])):
        idxs = np.argsort(incdec[j] * arr[:, j])
        text_rows[idxs[0]][j] = '\\textbf{' + text_rows[idxs[0]][j] + '}'
        text_rows[idxs[1]][j] = '\\underline{' + text_rows[idxs[1]][j] + '}'

    max_len = np.max(lens, axis=0)
    phantoms = max_len - lens
    for i in range(len(text_rows)):
        for j in range(len(text_rows[0])):
            if phantoms[i, j] > 0:
                text_rows[i][j] = '\\phantom{' + (phantoms[i, j] * '1') + '}' + text_rows[i][j]

    return [' & '.join(row) for row in text_rows]

def generate_table(dataset, feat):
    basenames = get_basenames(dataset)

    results_type = f'triplets-features_{feat}_noresize_2048-LG'

    results = []
    for basename in basenames:
        json_path = os.path.join('results', f'focal_{basename}-{results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            results.extend([x for x in json.load(f) if x['experiment'] in experiments])

    print("Data loaded")

    print(30 * '*')
    print(30 * '*')
    print(30 * '*')
    print("Printing: ", dataset)
    print(30 * '*')


    # print_results_summary(results, experiments)

    rows = get_rows(results, experiments)
    print_table_text(experiments, rows)

if __name__ == '__main__':
    # generate_table('pt', 'superpoint')
    generate_table('cambridge', 'superpoint')

