import json
import ntpath

import numpy as np
from prettytable import PrettyTable


def print_results(results):
    tab = PrettyTable(['metric', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20'])
    tab.align["metric"] = "l"
    tab.float_format = '0.2'
    err_names = ['f1_err', 'f2_err', 'f3_err']
    for err_name in err_names:
        errs = np.array([r[err_name] for r in results])
        errs[np.isnan(errs)] = 1.0 if err_name == 'f_err' else 180
        res = np.array([np.sum(errs * 100.0 < t) / len(errs) for t in range(1, 21)])
        tab.add_row([err_name, np.median(errs), np.mean(errs), np.mean(res[:5]), np.mean(res[:10]), np.mean(res)])

    for field in ['inlier_ratio', 'iterations', 'runtime', 'refinements']:
        xs = [r['info'][field] for r in results]
        tab.add_row([field, np.median(xs), np.mean(xs), '-', '-', '-'])
        # print(f'{field}: \t median: {np.median(xs):0.02f} \t mean: {np.mean(xs):0.02f}')

    print(tab)


def print_results_summary(results, experiments):
    tab = PrettyTable(['experiment', 'median', 'mean', 'AUC@5', 'AUC@10', 'AUC@20', 'Mean runtime', 'Med runtime'])
    tab.float_format = '0.2'

    for experiment in experiments:
        exp_results = [r for r in results if r['experiment'] == experiment]
        errs = np.array([r['f1_err'] for r in exp_results])
        # errs = np.array([r['P_err'] for r in exp_results])
        errs[np.isnan(errs)] = 1.0
        res = np.array([np.sum(errs * 100 < t ) / len(errs) for t in range(1, 21)])
        runtime = [r['info']['runtime'] for r in exp_results]
        tab.add_row([experiment, np.median(errs), np.mean(errs),
                     100 * np.mean(res[:5]), 100 * np.mean(res[:10]), 100 * np.mean(res),
                     np.mean(runtime), np.median(runtime)])

    print(tab)

def check_phone_contribution(results):
    # experiments = np.unique([r['experiment'] for r in results])
    experiments = ['6pf + p3p', '4pH + 4pH + 3vHf + p3p']
    cameras = np.unique([ntpath.basename(x['img1'].split(ntpath.sep)[0]) for x in results])

    tab = PrettyTable(['Camera', 'our median', 'baseline median', 'diff'])
    tab.float_format = '0.4'

    for camera in cameras:
        cam_res = [x for x in results if camera in x['img1']]
        baseline_results = [x for x in cam_res if x['experiment'] == '6pf + p3p']
        our_results = [x for x in cam_res if x['experiment'] == '4pH + 4pH + 3vHf + p3p']

        baseline_median = np.nanmedian([x['f_err'] for x in baseline_results])
        our_median = np.nanmedian([x['f_err'] for x in our_results])
        tab.add_row([camera, our_median, baseline_median, baseline_median-our_median])

    print(tab)

if __name__ == '__main__':
    scenes = ['Calib', 'Boats', 'Book']

    results = []
    for scene in scenes:
        res_path = f'results/focal_{scene}-graph-triplets-features_superpoint_2048_2048-LG.json'

        with open(res_path, 'r') as f:
            results.extend(json.load(f))

    check_phone_contribution(results)

