import json
import ntpath
import os

import numpy as np
from prettytable import PrettyTable

from dataset_utils.data import get_experiments, get_basenames, is_image, err_f1, err_f1f2, err_f1f3


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

method_name_dict = {'4pH + 4pH + 3vHfc1 + p3p': '\\hr{fff} & \\textbf{ours}',
                    '4pH + 4pH + 3vHfc1 + p3p + Haikkila': '\\hr{fff} & \\cite{heikkila2017using}',
                    '6p fEf + p3p': '\\fEfr & \\cite{tzamos2023relative}',
                    '6p fEf + p3p + degensac': '\\fEfpr & ',
                    '6p fEf (pairs)': '\\fEf & \\cite{kukelova2012polynomial}',
                    '6p fEf (pairs) + degensac': '\\fEfp & \\cite{torii2011six}',
                    '4pH + 4pH + 3vHfc2 + p3p': '\\hr{ff} & \\textbf{ours}',
                    '6p Ef + p3p': '\\Efr & ',
                    '6p Ef (pairs)': '\\Ef & \\cite{bujnak20093d}',
                    '4pH + 4pH + 3vHfc3 + p3p': '\\hr{f \\rho \\rho} & \\textbf{ours} &',
                    '4pH + 4pH + 3vHfc3 + prior': '\\h{f \\rho \\rho} + prior & \\textbf{ours} &',
                    '4pH + 4pH + 3vHfc4 + p3p': '\\hr{f \\rho} & \\textbf{ours} &',
                    '6p fEf + p4pf': '\\fEfrf & &',
                    '6p fEf + p4pf + degensac': '\\fEfprf & &',
                    '6p Ef + p4pf': '\\Efrf & &',
                    '4pH + 4pH + 3vHfc3 + p3p + LO(0)': '\\hr{f \\rho \\rho} & \\textbf{ours} &',
                    '4pH + 4pH + 3vHfc4 + p3p + LO(0)': '\\hr{f \\rho} & \\textbf{ours} & ',
                    '6p fEf + p4pf + LO(0)': '\\fEfrf & &',
                    '6p fEf + p4pf + degensac + LO(0)': '\\fEfprf & &',
                    '6p Ef + p4pf + LO(0)': '\\Efrf & &',
                    '4pH + 4pH + 3vHfc3 + p3p + FO(0.3)': '\\hr{f \\rho \\rho} & \\textbf{ours} & \\multirow{3}{*}{35.5$^\\circ$ - 61.5$^\\circ$}',
                    '4pH + 4pH + 3vHfc4 + p3p + FO(0.3)': '\\hr{f \\rho} & \\textbf{ours} & \\multirow{2}{*}{35.5$^\\circ$ - 61.5$^\\circ$}',
                    '6p fEf + p4pf + FO(0.3)': '\\fEfrf & &',
                    '6p fEf + p4pf + degensac + FO(0.3)': '\\fEfprf & &',
                    '6p Ef + p4pf + FO(0.3)': '\\Efrf & & ',
                    '4pH + 4pH + 3vHfc3 + p3p + FR': '\\hr{f \\rho \\rho} & \\textbf{ours} & \\multirow{3}{*}{50$^\\circ$ - 70$^\\circ$}',
                    '4pH + 4pH + 3vHfc4 + p3p + FR': '\\hr{f \\rho} & \\textbf{ours} &  \\multirow{2}{*}{50$^\\circ$ - 70$^\\circ$}',
                    '6p fEf + p4pf + FR': '\\fEfrf & &',
                    '6p fEf + p4pf + degensac + FR': '\\fEfprf & &',
                    '6p Ef + p4pf + FR': '\\Efrf & &',

                    }


def method_input(exp):
    pts = int(exp[0])
    if 'pairs' in exp:
        return f'{pts} pairs'
    else:
        return f'{pts} triplets'

def print_full_table(results1, results2, experiments1, experiments2, cases = (1, 2), err_fun = (err_f1, err_f1)):

    print('\\begin{tabular}{|l|rc|c|ccccc|} \\cline{2-9}')
    print('\\multicolumn{1}{c|}{} & \\multicolumn{2}{|c|}{Method} & Sample & Median $\\xi_f$ & Mean $\\xi_f$ & mAA$_f$(0.1) & mAA$_f$(0.2) & Runtime (ms) \\\\ \\hline')

    print('\\multirow{', len(experiments1), '}{*}{\\rotatebox[origin=c]{90}{\\case{', cases[0] ,'}}}')
    print_rows(experiments1, results1, err_fun[0])

    print('\\hline')

    print('\\multirow{', len(experiments2), '}{*}{\\rotatebox[origin=c]{90}{\\case{', cases[1],'}}}')
    print_rows(experiments2, results2, err_fun[1])

    print('\\hline')

    print('\\end{tabular}')
def print_full_table_case34(results3, results4, experiments1, experiments2, cases = (1, 2), err_fun = (err_f1, err_f1)):

    print('\\begin{tabular}{|l|rc|c|c|ccccc|} \\cline{2-10}')
    print('\\multicolumn{1}{c|}{} & \\multicolumn{2}{|c|}{Method} & FOV Filtering & Sample & Median $\\xi_f$ & Mean $\\xi_f$ & mAA$_f$(0.1) & mAA$_f$(0.2) & Runtime (ms) \\\\ \\hline')

    print('\\multirow{', len(experiments1), '}{*}{\\rotatebox[origin=c]{90}{\\case{', cases[0] ,'}}}')
    print_rows(experiments1, results3, err_fun[0])

    print('\\hline')

    print('\\multirow{', len(experiments2), '}{*}{\\rotatebox[origin=c]{90}{\\case{', cases[1],'}}}')
    print_rows(experiments2, results4, err_fun[1])

    print('\\hline')

    print('\\end{tabular}')


def print_rows(experiments1, results1, err_fun=err_f1):
    num_rows = []
    for i, exp in enumerate(experiments1):
        exp_res = [x for x in results1 if x['experiment'] == exp]
        errs = np.array([err_fun(r) for r in exp_res])
        errs[np.isnan(errs)] = 1.0
        times = np.array([r['info']['runtime'] for r in exp_res])
        res = np.array([np.sum(errs * 100 < t) / len(errs) for t in range(1, 21)])
        num_rows.append(
            [np.median(errs), np.mean(errs), np.mean(res[:10]) * 100, np.mean(res[:20]) * 100, np.mean(times)])
    incdec = [1, 1, -1, -1, 1]

    def format(x, i):
        if i < 2:
            return f'{x:0.4f}'
        return f'{x:0.2f}'

    text_rows = [[format(x, i) for i, x in enumerate(y)] for y in num_rows]
    lens = np.array([[len(x) for x in y] for y in text_rows])
    arr = np.array(num_rows)
    for j in range(len(text_rows[0])):
        idxs = np.argsort(incdec[j] * arr[:, j])
        text_rows[idxs[0]][j] = '\\textbf{' + text_rows[idxs[0]][j] + '}'
        # text_rows[idxs[1]][j] = '\\underline{' + text_rows[idxs[1]][j] + '}'
    max_len = np.max(lens, axis=0)
    phantoms = max_len - lens
    for i, exp in enumerate(experiments1):
        for j in range(len(text_rows[0])):
            if phantoms[i, j] > 0:
                text_rows[i][j] = '\\phantom{' + (phantoms[i, j] * '1') + '}' + text_rows[i][j]

        # print(f' & {method_name_dict[exp]} & {method_input(exp)} & {"&".join(text_rows[i])} \\\\ \\cline{{2 - 8}}')
        print(f' & {method_name_dict[exp]} & {method_input(exp)} & {"&".join(text_rows[i])} \\\\ ')


def eval_table(scenes=None, cases=(1, 2, 3, 4)):
    if 1 in cases and 2 in cases:
        experiments1, _ = get_experiments(1, include_pairs=True)
        experiments2, _ = get_experiments(2, include_pairs=True)

        if scenes is None:
            scenes = get_basenames('custom_planar')

        results1 = []
        for scene in scenes:
            res_path = f'results/focal_{scene}-triplets-case1-features_superpoint_noresize_2048-LG.json'

            with open(res_path, 'r') as f:
                results1.extend(json.load(f))

        results2 = []
        for scene in scenes:
            res_path = f'results/focal_{scene}-triplets-case2-features_superpoint_noresize_2048-LG.json'

            with open(res_path, 'r') as f:
                results2.extend(json.load(f))

        print_full_table(results1, results2, experiments1, experiments2)

    if 3 in cases and 4 in cases:
        experiments3_b, _ = get_experiments(3, include_pairs=True)
        experiments3 = experiments3_b.copy()
        experiments3.extend([f'{x} + FO(0.3)' for x in experiments3_b])
        experiments3.extend([f'{x} + FR' for x in experiments3_b])

        experiments4_b, _ = get_experiments(4, include_pairs=True)
        experiments4 = experiments4_b.copy()
        experiments4.extend([f'{x} + FO(0.3)' for x in experiments4_b])
        experiments4.extend([f'{x} + FR' for x in experiments4_b])

        # scenes = get_basenames('custom_planar')

        results3 = []
        for scene in scenes:
            res_path = f'results/focal_{scene}-triplets-case2-features_superpoint_noresize_2048-LG-c3.json'

            with open(res_path, 'r') as f:
                results3.extend(json.load(f))

        results4 = []
        for scene in scenes:
            res_path = f'results/focal_{scene}-triplets-case4-features_superpoint_noresize_2048-LG.json'

            with open(res_path, 'r') as f:
                results4.extend(json.load(f))

        print_full_table_case34(results3, results4, experiments3, experiments4, cases = [3, 4], err_fun = (err_f1f3, err_f1f2))

def eval_offplane_table_34():
    experiments3, _ = get_experiments(3, include_pairs=True)
    experiments4, _ = get_experiments(4, include_pairs=True)
    # scenes = get_basenames('custom_planar')

    res_path = f'results/focal_DinoBook-triplets-case2-features_superpoint_noresize_2048-LG-c3.json'

    with open(res_path, 'r') as f:
        results3 = json.load(f)

    res_path = f'results/focal_DinoBook-triplets-case4-features_superpoint_noresize_2048-LG.json'

    with open(res_path, 'r') as f:
        results4 = json.load(f)

    print_full_table_case34(results3, results4, experiments3, experiments4, cases = [3, 4], err_fun = (err_f1f3, err_f1f2))

    # cameras = set([x['img1'].split(ntpath.sep)[0] for x in results4])
    # cameras = set.union(set([x['img2'].split(ntpath.sep)[0] for x in results4]), cameras)
    # cameras = set.union(set([x['img3'].split(ntpath.sep)[0] for x in results4]), cameras)

    cameras = {'IPhoneZBHBack', 'IPhoneZBHfront', 'LenovoTabletBack', 'LenovoTabletFront', 'SamsungBack', 'SamsungFront',
     'SamsungGlossyBack', 'SamsungGlossyFront'}

    res_path = f'results/focal_Book-triplets-case2-features_superpoint_noresize_2048-LG-c3.json'
    with open(res_path, 'r') as f:
        results3 = json.load(f)

    results3 = [x for x in results3 if {x['img1'].split(ntpath.sep)[0], x['img2'].split(ntpath.sep)[0],
                                        x['img3'].split(ntpath.sep)[0]} <= cameras]

    res_path = f'results/focal_Book-triplets-case4-features_superpoint_noresize_2048-LG.json'
    with open(res_path, 'r') as f:
        results4 = json.load(f)

    results4 = [x for x in results4 if {x['img1'].split(ntpath.sep)[0], x['img2'].split(ntpath.sep)[0],
                                        x['img3'].split(ntpath.sep)[0]} <= cameras]

    print_full_table_case34(results3, results4, experiments3, experiments4, cases = [3, 4], err_fun = (err_f1f3, err_f1f2))

def eval_offplane_table_12():
    experiments1, _ = get_experiments(1, include_pairs=True)
    experiments2, _ = get_experiments(2, include_pairs=True)
    # scenes = get_basenames('custom_planar')

    res_path = f'results/focal_DinoBook-triplets-case1-features_superpoint_noresize_2048-LG.json'

    with open(res_path, 'r') as f:
        results1 = json.load(f)

    res_path = f'results/focal_DinoBook-triplets-case2-features_superpoint_noresize_2048-LG.json'

    with open(res_path, 'r') as f:
        results2 = json.load(f)

    print("DinoBook")
    print("********")
    print_full_table(results1, results2, experiments1, experiments2, cases = [1, 2], err_fun = (err_f1f3, err_f1f2))

    # cameras = set([x['img1'].split(ntpath.sep)[0] for x in results4])
    # cameras = set.union(set([x['img2'].split(ntpath.sep)[0] for x in results4]), cameras)
    # cameras = set.union(set([x['img3'].split(ntpath.sep)[0] for x in results4]), cameras)

    cameras = {'IPhoneZBHBack', 'IPhoneZBHfront', 'LenovoTabletBack', 'LenovoTabletFront', 'SamsungBack', 'SamsungFront',
     'SamsungGlossyBack', 'SamsungGlossyFront'}

    res_path = f'results/focal_Book-triplets-case1-features_superpoint_noresize_2048-LG.json'
    with open(res_path, 'r') as f:
        results1 = json.load(f)

    results1 = [x for x in results1 if {x['img1'].split(ntpath.sep)[0], x['img2'].split(ntpath.sep)[0],
                                        x['img3'].split(ntpath.sep)[0]} <= cameras]

    res_path = f'results/focal_Book-triplets-case2-features_superpoint_noresize_2048-LG.json'
    with open(res_path, 'r') as f:
        results2 = json.load(f)

    results2 = [x for x in results2 if {x['img1'].split(ntpath.sep)[0], x['img2'].split(ntpath.sep)[0],
                                        x['img3'].split(ntpath.sep)[0]} <= cameras]

    print("Book")
    print("****")
    print_full_table(results1, results2, experiments1, experiments2, cases = [1, 2], err_fun = (err_f1f3, err_f1f2))

def format_num(i):
    if i > 0:
        return str(i)
    else:
        return '\\ding{55}'

camera_descriptions = {'DellWide': 'Dell Precision 7650 notebook camera',
 'IPhoneOldBack': ' Apple IPhone SE (2nd generation) back camera',
 'IPhoneOldFront': ' Apple IPhone SE (2nd generation) front camera',
 'IPhoneZBHBack': ' Apple IPhone SE (3rd generation) back camera',
 'IPhoneZBHfront': ' Apple IPhone SE (3rd generation) front camera',
 'LenovoTabletBack': 'Tablet Lenovo TB-X505F back camera',
 'LenovoTabletFront': ' Tablet Lenovo TB-X505F front camera',
 'MotoBack': 'Motorola Moto E4 Plus back camera',
 'MotoFront': 'Motorola Moto E4 Plus front camera',
 'Olympus': 'Olympus uD600,S600 compact digital camera',
 'SamsungBack': 'Samsung Galaxy S5 Mini back camera',
 'SamsungFront': 'Samsung Galaxy S5 Mini front camera',
 'SamsungGlossyBack': 'Samsung Galaxy S III Mini back camera',
 'SamsungGlossyFront': 'Samsung Galaxy S III Mini front camera',
 'SonyTelescopic': 'Sony $\\alpha$5000 digital camera with 55-210mm Lens'}


def print_dataset_table(scenes, cameras, calib_dict, num_img_dict, num_c1_dict, num_c2_dict, num_c4_dict):
    ascenes = scenes[:-1]

    print('\\begin{tabular}{|c|c|ccc|ccccccc|ccc|}')
    print('\\cline{6-14}')
    print('\\multicolumn{5}{c}{} & \\multicolumn{7}{|c|}{Images} & \\multicolumn{2}{|c|}{Triplets} \\\\ \\hline')
    print(f'ID & Description & FOV & Width & Height & {" & ".join(scenes)} & \\case{{1}} & \\case{{2}}/III & \\case{{4}} \\\\ \\hline')
    for camera in cameras:
        print(f'{camera} & {camera_descriptions[camera]}  '
              f'& {calib_dict[camera]["fov"]:0.1f}$^\\circ$ & {calib_dict[camera]["width"]} & {calib_dict[camera]["height"]}'
              f'& {"&".join([format_num(num_img_dict[camera][s]) for s in scenes])} '
              f'& {np.sum([num_c1_dict[camera][s] for s in ascenes])}  & {np.sum([num_c2_dict[camera][s] for s in ascenes])}& {np.sum([num_c4_dict[camera][s] for s in ascenes])} \\\\ \\hline')

    # print('\\bottomrule')
    print(f'\\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{3}}{{|c|}}{{Total}} & {"&".join([format_num(np.sum([num_img_dict[c][s] for c in cameras])) for s in scenes])} &'
          f'{np.sum([np.sum([num_c1_dict[c][s] for c in cameras]) for s in ascenes])} &'
          f'{np.sum([np.sum([num_c2_dict[c][s]//2 for c in cameras]) for s in ascenes])} & {np.sum([np.sum([num_c4_dict[c][s] // 3 for c in cameras]) for s in ascenes])} \\\\ \\cline{{3-14}}')

    print(f'\\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{3}}{{|c|}}{{Triplets \\case{{1}}}} & {"&".join([format_num(np.sum([num_c1_dict[c][s] for c in cameras])) for s in ascenes])}'
          f'& \\ding{{55}} & \\multicolumn{{2}}{{c}}{{}} \\\\ \\cline{{3-12}}')
    print(f'\\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{3}}{{|c|}}{{Triplets \\case{{2}}}} & {"&".join([format_num(np.sum([num_c2_dict[c][s]//2 for c in cameras])) for s in ascenes])}'
          f'& \\ding{{55}} & \\multicolumn{{2}}{{c}}{{}} \\\\ \\cline{{3-12}}')
    print(f'\\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{3}}{{|c|}}{{Triplets \\case{{4}}}} & {"&".join([format_num(np.sum([num_c4_dict[c][s] // 3 for c in cameras])) for s in ascenes])}'
        f'& \\ding{{55}} & \\multicolumn{{2}}{{c}}{{}} \\\\ \\cline{{3-12}}')
    print('\\end{tabular}')





def dataset_table():
    dataset_path = '/home/kocurvik/data/H3vf/sym_scenes'
    matches_path = '/home/kocurvik/data/H3vf/sym_matches'
    scene_dict = {}

    cameras = [x for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))]

    scenes = get_basenames('custom_planar')
    scenes.append('Calib')

    with open(os.path.join(dataset_path, 'calib_data.json'), 'r') as f:
        calib_dict = json.load(f)

    num_img_dict = {}
    for camera in cameras:
        num_img_dict[camera] = {}
        for scene in scenes:
            try:
                subset_path = os.path.join(dataset_path, camera, scene)
                num_imgs = len([x for x in os.listdir(subset_path) if is_image(x)])

            except:
                num_imgs = 0
            num_img_dict[camera][scene] = num_imgs

    num_c1_dict = {k:{x: 0 for x in scenes} for k in cameras}
    num_c2_dict = {k:{x: 0 for x in scenes} for k in cameras}
    num_c4_dict = {k:{x: 0 for x in scenes} for k in cameras}

    for scene in scenes:
        c1_matches_path = os.path.join(matches_path, scene, 'triplets-case1-features_superpoint_noresize_2048-LG.txt')
        with open(c1_matches_path, 'r') as f:
            c1_lines = f.readlines();

        for line in c1_lines:
            camera = line.split(" ")[0].split(ntpath.sep)[0]
            num_c1_dict[camera][scene] += 1

    for scene in scenes:
        c2_matches_path = os.path.join(matches_path, scene, 'triplets-case2-features_superpoint_noresize_2048-LG.txt')
        with open(c2_matches_path, 'r') as f:
            c2_lines = f.readlines();

        for line in c2_lines:
            camera = line.split(" ")[0].split(ntpath.sep)[0]
            num_c2_dict[camera][scene] += 1
            camera = line.split(" ")[2].split(ntpath.sep)[0]
            num_c2_dict[camera][scene] += 1

    for scene in scenes:
        c4_matches_path = os.path.join(matches_path, scene, 'triplets-case4-features_superpoint_noresize_2048-LG.txt')
        try:
            with open(c4_matches_path, 'r') as f:
                c4_lines = f.readlines();

            for line in c4_lines:
                camera = line.split(" ")[0].split(ntpath.sep)[0]
                num_c4_dict[camera][scene] += 1
                camera = line.split(" ")[1].split(ntpath.sep)[0]
                num_c4_dict[camera][scene] += 1
                camera = line.split(" ")[2].split(ntpath.sep)[0]
                num_c4_dict[camera][scene] += 1
        except Exception:
            ...


    print_dataset_table(scenes, cameras, calib_dict, num_img_dict, num_c1_dict, num_c2_dict, num_c4_dict)

if __name__ == '__main__':
    # dataset_table()
    # print(20 * "*")
    eval_table()
    # eval_offplane_table_12()

