import json
import ntpath
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from tqdm import tqdm
import seaborn as sns


from dataset_utils.data import experiments, iterations_list, get_basenames, pose_err_max, get_experiments

# large_size = 12
# small_size = 10

large_size = 24
small_size = 20

print(sns.color_palette("tab10").as_hex())

plt.rcParams.update({'figure.autolayout': True})

# plt.rcParams.update({'figure.autolayout': True})
rc('font',**{'family':'serif','serif':['Times New Roman']})
# rc('font',**{'family':'serif'})
rc('text', usetex=True)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def draw_results_focal_auc(results, experiments, iterations_list, colors=None, title='', save=None):
    plt.figure(frameon=False, figsize=(6, 4.5))

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = [r['f1_err'] for r in iter_results]
            errs = np.array(errs)
            errs[np.isnan(errs)] = 1.0
            AUC10 = np.mean(np.array([np.sum(errs * 100 < t) / len(errs) for t in range(1, 11)]))

            xs.append(mean_runtime)
            ys.append(AUC10)
        if colors is None:
            plt.semilogx(xs, ys, label=experiment, marker='*')
        else:
            plt.semilogx(xs, ys, label=experiment, marker='*', color=colors[experiment])

    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('mAA$_f$(0.1)', fontsize=large_size)
    plt.ylim([0.2, 0.58])
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)

    if not save:
        plt.title(title, fontsize=8)
        plt.legend()
        plt.show()
    else:
        # plt.savefig(save, bbox_inches='tight', pad_inches=0)
        plt.savefig(save, pad_inches=0)


def draw_results_focal_med(results, experiments, iterations_list, colors=None, title='', save=None):
    plt.figure(frameon=False, figsize=(6, 4.5))

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = [r['f1_err'] for r in iter_results]
            errs = np.array(errs)
            errs[np.isnan(errs)] = 1.0


            xs.append(mean_runtime)
            ys.append(np.median(errs))
        if colors is None:
            plt.semilogx(xs, ys, label=experiment, marker='*')
        else:
            plt.semilogx(xs, ys, label=experiment, marker='*', color=colors[experiment])

    # title += f"Error: max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err']))"



    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('Median $\\xi_f$', fontsize=large_size)
    plt.ylim([0, 0.24])
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)

    if not save:
        plt.title(title, fontsize=8)
        plt.legend()
        plt.show()
    else:
        # plt.savefig(save, bbox_inches='tight', pad_inches=0)
        plt.savefig(save, pad_inches=0)

def draw_results_focal_cumdist(results, experiments, title='', save=None):
    cameras = np.unique([ntpath.basename(x['img1'].split(ntpath.sep)[0]) for x in results])

    for camera in cameras:
        cam_results = [x for x in results if camera in x['img1']]
        plt.figure()
        for experiment in tqdm(experiments):
            experiment_results = [x for x in cam_results if x['experiment'] == experiment]

            xs = np.arange(101)

            errs = [r['f1_err'] for r in experiment_results]
            # errs.extend([r['f2_err'] for r in experiment_results])
            # errs.extend([r['f3_err'] for r in experiment_results])
            errs = np.array(errs)
            errs[np.isnan(errs)] = 1.0
            res = np.array([np.sum(errs * 100 < t) / len(errs) for t in xs])

            plt.plot(xs / 100, res, label=experiment)

        # title += f"Error: max(0.5 * (out['R_12_err'] + out['R_13_err']), 0.5 * (out['t_12_err'] + out['t_13_err']))"

        plt.title(title + camera, fontsize=8)

        plt.xlabel('f_err', fontsize=large_size)
        plt.ylabel('Portion of samples', fontsize=large_size)
        plt.tick_params(axis='x', which='major', labelsize=small_size)
        plt.tick_params(axis='y', which='major', labelsize=small_size)
        plt.legend()
        if save:
            plt.savefig(save)
        else:
            plt.show()

def draw_results_focal_cumdist_all(results, experiments, colors=None, title='', save=None):
    cameras = np.unique([ntpath.basename(x['img1'].split(ntpath.sep)[0]) for x in results])

    plt.figure()
    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = np.arange(101)

        errs = [r['f1_err'] for r in experiment_results]
        errs = np.array(errs)
        errs[np.isnan(errs)] = 1.0
        res = np.array([np.sum(errs * 100 < t) / len(errs) for t in xs])

        if colors is None:
            plt.plot(xs / 100, res, label=experiment)
        else:
            plt.plot(xs / 100, res, label=experiment, color=colors[experiment])

    plt.xlabel('$f_{err}$', fontsize=large_size)
    plt.ylabel('Portion of samples', fontsize=large_size)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if save:
        plt.savefig(save)
    else:
        plt.title(title, fontsize=8)
        plt.legend()
        plt.show()


def draw_results_pose_auc10(results, experiments, iterations_list, title=None):
    # plt.figure(frameon=False)
    plt.figure()

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([pose_err_max(out) for out in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))
            # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

            xs.append(mean_runtime)
            ys.append(AUC10)

        # plt.semilogx(xs, ys, label=experiment, marker='*', color=colors[experiment])
        plt.semilogx(xs, ys, label=experiment, marker='*')

    # plt.xlim(xlim)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('AUC@10$^\\circ$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        plt.title(title)
        plt.savefig(f'figs/{title}_pose.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved pose: {title}')
    else:
        plt.legend()
        plt.show()


def draw_results_pose_portion(results, experiments, iterations_list, title=None):
    plt.figure(frameon=False)

    for experiment in tqdm(experiments):
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        iter_results = experiment_results
        mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
        errs = np.array([out['R_23_err'] for out in iter_results])
        # errs = np.array([0.5 * (out['t_12_err'] + out['t_13_err']) for out in iter_results])
        errs[np.isnan(errs)] = 180
        cum_err = np.array([np.sum(errs < t) / len(errs) for t in range(1, 181)])

        # AUC10 = np.mean([x['info']['inlier_ratio'] for x in iter_results])

        xs = np.arange(1, 181)
        ys = cum_err

        plt.plot(xs, ys, label=experiment, marker='*', color=colors[experiment])

    # plt.xlim([5.0, 1.9e4])
    plt.xlabel('Mean runtime (ms)', fontsize=large_size)
    plt.ylabel('AUC@10$^\\circ$', fontsize=large_size)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    if title is not None:
        # plt.legend()
        plt.savefig(f'figs/{title}_cumpose.pdf', bbox_inches='tight', pad_inches=0)
        print(f'saved pose: {title}')

    else:
        plt.legend()
        plt.show()

def generate_graphs(dataset, results_type, case, all=True):
    basenames = get_basenames(dataset)
    experiments, colors = get_experiments(case)

    results = []
    for basename in basenames:
        json_path = os.path.join('results', f'focal_{basename}-graph-{results_type}.json')
        print(f'json_path: {json_path}')
        with open(json_path, 'r') as f:
            if all:
                results.extend([x for x in json.load(f) if x['experiment'] in experiments])
            else:
                results = [x for x in json.load(f) if x['experiment'] in experiments]
                draw_results_focal_auc(results, experiments, iterations_list, f'{dataset}_{basename}_{results_type}')

    if all:
        draw_results_focal_auc(results, experiments, iterations_list, colors=colors, save=f'figs/all_case{case}_fauc.pdf')
        draw_results_focal_med(results, experiments, iterations_list, colors=colors, save=f'figs/all_case{case}_fmed.pdf')


    # results = []
    # for basename in basenames:
    #     json_path = os.path.join('results', f'focal_{basename}-{results_type}.json')
    #     print(f'json_path: {json_path}')
    #     with open(json_path, 'r') as f:
    #         if all:
    #             results.extend([x for x in json.load(f) if x['experiment'] in experiments])
    #         else:
    #             results = [x for x in json.load(f) if x['experiment'] in experiments]
    #             draw_results_focal_auc(results, experiments, iterations_list, f'{dataset}_{basename}_{results_type}')
    #
    # if all:
    #     draw_results_focal_cumdist_all(results, experiments, colors=colors, save=f'figs/all_case{case}_fcumdist.pdf')

if __name__ == '__main__':
    # generate_graphs('custom_planar', 'triplets-case1-features_superpoint_noresize_2048-LG', 1, all=True)
    # generate_graphs('custom_planar', 'triplets-case2-features_superpoint_noresize_2048-LG', 2, all=True)
    generate_graphs('custom_planar', 'triplets-case2-features_superpoint_noresize_2048-LG-c3', 3, all=True)

