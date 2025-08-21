# Three-view Focal Length Recovery From Homographies

This repo contains code for paper "Three-view Focal Length Recovery From Homographies" (CVPR 2025) Available on [CVF repository](https://openaccess.thecvf.com/content/CVPR2025/html/Ding_Three-view_Focal_Length_Recovery_From_Homographies_CVPR_2025_paper.html) or on arxiv: [2501.07499](https://arxiv.org/abs/2501.07499)

## Installation

Create an environment with pytorch and packaged from `requirements.txt`.

Install [PoseLib fork with implemented estimators](https://github.com/kocurvik/PoseLib-hf) into the environment:
```shell
git clone https://github.com/kocurvik/PoseLib-hf
git cd PoseLib-hf
pip install .
```

Before running the python scripts make sure that the repo is in your python path (e.g. `export PYTHONPATH=/path/to/repo/hf`)

## Dataset

You can download the HomoTT dataset with matches [here](https://doi.org/10.5281/zenodo.14638904). The GT focal lengths were obtained using `dataset_utils\calibrate.py` and matches were obtained using `prepare_custom.py`.

## Evaluation

To perform the evaluation on real data run:
```shell
for x in $( ls /path/to/dataset/sym_matches); do
  echo $x
  python eval_f.py -c 1 -nw 64 triplets-case1-features_superpoint_noresize_2048-LG /path/to/dataset/sym_matches/$x
  python eval_f.py -c 2 -nw 64 triplets-case2-features_superpoint_noresize_2048-LG /path/to/dataset/sym_matches/$x
  python eval_f.py -c 1 -g -nw 64 triplets-case1-features_superpoint_noresize_2048-LG /path/to/dataset/sym_matches/$x
  python eval_f.py -c 2 -g -nw 64 triplets-case2-features_superpoint_noresize_2048-LG /path/to/dataset/sym_matches/$x
done
```

You can run the synthetic experiments using `boxplot.py`.

## Citation
```
@inproceedings{ding2025three,
  title={Three-view Focal Length Recovery From Homographies},
  author={Ding, Yaqing and Kocur, Viktor and Haladov{\'a}, Zuzana Berger and Wu, Qianliang and Cai, Shen and Yang, Jian and Kukelova, Zuzana},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={11505--11514},
  year={2025}
}
```


