# Generic Event Boundary Detection in Video with Pyramid Features

This repo is for our paper at https://arxiv.org/abs/2301.04288

```
@article{huynh2023generic,
  title={Generic Event Boundary Detection in Video with Pyramid Features},
  author={Huynh, Van Thong and Yang, Hyung-Jeong and Lee, Guee-Sang and Kim, Soo-Hyung},
  journal={arXiv preprint arXiv:2301.04288},
  year={2023}
}
```

* The required packages for our repo can be found in `requirements.txt`. We also included 'Dockerfile` of our repo.
* To generate tfrecord files for using within this repo, refer to `core/dataloader.py#L342` for [Kinetics-GEBD](https://openaccess.thecvf.com/content/ICCV2021/papers/Shou_Generic_Event_Boundary_Detection_A_Benchmark_for_Event_Segmentation_ICCV_2021_paper.pdf) and `core/tapos_utils.py#L350` for [TAPOS](https://sdolivia.github.io/TAPOS/) datasets.
* The sample training scripts can be found in `scripts/train.sh` and `docker_train.sh`
