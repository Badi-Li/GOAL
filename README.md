# [NeurIPS 2025] GOAL: Distilling LLM Prior to Flow Model for Generalizable Agent’s Imagination in Object Goal Navigation
This repository contains Pytorch implementation of our paper: [Distilling LLM Prior to Flow Model for Generalizable Agent’s Imagination in Object Goal Navigation](http://arxiv.org/abs/2508.09423)

## Data and Model Weights Preparation
1. Download scene datasets of MP3D and HM3D according to the instructions [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md). Place the downloaded datasets under the directory `./data/scene_datastets`

2. For generating semantic maps dataset, please refer to the scripts offered by [PONI](https://github.com/srama2512/PONI). Note that for HM3D, minor modifications to the code are required due to differences in file formatting compared to MP3D. Alternatively, we also provide precomputed semantic maps at [MP3D](https://drive.google.com/file/d/1k4nreOA9xhC8PnKhk2FTlcsuAsZaAJki/view?usp=drive_link) and [HM3D](https://drive.google.com/file/d/174Vu2p97SRiRktLdfHV_4XaHFxucoz3F/view?usp=drive_link). After downloading, extract the files and place them in `./data/semantic_maps`.

3. Following [PONI](https://github.com/srama2512/PONI), the validation episode datasets are split into multiple parts for parallel processing. You may either download episodes datasets according to [instructions](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) and split the datasets yourself or download the pre-split datasets directly from [here](https://drive.google.com/drive/folders/1ziiEyBOnRO5A2XHm24XSt5ext8HlkCQH?usp=drive_link). Place the resulting files under `./data/datasets/objectnav`

4. We follow the common practices in [SGM](https://github.com/sx-zhang/SGM), [T-Diff](https://github.com/sx-zhang/T-diff) etc., to leverage area potential function from [PONI](https://github.com/srama2512/PONI) as a frontier based exploration strategy, when the prediction confidence of GOAL is low (e.g. at the very beginning of navigation with limited observations). You can download from official repo of [PONI](https://github.com/srama2512/PONI) or directly from [here](https://drive.google.com/file/d/1DpG4k7lFl6SV54Eud2CmPgva2CEQsVYD/view?usp=drive_link). Place the file as `./pretrained_models/area_potential.pth`. 

4. We provide pretrained models trained on MP3D and HM3D with ChatGPT Prior:

| Dataset | Models        |
|:-------:|:---------------:|
| MP3D    | [mp3d_chatgpt](https://drive.google.com/file/d/1t3d-EWvN4G6DcecRPRyWOehV4rPPHO07/view?usp=drive_link)    | 
| HM3D    | [hm3d_chatgpt](https://drive.google.com/file/d/1CELihXmObvU3jAgcGrcujsK8IcHONcAh/view?usp=drive_link)    |  


5. We provide model weights of sparse unet for segmentation [here](https://drive.google.com/file/d/14miU42QOVahzjIWCLijLxECke3X8MGeL/view?usp=drive_link). Place it as `./pretrained_models/spconv_state.pth`.

## Environments Setup
We recommend separate environments for training (generative flow) and evaluation (ObjectGoal navigation).
### training environment
1. Create environment:
```bash
conda create -n goal-train python=3.9
conda activate goal-train
```
2. Install Pytorch. For example, we use `v2.1.0` with `cuda11.8` for training.
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```
3. Install other necessary packages
```bash
# timm for necessary components of DiT; 
# sklearn for DBSCAN for clustering observed objects
pip install timm scikit-learn tensorboard
```

### evaluation environment
1. Create enviroment:
```bash
conda create -n goal-eval python=3.8
conda activate goal-eval
```
2. Install Pytorch. For example, we use `v1.12.0` with `cuda11.6` for evaluation. 
```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
3. Install [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab), we use version `v0.2.1`. However, we recommend installing habtiat-sim file directly from [here](https://anaconda.org/aihabitat/habitat-sim/files), as using official download scripts sometimes cause unexpected errors. 

4. Install [spconv](https://github.com/traveller59/spconv) for sparse unet. 
```bash
# adjust cuda version according to your setup 
pip install spconv-cu116
```

5. Install other necessary packages
```bash
pip install scikit-image scikit-fmm einops timm six torch_geometric torchdiffeq
```

6. Compile `pyastar` for local planning:
```bash
cd ./nav/astar_pycpp && make
```

We also provide the yaml files `train_env.yaml` and `eval.env.yaml` for reference.  

---

## Running Experiments

Experiment scripts with various configurations are available in the `./experiments_scripts` directory. You should first activate corresponding environments
```bash
# For training 
conda activate goal-train
# For evaluation 
conda activate goal-eval
```
Experiment scripts with various configurations are available in the `./experiments_scripts` directory. You should first activate corresponding environments
```bash
# For training 
conda activate goal-train
# For evaluation 
conda activate goal-eval
```

### Training

By default, training utilizes the first four GPUs. You may modify the visible GPU devices by editing the corresponding experiment scripts.

### Evaluation

1. Set the environment variable `GOAL_ROOT` to point to the root directory of the GOAL repository:
    ```bash
    export GOAL_ROOT=<YOUR_PATH_TO_GOAL>
    export PYTHONPATH=<YOUR_PATH_TO_GOAL>
    ```

2. Run the evaluation script as follows:
    ```bash
    sh script.sh <GPU_IDS> <THREADS_PER_GPU> <PARTS>
    ```

    - `<GPU_IDS>`: Comma-separated list of GPU device IDs, e.g., `0,1`  
    - `<THREADS_PER_GPU>`: Number of parallel threads per GPU  
    - `<PARTS>`: (Optional) Specify dataset splits to evaluate; if omitted, all splits will be used (11 splits for MP3D, 20 splits for HM3D)

    Example:
    ```bash
    sh script.sh 0,1 6
    ```
    This command runs all parts on GPUs 0 and 1, with 6 threads per GPU. Note that 6 threads per GPU corresponds approximately to a 22GB GPU memory requirement. Please adjust the thread count according to your hardware capacity.
    This command runs all parts on GPUs 0 and 1, with 6 threads per GPU. Note that 6 threads per GPU corresponds approximately to a 22GB GPU memory requirement. Please adjust the thread count according to your hardware capacity.

3. After evaluation completes, merge results from all parts to obtain overall performance statistics:
    ```bash
    python $GOAL_ROOT/nav/merge_results.py --path_format "$EXPT_ROOT/<EXP_NAME>/tb_seed_100_val_part_*/stats.json"
    ```

## Acknowledgements
Our work is built upon [PONI](https://github.com/srama2512/PONI), [SGM](https://github.com/sx-zhang/SGM), [flow_matching](https://github.com/facebookresearch/flow_matching), [astar_pycpp](https://github.com/srama2512/astar_pycpp).

## Citation 
If you find this codebase useful, please site us:
```bash
@article{li2025distilling,
  title={Distilling LLM Prior to Flow Model for Generalizable Agent's Imagination in Object Goal Navigation},
  author={Li, Badi and Lu, Ren-jie and Zhou, Yu and Meng, Jingke and Zheng, Wei-shi},
  journal={arXiv preprint arXiv:2508.09423},
  year={2025}
}
```
