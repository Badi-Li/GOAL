### Data and Model Weights Preparation
1. Download scene datasets and episodes datasets of MP3D and HM3D according to the instructions [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)

2. For generating semantic maps dataset, please refer to the scripts offered by [PONI](https://github.com/srama2512/PONI). Note that for HM3D, semantic maps construction requires minor adjustment to the codes as HM3D uses a different files formatting than MP3D. We also provide precomputed semantic maps dataset [here](https://drive.google.com/drive/folders/1ChP4ry1QUpQACt4DEfshKhn40uS2G3p4?usp=drive_link).

3. Following [PONI](https://github.com/srama2512/PONI), we split the validation episodes datasets into multiple parts and run them independently in parallel, so you should split the downloaded episodes datasets, or download directly from [here](https://drive.google.com/drive/folders/1ziiEyBOnRO5A2XHm24XSt5ext8HlkCQH?usp=drive_link).

4. We provide multiple model weights trained with different LLMs and on different datasets. You can find them [here](https://drive.google.com/drive/folders/1Lz8lYl-KZYWzwXnYF-623NKz04zjowL6?usp=drive_link), with file name scheduled as `{dataset}_{llm}`

5. We provide model weights of sparse unet for segmentation [here](https://drive.google.com/file/d/194ZN-eua0CjN9o1ymbf4_9eLY1uUhXyT/view?usp=drive_link).

6. Structure all downloaded data and model weights as follows
    ```bash
    GOAL
      |-- data
        |-- scene_datasets
        |-- datasets
            |-- objectnav
        |-- semantic_maps
      |-- pretrained_models
        |-- ...
    ```

### Environments Setup
We recommend separate environments for training (generative flow) and evaluation (ObjectGoal navigation).
#### training environment
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

#### evaluation environment
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

```

### Running
We provide experiments scripts with different settings in `./experiments_scipts`. 
#### Training
For training, we use the first 4 GPUs by default, and you can adjust the visible GPUs in the experiment scripts. 
#### Evaluation
For evaluation, the scripts is used as:
```bash
sh script.sh <GPUs ids> <threads per GPU> <parts>
```
By default, the parts will be all available splits (11 for MP3D and 20 for HM3D), for example
```bash
sh script.sh 0,1 6 
```
will run all parts on gpus 0 and 1, with 6 parts for each gpu. We evaluate 6 threads in parallel with a single GPU with 24GB memory. If the memory of your GPU is smaller, please adjust the number according to your setting. 