# A Combinatorial Interaction Testing Method for Multi-Label Image Classifier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13368486.svg)](https://doi.org/10.5281/zenodo.13368486)
[![DOI](https://zenodo.org/badge/DOI/10.1109/ISSRE62328.2024.00051.svg)](https://doi.org/10.1109/ISSRE62328.2024.00051)

This repository contains the source code of **LV-CIT**, a black-box testing method that applies Combinatorial Interaction Testing (CIT) to systematically test the ability of classifiers to handle such correlations. It also contains the replication package to reproduce the results reported in the ISSRE 2024 paper.

#### File Structure

```
.
├─data        # data and results
│  ├─lvcit    # images and results of LV-CIT
|  |  ├─0source_img  # source images from voc/coco, split into different dictionaries by labels 
|  |  |  ├─VOC
|  |  |  └─COCO
|  |  ├─1covering_array  # covering arrays generated by LV-CIT, Baseline, and ACTS
|  |  |  ├─adaptive random
|  |  |  ├─baseline
|  |  |  └─acts
|  |  ├─2matting_img   # matting images by YOLACT++ from source images
|  |  |  ├─VOC_output  # YOLACT++ output of VOC images
|  |  |  ├─VOC_output_model_pass  # results of first validation by DNN models
|  |  |  ├─VOC_library # object libraries of VOC images (results of second validation by human)
|  |  |  ├─COCO_output # YOLACT++ output of COCO images
|  |  |  ├─COCO_output_model_pass # results of first validation by DNN models
|  |  |  └─COCO_library # object libraries of COCO images (results of second validation by human)
|  |  ├─3composite_img  # composite images by LV-CIT
|  |  |  ├─VOC_20_v6_random_255_255_255_255_s1a0   # composite images of VOC by LV-CIT
|  |  |  └─COCO_80_v6_random_255_255_255_255_s1a0  # composite images of COCO by LV-CIT
|  |  ├─4results  # model outputs
|  |  |  ├─VOC_20_v6_random_255_255_255_255_s1a0   # results of different models on VOC
|  |  |  ├─COCO_80_v6_random_255_255_255_255_s1a0  # results of different models on COCO
|  |  |  └─atom  # results of ATOM provided by its authors
|  |  └─5res_analyse  # results of RQ1-RQ3
|  ├─coco/coco  # COCO2014 dataset
|  └─voc        # VOC2007 dataset
├─checkpoints # checkpoints released by authors of DNN models (see DNN Models Under Test)
|  ├─msrn
|  ├─asl
|  └─mlgcn
├─models       # for DNN models under test
├─dataloaders  # dataloaders for different datasets
├─img_classify2dir.py  # for splitting source images into different dictionaries by labels
├─ca_generator.py      # for generating covering arrays by LV-CIT and Baseline
├─run_acts.py          # for running ACTS
├─check_libraries.py   # for first validation step by DNN models (to build the object libraries)
├─compositer.py        # for generating composite images
├─lvcit_main.py        # for executing the tests for composite images
├─lvcit_runner.py
├─default_main.py      # for executing the tests for VOC/COCO test(validation) sets to get outputs of Random method
├─default_runner.py
├─plot.py              # for generating figures in the paper
├─ana_atom_info.py     # for analysing the results of ATOM
├─ana_train_val_info.py  # for analysing the results of VOC/COCO test(validation) sets
├─analyse.py           # for generating the tables in the paper
├─util.py
├─requirements.txt
└─README.md
```

## Experiment Setup

LV-CIT is implemented in Python (version 3.8). The code is developed and tested under a Windows platform (Windows 11), on a computer equipped with an AMD Ryzen 7 5800X 8-core CPU @ 3.8GHz, a 32GB RAM, and an NVIDIA GeForce RTX 4060Ti GPU with 8GB VRAM. The cuda and cudnn versions are 11.7 and 8.9.6, respectively.
The following steps are required to set up the environment and run the experiments:

1. Run the following command to install dependency packages (in the root directory of this repository):

```
conda create -n lvcit python=3.8
conda activate lvcit
pip install -r requirements.txt
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Note that direct installation of `inplace-abn` may fail due to the lack of dependencies. For Windows users, please install microsoft visual studio and add PATH_TO_BIN to system environment before installing inplace_abn; for Linux users, please download the [source code](https://github.com/mapillary/inplace_abn/releases) and install inplace_abn from source code (Note that the PyTorch and CUDA versions should be compatible).

```
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python setup.py install
cd scripts
pip install -r requirements.txt
```

2. Download the checkpoints of the DNN models under test: [MSRN](https://github.com/chehao2628/MSRN), [ML-GCN](https://github.com/megvii-research/ML-GCN), and [ASL](https://github.com/Alibaba-MIIL/ASL). Then put them in the `checkpoints/<model_name>/` directory where `<model_name>` is `msrn`, `mlgcn`, or `asl`, and rename them to `<dataset>_checkpoints.pth.tar` where `<dataset>` is `voc` or `coco` (See more details in [checkpoints](./checkpoints/)). Addtionally, download [ResNet-101 pretrained model](https://github.com/chehao2628/MSRN?tab=readme-ov-file#2-download-resnet-101-pretrained-model) for MSRN, put the model in the `checkpoints/msrn/` directory, and rename it to `resnet101_for_msrn.pth.tar`.


3. Download the [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) and [COCO2014](https://cocodataset.org) datasets and put them in the `data/<dataset>/tmp` directory, where `<dataset>` is `voc` or `coco/coco` (or you can just run `python default_main.py` in [step 4](#4-execute-tests) and the datasets will be download automatly).

4. Download our created [object libraries](https://drive.google.com/drive/folders/1_z7JdSVLbSxoTLke3J6249HhtKztXlhj) and put them in the `data/lvcit/2matting_img/` directory.

## Usage

### Quick Start

Our tool consists of three main components: label value covering array generation, composite images generation, and test execution. To quickly validate its general functionality, we provide a [sample dataset](https://drive.google.com/file/d/1iM_FxCASpqlMli3FoN3aOmgqawOCHuAA). Download it and replace the entire `data` folder, then execute the following code to validate each function individually.

#### 1. Generate Label Value Covering Arrays

Run the following code to generate a label value covering array for $n=6$, $k=3$ and $t=2$ (where $n$ indicates the size of label space, $k$ indicates the counting constraint value, and $t$ indicates the covering strength). The covering array will be saved in `data/lvcit/1covering_array/adaptive_random/`.

```bash
python ca_generator.py --all=False -m "adaptive random" -n 6 -k 3 -t 2
```

#### 2. Generate Composite Images

Run the following code to generate composite images. This step will use the covering array generated in the previous step and the object libraries provided in the sample dataset (`data/lvcit/3matting_img/VOC_library`) to generate composite images. The results will be saved in `data/lvcit/3composite_img`.

```bash
python compositer.py --demo=True
```

#### 3. Execute Tests

Run the following code to use the composite images generated by LV-CIT to test the DNN models. The results will be saved in `data/lvcit/4results`.

```bash
python lvcit_main.py --demo=True
```

### Reproducibility Instructions

#### 1. Generate Label Value Covering Arrays

> Note that the covering arrays used in the experiment are already included in the replication package.

Run the following code to generate label value covering arrays. The generation results can be found in `data/lvcit/1covering_array/` (the results of LV-CIT and Baseline are saved in `adaptive_random` and `baseline` subdirectories, respectively).

```bash
python ca_generator.py
```

Download the [ACTS](https://csrc.nist.gov/Projects/automated-combinatorial-testing-for-software) tool and run the following command to generate covering arrays by ACTS. Save the results to `data/lvcit/1covering_array/acts/`.

```bash
java -Ddoi=2 -Doutput=csv -Dchandler=solver -Dprogress=on -Drandstar=on -jar acts_3.2.jar args_<n>_<k>.txt ca_acts_<n>_<k>_2_<No.>.csv  # please replace <n>, <k>, and <No.> with the label space size, the counting constraint value, and the number of covering arrays, respectively.
```

Or, you can use our provided script `run_acts.py` to run ACTS. Please copy `run_acts.py` to the directory of ACTS (same with `acts_3.2.java`) and run the following command to generate covering arrays by ACTS:

```bash
python run_acts.py
```

Then save the results to `data/lvcit/1covering_array/acts/` and copy the generated file `ca_acts_info.csv` to `data/lvcit/1covering_array/` for further analysis.

#### 2. Build Object Libraries

> This step can be skipped if you choose to download our released object libraries.

First, get the matting images by the following steps:
1. Get the codes and pretrained models for [YOLACT++](https://github.com/dbolya/yolact) and replace `eval.py` with `yolact/eval.py` we provided.
2. Run the following code to split source images into different dictionaries by labels, and save them in `data/lvcit/0source_img/<dataset>/`, where `<dataset>` is `VOC` or `COCO`:

```bash
python img_classify2dir.py
```

3. Run the following code to generate matting images (see [YOLACT++](https://github.com/dbolya/yolact) for more details), and save these images in `data/lvcit/2matting_img/<dataset>_output/`, where `<dataset>` is `VOC` or `COCO`.

```bash
python eval.py --trained_model=weights/yolact_plus_base_54_800000.pth --score_threshold=0.15 --top_k=15 --display_masks=False --display_text=False --display_bboxes=False --display_scores=False --images=data/lvcit/0source_img/<dataset>:data/lvcit/2matting_img/<dataset>_output
```

Next, use the following steps to validate these images:

- Run the following code to validate object images by DNN models. The images that can be correctly classified will be saved in `data/lvcit/2matting_img/<dataset>_output_model_pass`:
```bash
python check_libraries.py
```
- Validate the above images by human based on the following criteria: 1) the prominent features of the target object are complete and easily recognisable, and 2) there is no other object in the image.


The final object libraries are saved in `data/lvcit/2matting_img/<dataset>_library`, where `<dataset>` is `VOC` or `COCO`.

#### 3. Generate Composite Images

Execute the following command to generate composite images:

```bash
python compositer.py
```

The execution of the above script relies on the label value covering arrays generated in Step 1 and the object libraries generated in Step 2. The results will be saved in `data/lvcit/3composite_img`.

#### 4. Execute Tests

Run the following code to use compsite images generated by LV-CIT to test the DNN models:

```bash
python lvcit_main.py
```

In addition to LV-CIT, run the following code to use randomly selected images (from test/validation sets of VOC and COCO) to test the DNN models (i.e., the Random method):

```bash
python default_main.py
```

The results of the above two methods will be saved in `data/lvcit/4results`. The [results of the ATOM](https://drive.google.com/drive/folders/1Tf6B5g0uUi1kdyuES6UxXWIljLkbGrrk) are provided by the authors of [ATOM](https://github.com/GIST-NJU/ATOM) and please download and save them in `data/lvcit/4results/atom`.

#### 5. Result Analysis

Run the following commands to get the figures and tables presented in the paper:

```bash
python plot.py
python ana_atom_info.py
python ana_train_val_info.py
python analyse.py
```
