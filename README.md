# Rescue VLM: Dual-Stream Vision–Language Framework for Disaster Assessment

This repository contains the implementation and experiment artifacts for a vision–language framework for disaster assessment using bi-temporal satellite imagery. The pipeline supports disaster caption generation and disaster recovery recommendation.

## Dataset and preparation

The experiments use the **DisasterM3** dataset. The repository's `data/` directory contains the JSON files used to organize the experimental samples:

```text
data/
├── train.json
├── validation.json
└── test.json
```

To prepare the image data, obtain the DisasterM3 training archive from the [official project repository](https://github.com/Junjue-Wang/DisasterM3) and extract it locally. Keep the extracted training images in the directory expected by `dataset_loaderv3.5.py` (for example, `data/train_images/` if that matches the paths configured in the loader). The JSON files define the sample records for the corresponding splits; the loader uses those records to locate the images and read the required annotations. If your extracted archive has a different directory layout, update the image-path mapping in the loader accordingly.

Before running an experiment, verify that the paths referenced by each JSON file resolve to the extracted images and that the expected fields match the dataset loader. Keep the JSON files, image-path mapping, preprocessing configuration, split assignments, and random seed unchanged when reproducing a reported run. The original image archive is not duplicated in this repository.

## Repository structure

```text
Rescue_VLM_Ultimate_Version3.5/
├── README.md
├── data/
│   ├── train.json
│   ├── validation.json
│   └── test.json
├── dataset_loaderv3.5.py
├── models/
│   └── projector_v3.5.py
├── scripts/
│   └── multimodal_v3/
│       ├── train_vlm_v3.5.py
│       ├── test_pipeline_v3.5.py
│       ├── evaluate_vlm_v3.5.py
│       ├── domain_eval_v3.5.py
│       └── test_qwen25vl.py
├── baseline_qwen25vl/
│   ├── test_qwen25vl.py
│   └── outputs/
├── outputs/
│   ├── multimodal_v3/
│   │   └── best_checkpoint/
│   ├── test_predictions_v4.json
│   ├── eval_scores_v4_local.json
│   └── eval_scores_v5_fast.json
├── bata_explosion_pre_0.png
├── bata_explosion_post_0.png
├── turkey_earthquake4_pre_733.png
└── turkey_earthquake4_sar_post_733.png
├── requirements.txt
```

The locally downloaded Qwen model weights and Hugging Face cache are not required to be committed to the public repository. Obtain pretrained model files from their official source and follow the relevant model licence.

## Environment setup

The following commands assume Linux and Python 3:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `requirements.txt` with the package versions validated for the experiments before publishing the repository. For GPU-based runs, record the GPU, driver, CUDA runtime, PyTorch, Transformers, and other relevant library versions.

## Training and evaluation

Check each script's supported arguments and configuration before execution:

```bash
python scripts/multimodal_v3/train_vlm_v3.5.py --help
python scripts/multimodal_v3/test_pipeline_v3.5.py --help
python scripts/multimodal_v3/evaluate_vlm_v3.5.py --help
python scripts/multimodal_v3/domain_eval_v3.5.py --help
python baseline_qwen25vl/test_qwen25vl.py --help
```

If a script does not support `--help`, inspect its argument definitions and configuration directly. Set the dataset, base-model, checkpoint, and output paths according to the options implemented in the code. Record the exact commands and configuration used for each reported experiment.

## Checkpoints and outputs

The `outputs/` directories contain checkpoint components, training logs, predictions, and evaluation files from different runs. The multimodal checkpoint includes vision-component weights, a LoRA adapter, projector weights, and training state. Verify the model version, evaluation split, sample count, metric definition, and aggregation procedure before using any output in a report. File names alone do not establish that results from different runs are directly comparable.

## Reproducibility

For reproducibility, document the DisasterM3 source/version, the JSON split files, the extracted-image path mapping, preprocessing settings, split or cross-validation configuration, random seed, base-model identifier and revision, software and hardware versions, training hyperparameters, checkpoint-selection rule, and evaluation metrics. Use the same data preparation and evaluation protocol when comparing the proposed method with the baseline.

## Data use and citation

Cite the original DisasterM3 publication and dataset source when using the dataset. Follow the dataset provider's current terms and restrictions. Check the licences of pretrained models and third-party components separately, and do not redistribute the original image archive or model weights unless their terms explicitly permit it.
