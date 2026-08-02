# RecViT Attention Rollout Manual

This project implements and evaluates an adapted Attention Rollout method for the Recurrent Vision Transformer (RecViT). The main purpose is to generate attention-based explanation maps for image classification, compare raw RecViT attention maps with recurrent rollout maps, and evaluate the generated maps against PET trimap-based foreground masks.

The project was created for a bachelor thesis about interpretability of Vision Transformer models, with a focus on adapting Attention Rollout to the recurrent structure of RecViT.

## Main features

- Load trained RecViT checkpoints for CIFAR-10, CIFAR-100, or Oxford-IIIT PET.
- Extract attention matrices from RecViT attention blocks using forward hooks.
- Compute standard Attention Rollout inside each recurrent step.
- Connect recurrent steps using a RecViT-specific rollout aggregation method.
- Compare two patch-transition assumptions across recurrent steps:
  - `identity`: patch positions are treated as corresponding across steps.
  - `zero`: patch tokens are treated as new independent inputs at each step.
- Generate final rollout heatmaps and step-wise rollout heatmaps.
- Generate raw attention maps for selected recurrent steps and layers in the grid experiment.
- Evaluate attention maps against trimap-derived foreground/background masks.
- Save visual outputs, NumPy masks, CSV score files, and run metadata.

## Project structure

Typical important files:

```text
.
├── rec_vit_explain.py              # Run RecViT rollout for one image/configuration
├── rec_vit_rollout.py              # Core RecViT-adapted Attention Rollout implementation
├── run_recvit_rollout_grid.py      # Main PET grid experiment script
├── attention_map_evaluation.py     # Trimap-based attention scoring utilities
├── reveal_segmentation.py          # Trimap generation and visualization utilities
├── experiment_cifar10.py           # Original/helper CIFAR-10 attention experiment script
├── examples_PET/                   # PET input images and matching trimap images
├── rollout_results/                # Default single-run output folder
└── rollout_PET_grid_results/       # Default grid-experiment output folder
```

## RecViT Dependency

This project depends on the original RecViT implementation by Pocos et al. All necessary scripts, model definitions, utilities, and trained checkpoints required to run the experiments are stored in the `rec_vit_model` directory.

The `rec_vit_model` directory contains the original RecViT codebase together with the trained model checkpoints used in the experiments. The scripts developed as part of this thesis, including `rec_vit_rollout.py`, `rec_vit_explain.py`, `run_recvit_rollout_grid.py`, `attention_map_evaluation.py`, and `reveal_segmentation.py`, build on top of this implementation and use its utilities for model loading and inference.

In particular, trained models are loaded through:

```python
from rec_vit_model.pckgs.networks.network_utils import load_trained_network
```

Therefore, the `rec_vit_model` directory must remain present in the project structure for the provided scripts to run correctly.

The original RecViT implementation is not the contribution of this thesis. The contribution of this work consists of the adapted recurrent Attention Rollout implementation, attention-map generation and evaluation scripts, experiment automation scripts, and the methodology for comparing raw attention maps and recurrent rollout maps.

Make sure the project is placed or launched so that this import works.

## Environment setup

The project is written in Python and uses PyTorch. A CUDA GPU is recommended for larger grid experiments, but a single run can also be executed on CPU.

Recommended Python packages:

```text
torch
torchvision
timm
numpy
opencv-python
Pillow
```

Example setup with Conda:

```bash
conda create -n recvit python=3.10
conda activate recvit
pip install torch torchvision timm numpy opencv-python pillow
```

If you use a CUDA version of PyTorch, install it according to the official PyTorch instructions for your system.

## Required data and checkpoints

### Input images

For PET experiments, the expected input folder is:

```text
examples_PET/
```

Each PET image must have a matching trimap-like image with the same stem and the suffix `_trimap.png`.

Example:

```text
examples_PET/
├── basset_hound_38.jpg
└── basset_hound_38_trimap.png
```

The grid script checks that each image has a corresponding trimap file.

### Trained RecViT checkpoints

The trained models are loaded by `load_trained_network(...)`. For the `tiny` model, checkpoint names are built from the dataset, model name, pretrained flag, number of recurrent loops, run number, and optional training flags.

For example, a PET pretrained tiny checkpoint with 3 recurrent loops and run number 0 is expected to follow a naming pattern similar to:

```text
pet_tiny_pretrained_k_3_run_0.pth
```

The exact checkpoint root is controlled by `NETWORKS_DIR` in the RecViT model package configuration.

## Running one image

Use `rec_vit_explain.py` to generate rollout maps for a single image and one selected configuration.

Example PET command:

```bash
python rec_vit_explain.py \
  --use_cuda \
  --image_path ./examples_PET/basset_hound_38.jpg \
  --trimap_path ./examples_PET/basset_hound_38_trimap.png \
  --output_dir ./rollout_results/basset_hound_38 \
  --model_name tiny \
  --dataset PET \
  --n_loops 3 \
  --run_no 0 \
  --pretrained \
  --tiny_patch 8 \
  --head_fusion mean \
  --discard_ratio 0.5 \
  --patch_attendance identity
```

Example CIFAR-10 command without trimap evaluation:

```bash
python rec_vit_explain.py \
  --image_path ./examples_CIFAR10/plane.png \
  --output_dir ./rollout_results/plane \
  --model_name tiny \
  --dataset CIFAR_10 \
  --n_loops 1 \
  --run_no 0 \
  --head_fusion max \
  --discard_ratio 0.9 \
  --patch_attendance identity
```

### Important arguments

| Argument | Meaning |
|---|---|
| `--use_cuda` | Use GPU if CUDA is available. If omitted, CPU is used. |
| `--image_path` | Path to the input image. |
| `--output_dir` | Folder where generated maps and evaluation files are saved. |
| `--head_fusion` | Method for fusing attention heads: `mean`, `max`, or `min`. |
| `--discard_ratio` | Ratio of lowest attention values discarded during rollout. |
| `--model_name` | RecViT model size, usually `tiny` or `extra_tiny`. |
| `--dataset` | Dataset checkpoint type: `CIFAR_10`, `CIFAR_100`, or `PET`. |
| `--n_loops` | Number of recurrent steps used by the loaded RecViT checkpoint. |
| `--run_no` | Checkpoint run number. |
| `--pretrained` | Load checkpoint trained from pretrained ViT initialization. |
| `--tiny_patch` | Patch size for the tiny model. Use `8` for PET pretrained checkpoints when needed. |
| `--trimap_path` | PET trimap-like input used for trimap-based evaluation. |
| `--patch_attendance` | Cross-step patch strategy: `identity` or `zero`. |
| `--use_different_inputs` | Experimental flag for different inputs across recurrent steps. The current main workflow uses the same input image at each step. |

### Single-run outputs

A single run saves files into `--output_dir`.

Typical output:

```text
rollout_results/<run_name>/
├── input.png
├── rec_rollout_<n_loops>_loops_<patch_strategy>_<discard_ratio>_<head_fusion>.png
├── rec_rollout_final_to_input_step_1_<n_loops>_loops_<patch_strategy>_<discard_ratio>_<head_fusion>.png
├── rec_rollout_final_to_input_step_2_<n_loops>_loops_<patch_strategy>_<discard_ratio>_<head_fusion>.png
├── ...
└── trimap/
    ├── generated_trimap_color.png
    ├── attention_scores.csv
    ├── final_rollout_attention_trimap_overlay.png
    ├── final_to_input_step_1_attention_trimap_overlay.png
    └── ...
```

The `trimap/` folder is created only for PET runs when `--trimap_path` is provided.

## Running the PET grid experiment

Use `run_recvit_rollout_grid.py` for the full PET grid experiment. The script reads all image/trimap pairs from `examples_PET/` and runs all configured combinations.

```bash
python run_recvit_rollout_grid.py
```

The default grid configuration is defined at the top of the script:

```python
IMAGE_DIR = Path("./examples_PET")
OUTPUT_ROOT = Path("./rollout_PET_grid_results")
USE_CUDA = True
SEED = 42

MODEL_NAMES = ["tiny"]
DATASETS = ["PET"]
TINY_PATCH = 8

N_LOOPS = [2, 3, 4]
RUN_NOS = [0]
HEAD_FUSIONS = ["mean", "max", "min"]
DISCARD_RATIOS = [0.0, 0.5, 0.9]
PATCH_ATTENDANCES = ["identity", "zero"]

PRETRAINED = [True]
METHOD2 = [False]
REG_1000 = [False]
ON_OFF = [False]
USE_DIFFERENT_INPUTS = [False]
```

For each image, the default grid produces:

```text
3 loop counts × 3 head-fusion methods × 3 discard ratios × 2 patch strategies = 54 configurations per image
```

The total number of runs depends on the number of valid image/trimap pairs in `examples_PET/`.

## Grid output structure

The grid script saves results under:

```text
rollout_PET_grid_results/
```

The folder structure is organized by model variant, image name, raw attention maps, and rollout configuration.

Example structure:

```text
rollout_PET_grid_results/
├── summary.json
└── PET_tiny_pretrained_loops3_run0/
    └── basset_hound_38/
        ├── input.png
        ├── trimap/
        │   ├── generated_trimap_raw.png
        │   └── generated_trimap_color.png
        ├── raw_attention/
        │   ├── fusion-mean/
        │   │   ├── step_1/
        │   │   │   ├── layer_01.npy
        │   │   │   ├── layer_01.png
        │   │   │   └── ...
        │   │   ├── step_last/
        │   │   │   ├── layer_01.npy
        │   │   │   ├── layer_01.png
        │   │   │   └── ...
        │   │   └── raw_attention_scores.csv
        │   ├── fusion-max/
        │   └── fusion-min/
        └── rollout/
            └── fusion-mean_discard-0p5_patch-identity/
                ├── final_rollout.npy
                ├── final_rollout.png
                ├── final_to_input_step_1.npy
                ├── final_to_input_step_1.png
                ├── final_to_input_step_2.npy
                ├── final_to_input_step_2.png
                ├── final_to_input_step_3.npy
                ├── final_to_input_step_3.png
                ├── vit_rollout_last_step.npy
                ├── vit_rollout_last_step.png
                ├── rollout_scores.csv
                ├── combined_attention_scores.csv
                └── run_metadata.json
```

### Important grid files

| File | Description |
|---|---|
| `summary.json` | List of all grid runs and their status. Re-created when the grid script is run. |
| `input.png` | Resized input image used for visualization. |
| `generated_trimap_raw.png` | Generated raw trimap mask. |
| `generated_trimap_color.png` | Colored trimap visualization. |
| `raw_attention_scores.csv` | Scores for raw layer-wise attention maps. |
| `rollout_scores.csv` | Scores for rollout maps in one configuration. |
| `combined_attention_scores.csv` | Raw attention and rollout scores combined for one configuration. |
| `run_metadata.json` | Configuration, output paths, number of saved maps, and per-step predictions. |
| `.npy` files | Raw numerical attention masks. |
| `.png` files | Heatmap overlays saved for visual inspection. |

## Attention map types

The project produces several kinds of maps.

### Raw attention maps

Raw attention maps are generated in the grid script. They are extracted from selected recurrent steps:

- first recurrent step: `step_1`
- final recurrent step: `step_last`

For each selected step, the script saves one map per Transformer layer after head fusion.

### Final recurrent rollout map

`final_rollout` is the main RecViT-adapted rollout map. It represents attention propagation from the final recurrent output back to the input of the first recurrent step.

### Step-input rollout maps

`final_to_input_step_<t>` maps represent attention propagation from the final recurrent output to the input of recurrent step `t`.

### Standard ViT rollout of the last step

`vit_rollout_last_step` is the standard per-step rollout computed only for the final recurrent step. It is useful as a comparison to the full recurrent rollout.

## Evaluation method

For PET experiments, the project evaluates attention maps using trimap-derived masks.

The trimap is converted to a score mask:

```text
background = -1.0
border     =  0.0
foreground =  1.0
```

Each attention map is normalized to `[0, 1]`, weighted by this score mask, and averaged by the attention mass. A higher score means that more attention is concentrated on the foreground region and less on the background.

The score is saved in CSV files such as:

```text
attention_scores.csv
raw_attention_scores.csv
rollout_scores.csv
combined_attention_scores.csv
```

## How the RecViT rollout works

The core implementation is in `rec_vit_rollout.py`.

The workflow is:

1. Register forward hooks on attention modules that contain `qkv` projections.
2. During each recurrent step, collect attention matrices separately.
3. For each recurrent step, fuse attention heads using `mean`, `max`, or `min`.
4. Optionally discard the lowest attention values according to `discard_ratio`.
5. Add the identity matrix to account for residual connections.
6. Normalize rows and multiply attention matrices across layers to obtain a local rollout matrix for the step.
7. Connect the local rollout matrices across recurrent steps using transition matrices.
8. Extract the class-token-to-patch relevance row and reshape it into a 2D heatmap.

The recurrent transition uses the previous step rollout for the class-token connection. Patch-token transitions are controlled by `patch_attendance`:

- `identity`: corresponding patch positions are linked across steps.
- `zero`: patch tokens are not directly propagated across steps.

## Reproducibility notes

The grid script fixes random seeds:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

For reproducible experiments, keep the same:

- input images,
- trimap images,
- trained checkpoints,
- model variant,
- number of recurrent loops,
- head-fusion method,
- discard ratio,
- patch-attendance strategy,
- patch size,
- CUDA/CPU environment.

## Common problems

### Import error for `rec_vit_model`

Make sure the RecViT model package is available in the Python path and that the project is launched from the correct directory.

### Checkpoint not found

Check that `NETWORKS_DIR` points to the correct checkpoint directory and that the checkpoint name matches the selected dataset, model, loop count, run number, and flags.

### CUDA requested but CPU is used

`--use_cuda` only enables GPU if CUDA is available through PyTorch. If CUDA is not available, the script falls back to CPU.

### Missing trimap in the grid script

For each PET image, the grid script expects a matching file named:

```text
<image_stem>_trimap.png
```

For example:

```text
boxer_139.jpg
boxer_139_trimap.png
```

### Existing outputs are overwritten

Running the scripts again with the same output directory can overwrite files with the same names. Use a new output directory if you want to preserve old runs.

## Recommended thesis workflow

For thesis experiments, a practical workflow is:

1. Test one image with `rec_vit_explain.py`.
2. Verify that model loading, prediction, rollout generation, and trimap evaluation work.
3. Run the full PET grid with `run_recvit_rollout_grid.py`.
4. Inspect `summary.json` for failed runs.
5. Use `combined_attention_scores.csv` files for quantitative comparison.
6. Use saved `.png` overlays for representative qualitative figures.
7. Use both raw attention maps and rollout maps, but avoid showing every generated map in the thesis text.

## License and attribution

This repository contains research code for RecViT attention-map analysis. Parts of the RecViT model code are based on or adapted from the original RecViT/timm implementation. Add the exact license and citation information of the original RecViT repository and any reused code before publishing the project publicly.
