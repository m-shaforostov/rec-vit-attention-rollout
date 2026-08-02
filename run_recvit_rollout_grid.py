import csv
import itertools
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from rec_vit_rollout import RecVITAttentionRollout
from rec_vit_model.pckgs.networks.network_utils import load_trained_network
from attention_map_evaluation import (
    trimap_to_score_mask,
    compute_attention_score,
)
from reveal_segmentation import make_trimap, save_colored_trimap


# ----------------------------
# CONFIG
# ----------------------------

IMAGE_DIR = Path("./examples_PET")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

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


# FOR REPRODUCIBILITY
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def safe_name(x):
    return str(x).replace("/", "_").replace(".", "p").replace(" ", "")


def collect_image_trimap_pairs(image_dir):
    files = list(image_dir.iterdir())

    input_images = [
        p for p in files
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.stem.endswith("_trimap")
    ]

    trimap_images = [
        p for p in files
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.stem.endswith("_trimap")
    ]

    if len(files) % 2 != 0:
        raise ValueError(
            f"Expected even number of files in {image_dir}, got {len(files)}"
        )

    if len(input_images) != len(trimap_images):
        raise ValueError(
            f"Number of input images ({len(input_images)}) "
            f"does not match number of trimaps ({len(trimap_images)})"
        )

    pairs = []

    for image_path in sorted(input_images):
        trimap_path = image_path.with_name(image_path.stem + "_trimap.png")

        if not trimap_path.exists():
            raise FileNotFoundError(
                f"Missing trimap for {image_path}: expected {trimap_path}"
            )

        pairs.append((image_path, trimap_path))

    return pairs


def make_variant_dir(config):
    parts = [
        config["dataset"],
        config["model_name"],
    ]

    if config["pretrained"]:
        parts.append("pretrained")
    if config["method2"]:
        parts.append("method2")
    if config["reg_1000"]:
        parts.append("reg1000")
    if config["on_off"]:
        parts.append("onoff")
    if config["use_different_inputs"]:
        parts.append("different_inputs")

    parts += [
        f"loops{config['n_loops']}",
        f"run{config['run_no']}",
    ]

    return "_".join(parts)


def make_rollout_tag(config):
    return (
        f"fusion-{config['head_fusion']}"
        f"_discard-{safe_name(config['discard_ratio'])}"
        f"_patch-{config['patch_attendance']}"
    )


def show_mask_on_image(img_bgr, mask):
    img = np.float32(img_bgr) / 255.0

    mask = np.asarray(mask, dtype=np.float32)
    mask = mask - mask.min()
    if mask.max() > 0:
        mask = mask / mask.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0

    cam = heatmap + img
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)


def save_attention_outputs(mask, img_bgr, npy_path, png_path):
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(mask, dtype=np.float32)

    np.save(npy_path, mask)

    resized_mask = cv2.resize(
        mask,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    overlay = show_mask_on_image(img_bgr, resized_mask)
    cv2.imwrite(str(png_path), overlay)


def attention_matrix_to_cls_patch_mask(attention_matrix):
    """
    Converts attention matrix [1, N, N] or [N, N]
    to a normalized CLS-to-patch 2D map.
    """
    if attention_matrix.ndim == 3:
        values = attention_matrix[0, 0, 1:]
    elif attention_matrix.ndim == 2:
        values = attention_matrix[0, 1:]
    else:
        raise ValueError(f"Unexpected attention matrix shape: {attention_matrix.shape}")

    width = int(values.shape[-1] ** 0.5)

    if width * width != values.shape[-1]:
        raise ValueError(
            f"Patch count {values.shape[-1]} cannot be reshaped to a square map."
        )

    mask = values.reshape(width, width)
    mask = mask.astype(np.float32)

    mask = mask - mask.min()
    if mask.max() > 0:
        mask = mask / mask.max()

    return mask


def fuse_raw_attention(attention, head_fusion):
    """
    Input attention shape: [1, heads, tokens, tokens]
    Output shape: [1, tokens, tokens]
    """
    if head_fusion == "mean":
        fused = attention.mean(dim=1)
    elif head_fusion == "max":
        fused = attention.max(dim=1)[0]
    elif head_fusion == "min":
        fused = attention.min(dim=1)[0]
    else:
        raise ValueError(f"Unsupported head_fusion: {head_fusion}")

    return fused.cpu().numpy()


def prepare_input(image_path, device):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)
    img_bgr = np.array(img)[:, :, ::-1]

    return img, img_bgr, input_tensor


def write_scores_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method",
            "map_name",
            "step",
            "layer",
            "score",
            "npy_path",
            "png_path",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_attention_mask(mask, trimap):
    score_mask, _ = trimap_to_score_mask(trimap, mask.shape)
    return compute_attention_score(mask, score_mask)


def save_raw_attention_maps(attention_rollout, config, image_root, img_bgr, trimap):
    raw_root = image_root / "raw_attention" / f"fusion-{config['head_fusion']}"
    raw_root.mkdir(parents=True, exist_ok=True)

    score_rows = []

    selected_steps = {
        1: attention_rollout.attentions[0],
        config["n_loops"]: attention_rollout.attentions[-1],
    }

    for step_idx, step_attentions in selected_steps.items():
        step_dir_name = "step_1" if step_idx == 1 else "step_last"
        step_dir = raw_root / step_dir_name

        for layer_idx, attention in enumerate(step_attentions, start=1):
            fused = fuse_raw_attention(attention, config["head_fusion"])
            mask = attention_matrix_to_cls_patch_mask(fused)

            map_name = f"raw_step_{step_idx}_layer_{layer_idx:02d}"

            npy_path = step_dir / f"layer_{layer_idx:02d}.npy"
            png_path = step_dir / f"layer_{layer_idx:02d}.png"

            save_attention_outputs(mask, img_bgr, npy_path, png_path)

            score_rows.append({
                "method": "raw_attention",
                "map_name": map_name,
                "step": step_idx,
                "layer": layer_idx,
                "score": score_attention_mask(mask, trimap),
                "npy_path": str(npy_path),
                "png_path": str(png_path),
            })

    write_scores_csv(raw_root / "raw_attention_scores.csv", score_rows)
    return score_rows


def save_rollout_maps(
    final_rollout_mask,
    final_to_step_input_masks,
    config,
    image_root,
    img_bgr,
    trimap,
    attention_rollout,
):
    rollout_tag = make_rollout_tag(config)
    rollout_root = image_root / "rollout" / rollout_tag
    rollout_root.mkdir(parents=True, exist_ok=True)

    score_rows = []

    # 1. Final rollout
    final_npy = rollout_root / "final_rollout.npy"
    final_png = rollout_root / "final_rollout.png"

    save_attention_outputs(final_rollout_mask, img_bgr, final_npy, final_png)

    score_rows.append({
        "method": "rollout",
        "map_name": "final_rollout",
        "step": "final",
        "layer": "",
        "score": score_attention_mask(final_rollout_mask, trimap),
        "npy_path": str(final_npy),
        "png_path": str(final_png),
    })

    # 2. Final output -> input of every recurrent step
    for step_idx, step_mask in enumerate(final_to_step_input_masks, start=1):
        map_name = f"final_to_input_step_{step_idx}"

        npy_path = rollout_root / f"{map_name}.npy"
        png_path = rollout_root / f"{map_name}.png"

        save_attention_outputs(step_mask, img_bgr, npy_path, png_path)

        score_rows.append({
            "method": "rollout",
            "map_name": map_name,
            "step": step_idx,
            "layer": "",
            "score": score_attention_mask(step_mask, trimap),
            "npy_path": str(npy_path),
            "png_path": str(png_path),
        })

    # 3. Standard ViT rollout of the final recurrent step only
    vit_last_mask = attention_rollout.step_rollout_masks[-1]

    vit_npy = rollout_root / "vit_rollout_last_step.npy"
    vit_png = rollout_root / "vit_rollout_last_step.png"

    save_attention_outputs(vit_last_mask, img_bgr, vit_npy, vit_png)

    score_rows.append({
        "method": "standard_vit_rollout_last_step",
        "map_name": "vit_rollout_last_step",
        "step": config["n_loops"],
        "layer": "",
        "score": score_attention_mask(vit_last_mask, trimap),
        "npy_path": str(vit_npy),
        "png_path": str(vit_png),
    })

    write_scores_csv(rollout_root / "rollout_scores.csv", score_rows)

    return rollout_root, score_rows


def save_trimap_outputs(image_root, trimap):
    trimap_dir = image_root / "trimap"
    trimap_dir.mkdir(parents=True, exist_ok=True)

    raw_path = trimap_dir / "generated_trimap_raw.png"
    color_path = trimap_dir / "generated_trimap_color.png"

    cv2.imwrite(str(raw_path), trimap)
    save_colored_trimap(trimap, color_path)


# Run one configuration
def run_one(config, run_number, total_runs):
    device = "cuda:0" if (config["use_cuda"] and torch.cuda.is_available()) else "cpu"

    variant_dir = make_variant_dir(config)
    image_stem = Path(config["image_path"]).stem

    image_root = OUTPUT_ROOT / variant_dir / image_stem
    image_root.mkdir(parents=True, exist_ok=True)

    img, img_bgr, input_tensor = prepare_input(config["image_path"], device)

    cv2.imwrite(str(image_root / "input.png"), img_bgr)

    trimap = make_trimap(Path(config["trimap_path"]))
    save_trimap_outputs(image_root, trimap)

    print(f"\nRunning {run_number}/{total_runs}")
    print(json.dumps(config, indent=2))

    model = load_trained_network(
        name=config["model_name"],
        dataset=config["dataset"],
        n_loops=config["n_loops"],
        run_no=config["run_no"],
        pretrained=config["pretrained"],
        device=device,
        method2=config["method2"],
        reg_1000=config["reg_1000"],
        on_off=config["on_off"],
        tiny_patch=config["tiny_patch"],
    )

    model.eval()

    attention_rollout = RecVITAttentionRollout(
        model,
        head_fusion=config["head_fusion"],
        discard_ratio=config["discard_ratio"],
        repeats=config["n_loops"],
        patch_attendance=config["patch_attendance"],
        use_different_inputs=config["use_different_inputs"],
    )

    try:
        final_rollout_mask, final_to_step_input_masks, per_step_logits = attention_rollout(input_tensor)

        raw_score_rows = save_raw_attention_maps(
            attention_rollout=attention_rollout,
            config=config,
            image_root=image_root,
            img_bgr=img_bgr,
            trimap=trimap,
        )

        rollout_root, rollout_score_rows = save_rollout_maps(
            final_rollout_mask=final_rollout_mask,
            final_to_step_input_masks=final_to_step_input_masks,
            config=config,
            image_root=image_root,
            img_bgr=img_bgr,
            trimap=trimap,
            attention_rollout=attention_rollout,
        )

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "status": "OK",
            "config": config,
            "device": device,
            "variant_dir": variant_dir,
            "image_root": str(image_root),
            "rollout_output_dir": str(rollout_root),
            "num_raw_attention_maps": len(raw_score_rows),
            "num_rollout_maps": len(rollout_score_rows),
            "per_step_predictions": [
                int(logit.argmax(dim=1).item())
                for logit in per_step_logits
            ],
        }

        with open(rollout_root / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        combined_scores = raw_score_rows + rollout_score_rows
        write_scores_csv(rollout_root / "combined_attention_scores.csv", combined_scores)

        return {
            "status": "OK",
            "config": config,
            "image_root": str(image_root),
            "rollout_output_dir": str(rollout_root),
        }

    except Exception as e:
        error_dir = image_root / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)

        error_path = error_dir / f"{make_rollout_tag(config)}_error.txt"

        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(e))

        print(f"FAILED: {e}")

        return {
            "status": "FAILED",
            "error": str(e),
            "config": config,
            "image_root": str(image_root),
        }

if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    image_trimap_pairs = collect_image_trimap_pairs(IMAGE_DIR)

    grid = list(itertools.product(
        image_trimap_pairs,
        MODEL_NAMES,
        DATASETS,
        N_LOOPS,
        RUN_NOS,
        HEAD_FUSIONS,
        DISCARD_RATIOS,
        PATCH_ATTENDANCES,
        PRETRAINED,
        METHOD2,
        REG_1000,
        ON_OFF,
        USE_DIFFERENT_INPUTS,
    ))

    total_runs = len(grid)
    results = []

    for i, (
            image_pair,
            model_name,
            dataset,
            n_loops,
            run_no,
            head_fusion,
            discard_ratio,
            patch_attendance,
            pretrained,
            method2,
            reg_1000,
            on_off,
            use_different_inputs,
    ) in enumerate(grid, start=1):

        if method2 and n_loops != 3:
            continue

        image_path, trimap_path = image_pair

        config = {
            "image_path": str(image_path),
            "trimap_path": str(trimap_path),
            "model_name": model_name,
            "dataset": dataset,
            "n_loops": n_loops,
            "run_no": run_no,
            "head_fusion": head_fusion,
            "discard_ratio": discard_ratio,
            "patch_attendance": patch_attendance,
            "pretrained": pretrained,
            "method2": method2,
            "reg_1000": reg_1000,
            "on_off": on_off,
            "use_different_inputs": use_different_inputs,
            "use_cuda": USE_CUDA,
            "tiny_patch": TINY_PATCH,
            "seed": SEED,
        }

        result = run_one(config, i, total_runs)
        results.append(result)

    summary_path = OUTPUT_ROOT / "summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    failed = sum(r["status"] == "FAILED" for r in results)

    print("\nFinished.")
    print(f"Total runs: {total}")
    print(f"Failed runs: {failed}")
    print(f"Summary saved to: {summary_path}")