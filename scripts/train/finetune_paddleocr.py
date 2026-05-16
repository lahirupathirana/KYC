"""
PaddleOCR fine-tuning configuration generator.

This script generates YAML configuration files for fine-tuning:
  1. PP-OCRv4 detection model (DB++)  — text region detection
  2. PP-OCRv4 recognition model (SVTR) — character recognition
  3. A custom Sinhala character dictionary for recognition fine-tuning

It also prints the exact PaddleOCR training commands to run.

Fine-tuning strategy:
  - Start from PP-OCRv4 pretrained weights (English/multilingual)
  - Freeze early layers; fine-tune classification head and decoder
  - Use synthetic SL document data + augmentation
  - Sinhala recognition: custom dict + synthetic Sinhala word images

Usage:
    python scripts/train/finetune_paddleocr.py \\
        --dataset datasets/paddleocr \\
        --output configs/paddleocr \\
        --gpu

Then run the generated commands to start training.

Requires PaddleOCR source:
    git clone https://github.com/PaddlePaddle/PaddleOCR
    cd PaddleOCR && pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


# ── Detection config (PP-OCRv4-det, DB++) ─────────────────────────────────────

DETECTION_CONFIG_TEMPLATE = """\
Global:
  use_gpu: {use_gpu}
  epoch_num: 500
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/det_ppocr_v4_sl/
  save_epoch_step: 50
  eval_batch_step: [0, 400]
  cal_metric_during_train: False
  pretrained_model: ./pretrain_models/en_PP-OCRv4_det_train/best_accuracy
  checkpoints:
  save_inference_dir:
  use_visualdl: False
  infer_img: doc/imgs_en/img_10.jpg
  save_res_path: ./checkpoints/det_db/predicts_db.txt
  distributed: False

Architecture:
  model_type: det
  algorithm: DB
  Transform:
  Backbone:
    name: ResNet
    in_channels: 3
    layers: 50
  Neck:
    name: DBFPN
    out_channels: 256
  Head:
    name: DBHead
    k: 50

Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 5.0e-05

PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5

Metric:
  name: DetMetric
  main_indicator: hmean

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {dataset_dir}/images/
    label_file_list:
      - {dataset_dir}/train_det.txt
    ratio_list: [1.0]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - DetLabelEncode:
      - IaaAugment:
          augmenter_args:
            - {{ "type": "Fliplr", "args": {{ "p": 0.5 }} }}
            - {{ "type": "Affine", "args": {{ "rotate": [-10, 10] }} }}
            - {{ "type": "Resize", "args": {{ "size": [0.5, 3] }} }}
      - EastRandomCropData:
          size: [640, 640]
          max_tries: 50
          keep_ratio: true
      - MakeBorderMap:
          shrink_ratio: 0.4
          thresh_min: 0.3
          thresh_max: 0.7
      - MakeShrinkMap:
          shrink_ratio: 0.4
          min_text_size: 8
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: hwc
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']
  loader:
    shuffle: True
    drop_last: False
    batch_size_per_card: 8
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {dataset_dir}/images/
    label_file_list:
      - {dataset_dir}/val_det.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - DetLabelEncode:
      - DetResizeForTest:
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: hwc
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'shape', 'polys', 'ignore_tags']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: 1
    num_workers: 2
"""


# ── Recognition config (PP-OCRv4-rec, SVTR_LCNet) ────────────────────────────

RECOGNITION_CONFIG_TEMPLATE = """\
Global:
  use_gpu: {use_gpu}
  epoch_num: 200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/rec_ppocr_v4_sl/
  save_epoch_step: 20
  eval_batch_step: [0, 2000]
  cal_metric_during_train: True
  pretrained_model: ./pretrain_models/en_PP-OCRv4_rec_train/best_accuracy
  checkpoints:
  save_inference_dir:
  use_visualdl: False
  infer_img: doc/imgs_words_en/word_10.png
  character_dict_path: {config_dir}/sl_char_dict.txt
  max_text_length: &max_text_length 64
  infer_mode: False
  use_space_char: True
  distributed: False
  save_res_path: ./output/rec/predicts_ppocrv4.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: MultiStepDecay
    learning_rate: 0.0005
    milestones: [60, 140, 180]
    gamma: 0.1
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 64
            depth: 2
            hidden_dims: 120
            use_guide: True
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 96
          max_text_length: *max_text_length

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: False

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {dataset_dir}/
    label_file_list:
      - {dataset_dir}/train_rec.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - RecConAug:
          prob: 0.5
          eps: 0.01
          max_size: 3
          word_num: 4
      - RecAug:
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']
  loader:
    shuffle: True
    batch_size_per_card: 128
    drop_last: True
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {dataset_dir}/
    label_file_list:
      - {dataset_dir}/val_rec.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: 128
    num_workers: 4
"""


# ── Sinhala character dictionary ──────────────────────────────────────────────

def build_sinhala_dict() -> str:
    """
    Generate a character dictionary for PaddleOCR recognition fine-tuning.

    Includes:
      - Standard ASCII printable characters (numbers, letters, punctuation)
      - Sinhala Unicode block U+0D80–U+0DFF (vowels, consonants, vowel signs)
    """
    chars: list[str] = []

    # ASCII printable
    for c in range(32, 127):
        chars.append(chr(c))

    # Sinhala independent vowels (U+0D85–U+0D96)
    for cp in range(0x0D85, 0x0D97):
        chars.append(chr(cp))

    # Sinhala consonants (U+0D9A–U+0DC6)
    for cp in range(0x0D9A, 0x0DC7):
        chars.append(chr(cp))

    # Sinhala vowel signs and dependent characters (U+0DCF–U+0DDF)
    for cp in range(0x0DCF, 0x0DE0):
        chars.append(chr(cp))

    # Sinhala digits (U+0DE6–U+0DEF)
    for cp in range(0x0DE6, 0x0DF0):
        chars.append(chr(cp))

    # Zero Width Joiner (U+200D) — used in Sinhala YANSAYA / REPAYA
    chars.append("‍")

    return "\n".join(chars)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PaddleOCR fine-tuning configs for Sri Lankan documents"
    )
    parser.add_argument("--dataset", default="datasets/paddleocr",
                        help="Path to prepared PaddleOCR dataset (from prepare_paddleocr_dataset.py)")
    parser.add_argument("--output", default="configs/paddleocr",
                        help="Directory to write config YAML files")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU training")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    config_dir = Path(args.output).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = "True" if args.gpu else "False"

    # Detection config
    det_cfg = DETECTION_CONFIG_TEMPLATE.format(
        use_gpu=use_gpu,
        dataset_dir=dataset_dir,
    )
    (config_dir / "sl_det_ppocr_v4.yml").write_text(det_cfg, encoding="utf-8")

    # Recognition config
    rec_cfg = RECOGNITION_CONFIG_TEMPLATE.format(
        use_gpu=use_gpu,
        dataset_dir=dataset_dir,
        config_dir=config_dir,
    )
    (config_dir / "sl_rec_ppocr_v4.yml").write_text(rec_cfg, encoding="utf-8")

    # Sinhala character dict
    char_dict = build_sinhala_dict()
    (config_dir / "sl_char_dict.txt").write_text(char_dict, encoding="utf-8")

    print(f"Configs written to: {config_dir}")
    print()
    print("=" * 70)
    print("NEXT STEPS — run from inside PaddleOCR/ source directory:")
    print("=" * 70)
    print()
    print("1. Download pretrained weights:")
    print("   mkdir -p pretrain_models")
    print("   wget -P pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_train.tar")
    print("   wget -P pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar")
    print("   cd pretrain_models && tar xf en_PP-OCRv4_det_train.tar && tar xf en_PP-OCRv4_rec_train.tar && cd ..")
    print()
    print("2. Fine-tune detection model:")
    print(f"   python tools/train.py -c {config_dir}/sl_det_ppocr_v4.yml \\")
    print(f"     -o Global.pretrained_model=pretrain_models/en_PP-OCRv4_det_train/best_accuracy")
    print()
    print("3. Fine-tune recognition model:")
    print(f"   python tools/train.py -c {config_dir}/sl_rec_ppocr_v4.yml \\")
    print(f"     -o Global.pretrained_model=pretrain_models/en_PP-OCRv4_rec_train/best_accuracy")
    print()
    print("4. Export to inference format:")
    print(f"   python tools/export_model.py -c {config_dir}/sl_det_ppocr_v4.yml \\")
    print(f"     -o Global.checkpoints=./output/det_ppocr_v4_sl/best_accuracy \\")
    print(f"     Global.save_inference_dir=./inference/sl_det/")
    print()
    print("5. Evaluate (run from this repo's ai_service/ directory):")
    print("   python scripts/evaluate_ocr.py \\")
    print("     --images datasets/synthetic/nic \\")
    print("     --ground-truth datasets/synthetic/nic/manifest.json \\")
    print("     --doc-type nic")


if __name__ == "__main__":
    main()
