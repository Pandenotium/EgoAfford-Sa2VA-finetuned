# Third-party notices

EgoAfford's original code and release modifications use [Apache-2.0](LICENSE). The components below retain their own copyrights and license terms. This file records software attribution; it does not license datasets or pretrained weights.

## Included source

| Component and location | Source | License text |
|---|---|---|
| Sa2VA snapshot: `third_party/sa2va/projects/`, `vlm/`, and supporting modules | [ByteDance/Sa2VA](https://github.com/bytedance/Sa2VA), recorded revision `b6fed1a7928ad4adfa124f1fe79031c1f58ec7a1` | [Apache-2.0](third_party/sa2va/LICENSE) |
| SAM2: `third_party/sa2va/third_parts/sam2/` | [Meta SAM2](https://github.com/facebookresearch/sam2) | [Apache-2.0](third_party/sa2va/third_parts/sam2/LICENSE) |
| Connected-components CUDA code: SAM2 `csrc/connected_components.cu` | SAM2 adaptation of [Connected_components_PyTorch](https://github.com/zsef123/Connected_components_PyTorch) | [BSD-3-Clause, as distributed by SAM2](third_party/sa2va/third_parts/sam2/LICENSE_cctorch) |
| OpenMMLab detection utilities: `third_party/sa2va/third_parts/mmdet/` | [MMDetection](https://github.com/open-mmlab/mmdetection), retained through Sa2VA | [Apache-2.0 with OpenMMLab notice](licenses/OpenMMLab-MMDetection-Apache-2.0.txt) |
| Video and data utilities with OpenMMLab headers, including `vlm/utils/video_io.py` and portions of `projects/sa2va/datasets/data_utils.py` | [MMCV](https://github.com/open-mmlab/mmcv) and the Sa2VA snapshot | [Apache-2.0 with OpenMMLab notice](licenses/OpenMMLab-MMCV-Apache-2.0.txt) |
| InternVL generation helper: `egoafford_sa2va/internvl_generation.py` | [OpenGVLab InternVL](https://github.com/OpenGVLab/InternVL); adapted from InternVL3-8B `modeling_internvl_chat.py`, `generate()` | [MIT](licenses/InternVL-MIT.txt) |
| DAVIS-style region and boundary metric functions: `third_party/sa2va/third_parts/revos/utils/metircs.py` | Retained through Sa2VA; [DAVIS evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) | [DAVIS BSD-3-Clause](licenses/DAVIS-BSD-3-Clause.txt), in addition to the surrounding Sa2VA source license |

Copyright notices retained in source include:

- `Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.`
- `Copyright (c) OpenMMLab. All rights reserved.`
- `Copyright (c) 2023 OpenGVLab` for the MIT-licensed InternVL helper.

The connected-components license retains the SAM2-distributed notice, `Copyright (c) 2020, the respective contributors, as shown by the AUTHORS file.` The DAVIS license retains `Copyright (c) 2020, DAVIS: Densely Annotated VIdeo Segmentation`. Complete conditions and disclaimers are in the linked local license files.

## Modifications in this release

The vendored source is a subset of the training snapshot. The recorded revision and original imported-file hashes are in [source provenance](docs/SOURCE_PROVENANCE.md), `third_party/sa2va/UPSTREAM_COMMIT`, and `docs/SOURCE_SHA256.json`. The hashes describe the imported files, before release adaptations; they are not checksums of the current modified files.

- `third_party/sa2va/projects/sa2va/models/__init__.py` and `models/extension/__init__.py`: remove unused SAM3 exports from this SAM2 release.
- `egoafford_sa2va/config.py`, `common.py`, `dataset_internvl.py`, and `evaluate_internvl.py`: portable paths and data compatibility, configurable launch settings, and checkpoint completeness checks, as detailed in source provenance.
- `batch_support.py`: select supervised logits before FP32 conversion while preserving per-sequence cross-entropy and gradients.
- `egoafford_sa2va/internvl_generation.py`: match the visual replacement tensor's dtype to the language embeddings. Its OpenGVLab copyright and MIT terms are retained separately.

Modified source files carry a short release-modification notice. Existing upstream copyright and license headers are preserved. These changes do not imply upstream endorsement.

## Dependencies and separately obtained components

Libraries installed through `requirements-sa2va.txt` or `pyproject.toml`, including XTuner, MMEngine, Transformers, PEFT, PyTorch and their dependencies, remain under their own licenses. Their installed distributions provide their license texts; this repository does not bundle those distributions. [XTuner](https://github.com/InternLM/xtuner) supplies model and training interfaces used by this implementation.

InternVL3-8B custom model code, tokenizer files and pretrained weights, and SAM2 pretrained weights are obtained separately. Follow the terms accompanying those exact downloads. The MIT notice above covers the included InternVL-derived helper; the code repository's Apache-2.0 license does not grant rights to external weights, EgoStandard media or annotations hosted elsewhere.
