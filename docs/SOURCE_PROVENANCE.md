# Source provenance

The release adapter comes from the Sa2VA source used for the paper's final 39,956-image training run. Its recorded Sa2VA upstream revision is:

`b6fed1a7928ad4adfa124f1fe79031c1f58ec7a1`

The remote training environment keeps a source snapshot with this revision marker, rather than a Git checkout. The original imported file hashes are recorded in `SOURCE_SHA256.json`. The vendored subset contains Sa2VA models/datasets, the finetuning config, and supporting `vlm`/`third_parts` modules. Unused demos, evaluation suites, HF export implementations and unrelated configs are omitted. Original headers and the source snapshot's root license are retained.

## Release adaptations

- `config.py`: resolve configurable paths, register the vendored import path, and derive accumulation from `WORLD_SIZE`. The default eight-GPU recipe, learning rate, warmup, optimizer and module settings are preserved.
- `common.py`: resolve manifest asset paths and accept explicit binary PNG GT masks alongside original packed NPZ masks. `source` is accepted as a `data_source` alias.
- `evaluate_internvl.py`: retain the set of trained parameter names for checkpoint completeness checks. Generation, SEG-hidden alignment, mask decoding and role mapping are unchanged.
- `scripts/`: add a local `torchrun` launcher, explicit checkpoint/image/manifest inference, and CPU configuration checks. Cluster schedulers, internal run paths, monitoring scripts and logs are excluded.
- Vendored `models/__init__.py` and `models/extension/__init__.py`: remove unused SAM3 exports so the SAM2 release does not import or distribute the SAM3 implementation. Unused SAM1 code and engine helpers are also omitted. The SAM2 model implementation is unchanged.

The remaining adapter modules retain the source implementation. In particular, prompt text, targets, image preprocessing, SEG parsing, prediction recovery, native mask losses and evaluation formulas are preserved. `batch_support.py` supplies the original per-sequence language-loss reduction.

## Verification scope

CPU tests cover mask/path compatibility, token-role parsing, PNG/NPZ equivalence, and metric calculations. Configuration/import checks use the original source environment without loading weights or launching training. A fresh installation and a GPU checkpoint/inference smoke test remain release checks; the README reference scores are the paper's archived results.
