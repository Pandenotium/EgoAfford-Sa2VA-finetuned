# Source provenance

The release adapter comes from the Sa2VA source used for the paper's final 39,956-image training run. Its recorded Sa2VA upstream revision is:

`b6fed1a7928ad4adfa124f1fe79031c1f58ec7a1`

The remote training environment keeps a source snapshot with this revision marker, rather than a Git checkout. The original imported file hashes are recorded in `SOURCE_SHA256.json`. The vendored subset contains Sa2VA models/datasets, the finetuning config, and supporting `vlm`/`third_parts` modules. Unused demos, evaluation suites, HF export implementations and unrelated configs are omitted. Original headers and the source snapshot's root license are retained.

## Release adaptations

- `config.py`: resolve configurable paths, register the vendored import path, and derive accumulation from `WORLD_SIZE`. The default eight-GPU recipe, learning rate, warmup, optimizer and module settings are preserved.
- `common.py`: resolve manifest asset paths and accept explicit binary PNG GT masks alongside original packed NPZ masks. `source` is accepted as a `data_source` alias.
- `evaluate_internvl.py`: retain the set of trained parameter names for checkpoint completeness checks. Generation, SEG-hidden alignment, mask decoding and role mapping are unchanged.
- `scripts/`: add a local `torchrun` launcher, explicit checkpoint/image/manifest inference, and CPU configuration checks. Cluster schedulers, internal run paths, monitoring scripts and logs are excluded.
- `batch_support.py`: select supervised token rows before converting logits to FP32 for cross-entropy. Ignored positions have zero loss and gradient; per-sequence normalization is preserved. Loss and gradient equivalence are tested in FP32 and BF16.
- Vendored `models/__init__.py` and `models/extension/__init__.py`: remove unused SAM3 exports so the SAM2 release does not import or distribute the SAM3 implementation. Unused SAM1 code and engine helpers are also omitted. The SAM2 model implementation is unchanged.

The remaining adapter modules retain the source implementation. In particular, prompt text, targets, image preprocessing, SEG parsing, prediction recovery, native mask losses and evaluation formulas are preserved. `batch_support.py` supplies the original per-sequence language-loss reduction.

## Verification scope

CPU tests cover mask/path compatibility, token-role parsing, PNG/NPZ equivalence, metric calculations, and FP32/BF16 loss/gradient equivalence. A separate source snapshot passed GPU single-image and batch-size-1/2 inference with the prepared safetensors checkpoint and named-channel NPZ data; native and portable evaluators agreed on four Synthetic examples. Binary dependencies were reused from the original environment. A one-GPU check reached AdamW state allocation but exceeded the 48 GB device's memory, including with reduced visual patches. A subsequent eight-H100 80 GB check preserved the recommended micro-batch 2, accumulation 2, effective batch 32 and default patch limit. It completed two optimizer updates and verified all 535 trainable keys were saved; a changed LoRA probe and optimizer step counter were restored on all eight ranks, with scheduler loading and unchanged learning rates. MMEngine resume used trusted pickle loading for that job's own checkpoint metadata. Fresh dependency installation remains unverified. The README reference scores are the paper's archived results.

Use batch size 1 for inference and keep batch settings fixed for comparisons.

## v2 export compatibility

The v2 export uses named-channel NPZ masks and nullable role names. The release readers accept these labels directly, resolve their paths relative to each manifest, and skip the legacy synthetic repair for this already finalized schema. The inference CLI also accepts safetensors state dictionaries and checks missing image inputs before loading a model. These are file/interface changes; the prediction, loss and metric formulas remain unchanged.

## Licensing

The repository code uses [Apache-2.0](../LICENSE). [Third-party notices](../THIRD_PARTY_NOTICES.md) identify retained Apache, MIT and BSD terms and link their complete local texts. Release-modification comments preserve the existing source bytes and do not change Python syntax trees. `SOURCE_SHA256.json` continues to record the original imported snapshot.
