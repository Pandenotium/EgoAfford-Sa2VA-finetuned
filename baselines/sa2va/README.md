# EgoAfford-trained Sa2VA-8B

Training and inference use the imported `egoafford_sa2va` implementation. See the [root README](../../README.md) for setup and commands, and [source provenance](../../docs/SOURCE_PROVENANCE.md) for the release edits.

The paper recipe is `configs/sa2va_8b.json`: InternVL3-8B + SAM2 Hiera-L, pretrained-component initialization, full-rule prompts, role-name/box supervision and emit-empty SEG targets. Epoch 18 is the validation-selected paper checkpoint. `release.json` records the reference recipe and results.
