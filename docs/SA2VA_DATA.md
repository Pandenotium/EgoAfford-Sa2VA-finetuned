# Native Sa2VA data interface

Each JSONL row identifies one image and one given action. The role order is always `direct_object`, `instrument`, `destination`.

## Original packed masks

```json
{"id":"task/image","image":"images/example.jpg","action":"Move the cup onto the tray","data_source":"real","names":{"direct_object":"cup","instrument":"None","destination":"tray"},"mask_path":"masks/task.npz","mask_index":0}
```

`mask_path` points to a sparse packed NPZ with fields `shape`, `nonempty_idx`, `nonempty_shape`, and `packed`. `shape` is `[N,3,H,W]`; `nonempty_idx` indexes the nonempty sample/role pairs. The selected `mask_index` yields three binary masks at the input image resolution. The loader uses the original bit-unpacking implementation.

## Explicit PNG masks

The same row may instead contain:

```json
{"id":"task/image","image":"images/example.jpg","action":"Move the cup onto the tray","source":"real","names":{"direct_object":"cup","instrument":"None","destination":"tray"},"masks":{"direct_object":"masks/example_o.png","instrument":"masks/example_i.png","destination":"masks/example_d.png"}}
```

All three GT masks are required, including empty masks. They must be binary, single-channel PNGs at the image resolution. `names` are required for training, with the string `None` for an absent role. Inference needs only `id`, `image` and `action`; `--evaluate` additionally needs masks. The public benchmark format's `source` is accepted as an alias for `data_source`.

Relative asset paths resolve against `EGOAFFORD_DATA_ROOT`, if set, or the manifest directory otherwise. This path convention does not alter labels or pixels. Keep `data_source`/`source` accurate: the original training path reconciles names and empty masks for synthetic samples, while real samples retain their supplied annotations.

Prompt construction and the supervised answer are in `egoafford_sa2va/protocol.py`. Image preprocessing is in `dataset_internvl.py`: InternVL dynamic patches (448 px, maximum 12 plus thumbnail), and a direct 1024 px resize for SAM2. Predictions are interpolated back to the original image dimensions before thresholding at 0.5.
