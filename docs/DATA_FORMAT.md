# Fixed-role release interface

The release exporter converts the runtime format to this portable JSONL interface, without changing sample IDs, actions, role identities or mask pixels. It must compare decoded masks before and after export. Stored packed/NPZ masks can remain as source assets; the portable evaluator consumes explicit binary PNG masks.

A GT record contains:

```json
{"id":"sample-id","source":"synthetic","group_id":"task-id","action":"Move the cup onto the tray","image":"images/sample.jpg","masks":{"direct_object":"masks/sample_o.png","instrument":"masks/sample_i.png","destination":"masks/sample_d.png"}}
```

Paths use `/` and are relative to the supplied root. Masks are single-channel PNGs containing 0 and 1 or 255; all three GT mask paths are required, including empty masks. Images and masks retain the exact model-input dimensions. `group_id` is the real episode or generated task grouping used in the split. An action string is required and must be the final caption version.

Prediction records contain `id` and the same `masks` mapping. A null value is permitted for an explicitly absent predicted role; omitting a key is an error. Prediction order is irrelevant because IDs determine pairing. RGB overlays, soft masks and size mismatches are rejected.

Real-data release records additionally retain a `reconstruction` object with the original source release, episode/video identifier, frame index or exact original extraction instruction, and the SHA-256 of the reconstructed image. The training agent must export the actual extraction procedure; timestamps or frame indices must not be guessed. Do not place internal absolute paths, credentials or private storage URLs in public manifests.

Fixed splits:

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| Real | 26,250 | 778 | 2,215 |
| Synthetic | 13,706 | — | 488 |

IDs and episode/generated-task groups must not cross train, validation and test within a source. A repeated action or object concept alone is not split leakage. The release validator must report missing group provenance as incomplete, rather than assuming no overlap.
