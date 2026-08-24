---
license: other
license_name: research-use-hico-det
task_categories:
  - object-detection
tags:
  - human-object-interaction
  - hico-det
  - siglip2
  - precomputed-features
pretty_name: SigLIP 2 visual tokens for HICO-DET (UMI-HOI)
size_categories:
  - 10K<n<100K
---

# SigLIP 2 visual tokens for HICO-DET (UMI-HOI)

Precomputed SigLIP 2 (`google/siglip2-giant-opt-patch16-384`, not the first-generation
SigLIP) image tokens for every HICO-DET image, used as the vision-language input of **UMI-HOI** (*Unifying Multimodal Information with Semantic Multi-Head Attention for
Human-Object Interaction Detection*, CVPR Findings 2026). Code:
https://github.com/ga94zot/UMI-HOI

## What is in each file

| | |
|---|---|
| Encoder | [`google/siglip2-giant-opt-patch16-384`](https://huggingface.co/google/siglip2-giant-opt-patch16-384), default `AutoProcessor` preprocessing (384x384, patch 16) |
| Per image | `Tensor(577, 1536)` float32: rows `0..575` = 24x24 patch tokens (`last_hidden_state`), row `576` = pooled image token (`get_image_features` pooled output) |
| Images | HICO-DET `train2015` (38,118) and `test2015` (9,658), keyed by image stem, e.g. `HICO_train2015_00000001` |
| Storage | `<split>/shard-XXXX.safetensors` (~10 GiB each, bit-exact fp32), `<split>/index.json` (stem -> shard), `shards.json` (sha256, sizes) |

The features were generated with `generate_feature.py` (included) by the UMI-HOI authors.

## Loading

Restore the per-image `.pt` layout that the UMI-HOI data loader reads:

```bash
python hf_download_features.py --repo-id <this repo> --out /data/hico_siglip_feature
python main.py ... --llava-token-path /data/hico_siglip_feature
```

Or read a shard directly:

```python
from safetensors import safe_open
with safe_open("train/shard-0000.safetensors", framework="pt") as f:
    tokens = f.get_tensor("HICO_train2015_00000001")   # (577, 1536)
patches, pooled = tokens[:-1], tokens[-1]
```

## License

The tensors are derived from HICO-DET images (Chao et al., research use only; see the
HICO-DET terms) with an Apache-2.0 model. They are provided for research use under the same
terms as HICO-DET.

## Citation

```bibtex
@inproceedings{wu2026umihoi,
  title     = {Unifying Multimodal Information with Semantic Multi-Head Attention for Human-Object Interaction Detection},
  author    = {Wu, Yuankai and others},
  booktitle = {CVPR Findings},
  year      = {2026}
}
```
