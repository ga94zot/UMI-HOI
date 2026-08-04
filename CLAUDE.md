# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup state of this working copy

All five git submodules (`detr`, `h_detr`, `hicodet`, `vcoco`, `pocket`) are currently **empty** — `git submodule update` has not been run here. Nothing in this repo imports successfully until they are populated, since [pvic.py](pvic.py) and [utils.py](utils.py) import `detr.models`, `h_detr.models`, `pocket.core`, `hicodet.hicodet` and `vcoco.vcoco` at module load. Static reasoning about those packages is not possible from this checkout alone.

```bash
git submodule init && git submodule update
pip install -r requirements.txt   # note: README historically said requirement.txt
pip install -e pocket
cd h_detr/models/ops && python setup.py build install   # CUDA op for MultiScaleDeformableAttention
```

Pinned versions matter: `transformers==4.37.2` is required both by the repo and by the SigLIP2 feature extraction path, which additionally needs a hand-patched forward in the installed `transformers` package (see [siglip/modified_forward.py](siglip/modified_forward.py)).

## The `DETR` environment variable

Every entry point ([main.py](main.py), [visualization.py](visualization.py), [inference.py](inference.py), [eval_answers.py](eval_answers.py)) raises `KeyError` unless `DETR` is set to `base` or `advanced`. It selects which argparse parent from [configs.py](configs.py) is used, and therefore which flags even exist:

- `DETR=base` → `base_detector_args()`, plain DETR-R50, `--raw-lambda` default 2.8
- `DETR=advanced` → `advanced_detector_args()`, Deformable/H-Deformable DETR (adds `--num-queries-one2one`, `--num-queries-one2many`, `--use-checkpoint`, `--drop-path-rate`, …), `--raw-lambda` default 1.7

The two parsers are near-duplicates with *different defaults* (e.g. `--dropout` 0.1 vs 0.0, `--set-cost-class` 1 vs 2). Changing a shared hyperparameter means editing both functions.

## Commands

There is no test suite, linter, or build config in this repo. `python setup.py build install` under `h_detr` is the only compile step.

[docs.md](docs.md) holds the full per-backbone command matrix. The shape is always:

```bash
# Train (HICO-DET, DETR-R50)
DETR=base python main.py --pretrained checkpoints/detr-r50-hicodet.pth \
  --output-dir outputs/run-name \
  --llava-answer-path text_folder --llava-token-path token_folder \
  --repr-dim 512 --sub-headnum 5 --obj-headnum 3 --train-type default \
  --world-size 1 --batch-size 1 --epoch 15 --lr-drop 7

# Evaluate — same args as training, plus:
  --eval --resume /path/to/model

# Cache detections for the official Matlab / V-COCO evaluation — same args, plus:
  --cache --resume /path/to/model --output-dir matlab      # HICO-DET
  --cache --resume /path/to/model --output-dir vcoco_cache # V-COCO

# V-COCO adds:
  --dataset vcoco --data-root vcoco/ --partitions trainval test
```

Visualization (attention maps written under `./visualization/`):

```bash
DETR=base python visualization.py --llava-answer-path text_folder --llava-token-path token_folder \
  --batch-size 1 --index start_idx --action-score-thresh 0.2 --example-num N \
  --avg-attn --repr-dim 512 --resume your_checkpoint
```

Constraints when composing commands:

- `--sub-headnum + --obj-headnum` must equal 8 — asserted in [main.py:195](main.py#L195).
- `--partitions` takes exactly two values; `partitions[0]` is the train split and `partitions[1]` the test split.
- Per-process batch size is `--batch-size // --world-size`, so `--batch-size` must be ≥ and divisible by `--world-size`.
- Training always goes through `mp.spawn` + NCCL, even at `--world-size 1`. `--port` picks the DDP rendezvous port; run concurrent jobs on different ports.
- W&B is disabled unless `--use-wandb` is passed (`WANDB_MODE=disabled` is set otherwise), but `wandb.init`/`wandb.log` are called unconditionally in [utils.py](utils.py), so the `wandb` package must be importable regardless.
- Checkpoints, `best.pth` and `log.txt` all land in `--output-dir`.

## Architecture

Two-stage HOI detector: a **frozen** object detector produces region proposals, and a trainable interaction head classifies human-object pairs. `model.freeze_detector()` is called in [main.py](main.py) before the optimizer is built, and the whole detector forward runs under `torch.no_grad()` in `PViC.forward` — only the head trains.

Forward path in [pvic.py](pvic.py):

1. `od_forward` — dispatches to `base_forward` or `advanced_forward` depending on detector type. These are static methods that re-implement the submodules' forward passes inline to also return backbone `features` and (advanced only) `memory`/`masks`/`pos`.
2. `prepare_region_proposals` ([ops.py](ops.py)) — thresholds and pads instances to `[min_instances, max_instances]`, permuting humans to the front.
3. `HumanObjectMatcher` — enumerates valid human×object pairs, builds spatial encodings plus modulated sinusoidal box positional embeddings, and fuses them into pair queries.
4. `FeatureHead` — maps the backbone C5 map (`--kv-src` selects C5/C4/C3) into the key/value sequence.
5. `TransformerDecoder` — the S-MHA stack ([transformers_.py](transformers_.py)).
6. `binary_classifier` — a single `nn.Linear(repr_size, num_verbs)`, trained with `binary_focal_loss_with_logits` on prior-composed logits.

`num_verbs` is set by dataset in [main.py](main.py): 117 for HICO-DET, 24 for V-COCO. Evaluation is over 600 HOI triplet classes via `DetectionAPMeter`.

### The head split is the core mechanism

The paper's contribution is that attention heads carry explicit semantic roles, implemented as a **channel-wise split of the 8 heads into a human branch and an object branch**, threaded through the entire head:

```
h_dim = repr_dim // 8 * sub_headnum      o_dim = repr_dim // 8 * obj_headnum
```

Both `HumanObjectMatcher` (`h_mmf`/`o_mmf`), `FeatureHead` (`h_mapping`/`o_mapping`, `h_layers`/`o_layers`) and `TransformerDecoderLayer` (every `*_h`/`*_o` projection pair) slice tensors at `h_dim` and concatenate along the last dim. Any change to `repr_dim`, `sub_headnum` or `obj_headnum` propagates through all three — the `view(..., head_num, dim // head_num // 2)` reshapes will silently misalign or throw if the arithmetic stops working out. Note the `// 2`: half of each head's channels are content, half are positional, concatenated per-head before the attention call.

### The decoder token sequence

`TransformerDecoderLayer.forward` ([transformers_.py:357](transformers_.py#L357)) does not do conventional cross-attention. It concatenates five token groups into one sequence and runs a single self-attention where only the pair queries act as queries:

```
[ cls_token (HO pair queries) | backbone_token (HW) | clip_token (VLM patches + CLS) | answer_token | register_token (256) ]
```

- `clip_token` comes from `llava_feature`, projected from a hard-coded input dim of **1536**. `llava_feature[:-1]` are patch tokens, `llava_feature[-1]` is the CLS token.
- `answer_token` indexes a learned `nn.Embedding(256, q_dim)` by `llava_answer_idx.unique()` — the LLaVA text answer enters only as a set of discrete class indices, not as text embeddings.
- `register_token` is 256 learned, input-independent tokens.
- `layer_id` changes the query adapter's input dim: layer 0 receives the split `h_dim`/`o_dim` query, later layers receive full `q_dim`. `TransformerDecoder.__init__` therefore *rebuilds* its layers from the prototype's hyperparameters rather than deep-copying the passed `decoder_layer` — the instance handed to it is only a config carrier and is discarded.

### VLM features are a hard dependency, precomputed offline

The model cannot run without `--llava-answer-path` and `--llava-token-path`. `DataFactory.__getitem__` returns a 4-tuple `(image, target, llava_answer, llava_feature)`, and `PViC.forward` indexes `llava_answer[i]` / `llava_feature[i]` unconditionally — passing `None` raises. Features are generated out-of-band by the separate LLaVA fork and by [siglip/generate_feature.py](siglip/generate_feature.py) (SigLIP2 giant, patch16-384 — hence the 24×24 patch grid assumed in [visualization.py](visualization.py)).

The paths in [siglip/generate_feature.py](siglip/generate_feature.py), [analyze_hico.py](analyze_hico.py) and [analyze_vcoco.py](analyze_vcoco.py) are hard-coded absolute paths from the original author's machine; they need editing before use.

### Training engine

`CustomisedDLE` ([utils.py](utils.py)) subclasses `pocket.core.DistributedLearningEngine`. The engine splits each collated batch into `inputs` and `targets` by treating the **last** tuple element as targets, so in `_on_each_iteration` the VLM feature tensor arrives as `self._state.targets` while `llava_answer` comes from `self._state.inputs[2]`. Changing the arity or order of `custom_collate`'s return breaks this mapping.

`_on_end_epoch` runs full `test_hico()` after every epoch and switches its reporting between rare/non-rare (`--train-type default`) and seen/unseen (zero-shot splits) based on `test_dataloader.dataset.dataset.train_type`.

### Zero-shot splits

`--train-type` ∈ `{default, RF_UC, NF_UC, UV, UO}` is passed straight through `DataFactory` into `HICODet` — the split logic lives in the `hicodet` submodule, not here. V-COCO ignores it.

## Duplicated and stale modules

- **[pvic_vis.py](pvic_vis.py) / [transformers_vis.py](transformers_vis.py) are forks of [pvic.py](pvic.py) / [transformers_.py](transformers_.py)**, used by [visualization.py](visualization.py) and [eval_answers.py](eval_answers.py). `transformers_vis.py` is currently byte-identical to `transformers_.py`, and `pvic_vis.py` differs only in its import line and one type annotation. **Model changes must be mirrored into both forks**, or the visualization entry points will load checkpoints with a mismatched `state_dict`.
- **[attention.py](attention.py)** is a vendored copy of `torch.nn.MultiheadAttention` that returns un-averaged per-head attention weights (plus a `MultiheadAttentionSigmoid` variant that is imported but unused). Attention visualization depends on this; swapping in stock PyTorch attention breaks it.
- **[inference.py](inference.py) is dead code** inherited from upstream PViC. It calls `DataFactory` without the three now-required LLaVA arguments, hooks a `decoder.layers[-1].qk_attn` module that no longer exists, and calls `model([image])` without VLM features. Use [visualization.py](visualization.py) instead.
- **`main.py --sanity`** is likewise broken: `sanity_check` constructs `DataFactory` with three missing required arguments.
- **[attn.py](attn.py)** is a standalone collection of plotting experiments, imported by nothing.

[visualization.py](visualization.py) extracts attention via `register_forward_hook` on named submodules (`ho_matcher.h_mmf`, `decoder.layers[i].backbone_pos_proj_h`, `decoder.layers[i].self_attn`, …) and reconstructs the token-group offsets arithmetically to slice the attention matrix. Renaming any of those submodules, or changing the token concatenation order, silently produces wrong heatmaps rather than an error.

## Attribution

The codebase derives from PViC (BSD-3-Clause); [LICENSE](LICENSE) retains the original copyright notice. Module docstrings still carry upstream author headers — leave them intact.
