# SimpleVLA-RL dependency changes (RTX 5090 / ECE780 container)

Date: 2026-10-28. Goal: make `examples/run_openvla_oft_rl_libero.sh` run in the ECE780 Docker image.

## Summary

SimpleVLA-RL now runs in its **own venv at `/opt/simplevla/.venv`**, the same isolation pattern
the Dockerfile already uses for openpi. It can't share system Python because the Dockerfile
deliberately keeps `robosuite==1.5.2` there, and LIBERO needs `robosuite==1.4.x`.

Frozen package set: [`requirements-5090.lock.txt`](requirements-5090.lock.txt)

## Problems found and fixes

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | Every CUDA op fails (`no kernel image`) | GPU is RTX 5090 (sm_120); torch 2.4.0+cu124 ships kernels only up to sm_90 | `torch==2.7.1` / `torchvision==0.22.1` from the **cu128** index |
| 2 | Can't upgrade torch | `vllm==0.6.3` hard-pins torch 2.4 | Dropped vllm (and xformers). The LIBERO script uses `rollout.name=hf`; vllm is only imported for `rollout.name=vllm` |
| 3 | Upstream veRL v0.2 (`examples/verl`, pip `-e`) pins `vllm<=0.6.3`, `tensordict<0.6` | SETUP.md install path | Not installed in the venv. SimpleVLA-RL ships its own `verl/`; the script now `cd`s to the repo root and sets `PYTHONPATH` so that copy is used (including by Ray workers) |
| 4 | `ImportError: cannot import name 'ForkingPickler'` | `tensordict==0.5.0` is incompatible with torch 2.7 | `tensordict==0.8.3`. `DataProto` chunk/concat/select was smoke-tested |
| 5 | transformers 5.20.2 / tokenizers 0.23 / timm missing | `modeling_prismatic.py` requires `timm` 0.9.10–0.9.16 (hard error) and expects `transformers==4.40.1`, `tokenizers==0.19.1` | openvla-oft transformers fork (4.40.1) + `tokenizers==0.19.1` + `timm==0.9.10` + `peft==0.11.1` + `accelerate==0.34.2` |
| 6 | `flash_attn`, `tensorflow`, `json_numpy` missing | Imported at module level (`dp_rob.py`, `rob_rollout.py`, `openvla_utils.py`) | flash-attn 2.8.3 prebuilt wheel (torch2.7, cxx11abi=TRUE; kernel verified on sm_120), `tensorflow-cpu==2.15.1`, `json-numpy` |
| 7 | `No module named robosuite.environments.manipulation.single_arm_env` | LIBERO is incompatible with robosuite 1.5 | `robosuite==1.4.1`, `mujoco==3.2.3`, `bddl==1.0.1`, `gym==0.25.2`, `easydict==1.9` (same as openpi's LIBERO venv). LIBERO itself: `-e /opt/openpi/third_party/libero --no-deps` |
| 8 | `UnpicklingError: Weights only load failed` loading LIBERO init states | torch ≥2.6 defaults `torch.load(weights_only=True)` | `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the run script (avoids patching LIBERO, which openpi shares) |
| 9 | raylet `Fatal Python error: Aborted`, `thread: Resource temporarily unavailable` | Container cgroup `pids.max=2048`; Ray prestarts 32 workers (one per CPU) | Workaround in script: `RAY_prestart_worker_first_driver=0`, `RAY_enable_worker_prestart=0` (peak dropped to ~785). **Proper fix: start the container with `--pids-limit=-1` (or e.g. 16384)** |

### Run-script config fixes (`examples/run_openvla_oft_rl_libero.sh`), not dependencies

- `ROBOT_PLATFORM=ALOHA` → `LIBERO`. ALOHA selects 14-dim / 25-step action chunks; this checkpoint is 7-dim / 8-step.
- `SFT_MODEL_PATH` → `models/Openvla-oft-SFT-libero10-traj1` (the folder with `config.json`).
  Note: earlier runs copied `*_prismatic.py`, `constants.py`, and `train_utils.py` into `models/` itself; those copies are leftovers and safe to delete.
- `NUM_GPUS=8` → `1`.
- `ALIGN_PATH` placeholder contained spaces, which broke Hydra parsing (`Error parsing override 'PATH'`); it now points at `align.json`.

## Memory: switched to LoRA training

Full fine-tuning does not fit on one 32 GB GPU. The fp32 actor is ~30 GB of weights alone, and a training
step needs ~120 GB (weights + grads + Adam). `param_offload` / `optimizer_offload` don't help: this codebase only
parks state on CPU *between* phases and moves it all back to the GPU for the update (`fsdp_workers.py`, `update_actor`).

The script now trains LoRA adapters on a frozen bf16 base:
- `actor_rollout_ref.model.lora_rank=32`, `lora_alpha=32` (target modules: all linear layers; 110.8M trainable params, 1.45%)
- `+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16` (the `+` is needed because the key isn't in `ppo_trainer.yaml`)
- `optim.lr=5e-5` (the full fine-tuning script used 5e-6; LoRA usually needs a higher LR, but this value is untuned)
- Code change, `verl/workers/fsdp_workers.py`: after `get_peft_model`, trainable LoRA params are upcast to fp32.
  peft creates adapters in the base dtype (bf16), and bf16 weights + Adam would round away updates of ~lr size.

Measured (2 tasks × 8 samples smoke test): the model loads, LoRA is applied, and LIBERO envs start. Rollout uses
**~19–20 GB** on the GPU. It OOM'd only because a process outside the container held ~11.7 GB at the time.
**The training update step's memory is not yet verified.** If it OOMs on a free GPU, set
`actor_rollout_ref.model.enable_gradient_checkpointing=True` or lower `actor.traj_mini_batch_size`.

## Rendering: EGL → OSMesa

During RL rollouts every LIBERO env failed to create its EGL context (`EGL_BAD_ALLOC` in `eglCreateContext`).
Stand-alone tests all passed: 16 concurrent envs, inside Ray, with NCCL initialized, and with the real model on the GPU.
So the exact trigger is unconfirmed; GPU memory contention during rollout is the likely cause.
- The script sets `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa` (CPU rendering; `libosmesa6` is already in the Dockerfile).
  Measured 58 ms/step vs 8 ms/step with EGL.
- Code change, `verl/workers/rollout/rob_rollout.py` (`env_worker`): the bare `except:` around env creation now prints
  the exception. Before, it retried forever and printed only `*** env initialization failed ***`.

## Dockerfile block

Append after the LIBERO config block (needs `/opt/openpi/third_party/libero` and `uv`, both already in the image).
Copy `requirements-5090.lock.txt` into the build context first.

```dockerfile
# ---------------------------------------------------------------------------
# SimpleVLA-RL (OpenVLA-OFT RL on LIBERO), isolated venv:
#   - torch 2.7.1+cu128 (RTX 5090 / sm_120 needs CUDA >= 12.8)
#   - robosuite 1.4.1 for LIBERO (system Python keeps robosuite 1.5.2)
#   - no vllm (pins torch 2.4; LIBERO script uses rollout.name=hf)
# ---------------------------------------------------------------------------
COPY requirements-5090.lock.txt /tmp/simplevla-requirements.txt
RUN uv venv --python 3.10 /opt/simplevla/.venv \
    && VIRTUAL_ENV=/opt/simplevla/.venv uv pip install -r /tmp/simplevla-requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match \
    && VIRTUAL_ENV=/opt/simplevla/.venv uv pip install --no-deps -e /opt/openpi/third_party/libero \
    && uv cache clean
```

Equivalent unpinned install, if you prefer it over the lock file:

```bash
uv venv --python 3.10 /opt/simplevla/.venv
export VIRTUAL_ENV=/opt/simplevla/.venv
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install \
  "torch==2.7.1" "torchvision==0.22.1" \
  "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git" \
  "tokenizers==0.19.1" "timm==0.9.10" "peft==0.11.1" "accelerate==0.34.2" sentencepiece \
  "numpy==1.26.4" "tensordict==0.8.3" "ray[default]==2.40.0" "hydra-core==1.3.2" \
  codetiming dill pandas "pyarrow>=15.0.0" pybind11 pylatexenc datasets wandb einops packaging ninja \
  "tensorflow-cpu==2.15.1" json-numpy \
  "robosuite==1.4.1" "mujoco==3.2.3" "bddl==1.0.1" "easydict==1.9" "gym==0.25.2" cloudpickle future "imageio[ffmpeg]" matplotlib opencv-python-headless \
  "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --no-deps -e /opt/openpi/third_party/libero
```

`docker run` flag: add `--pids-limit=-1` (the Ray env-var workaround is then optional).
