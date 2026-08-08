# Agent Guide

## Project Objective

This repository builds a deployable Stable Diffusion implementation in Rust. Candle is the only tensor/inference runtime. Do not add Python, PyTorch, TensorFlow, ONNX Runtime, or another large DL engine to the service path.

## Workspace Ownership

- `projects/diffusion-types`: public request/result types, model families, validation, cancellation, progress.
- `projects/diffusion-models`: safetensors inspection, metadata, architecture detection, checkpoint conversion.
- `projects/diffusion-extensions`: LoRA, textual inversion, ControlNet and image annotators.
- `projects/diffusion-registry`: well-known sources, CivitAI metadata, resumable downloads and checksums.
- `projects/diffusion-tools`: the `sd` binary and the shared local Candle pipeline.
- `projects/diffusion-server`: local Axum API and in-process task state.

Do not create a second CLI crate or a second inference implementation. The binary name is `sd` and it lives in `diffusion-tools`.

## Runtime Layout

Runtime assets are beside `sd.exe`:

```text
models/   downloaded model weights; ignored by Git
outputs/  generated images and logs; ignored by Git
pages/    built Vue/Vite frontend assets
```

`models/` and `outputs/` contain `.gitkeep` and README files only in source control. Never commit model weights, generated images, API tokens, or download fragments.

## Required Checks

Run these after relevant changes:

```powershell
cargo check --workspace
cargo test -p diffusion-types -p diffusion-extensions
cargo build --release -p diffusion-tools --bin sd
```

For generation changes, require a real decodable PNG and verify dimensions. A process exit or a one-step noise image is not an end-to-end success.

## Behavioral Rules

- Do not silently ignore CLI or HTTP extension parameters. Reject unsupported inputs with an explicit error until their runtime path exists.
- Validate dimensions, steps, CFG, ControlNet ranges, tensor shapes, dtype, and model family before allocating large model components.
- Keep the base model immutable for request-level extensions.
- Use atomic temporary-file writes for generated model artifacts and metadata.
- Preserve user changes in a dirty worktree; do not reset or checkout unrelated files.
- Prefer existing Candle and repository APIs over new abstractions.
- Use ASCII for source and documentation unless an existing file requires another encoding.

## Model Support Status

SD 1.5 Diffusers CPU text-to-image is the verified baseline. SD 2.1 and SDXL configuration paths exist but require their model assets. LoRA, textual inversion, ControlNet, depth/pose annotators, CivitAI downloads, and single-file checkpoint conversion are tracked in `TODO.md`; do not claim them as supported until they alter sampling and produce verified PNG output.

## Extension Implementation Order

1. Validate and load precomputed conditioning.
2. Implement LoRA key normalization, derived content-addressed weights, and UNet/text encoder application.
3. Implement SD 1.5 ControlNet residuals and Canny, then depth/OpenPose.
4. Add single-file checkpoint splitting and architecture detection.
5. Add CivitAI API/token handling and Pony/SDXL end-to-end tests.

## Security

Read `CIVITAI_API_TOKEN` from the environment only. Never write it to `.sd-download.json`, logs, task status, or generated metadata. Bind the local server to `127.0.0.1` by default.
