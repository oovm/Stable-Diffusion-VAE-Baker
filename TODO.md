# Stable Diffusion Rust TODO

## Extension Runtime

- [x] Add shared request types for LoRA, embeddings, conditioning, annotators, and ControlNet.
- [x] Validate generation dimensions, steps, CFG, ControlNet weights, and step ranges.
- [x] Load precomputed `prompt_embeds` and `negative_prompt_embeds` from safetensors.
- [x] Add basic LoRA safetensors key discovery and SHA-256 helpers.
- [ ] Apply LoRA deltas to UNet and text encoder weights at request time.
- [ ] Add content-addressed, atomic derived-weight cache for LoRA combinations.
- [ ] Load textual inversion `.safetensors` and `.pt` files, including multi-vector tokens.
- [ ] Inject textual inversion vectors into the CLIP tokenizer/embedding path.
- [ ] Implement SD 1.5 ControlNet residual generation.
- [ ] Inject ControlNet down-block and mid-block residuals into the Candle UNet.
- [ ] Support multiple ControlNets with independent weights and step ranges.
- [ ] Wire Canny, depth, and OpenPose annotators into the generation request.
- [ ] Return explicit errors for missing annotator weights and invalid extension shapes.

## Third-Party Models

- [ ] Add CivitAI API metadata and model-version download command.
- [ ] Read `CIVITAI_API_TOKEN` without persisting it in metadata or logs.
- [ ] Add SHA-256 verification and atomic writes to CivitAI downloads.
- [ ] Implement single-file A1111/ComfyUI/CivitAI safetensors loading.
- [ ] Normalize `model.diffusion_model.*`, `first_stage_model.*`, and `conditioner.*` keys.
- [ ] Detect SD 1.x, SD 2.x, and SDXL from metadata, keys, and tensor shapes.
- [ ] Preserve derivative metadata such as Pony, Illustrious, and custom model labels.
- [ ] Split single-file checkpoints into reusable UNet, VAE, and text encoder components.
- [ ] Generate a real PNG from a public CivitAI SD 1.5 checkpoint.
- [ ] Generate a real PNG from a Pony/SDXL checkpoint.

## API and Tests

- [ ] Parse `extra_body` into the same `GenerationRequest` used by the CLI.
- [ ] Reject unknown extension fields instead of silently ignoring them.
- [ ] Add task details for extension configuration, cache hits, and failure reasons.
- [ ] Add cancellation checks at every diffusion step and ControlNet boundary.
- [ ] Add LoRA zero-weight, alpha, rank, and multi-LoRA tests.
- [ ] Add textual inversion shape and multi-vector token tests.
- [ ] Add ControlNet zero-weight, residual merge, and step-boundary tests.
- [ ] Add fixed-seed baseline, LoRA, embedding, Canny, depth, and OpenPose PNG outputs.
- [ ] Verify extension outputs differ from baseline and remain valid PNGs.

