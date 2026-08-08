# stable-diffusion.rs

轻量化的 Rust Stable Diffusion 工作区，使用 Candle 作为唯一张量和推理运行时，不依赖 PyTorch、TensorFlow、ONNX Runtime 或其他大型深度学习引擎。

项目当前已经使用本地 SD 1.5 Diffusers 权重完成 CPU 端真实 `512x512`、20-step 文生图，并提供 `sd` CLI 和本地 HTTP 服务。SD 2.1、SDXL、CivitAI 单文件模型及 LoRA/ControlNet 扩展正在持续接入，详细状态见 [TODO.md](TODO.md)。

## 特性

- Candle CPU 推理；CUDA/Metal 适配保留为后续设备 feature 扩展点。
- SD 1.5 Diffusers 目录加载与 DDIM 采样。
- SD 2.1 和 SDXL 配置入口，SDXL 使用双 text encoder 目录约定。
- 预计算 text conditioning safetensors：`prompt_embeds` 与 `negative_prompt_embeds`。
- 并发、断点续传和进度条模型下载基础设施。
- Canny 图像 annotator 基础实现。
- 本地 HTTP 任务服务和静态 `pages/` 目录服务。
- 模型、输出和下载元数据与推理代码分离，便于部署。

## Workspace

```text
projects/
  diffusion-types/       公共请求、模型族、pipeline 和校验类型
  diffusion-models/      safetensors 检查和模型架构识别
  diffusion-extensions/  annotator、LoRA/conditioning 数据层
  diffusion-registry/    well-known 模型和断点续传下载
  diffusion-tools/       sd CLI 和 Candle 生成 pipeline
  diffusion-server/      Axum HTTP 服务与任务状态
  diffuser-edit/         兼容的离线 VAE 工具
  diffuser-exif/         兼容的图像元数据工具
```

## 运行目录

打包后，`models/`、`outputs/`、`pages/` 默认位于 `sd.exe` 同目录：

```text
sd.exe/
  sd.exe
  models/
    sd15/
  outputs/
  pages/
```

模型目录说明见 [models/README.md](models/README.md)，输出目录说明见 [outputs/README.md](outputs/README.md)。模型权重和生成图片已加入 Git ignore，只保留 `.gitkeep` 和说明文档。

## 构建

```powershell
cargo build --release -p diffusion-tools --bin sd
cargo check --workspace
cargo test -p diffusion-types -p diffusion-extensions
```

## 下载 SD 1.5

```powershell
.\target\release\sd.exe download sd15 --output-dir E:\models
```

默认不指定 `--output-dir` 时，模型写入 `sd.exe` 同目录的 `models/`。

## 生成图片

```powershell
.\target\release\sd.exe generate `
  --model-dir E:\models\sd15 `
  --family sd15 `
  --prompt "a red cabin in a snowy forest at dawn" `
  --negative-prompt "blurry, low quality, watermark" `
  --width 512 `
  --height 512 `
  --steps 20 `
  --cfg-scale 7.5 `
  --seed 42 `
  --output .\outputs\cabin.png
```

预计算 conditioning 文件可以通过 `--conditioning` 传入，要求包含：

```text
prompt_embeds
negative_prompt_embeds
```

当前 CLI 会校验其 shape 和模型族 hidden size；LoRA、ControlNet 和 textual inversion 参数在执行路径尚未完整接入前会明确报错，不会静默忽略。

## HTTP 服务

```powershell
.\target\release\sd.exe serve `
  --model-dir E:\models\sd15 `
  --output-dir .\outputs `
  --pages-dir .\pages `
  --address 127.0.0.1:3000
```

可用端点：

- `GET /health`
- `POST /v1/images/generations`
- `GET /v1/tasks/{id}`

服务默认只监听 `127.0.0.1`。任务状态保存在进程内，重启后任务不会恢复。

## 设计约束

- 推理运行时只使用 Rust/Candle。
- 外部模型权重按需下载，不编译进二进制。
- 基础模型文件不可被请求级扩展修改；LoRA 将使用派生缓存实现隔离。
- 未识别的 checkpoint 架构必须报错，不得伪装成 SD 1.x 或 SDXL。
- 任何扩展只有在真正影响采样并产出可验证 PNG 后，才能标记为已支持。

## 当前限制

- 真实端到端验证目前是 SD 1.5 Diffusers CPU 路径。
- SD 2.1/SDXL 需要对应模型目录和完整双 encoder 资产。
- LoRA、ControlNet、textual inversion、CivitAI 单文件 checkpoint loader 仍在 TODO 中。
- CPU 推理速度较慢，固定 seed 在 CPU 上还没有完全解决 Candle 的设备 RNG 限制。

## License

见 [License.md](License.md)。第三方模型和 CivitAI 资源遵循各自模型许可证，使用前请自行确认授权条件。
