---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-to-video
tags:
- zen-ai
- video-generation
- text-to-video
- image-to-video
- diffusion
base_model: zenlm/zen-video
library_name: diffusers
---

# Zen Director 5B

**Zen Director** is a 5B parameter text-to-video and image-to-video generation model. It creates high-quality videos from text descriptions and images with controllable motion.

## Base Model

Text-to-Image-to-Video model with 5B parameters.

## Capabilities

- **Text-to-Video**: Generate videos from text descriptions
- **Image-to-Video**: Animate static images into videos
- **High Resolution**: Supports high-quality video generation
- **Efficient**: Optimized MoE architecture for fast inference

## Model Details

- **Architecture**: Mixture-of-Experts (MoE) Transformer
- **Parameters**: 5B total
- **Base**: Zen Director TI2V
- **Resolution**: Up to 1280x720
- **Frame Rate**: 24 FPS
- **Duration**: Up to 5 seconds

## Installation

```bash
pip install diffusers transformers accelerate torch
pip install av opencv-python pillow
```

## Usage

### Text-to-Video

```python
from diffusers import DiffusionPipeline
import torch

# Load the model
pipe = DiffusionPipeline.from_pretrained(
    "zenlm/zen-director",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate video from text
prompt = "A serene sunset over a calm ocean with waves gently lapping at the shore"
video = pipe(prompt, num_frames=120, height=720, width=1280).frames

# Save video
from diffusers.utils import export_to_video
export_to_video(video, "output.mp4", fps=24)
```

### Image-to-Video

```python
from PIL import Image

# Load starting image
image = Image.open("input.jpg")

# Generate video from image
video = pipe(
    prompt="Animate this image with gentle camera movement",
    image=image,
    num_frames=120
).frames

export_to_video(video, "animated.mp4", fps=24)
```

## Performance

- **Inference Speed**: ~2-3 seconds/frame on A100
- **Memory**: Requires 24GB+ VRAM for full resolution
- **Quantization**: FP16 recommended for consumer GPUs

## Roadmap

- ✅ **v1.0** - TI2V-5B base (current)
- 🔄 **v2.0** - Next generation upgrade
- 📋 **Future** - Fine-tuning for specific styles and domains

## Limitations

- Requires high-end GPU (24GB+ VRAM recommended)
- Video duration limited to 5 seconds
- Best results with detailed, specific prompts
- Some motion artifacts in complex scenes

## Links

- **GitHub**: https://github.com/zenlm
- **Zen Gym** (Training): https://github.com/zenlm/zen-gym
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine

## Citation

```bibtex
@misc{zen-director-2025,
  title={Zen Director 5B: Text-to-Video Generation Model},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://huggingface.co/zenlm/zen-director-5b}}
}

@article{zendirector2024,
  title={Zen Director: High-Quality Video Generation},
  author={Zen Team},
  journal={arXiv preprint},
  year={2024}
}
```

## License

Apache 2.0

---



Part of the **[Zen AI](https://github.com/zenlm)** ecosystem.
