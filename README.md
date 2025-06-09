# 🎨 Color Matching Engine for Hair Swatch Recommendation

This repository provides a modular pipeline to analyze a user's hair from portrait images and recommend the most similar hair swatch using either classical computer vision (CV) or vision-language models (VLMs) like Qwen2.5-VL or ColPali.

---

## 📁 Repository Structure

```
color_matching/
├── common/                  # Base interfaces for inference components
├── config/
│   ├── loader/              # Configuration loader (settings.yml)
│   └── files/               # Optional swatch files or templates
├── models/                 # Pre-built inference models (hair segmenter, VLMs)
├── src/
│   ├── HairMatchGeneratorCV.py     # Classic CV pipeline
│   ├── SwatchMatchGenerator.py     # VLM-based swatch matcher
│   └── helpers/                    # Feature matchers and utilities
├── local_test.py           # Entrypoint script to run a full inference
├── local_test_params.yml   # Local test configuration file
├── setup_env.sh            # One-click dependency installer
└── README.md
```

---

## ⚙️ Installation


> ⚠️ Before running anything, ensure you execute the setup script:
> ```bash
> bash setup_env.sh
> ```
```bash
git clone https://github.com/your-org/color_matching.git
cd color_matching
bash setup_env.sh
```

---

## 🧪 Running a Test

Use the built-in test runner:

```bash
python local_test.py
```

This script loads configuration from `local_test_params.yml` and `settings.yml`.

---

## 🛠️ Configuration Guide

### 🔧 `config/loader/settings.yml`

This YAML controls system-level configuration and model settings.

```yaml
general:
  artefacts_dir: ./artefacts     # Where input/mask/results will be saved

hair_match_generator:
  args:
    swatch_path: ./swatches       # Folder containing swatch images
```

If you're using VLMs instead of CV:

```yaml
swatch_match_generator:
  args:
    swatch_path: ./swatches
    device: cpu                   # or mps / cuda
    vlm_candidate: QwenV25Infer   # Model class name from models/
    models: [QwenV25Infer]        # List of model classes to load
```

---

### 📄 `local_test_params.yml`

This file tells the test script what image and pipeline to use:

```yaml
run_mode: cv     # or vlm

input_image: ./dataset/portraits/sample1.jpg

generator:
  name: HairMatchGeneratorCV      # or SwatchMatchGenerator for VLM
```

---

## 📂 Dataset Organization

Swatches should be stored as cropped square color patches:

```
./swatches/
├── black.png
├── light_blonde.jpg
├── golden_brown.jpeg
```

Portraits (user-uploaded inputs):

```
./dataset/portraits/
├── sample1.jpg
├── user_test.png
```

Make sure these are all front-facing and well-lit, ideally cropped to head-and-shoulder shots.

---

## 🧩 Core Components

### `HairMatchGeneratorCV`

Classical pipeline:
- Uses `HairSegmenter` to extract hair mask
- Matches to `HairSwatchMatcherCV` using LAB color stats

**Artifacts saved to**: `./artefacts/HairMatchGeneratorCV/`

### `SwatchMatchGenerator`

VLM-based approach:
- Uses ColPali or Qwen2.5-VL to extract embeddings
- Matches to swatches using text/image embedding similarity

**Artifacts saved to**: `./artefacts/SwatchMatchGenerator/`

---

## 🧠 Customization

- Add new matchers in `src/`
- Add new models in `models/`, register in `ModelManager`
- Extend segmentation logic in `HairSegmenter.py`
- Preprocess portrait images in `local_test.py`

---

## 🧾 Outputs

Each run saves:

```
artefacts/
└── HairMatchGeneratorCV/
    ├── HairMatchGeneratorCV_<timestamp>_input.png
    ├── HairMatchGeneratorCV_<timestamp>_hair_mask.png
    └── [optional future] result.json
```

---

## 📜 License

MIT License
