# 🧠 FOV reconstruction for VR experiment


## 📦 Installation

### 1. Create the conda environment

```bash
conda create -n avr_env python=3.9.23 -y          
conda activate avr_env
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

## ▶️ Runing the Code

### 0. Obtain VR video frames

```bash
python video_to_frames.py --stimuli_dir /path/to/stim_dir --video_fps 30 --desired_fps 1
```

| Argument | Default | Description |
|---|---|---|
| `--stimuli_dir` | — | Path to directory containing raw stimulus videos |
| `--video_fps` | `30` | Original frame rate of the source videos |
| `--desired_fps` | `1` | Frame rate to downsample to when extracting frames |
| `--crop_half` | `False` | If set, crops each frame to half width (for stereo VR footage) |
| `--verbose` | `False` | Print progress per video during extraction |

### 1. Preprocess VR experimental data

```bash
python preprocess_exp_data.py
```

