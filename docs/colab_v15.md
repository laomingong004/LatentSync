# LatentSync 1.5 Colab Deployment Guide

A ready-to-run notebook version of these steps is bundled as
[`notebooks/latentsync_v15_colab.ipynb`](../notebooks/latentsync_v15_colab.ipynb).
Follow the cells below if you prefer to copy commands manually into your own
Colab notebook. All commands assume you have selected a GPU runtime.

1. **Inspect the GPU**
   ```python
   !nvidia-smi
   ```

2. **Install runtime dependencies**
   ```python
   !apt-get -y install ffmpeg
   !pip install -q "numpy<2.0"
   !pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   !pip install -q diffusers==0.32.2 transformers==4.48.0 accelerate==0.32.1 \
     einops==0.7.0 omegaconf==2.3.0 decord==0.6.0 opencv-python==4.9.0.80 \
     mediapipe==0.10.11 python_speech_features==0.6 librosa==0.10.1 scenedetect==0.6.1 \
     ffmpeg-python==0.2.0 imageio==2.31.1 imageio-ffmpeg==0.5.1 lpips==0.1.4 \
     face-alignment==1.4.1 gradio==5.24.0 huggingface-hub==0.30.2 \
     insightface==0.7.3 onnxruntime-gpu==1.21.0 DeepCache==0.1.1
   ```

3. **Clone the repository**
   ```python
   %cd /content
   !git clone https://github.com/bytedance/LatentSync.git
   %cd LatentSync
   ```

4. **Download the LatentSync 1.5 weights and Whisper Tiny checkpoint**
   ```python
   from huggingface_hub import snapshot_download

   snapshot_download(
       repo_id="ByteDance/LatentSync-1.5",
       local_dir="checkpoints",
       local_dir_use_symlinks=False,
       allow_patterns=[
           "latentsync_unet.pt",
           "whisper/tiny.pt",
       ],
   )
   ```

5. **Ensure the bundled `app_v15.py` script is clean**
   The repository already contains the correct Gradio launcher. If you previously followed an older notebook snippet that produced escaped sequences such as `\"\"\"`, restore the clean version before running:
   ```python
   !git checkout -- app_v15.py
   ```

6. **(Optional) Upload the video and audio assets you want to process**
   ```python
   from google.colab import files

   uploads = files.upload()  # Provide a 25 FPS face video and a 16 kHz audio file
   ```

7. **Launch the Gradio WebUI with a public share URL**
   ```python
   !python app_v15.py
   ```


> [!NOTE]
> Google Colab currently ships with NumPy 2.x and an older Accelerate build. Downgrading NumPy ensures that
> OpenCV and MediaPipe import without `_ARRAY_API` errors, and bumping Accelerate satisfies the `clear_device_cache` symbol
> expected by Diffusers 0.32.2.
Once step 7 runs, Gradio will print a `https://xxxxx.gradio.live` link in the cell output. Open the link to access the interface, upload your media, and click **Process Video** to generate the synced result. Outputs are saved inside the notebook's `temp/` directory.

> **Why the earlier error appears**
> A previous draft of the instructions suggested generating `app_v15.py` via a raw string that escaped triple quotes (`\"\"\"`). Running that cell writes the escaped characters directly into the file, and Python treats them as a stray backslash followed by quotes, leading to `SyntaxError: unexpected character after line continuation character`. Restoring the tracked version of `app_v15.py` removes those backslashes and resolves the issue.
