"""LatentSync 1.5 Gradio demo for Colab deployments.

This module mirrors the functionality showcased in the documentation snippet
shared with users, but keeps the Gradio layout identical to the official
`gradio_app.py` while loading the Stage2 configuration that matches the v1.5
checkpoints (256×256).

Run with:
    python app_v15.py

It will automatically enable Gradio's share mode.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gradio as gr
from omegaconf import OmegaConf

from scripts.inference import main

CONFIG_PATH = Path("configs/unet/stage2.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def process_video(
    video_path: str,
    audio_path: str,
    guidance_scale: float,
    inference_steps: int,
    seed: int,
) -> str:
    """Run LatentSync inference and return the path to the output video."""
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_file_path = Path(video_path)
    video_path = video_file_path.resolve().as_posix()
    audio_path = Path(audio_path).resolve().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{video_file_path.stem}_{current_time}.mp4"

    config = OmegaConf.load(CONFIG_PATH)
    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )

    args = _create_args(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path.as_posix(),
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    main(config=config, args=args)
    return output_path.as_posix()


def _create_args(
    *,
    video_path: str,
    audio_path: str,
    output_path: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            CHECKPOINT_PATH.resolve().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--temp_dir",
            "temp",
            "--enable_deepcache",
        ]
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="LatentSync 1.5 Demo") as demo:
        gr.Markdown(
            """
            <h1 align="center">LatentSync 1.5</h1>
            <p align="center">Upload a 25 FPS face video and a 16 kHz audio file to perform lip sync.</p>
            """
        )

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input Video")
                audio_input = gr.Audio(label="Input Audio", type="filepath")

                with gr.Row():
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=1.5,
                        step=0.1,
                        label="Guidance Scale",
                    )
                    inference_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Inference Steps",
                    )

                seed = gr.Number(value=1247, label="Random Seed", precision=0)
                process_btn = gr.Button("Process Video")

            with gr.Column():
                video_output = gr.Video(label="Output Video")

        process_btn.click(
            fn=process_video,
            inputs=[video_input, audio_input, guidance_scale, inference_steps, seed],
            outputs=video_output,
        )

    return demo


def main_entry() -> None:
    demo = build_demo()
    demo.queue().launch(share=True)


if __name__ == "__main__":
    main_entry()
