"""Convert a HuggingFace Whisper checkpoint to CTranslate2 (Faster Whisper) format."""

import argparse
import os
import subprocess
import sys

from .constants import MODEL_DIR


def convert(model_path: str = None, output_dir: str = None, quantization: str = "float16"):
    model_path = model_path or os.path.join(MODEL_DIR, "final")
    output_dir = output_dir or os.path.join(MODEL_DIR, "ct2")

    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", model_path,
        "--output_dir", output_dir,
        "--quantization", quantization,
        "--force",
    ]

    print(f"Converting {model_path} -> {output_dir} (quantization={quantization})")
    subprocess.run(cmd, check=True)
    print(f"Done. CT2 model saved to {output_dir}")
    print(f"Use with faster-whisper:\n  from faster_whisper import WhisperModel\n  model = WhisperModel(\"{output_dir}\")")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Whisper model to CTranslate2")
    parser.add_argument("--model", default=None, help=f"HF model path (default: {MODEL_DIR}/final)")
    parser.add_argument("--output", default=None, help=f"Output dir (default: {MODEL_DIR}/ct2)")
    parser.add_argument("--quantization", default="float16", choices=["float16", "float32", "int8", "int8_float16"])
    args = parser.parse_args()

    convert(args.model, args.output, args.quantization)
