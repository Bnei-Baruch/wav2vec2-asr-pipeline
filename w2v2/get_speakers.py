from pyannote.audio import Pipeline
import os
import torchaudio
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier


HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_ID = "pyannote/speaker-diarization-3.1"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)


def run(audio_path: str):
    output = diarization(audio_path)
    for turn, _, speaker in output.itertracks(yield_label=True):
        print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")

        identify_speakers(audio_path, "rav.wav")


def get_clip_from_output(output: Output):
    for turn, _, speaker in output.itertracks(yield_label=True):
        print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
        clip = clip_from_turn(turn)


def clip_from_turn(turn: Turn):
    return audio_path[turn.start : turn.end]


def identify_speakers(clip_path: str, ref_path: str):
    ref_emb = get_embedding(ref_path)
    unknown_emb = get_embedding(clip_path)

    score = F.cosine_similarity(ref_emb, unknown_emb)
    print(f"Сходство: {score.item():.4f}")


def get_embedding(file_path):
    signal, fs = torchaudio.load(file_path)
    embedding = classifier.encode_batch(signal)
    return embedding


def diarization(audio_path: str):
    pipeline = Pipeline.from_pretrained(MODEL_ID, token=HUGGINGFACE_TOKEN)

    output = pipeline(audio_path)
    return output.speaker_diarization


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarization")
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()
    run(args.audio_path)
