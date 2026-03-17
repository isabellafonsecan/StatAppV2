import os
from pyannote.audio import Pipeline


def main():
    audio_path = "audio.mp3"

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "Missing HF_TOKEN environment variable.\n"
            "Set it like: export HF_TOKEN='your_huggingface_token'\n"
            "Then re-run."
        )

    # Requires HF access; commonly used diarization pipeline:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    diarization = pipeline(audio_path)

    # Print RTTM-like output: start, end, speaker label
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        end = turn.end
        print(f"{start:.2f}\t{end:.2f}\t{speaker}")


if __name__ == "__main__":
    main()