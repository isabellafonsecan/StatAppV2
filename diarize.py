import os
from pyannote.audio import Pipeline


def main():
    audio_path = "audio.mp3"

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "Missing HF_TOKEN environment variable.\n"
            'Set it like: export HF_TOKEN="hf_..." then re-run.'
        )

    # Newer pyannote versions use `token=...`; older ones use `use_auth_token=...`.
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )

    output = pipeline(audio_path)

    # In your version, `pipeline(...)` returns DiarizeOutput with attribute `speaker_diarization`
    diarization = output.speaker_diarization

    # Iterate segments
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"{turn.start:.2f}\t{turn.end:.2f}\t{speaker}")


if __name__ == "__main__":
    main()