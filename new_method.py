from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.0",
  use_auth_token="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")

# run the pipeline on an audio file
diarization = pipeline("./sound/video.wav", num_speakers=2)

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

# Type -- segment type; should always by SPEAKER
# File ID -- file name; basename of the recording minus extension (e.g., rec1_a)
# Channel ID -- channel (1-indexed) that turn is on; should always be 1
# Turn Onset -- onset of turn in seconds from beginning of recording
# Turn Duration -- duration of turn in seconds
# Orthography Field -- should always by < NA >
# Speaker Type -- should always be < NA >
# Speaker Name -- name of speaker of turn; should be unique within scope of each file
# Confidence Score -- system confidence (probability) that information is correct; should always be < NA >
# Signal Lookahead Time -- should always be < NA >