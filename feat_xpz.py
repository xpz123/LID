from pydub import AudioSegment


def load_pcm(pcmfile, framerate=16000, samplewidth=2, channel=1):
   return AudioSegment.from_raw(pcmfile, frame_rate=framerate, sample_width=samplewidth, channels=channel)


def seg_audio(audio, seglen=3000):
    duration = len(audio)
    segs = duration / seglen
    for i in 
