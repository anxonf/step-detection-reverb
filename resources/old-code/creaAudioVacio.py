import wave
import pydub
from audioop import add
from pydub import AudioSegment
from pydub.playback import play

salida = "silencio.wav"

# create 1 sec of silence audio segment
segment = AudioSegment.silent(duration=30000)

#Either save modified audio
segment.export(salida, format="wav")