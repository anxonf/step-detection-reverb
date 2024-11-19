import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Python code to preprocess audio for deep learning
file = "..\sounds\Pasos.wav"


# STEP 1: Waveform  (time domain)
signal, sr = librosa.load(file)
librosa.display.waveshow(signal, sr=sr)

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# STEP 2: fft -> spectrum (frequency domain)
fft = np.fft.fft(signal)
magnitude = np.abs(fft) # get absolute values of the complex values: the contribution of each frequency to the overall sound
frequency = np.linspace(0, sr, len(magnitude)) # gives a evenly spaced list of numbers of size "len(magnitude)" between the values "0" and "sr"

left_frequency = frequency[:int(len(frequency)/2)] # as the spectrum is simetric to the middle frequency, we only need to print the first half of samples
left_magnitude = magnitude[:int(len(magnitude)/2)] # as the spectrum is simetric to the middle frequency, we only need to print the first half of samples

#print("Frequencies: {}".format(left_frequency))
#print("Magnitudes: {}".format(left_magnitude))
plt.plot(left_frequency, left_magnitude)
plt.xscale('linear')
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()


# STEP 3: stft -> spectrogram
n_fft = 2048 # number of samples for fft -> the window of samples we are considering for each fft while running through the whole signal
hop_length = 512 # number of samples we are shifting each fft to the right -> the number of samples we move the window to compute the stft

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft) # compute the stft
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram) # obtain the spectrogram in a logarithmical scale

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length) # visualize spectrogram data as a heatmap
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()


# STEP 4: MFCCs
MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()