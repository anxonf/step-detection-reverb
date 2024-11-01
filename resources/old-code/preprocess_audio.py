import sys
import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import noisereduce as nr
import pylab as pl
import threading
import pyroomacoustics as pra
import IPython
from scipy.signal import fftconvolve
from scipy.io import wavfile


##### FUNCTIONS SECTION #####

# Function to play the audio signal
def play_audio(audio_signal, sample_rate):
    sd.play(audio_signal, sample_rate)
    sd.wait()

#Function to convert multichannel signals to mono
def convert_to_mono(audio_signal):
    # Si la señal ya es mono, retornarla sin cambios
    if audio_signal.ndim == 1:
        return audio_signal

    # Obtener el número de canales y la longitud de la señal
    num_channels = audio_signal.shape[1]
    signal_length = audio_signal.shape[0]

    # Calcular la media de los canales para obtener el canal mono
    mono_signal = np.mean(audio_signal, axis=1)

    # Redimensionar la señal para que sea un vector unidimensional
    mono_signal = np.reshape(mono_signal, (signal_length,))

    return mono_signal

# Function to plot the time domain representation of an audio signal
def plot_audio_analysis(audio_signal, sample_rate):
    # Compute the spectrum
    freq, spectrum = signal.periodogram(audio_signal, fs=sample_rate)
    print("LONGITUD ARRAY FREQS: " + str(len(freq)))
    print("LONGITUD ARRAY SPECTRUM: " + str(len(spectrum)))
    # Create a single plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

    # Plot the spectrum
    ax1.plot(freq, 10 * np.log10(spectrum))
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Spectrum')

    # Plot the time domain representation
    duration = len(audio_signal) / sample_rate
    print("DURACION: " + str(duration))

    time = np.linspace(0, duration, len(audio_signal))
    print("ARRAY TIEMPO: ", time)

    ax2.plot(time, audio_signal)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Time Domain')

    ax2.set_xlim([0, duration])

    # Display the plot
    plt.grid(True)
    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()

# Function to apply a noise filter to the audio signal
def apply_noise_filter(audio_signal):
    # Compute the power spectrum of the audio signal
    freq, spectrum = signal.periodogram(audio_signal, fs=sample_rate)

    # Estimate the noise floor level
    noise_floor = np.mean(spectrum)

    # Set a threshold to distinguish between signal and noise
    threshold = noise_floor * 10

    # Apply the threshold to remove the noise
    filtered_spectrum = np.where(spectrum > threshold, spectrum, 0)

    # Reconstruct the filtered audio signal
    filtered_audio = signal.spectrogram(filtered_spectrum, fs=sample_rate, return_onesided=False)[2]

    plot_audio_analysis(filtered_audio, sample_rate)

    return filtered_audio

# Function to create the Spectrogram, calculate background noise and remove it
def spectral_substraction(audio_signal, sample_rate):
    # Create the Spectrogram
    #f, t, spectrogram = np.abs(signal.stft(audio_signal))

    # Estimate noise
    noise = audio_signal[:sample_rate*4]

    freq, time, noise_spec = np.abs(signal.stft(noise, fs = 10000, nperseg = 256)) 
    noise_power = np.mean(np.square(noise_spec), axis=1)

    # Subtract the noise PSD from the signal PSD 
    freq, time, signal_spec = np.abs(signal.stft(audio_signal, fs = 10000, nperseg = 256)) 
    signal_power = np.square(signal_spec) 
    signal_power_denoised = np.maximum(signal_power - 2*noise_power[:, np.newaxis], 0) 

    # Reconstruct the denoised audio signal 
    signal_spec_denoised = np.sqrt(signal_power_denoised) * np.exp(1j*np.angle(signal_spec)) 
    time, audio_denoised = signal.istft(signal_spec_denoised, fs = 10000, nperseg = 256) 

    return audio_denoised

# Noise reduction function
def noise_reduction(audio_signal, sample_rate):

    noisy_section = audio_signal[0:5000]

    noise_reduced = nr.reduce_noise(y= audio_signal, y_noise= noisy_section, sr= sample_rate)

    return noise_reduced

# Create Beamform on a room
def createBeamform(room):
    print("Plotting Beamformed")
    room.plot(freq=[1000, 2000, 4000, 8000], img_order=4)
    plt.show()


def simulateStreamingAudio(signal, fs):
    h_len = 50
    h = np.ones(h_len)
    h /= np.linalg.norm(h)

    # stft parameters
    fft_len = 512
    block_size = fft_len - h_len + 1  # make sure the FFT size is a power of 2
    hop = block_size // 2  # half overlap
    window = pra.hann(block_size, flag='asymmetric', length='full') 

    # Create the STFT object + set filter and appropriate zero-padding
    stft = pra.transform.STFT(block_size, hop=hop, analysis_window=window, channels=1)
    stft.set_filter(h, zb=h.shape[0] - 1)

    processed_audio = np.zeros(signal.shape)
    n = 0
    while  signal.shape[0] - n > hop:

        stft.analysis(signal[n:n+hop,])
        stft.process()  # apply the filter
        processed_audio[n:n+hop,] = stft.synthesis()
        n += hop
        
    # plot the spectrogram before and after filtering
    fig = plt.figure()
    fig.set_size_inches(20, 8)
    plt.subplot(2,1,1)
    plt.specgram(signal[:n-hop].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30)
    plt.title('Original Signal', fontsize=22)
    plt.subplot(2,1,2)
    plt.specgram(processed_audio[hop:n], NFFT=256, Fs=fs, vmin=-20, vmax=30)
    plt.title('Lowpass Filtered Signal', fontsize=22)
    plt.tight_layout(pad=0.5)
    plt.show()



############################################    
############### MAIN PROGRAM ###############
############################################    

# Check if the audio file path is provided as a command-line argument
if len(sys.argv) != 2:
    print("Please provide the path to the audio file as a command-line argument.")
    print("Usage: python preprocess_audio.py <audio_file.wav>")
    sys.exit(1)

# Get the audio file path from the command-line argument
audio_file_path = sys.argv[1]
audio_file_name = os.path.basename(audio_file_path)
audio_file_path=audio_file_path.replace(audio_file_name,"")
print("AUDIO FILE PATH: " + audio_file_path)
print("AUDIO FILE NAME: " + audio_file_name)

# Get the audio file path from the command-line argument
audio_file = sys.argv[1]

# Load the audio file
try:
    sample_rate, audio_signal = wavfile.read(audio_file)

except Exception as e:
    print("Error loading the audio file:", str(e))
    sys.exit(1)

# Create a thread for the audio playback
audio_thread = threading.Thread(target=play_audio, args=(audio_signal, sample_rate))

# Creating the room to run the simulation
print("CREATING ROOM")

#fft_len = 512
Lg_t = 0.100                # filter size in seconds
Lg = np.ceil(Lg_t*sample_rate)       # in samples

room = pra.ShoeBox([5,30,4], fs=sample_rate, ray_tracing=True, air_absorption=False, materials=pra.Material(0.01, 0.1))

# add source
room.add_source([2.5,0.5,1.0], delay=5., signal=audio_signal)

# add noise source
fsn, noise = wavfile.read("TEST_WhiteNoise.wav")
room.add_source([2.5, 15., 2.], delay=0., signal=noise[:len(audio_signal)])

print("Tasa muestreo fuente: " + str(sample_rate))
print("Tasa muestro noise: " + str(fsn))

print("CREATING MICROPHONE ARRAY")
# add microphone array
R = np.c_[
    [2.5, 3.0, 3],  # mic 1
    [2.5, 6.0, 3],  # mic 2
    [2.5, 9.0, 3],  # mic 3
    [2.5, 12.0, 3],  # mic 4
    [2.5, 15.0, 3],  # mic 5
    [2.5, 18.0, 3],  # mic 6
    [2.5, 21.0, 3],  # mic 7
    [2.5, 24.0, 3],  # mic 8
    [2.5, 27.0, 3],  # mic 9
    ]

print(R)

#mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)
mics = pra.Beamformer(R, room.fs)
room.add_microphone_array(mics)

# Compute DAS weights

#mics.rake_delay_and_sum_weights(room.sources[0][:1])
#room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])

# plot for the room
room.image_source_model()

fig, ax = room.plot(img_order=3)
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 31])
ax.set_zlim([-1, 5])
plt.axis('scaled')

# simulate propagation
room.compute_rir()

# rir of each michrophone
room.plot_rir()
fig = plt.gcf()

plt.show()

# reverberation time of the rir
t60 = pra.experimental.measure_rt60(room.rir[0][0], fs=room.fs, plot=True)
print(f"The RT60 is {t60 * 1000:.0f} ms")

plt.show()

# simulate the signal convolved with impulse responses
room.simulate()
print("Room Mic_Array Signals Shape: ")
print(room.mic_array.signals.shape)
plt.show()

print("First Mic:")
#sd.play(room.mic_array.signals[1,:],sample_rate)
#sd.play(room.mic_array.signals[0,:],sample_rate)
#sd.wait()

# DAS beamforming improve
#signal_das = mics.process(FD=False)
#print("DAS Beamformed Signal:")
#sd.play(signal_das, sample_rate)
#sd.wait()

#print("Removing RIR from the signal on the microphone 1")
#deconvolution = signal.deconvolve(room.mic_array.signals[0,:], room.rir[1][0])

#sd.play(deconvolution, sample_rate)
#sd.wait()

# Plot the spectrum and time domain representation of the audio signal
plot_audio_analysis(room.mic_array.signals[1,:], sample_rate)

# Apply noise filter to the audio signal
print("Aplicando filtrado de eliminación de ruido de fondo...")

denoised_signal = noise_reduction(room.mic_array.signals[1,:], sample_rate)
output_file = "denoised_" + audio_file_name

# Save the cleaned audio to a new file
try:
    sf.write(output_file, denoised_signal, sample_rate)
    print("Filtered audio saved to:", output_file)
except Exception as e:
    print("Error saving the filtered audio:", str(e))

audio_thread = threading.Thread(target=play_audio, args=(denoised_signal, sample_rate))

# Start the audio playback thread
print("Reproduciendo señal filtrada...")
audio_thread.start()

# Plot the spectrum and time domain representation of the filtered audio signal
plot_audio_analysis(denoised_signal, sample_rate)

createBeamform(room)

#simulateStreamingAudio(audio_signal, sample_rate)