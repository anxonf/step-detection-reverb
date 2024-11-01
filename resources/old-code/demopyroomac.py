import sys
import os
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra

#####################
# Create 2D/3D Room #
#####################

def convert_to_mono(audio_signal):

    # Si la señal ya es mono, retornarla sin cambios
    if audio_signal.ndim == 1:
        print("Señal MONO")
        return audio_signal

    print("Señal Estéreo, procesando para obtener la señal MONO")
    # Obtener el número de canales y la longitud de la señal
    num_channels = audio_signal.shape[1]
    signal_length = audio_signal.shape[0]

    # Calcular la media de los canales para obtener el canal mono
    mono_signal = np.mean(audio_signal, axis=1)

    # Redimensionar la señal para que sea un vector unidimensional
    mono_signal = np.reshape(mono_signal, (signal_length,))

    return mono_signal


####### CONSTANTS ########
# Location of sources
azimuth = np.array([61., 270.]) / 180. * np.pi
distance = 2.  # meters

c = 343.    # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_range = [300, 3500]

snr_db = 5.    # signal-to-noise ratio
sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2

####### END OF CONSTANTS ########


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

# specify signal source
fs, signal = wavfile.read(audio_file)

# listen the output original signal
print("Original WAV:")
sd.play(signal, fs)

room = pra.ShoeBox([5,30,4], fs=fs, ray_tracing=True, air_absorption=True, materials=pra.Material(0.2, 0.15))

# Set the ray tracing parameters
room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)

# add source
room.add_source([2.5,0.5,1.0], signal=signal)
# specify noise source
fsn, noise = wavfile.read("D:/TELECO/GRADO/TFG/step-detection-reverb/resources/sounds/noise/TEST_WhiteNoise.wav")  # may spit out a warning when reading but it's alright!
room.add_source([2.5, 15., 0.01], signal=noise[:len(signal)])
print("fs: " + str(fs))
print("fsn: " + str(fsn))

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

#room.add_microphone_array(pra.MicrophoneArray(R, room.fs))


Lg_t = 0.100                # filter size in seconds
Lg = np.ceil(Lg_t*fs)       # in samples

fft_len = 512

mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)
room.add_microphone_array(mics)

# Compute DAS weights
mics.rake_delay_and_sum_weights(room.sources[0][:1])


#mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)
#room.add_microphone_array(R)

# Compute DAS weights
#R.rake_delay_and_sum_weights(room.sources[0][:1])

# plot for the room
room.image_source_model()

fig, ax = room.plot(img_order=3)
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 31])
ax.set_zlim([-1, 5])
plt.axis('scaled')

# simulate propagation and listen the center michrophone
room.compute_rir()
# rir of each michrophone
room.plot_rir()
fig = plt.gcf()

plt.show()

# reverberation time of the rir
t60 = pra.experimental.measure_rt60(room.rir[0][0], fs=room.fs, plot=True)
print(f"The RT60 is {t60 * 1000:.0f} ms")

plt.show()


###############
# Beamforming #
###############

# simulate the signal convolved with impulse responses
room.simulate()
print("Room Mic_Array Signals Shape: ")
print(room.mic_array.signals.shape)
plt.show()

print("First Mic:")
sd.play(room.mic_array.signals[0,:],fs)
sd.wait()
#IPython.display.Audio(room.mic_array.signals[0,:], rate=fs)

# DAS beamforming improve
signal_das = R.process(FD=False)
print("DAS Beamformed Signal:")
sd.play(signal_das, fs)
sd.wait()

"""
########################
# Direction of Arrival #
########################

X = pra.transform.stft.analysis(room.mic_array.signals.T, nfft, nfft // 2)
X = X.transpose([2, 1, 0])

algo_names = ['SRP', 'MUSIC', 'FRIDA', 'TOPS']
spatial_resp = dict()

# loop through algos
for algo_name in algo_names:
    # Construct the new DOA object
    # the max_four parameter is necessary for FRIDA only
    doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, num_src=2, max_four=4)

    # this call here perform localization on the frames in X
    doa.locate_sources(X, freq_range=freq_range)
    
    # store spatial response
    if algo_name == 'FRIDA':
        spatial_resp[algo_name] = np.abs(doa._gen_dirty_img())
    else:
        spatial_resp[algo_name] = doa.grid.values
        
    # normalize   
    min_val = spatial_resp[algo_name].min()
    max_val = spatial_resp[algo_name].max()
    spatial_resp[algo_name] = (spatial_resp[algo_name] - min_val) / (max_val - min_val)

# plotting param
base = 1.
height = 10.
true_col = [0, 0, 0]

# loop through algos
phi_plt = doa.grid.azimuth
for algo_name in algo_names:
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    c_phi_plt = np.r_[phi_plt, phi_plt[0]]
    c_dirty_img = np.r_[spatial_resp[algo_name], spatial_resp[algo_name][0]]
    ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=3,
            alpha=0.55, linestyle='-',
            label="spatial spectrum")
    plt.title(algo_name)
    
    # plot true loc
    for angle in azimuth:
        ax.plot([angle, angle], [base, base + height], linewidth=3, linestyle='--',
            color=true_col, alpha=0.6)
    K = len(azimuth)
    ax.scatter(azimuth, base + height*np.ones(K), c=np.tile(true_col,
               (K, 1)), s=500, alpha=0.75, marker='*',
               linewidths=0,
               label='true locations')

    plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, framealpha=0.5,
              scatterpoints=1, loc='center right', fontsize=16,
              ncol=1, bbox_to_anchor=(1.6, 0.5),
              handletextpad=.2, columnspacing=1.7, labelspacing=0.1)

    ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
    ax.xaxis.set_label_coords(0.5, -0.11)
    ax.set_yticks(np.linspace(0, 1, 2))
    ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
    ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
    ax.set_ylim([0, 1.05 * (base + height)])
    
plt.show()


print("Simulated propagation to first mic:")
plt.plot(room.mic_array.signals[0,:])

plt.show()
sd.play(room.mic_array.signals[0,:],fs)
sd.wait()

print("Simulated propagation to second mic:")
plt.plot(room.mic_array.signals[1,:])

plt.show()

sd.play(room.mic_array.signals[1,:],fs)
sd.wait()

print("Simulated propagation to third mic:")
plt.plot(room.mic_array.signals[2,:])

plt.show()

sd.play(room.mic_array.signals[2,:])
sd.wait()

print("Simulated propagation to fourth mic:")
plt.plot(room.mic_array.signals[3,:])

plt.show()

sd.play(room.mic_array.signals[3,:])
sd.wait()

print("Simulated propagation to fifth mic:")
plt.plot(room.mic_array.signals[4,:])

plt.show()

sd.play(room.mic_array.signals[4,:])
sd.wait()

print("Simulated propagation to sixth mic:")
plt.plot(room.mic_array.signals[5,:])

plt.show()

sd.play(room.mic_array.signals[5,:])
sd.wait()

print("Simulated propagation to seventh mic:")
plt.plot(room.mic_array.signals[6,:])

plt.show()

sd.play(room.mic_array.signals[6,:])
sd.wait()

print("Simulated propagation to eighth mic:")
plt.plot(room.mic_array.signals[7,:])

plt.show()

sd.play(room.mic_array.signals[7,:])
sd.wait()

print("Simulated propagation to nineth mic:")
plt.plot(room.mic_array.signals[8,:])

plt.show()

sd.play(room.mic_array.signals[8,:])
sd.wait()

print("Simulated propagation to tenth mic:")
plt.plot(room.mic_array.signals[9,:])

plt.show()

sd.play(room.mic_array.signals[9,:])
sd.wait()

"""



""""







######################
# Adaptive Filtering #
######################

# parameters
length = 15        # the unknown filter length
n_samples = 4000   # the number of samples to run
SNR = 15           # signal to noise ratio
fs = 16000

# the unknown filter (unit norm)
w = np.random.randn(length)
w /= np.linalg.norm(w)

# create a known driving signal
x = np.random.randn(n_samples)

# convolve with the unknown filter
d_clean = fftconvolve(x, w)[:n_samples]

# add some noise to the reference signal
d = d_clean + np.random.randn(n_samples) * 10**(-SNR / 20.)

# create a bunch adaptive filters
adfilt = dict(
    nlms=dict(
        filter=pra.adaptive.NLMS(length, mu=0.5), 
        error=np.zeros(n_samples),
        ),
    blocklms=dict(
        filter=pra.adaptive.BlockLMS(length, mu=1./15./2.), 
        error=np.zeros(n_samples),
        ),
    rls=dict(
        filter=pra.adaptive.RLS(length, lmbd=1., delta=2.0),
        error=np.zeros(n_samples),
        ),
    blockrls=dict(
        filter=pra.adaptive.BlockRLS(length, lmbd=1., delta=2.0),
        error=np.zeros(n_samples),
        ),
    )

for i in range(n_samples):
    for algo in adfilt.values():
        algo['filter'].update(x[i], d[i])
        algo['error'][i] = np.linalg.norm(algo['filter'].w - w)

plt.figure()
for algo in adfilt.values():
    plt.semilogy(np.arange(n_samples)/fs, algo['error'])
plt.legend(adfilt, fontsize=16)
plt.ylabel("$\||\hat{w}-w\||_2$", fontsize=20)
plt.xlabel("Time [seconds]", fontsize=20)
ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(16) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16) 
plt.grid()
fig = plt.gcf()
fig.set_size_inches(20, 10)

##################################################
# Short Time Fourier Transform (STFT) processing #
##################################################

# filter to apply (moving average / low pass)
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

fs, signal = wavfile.read("arctic_a0010.wav")

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

print("Original:")
IPython.display.Audio(signal, rate=fs)

print("LPF'ed:")
IPython.display.Audio(processed_audio, rate=fs)
"""