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
import math

# Function to play the audio signal
def play_audio(audio_signal, sample_rate):
    sd.play(audio_signal, sample_rate)
    sd.wait()

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

# Function to generate Spectrogram of a given signal
def plot_spectrogram(audio_signal, sample_rate):
  
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

    #processed_audio = np.zeros(audio_signal.shape)
    n = 0
    while  audio_signal.shape[0] - n > hop:
        stft.analysis(audio_signal[n:n+hop,])
        stft.process()  # apply the filter
        #processed_audio[n:n+hop,] = stft.synthesis()
        n += hop

    # plot the spectrogram before and after filtering
    fig = plt.figure()
    #plt.subplot(2,1,1)
    plt.specgram(audio_signal[:n-hop].astype(np.float32), NFFT=256, Fs=sample_rate)
    plt.title('Original Signal', fontsize=22)
    #plt.subplot(2,1,2)
    #plt.specgram(processed_audio[hop:n], NFFT=256, Fs=sample_rate)
    #plt.title('Lowpass Filtered Signal', fontsize=22)
    #plt.tight_layout(pad=0.5)
    plt.show()

    #audio_thread = threading.Thread(target=play_audio, args=(processed_audio, sample_rate))
    #audio_thread.start()



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

print("Cargando fichero de audio")
# Load the audio file
try:
    sample_rate, audio_signal = wavfile.read(audio_file)
    print("Tasa muestreo fuente: " + str(sample_rate))

    # specify signal and noise source
    #noise_fs, noise_signal = wavfile.read("D:/TELECO/GRADO/TFG/step-detection-reverb/resources/sounds/noise/silencio.wav")  # may spit out a warning when reading but it's alright!

except Exception as e:
    print("Error loading the audio file:", str(e))
    sys.exit(1)


Lg_t = 0.100                # filter size in seconds
Lg = np.ceil(Lg_t*sample_rate)       # in samples
fft_len = 512

# Creating the room to run the simulation
print("CREATING ROOM")

room = pra.ShoeBox([5,30,4], fs=sample_rate, ray_tracing=True, air_absorption=False, materials=pra.Material(0.01, 0.1))

# Set the ray tracing parameters
room.set_ray_tracing(receiver_radius=0.5, n_rays=100000)

# add source
room.add_source([2.5,0.5,1.0], delay=0., signal=audio_signal) #room.add_source([2.5,0.5,1.0], delay=5., signal=audio_signal)
#room.add_source([2.5,15,2], delay=0., signal=noise_signal[:len(audio_signal)])


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

mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)
room.add_microphone_array(mics)

# Compute DAS weights
mics.rake_delay_and_sum_weights(room.sources[0][:1])

room.image_source_model()

fig, ax = room.plot(freq=[500, 1000, 2000, 4000], img_order=3)
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 31])
ax.set_zlim([-1, 5])
plt.axis('scaled')

fig = plt.gcf()

plt.show()

#############################################################
############## RESPUESTA AL IMPULSO DE LA SALA
#############################################################
room.plot_rir()

fig = plt.gcf()

plt.show()

#############################################################
############## TIEMPO DE REVERBERACIÓN T60
#############################################################
t60 = pra.experimental.measure_rt60(room.rir[0][0], fs=room.fs, plot=True)
print(f"The RT60 is {t60 * 1000:.0f} ms")

fig = plt.gcf()

plt.show()

#############################################################
############## SIMULACION CONVOLUCION ReF y SEÑAL
#############################################################
room.simulate()
print(room.mic_array.signals.shape)

#############################################################
############## ESCUCHAMOS SEÑALES
#############################################################

chooseSignal = 0

print("¿Desea escuchar alguna de las señales captadas por los micros?(y/n)")
keepGoing = input()

while keepGoing != 'n':

    print("Introduzca el número del micro que quiera escuchar (0 para señal original y 9 para la señal de ruido):")
    chooseSignal = int(input())

    match chooseSignal:
        case 0:
            print("Original Signal")
            audio_thread = threading.Thread(target=play_audio, args=(audio_signal, sample_rate))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            #plot_audio_analysis(audio_signal, sample_rate)
            #plot_spectrogram(audio_signal, sample_rate)
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input()
        case 1:
            print("Simulated propagation to first mic:")
            audio_thread = threading.Thread(target=play_audio, args=(room.mic_array.signals[0,:], sample_rate))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            #plot_audio_analysis(room.mic_array.signals[0,:], sample_rate)
            #plot_spectrogram(room.mic_array.signals[0,:], sample_rate)
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input()
        case 2:
            print("Simulated propagation to second mic:")
            audio_thread = threading.Thread(target=play_audio, args=(room.mic_array.signals[1,:], sample_rate))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            #plot_audio_analysis(room.mic_array.signals[1,:], sample_rate) 
            #plot_spectrogram(room.mic_array.signals[1,:], sample_rate)           
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input()
        case 3:
            print("Simulated propagation to third mic:")
            audio_thread = threading.Thread(target=play_audio, args=(room.mic_array.signals[2,:], sample_rate))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            #plot_audio_analysis(room.mic_array.signals[2,:], sample_rate)
            #plot_spectrogram(room.mic_array.signals[2,:], sample_rate)
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input()            
        case 4:
            print("Simulated propagation to fourth mic:")
            audio_thread = threading.Thread(target=play_audio, args=(room.mic_array.signals[3,:], sample_rate))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            #plot_audio_analysis(room.mic_array.signals[3,:], sample_rate)   
            #plot_spectrogram(room.mic_array.signals[3,:], sample_rate)         
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input()
        case 5:
            print("Simulated propagation to fifth mic:")
            audio_thread = threading.Thread(target=play_audio, args=(room.mic_array.signals[4,:], sample_rate))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            #plot_audio_analysis(room.mic_array.signals[4,:], sample_rate)    
            #plot_spectrogram(room.mic_array.signals[4,:], sample_rate)        
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input()
        case 6:
            print("Simulated propagation to sixth mic:")
            audio_thread = threading.Thread(target=play_audio, args=(room.mic_array.signals[5,:], sample_rate))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            #plot_audio_analysis(room.mic_array.signals[5,:], sample_rate)      
            #plot_spectrogram(room.mic_array.signals[5,:], sample_rate)      
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input()
        case 7:
            print("Simulated propagation to seventh mic:")
            audio_thread = threading.Thread(target=play_audio, args=(room.mic_array.signals[6,:], sample_rate))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            #plot_audio_analysis(room.mic_array.signals[6,:], sample_rate)    
            #plot_spectrogram(room.mic_array.signals[6,:], sample_rate)        
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input()
        case 8:
            print("Simulated propagation to eigth mic:")
            audio_thread = threading.Thread(target=play_audio, args=(room.mic_array.signals[7,:], sample_rate))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            #plot_audio_analysis(room.mic_array.signals[7,:], sample_rate)    
            #plot_spectrogram(room.mic_array.signals[7,:], sample_rate)        
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input()

"""         case 9:
            print("Original Noise Signal")
            audio_thread = threading.Thread(target=play_audio, args=(noise_signal, noise_fs))
            audio_thread.start()
            # Plot the spectrum and time domain representation of the audio signal
            plot_audio_analysis(noise_signal, sample_rate)            
            plot_spectrogram(noise_signal, sample_rate)
            print("¿Desea continuar escuchando señales?(y/n)")
            keepGoing = input() """


print("------------- FIN DE LA EJECUCIÓN -------------")

