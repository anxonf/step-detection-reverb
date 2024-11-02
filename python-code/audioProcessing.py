import sys
import os
import numpy as np
import scipy as sp
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import noisereduce as nr
import pylab as pl
import threading
import pyroomacoustics as pra
import IPython
import IPython.display as ipd
from scipy.signal import fftconvolve, hilbert
from scipy.io import wavfile
import math
import wave
import librosa, librosa.display

# Constant definition for the framing and overlapping operations, AE, RMSE

FRAME_SIZE = 1024
HOP_LENGTH = 512

# Path to the test audio files
SOUNDS_PATH = "../resources/sounds"

# BASIC AND COMMON FUNCTIONS

def play_audio(audioSignal, sampleRate):
    sd.play(audioSignal, sampleRate)
    sd.wait()

#########################################################################
#########################################################################
########                                                         ########
######## PROCESADO DE LA SEÑAL DE AUDIO EN EL DOMINIO DEL TIEMPO ########
########                                                         ########
#########################################################################
#########################################################################

def plot_timeDomain(audioSignal):
    plt.figure(figsize=(10,8))
    librosa.display.waveshow(audioSignal, alpha=0.5)
    plt.title("Señal: Pasos.wav")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim((-1,1))
    plt.show()

def amplitude_envelope(audioSignal, frame_size, hop_length):
    return np.array([max(audioSignal[i:i+frame_size]) for i in range(0, audioSignal.size, hop_length)])

def plot_amplitudeEnvelope(audioSignal, envelopeSignal):

    frames = range(0, envelopeSignal.size)
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

    plt.figure(figsize=(10,8))
    librosa.display.waveshow(audioSignal, alpha=0.5)
    plt.plot(t, envelopeSignal, color="r")
    plt.title("Señal: Pasos.wav + Envelope")
    plt.ylim((-1,1))
    plt.show()

def compute_RMSE(audioSignal, frame_size, hop_length):
    compute_RMSE = []

    for i in range(0, len(audioSignal), hop_length): 
        rmse_current_frame = np.sqrt(sum(audioSignal[i:i+frame_size]**2) / frame_size)
        compute_RMSE.append(rmse_current_frame)
    return np.array(compute_RMSE)

def plot_RMSE(audioSignal, RMSESignal):
    frames = range(0, RMSESignal.size)
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

    plt.figure(figsize=(10,8))
    librosa.display.waveshow(audioSignal, alpha=0.5)
    plt.plot(t, RMSESignal, color="r")
    plt.title("Señal: Pasos.wav + RMSE")
    plt.ylim((-1,1))
    plt.show()

def plot_ZCR(ZCRSignal):
    frames = range(0, ZCRSignal.size)
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

    plt.figure(figsize=(10,8))
    #NORMALIZADO ENTRE 0 y 1, para el valor real, hay que multiplicar el zcr_signal por FRAME_SIZE y jugar con los márgenes del eje y
    plt.plot(t, ZCRSignal, color="r")
    plt.title("ZCR de la señal Pasos.wav")
    plt.ylim((0,1))
    plt.show()

#########################################################################
#########################################################################
#####                                                               #####
##### PROCESADO DE LA SEÑAL DE AUDIO EN EL DOMINIO DE LA FRECUENCIA #####
#####                                                               #####
#########################################################################
#########################################################################

def plot_spectrum(audioSignal, sampleRate, fRatio):

    # Cálculo de la Transformada de Fourier
    FTSignal = sp.fft.fft(audioSignal)
    magSpectrum = np.absolute(FTSignal)
    freqSpectrum = np.linspace(0, sampleRate, len(magSpectrum))
    numFreqBins = int(len(freqSpectrum) * fRatio) #Seleccionamos el número relevante de frecuencias para evitar no mostrar la redundancia de la DFT variando fRatio

    #print("Valor 0 del array de la señal: " + str(ft_signal[0]))
    #print("1st coef of MagSpectrum " + str(magSpectrum[0]))
    #print("1st coef of FreqSpectrum " + str(freqSpectrum[0]))
    #print("Longitud Magnitud: " + str(len(magnitude)))
    #print("Longitud Frecuencia: " + str(len(frequency)))
   
    plt.figure(figsize=(10,8))
    plt.plot(freqSpectrum[:numFreqBins], magSpectrum[:numFreqBins])
    plt.title("Espectro de la señal Pasos.wav")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.show()

def plot_spectrogram(audioSignal, sampleRate, hop_length):

    STFTSignal = librosa.stft(audioSignal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    spectrogramSignal = np.abs(STFTSignal) ** 2

    spectrogramP2dB_Log = librosa.power_to_db(spectrogramSignal)

    plt.figure(figsize=(10,8))  
    plt.title("Espectrograma Pasos.wav")

    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogramP2dB_Log, sr=sampleRate, hop_length=hop_length, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.f dB")

    spectrogramA2Db_Log = librosa.amplitude_to_db(np.abs(librosa.stft(audioSignal)))
    #spectrogramA2Db_Log = librosa.amplitude_to_db(librosa.stft(audioSignal))
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(spectrogramA2Db_Log, sr=sampleRate, hop_length=hop_length, x_axis= "time", y_axis="log")
    plt.colorbar(format="%+2.f dB")
    plt.show()

def plot_MEL_spectrogram(audioSignal, sampleRate):

    #Creación y presentación de los filtros MEL
    filterBanks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10) #n_fft = frame size -->> REVISAR PARAMETROS DE LOS FILTROS
    
    plt.figure(figsize=(10,8))
    plt.title("Banco de Filtros MEL")
    librosa.display.specshow(filterBanks, sr=sampleRate, x_axis="linear")
    plt.colorbar(format="%+2.f")
    plt.show()
    
    #Creación y presentación del Espectrograma MEL
    melSpectrogram = librosa.feature.melspectrogram(y=audioSignal, sr=sampleRate, n_fft=2048, hop_length=512, n_mels=10)
    melSpectrogram_Log = librosa.power_to_db(melSpectrogram)

    plt.figure(figsize=(10,8))
    plt.title("Espectrograma MEL de la Señal Pasos.wav")
    librosa.display.specshow(melSpectrogram_Log, x_axis="time", y_axis="mel",  sr=sampleRate)
    plt.colorbar(format="%+2.f")

    plt.show()

def plot_MFCC_spectrogram(audioSignal, sampleRate, title):

    plt.title(title)
    librosa.display.specshow(MFCCsSignal, x_axis= "time", sr=sampleRate)
    plt.colorbar()

def calculate_split_frequency_bin(splitFrequency, sampleRate, numFrequencyBins):
    
    frequencyRange = sampleRate / 2
    frequencyDeltaBin = frequencyRange / numFrequencyBins
    splitFrequencyBin = math.floor(splitFrequency / frequencyDeltaBin)
    return int(splitFrequencyBin)

def calculate_BER(audioSignal, splitFrequency, sampleRate):
    
    STFTSignal = librosa.stft(audioSignal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    
    splitFrequencyBin = calculate_split_frequency_bin(splitFrequency, sampleRate, len(STFTSignal[0]))
    calculate_BER = []
    
    # calculate power spectrogram
    powerSpectrogram = np.abs(STFTSignal) ** 2
    powerSpectrogram = powerSpectrogram.T
    
    # calculate BER value for each frame
    for frame in powerSpectrogram:
        sumPowerLowFrequencies = frame[:splitFrequencyBin].sum()
        sumPowerHighFrequencies = frame[splitFrequencyBin:].sum()
        BER_currentFrame = sumPowerLowFrequencies / sumPowerHighFrequencies
        calculate_BER.append(BER_currentFrame)
    
    return np.array(calculate_BER)

def plot_BER(BERSignal):

    frames = range(len(BERSignal))
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

    plt.figure(figsize=(10,8))
    plt.title("BER de la Señal Pasos.wav")
    plt.plot(t, BERSignal, color='b')
    plt.show()

def plot_spectral_centroid(audioSignal, sampleRate):

    spectralCentroid = librosa.feature.spectral_centroid(y=audioSignal, sr=sampleRate, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

    frames = range(len(spectralCentroid))
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
    
    plt.title("Spectral Centroid of the signal Pasos.wav")
    plt.plot(t, spectralCentroid, color='blue')
    #plt.show()

def plot_spectral_bandwidth(audioSignal, sampleRate):

    spectralBandwidth = librosa.feature.spectral_bandwidth(y=audioSignal, sr=sampleRate, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    
    frames = range(len(spectralBandwidth))
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

    plt.title("Spectral Bandwidth of the signal Pasos.wav")
    plt.plot(t, spectralBandwidth, color='blue')
    #plt.show()

##############################
##############################
#####                    #####
##### PROGRAMA PRINCIPAL #####
#####                    #####
##############################
##############################

fileName = "Pasos.wav"

sampleRate, audioSignal = wavfile.read(os.path.join(SOUNDS_PATH, fileName))
duration = len(audioSignal) / sampleRate
time = np.linspace(0, duration, len(audioSignal))

print("Duración de la señal: " + str(duration) + " segundos")
print("Tamaño: " + str(audioSignal.size) + " muestras")

#Duración de 1 Muestra
sampleDuration = 1/sampleRate
print("Duración de 1 muestra: " + str(sampleDuration) + " segundos")

plot_timeDomain(audioSignal)

envelopeSignal = amplitude_envelope(audioSignal, FRAME_SIZE, HOP_LENGTH)
print("Duración envolvente: " + str(len(envelopeSignal)))
plot_amplitudeEnvelope(audioSignal, envelopeSignal)

RMSESignal = compute_RMSE(audioSignal, FRAME_SIZE, HOP_LENGTH)
plot_RMSE(audioSignal, RMSESignal)

ZCRSignal = librosa.feature.zero_crossing_rate(audioSignal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
plot_ZCR(ZCRSignal)

fRatio=0.1
plot_spectrum(audioSignal, sampleRate, fRatio)

plot_spectrogram(audioSignal, sampleRate, HOP_LENGTH)

plot_MEL_spectrogram(audioSignal, sampleRate)

# Extraemos los Mel Frequency Cepstrum Coeficients
MFCCsSignal = librosa.feature.mfcc(y=audioSignal, n_mfcc=13, sr=sampleRate)
# Delta MFCCs
deltaMFCCsSignal = librosa.feature.delta(MFCCsSignal)
# Delta2 MFCCs
delta2MFCCsSignal = librosa.feature.delta(MFCCsSignal, order=2)

plt.figure(figsize=(10,8))
plt.subplot(3, 1, 1)
plot_MFCC_spectrogram(MFCCsSignal, sampleRate, "MFCCs Señal Pasos.wav")
plt.subplot(3, 1, 2)
plot_MFCC_spectrogram(deltaMFCCsSignal, sampleRate, "Delta MFCCs Señal Pasos.wav")
plt.subplot(3, 1, 3)
plot_MFCC_spectrogram(delta2MFCCsSignal, sampleRate, "Delta2 MFCCs Señal Pasos.wav")
plt.show()    

splitFrequency=2500
BERSignal = calculate_BER(audioSignal, splitFrequency, sampleRate)
plot_BER(BERSignal)

plt.figure(figsize=(10,8))
plt.subplot(2, 1, 1)
plot_spectral_centroid(audioSignal, sampleRate)
plt.subplot(2, 1 ,2)
plot_spectral_bandwidth(audioSignal, sampleRate)
plt.show()

audio_thread = threading.Thread(target=play_audio, args=(audioSignal, sampleRate))
audio_thread.start()