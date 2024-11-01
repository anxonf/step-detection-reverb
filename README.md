# step-detection-reverb
Proposal for an intrusion detection system in reverberant environments

FILES:

python-code/roomSimulation.py

	Configuration of the room, its surfaces, the sound sources, and the microphones
	Calculates reverberation time
	Calculates impulse response of the room
	Computes the result of the signal being affected by the impulse response of the room
	Chooses the most relevant signal / group of signals from the microphones to be processed

python-code/audioProcessing.py

	Receives a signal and processes it obtainin its most relevant acoustic parameters
	Time domain parameters: waveform, envelope, RMSE, ZCR
	Frequency domain parameters: spectrum, spectrogram, MEL, MFCC, BER, Spectral Centroid, Spectral Bandwith
	#PENDING: Implement a funtion that obtains the correlation between the signals of two different microphones to study the location of the source on the room
	Sends all this information to the ML algorithm that will define if the signal is a human step and where it came from

python-code/mlTraining.py

	Receives the processed information of the signal / signals at study
	Determines whether it is a step or not
	If it is a step, it locates the sector where the source was found

