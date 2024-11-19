import os
import librosa, librosa.display
import math
import json

DATASET_PATH="D:\TELECO\GRADO\TFG\Data\genres_original"
JSON_PATH="data.json"

SAMPLE_RATE=22050
DURATION=30
SAMPLES_PER_TRACK=SAMPLE_RATE*DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # num_segments -> cuando se tienen pocos ficheros en el dataset nos permite dividir cada uno de ellos en ese número de segmentos

    # STEP 1: Dictionary to store data

    data = {
        "mapping":[], # mapeos de las señales
        "labels":[], # outputs of the training
        "mfcc":[] # training data
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # STEP 2: loop through all the dataset
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # for that recursively checks all the files on our dataset
        # dirpath -> the path we are currently in
        # dirnames -> all the names of the subfolders on dirpath
        # filenames -> all the files in dirpath

        # ensure we are not at the root level
        if dirpath is not dataset_path:

            # save the semantic label -> save in mappings the type of map for the dataset samples
            dirpath_components = dirpath.split("\\") # si en la ruta estamos en la carpeta genre/blues devuelve: ["genre", "blues"]
            semantic_label = dirpath_components[-1] # nos quedamoscon la última posición del array para guardarlo en mapping -> "blues"
            data["mapping"].append(semantic_label) # lo añadimos a la lista de mapeos
            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:

                # load the audio file
                file_path = os.path.join(dirpath, f) 
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                                    
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], 
                                                sr=sr,
                                                n_mfcc=n_mfcc)
            
                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expeted length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment: {}".format(file_path, s+1))

    #save to json file
    with open(json_path,"w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
