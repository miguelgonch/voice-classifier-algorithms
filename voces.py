import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    i = 0 
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            # Short-time fourier transform
            # frequency information that is averaged over the entire time domain
            # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C2/C2_STFT-Basic.html
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        # Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        # chroma: Pertains to the 12 different pitch classes
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        # mel: Mel Spectrogram Frequency
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2,split=True):
    if split:
        x,y=[],[]
        i=0
        for file in glob.glob("Parciales\\Proyecto Final\\ravdess-data\\Actor_*\\*.wav"):
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            i = i + 1
        print("Files: ",i)
        return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
    else:
        x,y=[],[]
        i=0
        for file in glob.glob("Parciales\\Proyecto Final\\my-data\\Actor_*\\*.wav"):
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
            i = i + 1
        print("New files: ",i)
        return np.array(x), y

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25,split=True)

#DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#DataFlair - Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01,solver='adam', learning_rate_init=0.001,batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#DataFlair - Train the model
model.fit(x_train,y_train)

#DataFlair - Predict for the test set
y_pred=model.predict(x_test)

#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))

gmm = GaussianMixture(n_components=8,tol=1e-4).fit(x_train)
#labels = gmm.predict(x_train)
#labels2 = gmm.fit_predict(x_train,y_train)
gmmLabels = gmm.predict(x_test)

def codifyLabels(labels):
    codiLabels = []
    transLabels = {}
    newLabels = []
    for label in labels:
        if label not in codiLabels:
            codiLabels.append(label)
            transLabels[label]=len(codiLabels)-1
    for label in labels:
        newLabels.append(transLabels[label])
    return newLabels

# GMM labels
newGMMLabels = codifyLabels(gmmLabels)

# MLPClassifier labels
newMLPLabels = codifyLabels(y_pred)

# codify test labels
newTestLabels= codifyLabels(y_test)

accuracy2 =accuracy_score(y_true=newTestLabels, y_pred=newGMMLabels)
print("Accuracy: {:.2f}%".format(accuracy2*100))

comparison=accuracy_score(y_true=newGMMLabels, y_pred=newMLPLabels)
print("Comparison: {:.2f}%".format(comparison*100))