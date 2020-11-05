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

# Codify / normalize labels
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
    return newLabels,transLabels

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
#observed_emotions=['calm', 'happy', 'fearful', 'disgust']
observed_emotions=['happy', 'angry','fearful','neutral']
#observed_emotions=['fearful', 'happy','neutral','angry']
#observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2,split=True):
    if split:
        x,y=[],[]
        i=0
        for file in glob.glob("ravdess-data\\Actor_*\\*.wav"):
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
        for file in glob.glob("my-data\\Actor_*\\*.wav"):
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
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
model=MLPClassifier(alpha=0.001,solver='adam',tol=1e-4, learning_rate_init=0.001,batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=900)

#DataFlair - Train the model
model.fit(x_train,y_train)

#DataFlair - Predict for the test set
y_pred=model.predict(x_test)

#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#DataFlair - Print the accuracy
print("MLP Accuracy: {:.2f}%".format(accuracy*100))

gmm = GaussianMixture(n_components=len(observed_emotions),tol=1e-4, max_iter=700).fit(x_train)
gmmLabels = gmm.predict(x_test)

# GMM labels
newGMMLabels = codifyLabels(gmmLabels)

# MLPClassifier labels
newMLPLabels = codifyLabels(y_pred)

# codify test labels
newTestLabels= codifyLabels(y_test)

# fig, axs = plt.subplots(3)
# # GMM
# axs[0].scatter(range(168),x_test[:, [41]], c=newGMMLabels[0], s=40, cmap='viridis')

# # MLP
# axs[1].scatter(range(168), x_test[:, [41]], c=newMLPLabels[0], s=40, cmap='viridis')

# # Original
# axs[2].scatter(range(168), x_test[:, [41]], c=newTestLabels[0], s=40, cmap='viridis')
# plt.show()  

# Seria mejor con covarianza
accuracy2 = accuracy_score(y_true=newTestLabels[0], y_pred=newGMMLabels[0])
print("GMM Accuracy: {:.2f}%".format(accuracy2*100))

comparison= accuracy_score(y_true=newGMMLabels[0], y_pred=newMLPLabels[0])
print("GMM vs MLP Comparison: {:.2f}%".format(comparison*100))
print("\n")

if input("Test own recordings?")=="Y":
    pass    

myAudio,myLabels = load_data(split=False)
MLPPrediction = model.predict(myAudio)
GMMrediction = gmm.predict(myAudio)
resultsMlpo = []
resultsGMM = []
i = 0
for y_pred2 in MLPPrediction:
    resultsMlpo.append([y_pred2,newMLPLabels[1][y_pred2]])
    i=+1
    #print("MLP Guess: {} {}".format(y_pred2,newMLPLabels[1][y_pred2]))
i = 0
for y_pred3 in GMMrediction:    
    #resultsGMM.append([observed_emotions[newGMMLabels[1][y_pred3]],newGMMLabels[1][y_pred3]])
    resultsGMM.append([newGMMLabels[1][y_pred3]])
    i=+1
    #print("GMM Guess: {}".format(observed_emotions[newGMMLabels[1][y_pred3]]))

for num in range(len(myLabels)):
    print("MLP Guess: {}".format(resultsMlpo[num]))
    print("GMM Guess: {}".format(resultsGMM[num]))
    print("Original Label: {}\n".format(myLabels[num]))
