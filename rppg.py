import numpy as np
import cv2
import sys

#definisi unutuk membangun gaussian pyramid
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        #menggunakan pyrDown untuk Gaussian pyramid downscaling
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

#definisi untuk reconstruct frame
def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        #menggunakan pyrUp untuk Gaussian pyramid upscaling
        filteredFrame = cv2.pyrUp(filteredFrame)
    
    # Tambahkan Gaussian Blur
    filteredFrame = cv2.GaussianBlur(filteredFrame, (5, 5), 0)
    
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

# Webcam Parameters
webcam = cv2.VideoCapture(0) #aktifkan webcam

# panjang dan lebar asli
realWidth = 360
realHeight = 240

# panjanh dan lebar yang diinginkan
videoWidth = 180
videoHeight = 120

videoChannels = 3
videoFrameRate = 15

# Set the webcam parameters
webcam.set(3, realWidth)  # Set the width
webcam.set(4, realHeight)  # Set the height
#webcam.set(5, videoFrameRate)  # Set the frame rate

# Color Magnification Parameters
levels = 3
alpha = 170

# range frekuensi
minFrequency = 1.0 #disesuaikan
maxFrequency = 1.6 #disesuaikan

bufferSize = 150
bufferIndex = 0

# font yang digunakan
font = cv2.FONT_HERSHEY_DUPLEX #jenis font

# lokasi font
loadingTextLocation = (140, 70) # lokasi font loading
bpmTextLocation = (videoWidth//2 + 127, 70) # lokasi font BPM

# ukuran font
fontScale = 1 

# warna font
fontColor = (255,255,255)

# ketebalan font
lineType = 1

# kotak deteksi
boxColor = (0, 255, 0) # warna kotak
boxWeight = 5 #tebal kotak

# memanggil face detection model dengan cascade
cascade_path = '/Users/khansafca/Documents/projek_ekfis_khansa/haarcascade_frontalface_default.xml'
model = cv2.CascadeClassifier(cascade_path)

# meninisiasi Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels)) 
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter untuk range frekuensi
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 25 
bpmBufferIndex = 0
bpmBufferSize = 20
bpmBuffer = np.zeros((bpmBufferSize))

i = 0

# looping utama
while (True):
    ret, frame = webcam.read()
    if ret == False:
        break
    
    # Detect faces
    faces = model.detectMultiScale(frame)

    # membentuk rectangles mengikuti muka
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    
    # membentuk kotak deteksi
    detectionFrame = frame[videoHeight//2 +59:realHeight-videoHeight//2 +59, videoWidth//2 +59:realWidth-videoWidth//2 +59, :]


    # membangun Gaussian Pyramid
    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
    fourierTransform = np.fft.fft(videoGauss, axis=0)

    # Bandpass Filter sesuai dengan range frekuensi
    # membuang atau mendefinisi 0 untuk yang di luar range frekuensi
    fourierTransform[mask == False] = 0

    # Detak Jantung
    if bufferIndex % bpmCalculationFrequency == 0:
        i = i + 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
        hz = frequencies[np.argmax(fourierTransformAvg)]
        bpm = 60.0 * hz
        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
    
    print(hz)

    # Amplify
    filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    filtered = filtered * alpha

    # Reconstruct Resulting Frame
    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    outputFrame = detectionFrame + filteredFrame
    outputFrame = cv2.convertScaleAbs(outputFrame)

    bufferIndex = (bufferIndex + 1) % bufferSize

    #menampilkan frame ke layar
    frame[videoHeight//2 +59:realHeight-videoHeight//2 +59, videoWidth//2 +149:realWidth-videoWidth//2 +149, :] = outputFrame

    # menampilkan kotak deteksi
    cv2.rectangle(frame, (videoWidth//2 +149, videoHeight//2 +59), (realWidth-videoWidth//2 +149, realHeight-videoHeight//2 +59), boxColor, boxWeight)

    if len(faces) == 0:
        cv2.putText(frame, "tidak ada muka terdeteksi", loadingTextLocation, font, fontScale, fontColor, lineType)

    # menampilkan BPM
    elif i > bpmBufferSize:
        cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
    else:
        cv2.putText(frame, "menghitung detak jantung...", loadingTextLocation, font, fontScale, fontColor, lineType)
    
    # menampilkan frame ke layar
    if len(sys.argv) != 2:
        cv2.imshow("Pedeteksi denyut Jantung dengan rPPG", frame)

    # menutup layar dengan press key esc
    if cv2.waitKey(1) == 27:
        break
    
webcam.release()
cv2.destroyAllWindows()
