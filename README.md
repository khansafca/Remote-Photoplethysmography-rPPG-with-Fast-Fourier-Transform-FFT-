# Remote-Photoplethysmography-rPPG-with-Fast-Fourier-Transform-FFT

This Project is about Remote Photoplethysmography (rPPG) that use Fast Fourier Transform (FFT) to enabling heart rate detection without physical contact.

The project's key functionalities include:

A. Face Detection and Rectangle Formation:
* Utilized Python code to capture camera feed and perform face detection.
* Created a detection rectangle around the identified human face.

B. Gaussian Pyramid Construction and Frame Reconstruction:
* Built a Gaussian pyramid to efficiently process and reconstruct frames.

C. rPPG Signal Extraction:
* Extracted rPPG signals by selecting specific pixels or areas within the rectangle.
* Recorded changes in light within this area as a time signal.

D. Fast Fourier Transform (FFT) Implementation:
* Applied Fast Fourier Transform (FFT) to the rPPG time signal.
* Transformed the time signal into the frequency domain.

E. Heart Rate Calculation:
* Calculated the heart rate by converting the identified frequency into beats per minute (BPM).
