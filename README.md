# MLsimpleexamples
This is a repository with simple examples of Machine Learning.
There are three of the examples currently.
Digitrecognition.py file learns to recognize digits based on the MNIST dataset. There are also 10 BMP files, one per digit, in order to show the program recognizing the digits. 
It is not infallible, as for example 9 is sometimes recognized as 5. An interesting phenomenon takes place, because the program's guess probabilities differ each time it's run.
This is due to the training dataset being shuffled each time.
FaceBodyrecognition.py uses OpenCV in order to find faces and bodies on images and videos. It has four possible combinations, however zad3 runs for a very long time. 
Roadlinesrecognition.py uses OpenCV and Moviepy in order to find any lines on a road seen in a road video. It finds the left and right end of the road and the lines 
between the lanes of the road. It can also detect boundaries of shades.
