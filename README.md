# Face Verification 

This project focuses on building a neural network that identifies whether a person from two different images are the same or not. We forked the LFW_fuel master repo to retrieve the data. It is a very convenient repo which transforms the original image data into a fuel-compatible hdf5 file that is usable by blocks or Keras. For a better description of the dataset and how to run this repo on your environment, please refer to the README.md file inside lfw_fuel-master folder. 
To identify whether an idividual is the same person or not from two different images, we trained a 5-layer Siamese Neural Network which achieves 83% recall. Please refer to siameses_net.py in lfw_fuel-master/example folder to see complete code. To see our result, please refer to project_report.pdf.

