# CRNN_CTC_Tensorflow
This software implements the Convolutional Recurrent Neural Network (CRNN), a combination of CNN, RNN and CTC loss for image-based sequence recognition tasks, such as scene text recognition and OCR. For details, please refer to our paper 
https://arxiv.org/abs/1507.05717


# Dependencies
All dependencies should be installed as follow:
* tensorflow >= 1.3
* opencv-python
* numpy

Required packages may be installed with
```bash
pip install -r requirements.txt
```


# Run demo

# Train a new model

Data Preparation
* Firstly you need to store all your image data in a folder. 
* Then supply a txt file to specify the relative path to the image data dir and it's corresponding text label.   

For example:
```bash
path/1/2/373_coley_14845.jpg coley
path/17/5/176_Nevadans_51437.jpg nevadans
```
* Thirdly you are supposed to convert your dataset into tensorflow records which can be done by
```bash
python tools/
```
