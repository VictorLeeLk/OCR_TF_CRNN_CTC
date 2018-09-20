# CRNN_CTC_Tensorflow
This software implements the Convolutional Recurrent Neural Network (CRNN), a combination of CNN, RNN and CTC loss for image-based sequence recognition tasks, such as scene text recognition and OCR.  

"An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" : https://arxiv.org/abs/1507.05717  

More details for CRNN and CTC loss (chinese): https://zhuanlan.zhihu.com/p/43534801


# Dependencies
All dependencies should be installed are as follow:
* tensorflow >= 1.3
* opencv-python
* numpy

Required packages can be installed with
```bash
pip install -r requirements.txt
```


# Run demo

Asume your current work directory is CRNN_CTC_Tensorflow：
```bash
cd path/to/your/CRNN_CTC_Tensorflow/
```
Export current work directory path into PYTHONPATH and run inference demo:
```bash
export PYTHONPATH=$PYTHONPATH:./
python tools/inference_crnn_ctc.py \
  --image_dir ./test_data/images/ --image_list ./test_data/image_list.txt \
  --model_dir ./
```
Result is:


# Train a new model

### Data Preparation
* Firstly you need to download [Synth90k](http://www.robots.ox.ac.uk/~vgg/data/text/) dataset and extract it into a folder. 
```bash
sh dowload_Synth90k_data.sh
```
* Secondly supply a txt file to specify the relative path to the image data dir and it's corresponding text label.   

For example: image_list.txt
```bash
path/1/2/373_coley_14845.jpg coley
path/17/5/176_Nevadans_51437.jpg nevadans
```
* Then you are supposed to convert your dataset into tensorflow records which can be done by
```bash
python tools/create_crnn_ctc_tfrecord.py \
  --image_dir path/to/image/dir/ --anno_file path/to/image_list.txt --data_dir ./tfrecords/ \
  --validation_split_fraction 0.1
```
All training image will be scaled into (32, 100, 3) and write to tfrecord file.  
The dataset will be divided into train and validation set and you can change the parameter to control the ratio of them.

### Train model
```bash
python tools/train_crnn_ctc.py --data_dir ./tfrecords/ --model_dir ./model/ --batch_size 32
```

# Todo
