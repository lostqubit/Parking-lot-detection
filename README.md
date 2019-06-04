# Introduction
A Real-time car parking system model using Deep learning applied on CCTV camera images

### Dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.3 and later, also install other various technology stacks as mentioned in the pdf 

3. All the scripts were written in and tested in a linux based system so it is recommended to run the scripts in a linux based system since unix style paths wont support windows style paths



### Usage of scripts

1. **Download and unpack the dataset in data folder**:
   ```
   http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
   ```

2. **Run the preprocessing.py to produce train and test splits of data**:
   ```
   python preprocessing.py
   ```

3. **Train the model**:
   ```
   python train.py
   ```


