# TSP-Graph-ConvNet
This is repository is created for the course project of CS 6045-Advanced Algorithm


# Set Up Environment

* Create Virtual Environment  
```
virtualenv venv --python=3.7
```
* Activate Virtual Environment  
```
venv\Scripts\activate
```
* Install Pytorch
```
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
* Install Other Dependencies
```
pip install -r requirements.txt
```

# Download the Dataset

Donwnload [data.zip](https://kennesawedu-my.sharepoint.com/:u:/g/personal/jli36_students_kennesaw_edu/EWPJRJxyKe1Avyuz5ZzD8loBO_1eKnOll7Q8Z7w-9HWcWQ?e=zDpEf4), unzip it under the folder TSP-GRAPH-CONVNET/data/.

# Try Demo Code
```
python demo.py -m gd
```

# Try evaluation Code
```
python evaluation.py -s 20 -b 100 -m gd
```
