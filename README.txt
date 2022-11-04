[*]Enviorment

1.Python
2.Tensorflow
3.numpy
4.opencv-python




For training
#change training data path in main.py
train_low_data_names = glob(r'D:\Low-light_data\RGB\Low_rgb\*.png') #training data path
train_high_data_names = glob(r'D:\Low-light_data\RGB\Normal_rgb\*.png') #ground truth

For testing
inset images in "data" folder to get results in "Enhanced_result" folder

