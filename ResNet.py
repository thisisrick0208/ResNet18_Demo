import tensorflow as tf
from BasicBlock import BasicBlock
from tensorflow import keras
from tensorflow.keras import layers,Sequential


class ResNet(keras.Model):
    def __init__(self,layer_dims,num_classes=10):
        super(ResNet, self).__init__()
        # 预处理层
        self.stem=Sequential([
            layers.Conv2D(64,(3,3),strides=(1,1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
        ])
        # resblock
        self.layer1=self.build_resblock(64,layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1],stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # there are [b,512,h,w]
        # 自适应
        self.avgpool=layers.GlobalAveragePooling2D()
        self.fc=layers.Dense(num_classes)



    def call(self,input,training=None):
        x=self.stem(input)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        # [b,c]
        x=self.avgpool(x)
        x=self.fc(x)
        return x

    def build_resblock(self,filter_num,blocks,stride=1):
        res_blocks= Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num,stride))
        # just down sample one time
        for pre in range(1,blocks):
            res_blocks.add(BasicBlock(filter_num,stride=1))
        return res_blocks

def resnet18():
    return  ResNet([2,2,2,2])

