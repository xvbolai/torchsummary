import torchsummary
import torch
from torchsummary.summarizer import SMA

from model.Lenet import Model as Lenet

model = Lenet()
model.eval()
shape=(1, 3, 32, 32)

def calibrate_func(model):
    input1 = torch.rand([1,3,32,32])
    output = model(input1)


sm = SMA(model, shape)
sm.init()
sm.prepare()
sm.calibrate(calibrate_func)
sm.concise("Lenet")
