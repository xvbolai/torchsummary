## ![torchsummary](https://ndqzpq.dm2304.livefilestore.com/y4mF9ON1vKrSy0ew9dM3Fw6KAvLzQza2nL9JiMSIfgfKLbqJPvuxwOC2VIur_Ycz4TvVpkibMkvKXrX-N9QOkyh0AaUW4qhWDak8cyM0UoLLxc57apyhfDaxflLlZrGqiJgzn1ztsxiaZMzglaIMhoo8kjPuZ5-vY7yoWXqJuhC1BDHOwgNPwIgzpxV1H4k1oQzmewThpAJ_w_fUHzianZtMw?width=35&height=35&cropmode=none) torchsummary

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

torchsummary is a tool to summarize CNN model parameters, such as input shape, weight shape and attributes.

## installation

```shell
git clone https://github.com/xvbolai/brocolli.git
```

## How to use

```python
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
```

## License
Licensed under the [MIT](https://github.com/xvbolai/torchsummary/blob/master/LICENSE) license.