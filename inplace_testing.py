
import torch
from torch.autograd import Variable
from torch.nn import Linear
from allennlp.nn.util import get_dropout_mask


layer = Linear(5, 10)


loss = torch.nn.CrossEntropyLoss()
sgd = torch.optim.SGD(layer.parameters(), 0.01)

for i in range(100):
    input = Variable(torch.randn(3, 5), requires_grad=True)
    target = Variable(torch.LongTensor(3).random_(5))
    sgd.zero_grad()
    outputs = layer(input)
    #print(outputs)
    output = loss(input, target)
    print(output)
    output.backward()
    print(layer.weight.grad)
    sgd.step()


