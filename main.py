from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.cpp_extension import load
print(torch.__version__)

# Load custom module
ops = load(name='weithed_average', sources=['ops.cu'], verbose=True)

# Load example image
img = Image.open('pic.jpg')
img = (np.array(img) / 255).astype('float32')
img = torch.as_tensor(img).cuda()
img = torch.permute(img, (2, 0, 1))


class WeightedAverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input, weights)
        output = ops.forward(input, weights)
        return output

    @staticmethod
    def backward(ctx, gradPrev):
        input, weights = ctx.saved_tensors
        grads = ops.backward(input, weights, gradPrev)

        return None, grads[1]


class Network(torch.nn.Module):
    def __init__(self, kernelWidth):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv6 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv7 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv8 = torch.nn.Conv2d(
            64, kernelWidth * kernelWidth, kernel_size=5, padding=2)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        w = F.softmax(self.conv8(x))

        # Weighed average
        out = WeightedAverage.apply(input.detach(), w)

        return out


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)


# Generate network
net = Network(kernelWidth=5).cuda()
net.apply(weight_init_xavier_uniform)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-5)

# Training
for epoch in range(101):
    optimizer.zero_grad()

    input = img.clone()
    ref = img.clone()

    output = net(input)
    loss = criterion(output, ref)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'[Epoch {epoch+1:04d}] loss: {loss.item():.6f}')

        out = output.detach().cpu().permute(1, 2, 0).numpy()
        out = (out * 255).astype('uint8')
        out = Image.fromarray(out)
        out.save(f'{epoch:04d}.jpg')
