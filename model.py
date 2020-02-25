from torch import nn
from torch.nn import functional as F


class Model(nn.Module):

    def __init__(self, state_shape, n_actions):
        super(Model, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        w, h, c = state_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        flatten_num = (self.conv_shape(self.conv_shape(w, 8, 4), 4, 2) ** 2) * 32

        self.fc = nn.Linear(in_features=flatten_num, out_features=256)
        self.v = nn.Linear(in_features=256, out_features=1)
        self.policy = nn.Linear(in_features=256, out_features=self.n_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        value = self.v(x)
        pi = F.softmax(self.policy(x), dim=1)

        return pi, value

    @staticmethod
    def conv_shape(input, kernel_size, stride, padding=0):
        return (input + 2 * padding - kernel_size) // stride + 1
