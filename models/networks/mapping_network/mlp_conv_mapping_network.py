from torch import nn, randn

from models.networks.traceable_network import TraceableNetwork


class View(nn.Module):
    """
    Workaround to trace nn.Unflatten. With the use of nn.Unflatten this Error is raised:
    RuntimeError: NYI: Named tensors are not supported with the tracer.

    From https://github.com/pytorch/pytorch/issues/49538#issuecomment-989190281
    """

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class MLPConvMappingNetwork(TraceableNetwork):

    def __init__(self, num_landmark_vars, nmf=64, num_upsamples=5, norm_layer=nn.BatchNorm2d):
        super(MLPConvMappingNetwork, self).__init__()
        use_bias = True
        self.num_landmark_vars = num_landmark_vars

        num_filters = 4
        self.net = [
            nn.Flatten(),
            nn.Linear(num_landmark_vars, num_filters * 8 * 8),
            nn.ReLU(),
            nn.Linear(num_filters * 8 * 8, num_filters * 16 * 16),
            nn.ReLU(),
            # nn.Unflatten(1, (num_filters, 16, 16)),
            View(1, num_filters, 16, 16),  # Workaround to trace nn.Unflatten
        ]

        # upsample
        for i in range(num_upsamples):
            num_filters_up = num_filters * 2 if num_filters * 2 <= nmf else nmf
            self.net += [
                nn.ConvTranspose2d(num_filters, num_filters_up, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(num_filters_up),
                nn.ReLU(True),
            ]
            num_filters = num_filters_up

        while num_filters < nmf:
            num_filters_up = num_filters * 2 if num_filters * 2 <= nmf else nmf
            self.net += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(num_filters, num_filters_up, kernel_size=3, stride=1, padding=0, bias=use_bias),
                norm_layer(num_filters_up),
                nn.ReLU(True),
            ]
            num_filters = num_filters_up

        self.net += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.net = nn.Sequential(*self.net)

    def input_noise(self, metadata):
        return randn((1, self.num_landmark_vars)).to(metadata["device"])

    def forward(self, landmarks):
        # TODO check if crashes with batch size > 1
        return self.net(landmarks)
