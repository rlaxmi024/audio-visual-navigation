import torch
import numpy as np


class AudioCNN(torch.nn.Module):
    def __init__(self, spectrogram_shape):
        super().__init__()
        # Define the module parameters
        self._n_input_audio = spectrogram_shape[2]
        cnn_dims = np.array(spectrogram_shape[: 2], dtype=np.float32)

        # Choose the appropriate cnn kernel sizes and strides
        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        # Compute the output dimensions after convolution layers
        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        # Define the CNN model
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            torch.nn.Flatten()
        )

        # Define the classification and embedding layers
        self.contrastive_layer = torch.nn.Linear(32 * cnn_dims[0] * cnn_dims[1], 128)
        self.angle_layer = torch.nn.Linear(2 * 32 * cnn_dims[0] * cnn_dims[1], 4)
        self.eu_dist_layer = torch.nn.Linear(2 * 32 * cnn_dims[0] * cnn_dims[1], 1)

        # Initialize the weights of the model
        self._initialize_weights()

    def _conv_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        # Compute the output dimensions after passing through the convolution layers
        out_dimension = []

        for i in range(len(dimension)):
            out_dimension.append(int(np.floor(((
                dimension[i]
                + 2 * padding[i]
                - dilation[i] * (kernel_size[i] - 1)
                - 1
            ) / stride[i]) + 1 )))

        return tuple(out_dimension)

    def forward(self, x1, x2):
        outputs = {}

        # Concatenate the spectrograms for a single forward pass
        x = torch.cat([x1, x2])

        # Permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        x = x.permute(0, 3, 1, 2)

        # Perform forward pass
        x = self.cnn(x)

        # Generate embeddings for contrastive loss
        x1, x2 = torch.chunk(self.contrastive_layer(x), 2)

        # Concatenate the embeddings for corresponding spectograms
        x = torch.stack(torch.chunk(torch.cat(torch.chunk(x, 2), axis=1).flatten(), x.shape[0] // 2))

        # Return the contrastive embeddings and labels for the angle and euclidean distance
        outputs['angle_label'] = self.angle_layer(x)
        outputs['eu_dist_label'] = self.eu_dist_layer(x)
        outputs['contrastive_embedding'] = x1, x2

        return outputs

    def _initialize_weights(self):
        # Initialize the weights of the network
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def save_backbone(self, model_path):
        torch.save(self.cnn.state_dict(), model_path)