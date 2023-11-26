import torch
import torch.nn as nn

class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()
		# Conv2D (in_channel, out_channel, kernel_size, stride)
		self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 4, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
		)

		self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, bias=False),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, (2,2), stride=(1,1)),
            nn.ConvTranspose2d(8, 3, (2,2), stride=(1,1))
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
