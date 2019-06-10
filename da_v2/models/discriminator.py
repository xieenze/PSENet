import torch.nn as nn
import torch.nn.functional as F
import torch

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=3, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x

def FCDiscriminator2(num_classes,ndf = 64):
	conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=2, padding=1)
	conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1)
	conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1)
	conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=2, padding=1)
	classifier = nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=2, padding=1)
	leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	return nn.Sequential(conv1,leaky_relu,conv2,leaky_relu,conv3,leaky_relu,conv4,leaky_relu,classifier)
#test
if __name__ == "__main__":
	from IPython import embed
	imgs = torch.randn(1,7,160,160)
	model_1 = FCDiscriminator(7)
	model_2 = FCDiscriminator2(7)
	out1 = model_1(imgs)
	out2 = model_2(imgs)
	embed()
