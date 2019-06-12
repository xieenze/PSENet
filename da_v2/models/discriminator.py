import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)

		self.bn1 = nn.BatchNorm2d(ndf)
		self.bn2 = nn.BatchNorm2d(ndf*2)
		self.bn3 = nn.BatchNorm2d(ndf*4)
		self.bn4 = nn.BatchNorm2d(ndf*8)


		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=3, stride=2, padding=1)
		self.avgpool = nn.AvgPool2d(40)
		self.classifier = nn.Linear(ndf*8, 1)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.leaky_relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.leaky_relu(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.leaky_relu(x)

		x = self.conv4(x)
		x = self.bn4(x)
		x = self.leaky_relu(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
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
	imgs = torch.randn(1,7,640,640)
	model_1 = FCDiscriminator(7)
	# model_2 = FCDiscriminator2(7)
	out1 = model_1(imgs)
	# out2 = model_2(imgs)

	bce_loss = torch.nn.BCEWithLogitsLoss()
	pred = out1
	gt = torch.FloatTensor(pred.size()).fill_(1)

	embed()
