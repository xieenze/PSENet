import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 256):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)

		self.bn1 = nn.BatchNorm2d(ndf)
		self.bn2 = nn.BatchNorm2d(ndf)
		self.bn3 = nn.BatchNorm2d(ndf)
		self.bn4 = nn.BatchNorm2d(ndf)


		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=3, stride=2, padding=1)
		self.avgpool = nn.AvgPool2d(10)
		self.classifier = nn.Linear(ndf, 1)

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


#test
if __name__ == "__main__":
	from IPython import embed
	imgs = torch.randn(2,256,160,160)
	model_1 = FCDiscriminator(256)
	out1 = model_1(imgs)

	bce_loss = torch.nn.BCEWithLogitsLoss()
	pred = out1
	gt = torch.FloatTensor(pred.size()).fill_(1)

