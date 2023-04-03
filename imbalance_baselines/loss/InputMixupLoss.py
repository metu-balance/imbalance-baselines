import numpy as np
import torch


class InputMixupLoss:
    def __init__(self, alpha, model):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.alpha = alpha

    def __call__(self, image, label):
        l = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(image.size(0))
        image_a, image_b = image, image[idx]
        label_a, label_b = label, label[idx]
        mixed_image = l * image_a + (1 - l) * image_b

        output = self.model(mixed_image)
        loss = l * self.criterion(output, label_a) + \
            (1 - l) * self.criterion(output, label_b)

        return loss
