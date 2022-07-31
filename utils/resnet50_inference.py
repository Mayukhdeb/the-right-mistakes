import torch
import os
from PIL import Image
from torchvision import transforms
import torchvision.models as models

class Resnet50Inference:
    def __init__(self,device = 'cpu', imagenet_classes_filename = 'imagenet_classes.txt'):
        
        assert os.path.exists(imagenet_classes_filename), f'Expected imagenet_classes_filename to exist: {imagenet_classes_filename}\nMaybe run: $wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        # Read the categories
        with open(imagenet_classes_filename, "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
            
        self.model = models.resnet50(pretrained = True).to(device)
        self.model.eval()
        
        self.transforms = preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = device
        
    def run(self, filename: str, num_top_categories = 1):
        input_image = Image.open(filename)
        input_tensor = self.transforms(input_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device) # create a mini-batch as expected by the model


        with torch.no_grad():
            output = self.model(input_batch)
            
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        
        # Show top categories per image
        top_prob, top_catid = torch.topk(probabilities, num_top_categories)
        
        return {
            'categories': [self.categories[top_catid[i]] for i in range(len(top_catid))],
            'probabilities': [top_prob[i].item() for i in range(len(top_prob))]
            
        }