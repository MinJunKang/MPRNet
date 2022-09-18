import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from pathlib import Path
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict


class Run_task(object):
    
    def __init__(self, rootpath, task):
        
        load_file = run_path(str(Path(rootpath) / task / "MPRNet.py"))
        self.model = load_file['MPRNet']()
        self.model.cuda()
        self.img_multiple_of = 8
        
        weights = str(Path(rootpath) / task / "pretrained_models" / ("model_"+task.lower()+".pth"))
        self.load_checkpoint(weights)
        self.model.eval()
        
    def load_checkpoint(self, weights):
        checkpoint = torch.load(weights)
        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            
    def forward(self, img_):
        
        img = Image.fromarray(img_).convert('RGB')
        input_ = TF.to_tensor(img).unsqueeze(0).cuda()
        
        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+self.img_multiple_of)//self.img_multiple_of)*self.img_multiple_of, ((w+self.img_multiple_of)//self.img_multiple_of)*self.img_multiple_of
        padh = H-h if h%self.img_multiple_of!=0 else 0
        padw = W-w if w%self.img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        with torch.no_grad():
            restored = self.model(input_)
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        
        return restored
        