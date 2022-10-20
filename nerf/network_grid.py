import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize
from pdb import set_trace
import clip
import GPUtil
import gc
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, nerf_conditioning = False, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.nerf_conditioning = nerf_conditioning
        net = []


        for l in range(num_layers):
            if self.nerf_conditioning:
                net.append(nn.Linear(self.dim_in + 512  if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))
            else:
                net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)

        
    def forward(self, x, conditioning_vector = None):

        for l in range(self.num_layers):
            if l == 0:
                if self.nerf_conditioning :
                    x = torch.cat((x,conditioning_vector['input_vec'].repeat(x.shape[0],1)), dim=1)
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.conditioning_model = opt.conditioning_model
        if opt.conditioning_model == 'CLIP':
            self.clip_model, clip_preprocess = clip.load("ViT-B/16", device='cuda', jit=False)
            for parameters in self.clip_model.parameters():
                parameters.requires_grad = False
            self.nerf_conditioning = True
        elif opt.conditioning_model == 'T5':
            from transformers import AutoTokenizer, AutoModelWithLMHead
            self.text_model_tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.text_model = AutoModelWithLMHead.from_pretrained("t5-small").cuda()
            
            for parameters in self.text_model.parameters():
                parameters.requires_grad = False
            self.nerf_conditioning = True
 
        elif opt.conditioning_model is None:
            self.nerf_conditioning = False 

        if self.nerf_conditioning:
            self.conditioning_vector = self.get_conditioning_vec()
        else:
            self.conditioning_vector = None

        self.encoder, self.in_dim = get_encoder('tiledgrid', input_dim=3, desired_resolution=2048 * self.bound)

        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, self.nerf_conditioning,bias=True)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            # self.encoder_bg, self.in_dim_bg = get_encoder('tiledgrid', input_dim=2, num_levels=4, desired_resolution=2048)
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)

            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def get_conditioning_vec(self,index=0):
        if self.conditioning_model == 'CLIP':
            ref_text = self.opt.text[index]
            conditioning_vector = {}
            #if self.opt.dir_text:
            #    print('not implemented')
            text_token = clip.tokenize(ref_text).to('cuda')
            with torch.no_grad():
                conditioning_tokens = self.clip_model.encode_text(text_token)
            conditioning_tokens = conditioning_tokens/ conditioning_tokens.norm(dim=-1, keepdim =True)
            conditioning_vector['input_vec']  = conditioning_tokens
            conditioning_vector['input_tokens'] = conditioning_tokens

        elif self.conditioning_model == 'T5':
            ref_text = self.opt.text[index]
            conditioning_vector = {}
            #if self.opt.dir_text:
            #    print('not implemented')
            text_token =torch.tensor(self.text_model_tokenizer([ref_text])['input_ids']).to('cuda')
            decoder_input_ids = self.text_model_tokenizer("dummy text to be ignored", return_tensors="pt").input_ids.cuda()
            if False: #TODO change later sud
                conditioning_tokens = self.text_model(text_token, decoder_input_ids = decoder_input_ids)['encoder_last_hidden_state']
            else:
                with torch.no_grad():
                    conditioning_tokens = self.text_model(text_token, decoder_input_ids = decoder_input_ids)['encoder_last_hidden_state']
            conditioning_vector['input_vec'] = conditioning_tokens.mean(dim=1)/conditioning_tokens.mean(dim=1).norm(dim=-1, keepdim=True)
            conditioning_vector['input_tokens'] = conditioning_tokens/conditioning_tokens.norm(dim=-1, keepdim = True )
        return conditioning_vector    


    # add a density blob to the scene center
    def gaussian(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        if self.opt.mem:
            torch.cuda.empty_cache()
            gc.collect() 
        h = self.encoder(x, bound=self.bound)

        h = self.sigma_net(h, conditioning_vector = self.conditioning_vector)

        sigma = trunc_exp(h[..., 0] + self.gaussian(x))
        albedo = torch.sigmoid(h[..., 1:])
        if self.opt.mem:
            torch.cuda.empty_cache()
            gc.collect() 
        return sigma, albedo
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if shading == 'albedo':
            # no need to query normal
            sigma, color = self.common_forward(x)
            normal = None
        
        else:
            # query normal

            sigma, albedo = self.common_forward(x)
            normal = self.finite_difference_normal(x)

            # with torch.enable_grad():
            #     x.requires_grad_(True)
            #     sigma, albedo = self.common_forward(x)
            #     # query gradient
            #     normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]

            # normalize...
            normal = safe_normalize(normal)
            normal[torch.isnan(normal)] = 0

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ -l).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]        

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params
