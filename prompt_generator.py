from pdb import set_trace
import argparse
import torch
import os
parser = argparse.ArgumentParser()
parser.add_argument('--texture', default=None, help="text prompt")
parser.add_argument('--object', default=None, help="text prompt")
parser.add_argument('--num_prompts', type =int,default=None, help="text prompt")
parser.add_argument('--partition', default=None, help="text prompt")
parser.add_argument('--num_layers', type=int, default=3, help="render width for NeRF in training")
parser.add_argument('--hidden_dim', type=int, default=64, help="render width for NeRF in training")
parser.add_argument('--clip_grad', action='store_true', help="overwrite current experiment")
parser.add_argument('--skip', action='store_true', help="overwrite current experiment")
parser.add_argument('--WN', type = str, default = None)
parser.add_argument('--init', type = str, default = None)
parser.add_argument('--LN', type = str, default = None)
parser.add_argument('--clip_grad_val', type=int, default=1, help="render width for NeRF in training")
parser.add_argument('--lr', type=float, default=1e-3, help="render width for NeRF in training")   

opt = parser.parse_args()


#prefixes = ['Award-Winning Art of']
prefixes = ['Detailed 3d render of']
#prefixes = ['Award-Winning 3d Render of', 'Award-Winning Art of', 'Detailed 3d render of', 'Detailed Art of', "Award-Winning Fanart of", 
#             "Detailed Fanart of", "Oil painting of"]
#suffixes = ['Trending on ArtStation', "Unreal Engine", "DeviantArt"]
suffixes = ['DeviantArt'] 
#suffixes = ['Trending on ArtStation']
# Generate n engineered samples of prompts
def engineer_prompt(prompt, prefixes, suffixes, n=1):
    from itertools import permutations 
    import random 
     
    perms = [(p, s) for p in prefixes for s in suffixes]
    random.shuffle(perms)
    new_prompts = [(f"{p} {prompt}, {s}",p,s) for p, s in perms[:n]]
     
    return new_prompts

if len(opt.texture.split(" "))>1:
    opt.texture_path = '_'.join(opt.texture.split(" "))
else:
    opt.texture_path = opt.texture
 
#prompts = engineer_prompt("a {} in the shape of {}. {} imitating {}".format(opt.object, opt.texture, opt.object, opt.texture), prefixes, suffixes,opt.num_prompts )
#prompts = engineer_prompt("a {} made of {}. {} with the texture of {} ".format(opt.object, opt.texture, opt.object, opt.texture), prefixes, suffixes,opt.num_prompts )
prompts = engineer_prompt("a {} made of {}".format(opt.object, opt.texture, opt.object, opt.texture), prefixes, suffixes,opt.num_prompts )
for prompt,p,s in prompts:
    p_id = prefixes.index(p)
    
    s_id = suffixes.index(s)
    p_id = 1
    s_id =1
    if not os.path.exists('{}/{}/'.format(opt.object,opt.texture_path,str(p_id),str(s_id))):
        os.makedirs('{}/{}/'.format(opt.object,opt.texture_path,str(p_id),str(s_id)))

    with open('{}/{}/{}_{}_prompt.txt'.format(opt.object, opt.texture_path, str(p_id), str(s_id)), 'w') as f:
        f.write('{}'.format(prompt))
    file_name = '{}/{}/{}_{}'.format(opt.object,opt.texture_path,str(p_id),str(s_id))
    if opt.skip:
        file_name = file_name + '_skip'  
    if opt.WN is not None:
        file_name = file_name + '_'+str(opt.WN)
    if opt.LN is not None:
        file_name = file_name + '_'+str(opt.LN)
    if opt.init is not None:
        file_name = file_name + '_'+str(opt.init)

    file_name = file_name + '_'+str(opt.lr).split('.')[-1]+'.sh'
    
        
    with open(file_name, 'w') as f:
        f.write('#!/bin/bash \n')
        f.write(". env.sh \n")
        if not opt.skip:
            command = 'python main.py --text {}/{}/{}_{}_prompt.txt --iters 20000 -O --ckpt scratch --project_name {}  -O --eval_interval 10 --wandb_flag --workspace {} --num_layers {} --hidden_dim {} --lr {}'.format(opt.object,opt.texture_path,str(p_id),str(s_id),opt.object,  str(opt.object+'_'+opt.texture_path), str(opt.num_layers), str(opt.hidden_dim), str(opt.lr))
            exp_name ='{}_{}_{}_{}'.format(opt.object,opt.texture_path,str(p_id), str(s_id))
            
        else:
            command = 'python main.py --text {}/{}/{}_{}_prompt.txt --iters 20000 -O --ckpt scratch --project_name {} -O --eval_interval 10 --wandb_flag --workspace {} --num_layers {} --hidden_dim {} --skip --lr {}'.format(opt.object,opt.texture_path,str(p_id),str(s_id), str(opt.object), str(opt.object+'_'+opt.texture_path), str(opt.num_layers), str(opt.hidden_dim), str(opt.lr))
            exp_name ='{}_{}_{}_{}_skip'.format(opt.object,opt.texture_path,str(p_id),str(s_id))

        if opt.WN is not None:
            command = command+" --WN {}".format(opt.WN)
            exp_name = exp_name+"_{}".format(opt.WN)
        
        if opt.clip_grad:
            if opt.clip_grad_val ==1:
                command = command + ' --clip_grad --clip_grad_val 1.0'
            else:
                command = command + ' --clip_grad --clip_grad_val 2.0'
            exp_name = exp_name+"_grad_clip_{}".format(opt.clip_grad_val)

        if opt.init is not None:
            command = command + ' --init {}'.format(opt.init)          
            exp_name = exp_name+"_{}".format(opt.init)

        if  opt.LN is not None:
            command = command + ' --normalization {}'.format(opt.LN)
            exp_name = exp_name+"_{}".format(opt.LN)
        
        exp_name = exp_name+ "_{}".format(str(opt.lr).split('.')[-1])
        command = command+ ' --exp_name {}'.format(exp_name)
        
        f.write("{}".format(command))
    if not os.path.exists("{}/".format(opt.object)):
        os.makedirs('{}/'.format(opt.object))
    with open('{}/{}/exe.sh'.format(opt.object, opt.texture_path), 'a') as f:
        if opt.partition is None:
            flag_ = int(torch.bernoulli(torch.ones(1)*.7).item())
            if flag_ ==0:
                f.write("sbatch -p dreamfields-gpu  {} \n".format(file_name))
            else:
                f.write("sbatch -p greg-gpu -C 48g  {} \n".format(file_name))

        else:
            if 'dream' in opt.partition:
                f.write("sbatch -p dreamfields-gpu   {} \n".format(file_name))
            elif 'greg' in opt.partition:
                f.write("sbatch -p greg-gpu -C 48g  {} \n".format(file_name))

