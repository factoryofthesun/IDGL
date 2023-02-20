import os
import torch
import argparse
from pdb import set_trace
import itertools
parser = argparse.ArgumentParser()
parser.add_argument('--texture', default=None, help="text prompt") 
parser.add_argument('--object', default=None, help="text prompt")
parser.add_argument('--num_prompts', type =int,default=None, help="text prompt")
parser.add_argument('--partition', default=None, help="text prompt")  
parser.add_argument('--skip', action = 'store_true')
parser.add_argument('--clip_grad', action = 'store_true') 
opt = parser.parse_args()

clip_grad_options = [True, False]
skip_options = [True, False]
#$WN_options = ['full', 'not_first', 'not_last', 'not_first_not_last', None]
WN_options = [None]
LN_options = [ 'No']
#lr_options = [1e-2,1e-3, 1e-4]
lr_options = [1e-2,1e-1]

pos_enc_ins = [2,3,0]
conditioning_mode = ['cat', 'sum']

#init_options = ['extend_ortho', 'ortho', 'extend_eye', None]
init_options = [None]
experiments = [element for element in itertools.product(*[WN_options, LN_options, lr_options, init_options, pos_enc_ins])]
set_trace()

print("#!/bin/bash")
print(". env.sh")
with open('generate_following.txt', 'w') as f:
    print('python prompt_generator.py --texture {} --object {}  --num_prompts 2 --partition {} --num_layers 3 --hidden_dim 64 --WN None --LN No --lr 0.001 --init None --skip'.format(opt.texture, opt.object, opt.partition))

    for vals in experiments:
        command = 'python prompt_generator.py --texture {} --object {}  --num_prompts 2 --partition {} --num_layers 6 --hidden_dim 64  '.format(opt.texture, opt.object, opt.partition) 
        if opt.skip :
            command = command + " --skip"
        if opt.clip_grad:
            command = command + " --clip_grad"
        command = command + " --WN {} --LN {} --lr {} --init {}".format(*vals)
        print(command)
'''
with open ('generate_following.txt', 'w') as f:
    for clip_grad_val in clip_grad_options:
        command = 'prompt_generator.py --texture {} --object {}  --num_prompts 2 --partition {} --num_layers 6 --hidden_dim 64  '.format(opt.texture, opt.object, opt.partition)
        for clip_grad_val in clip_grad_options:
            if clip_grad_val:
                command = command + ' --clip_grad'        
            for val in [1,2]:
                if clip_grad_val:
                    command = command + '--clip_grad_val {}'.format(val)

                for skip_val in skip_options:
                    if skip_val:
                        command = command + ' --skip'
                    for WN_val in WN_options:
                        command = command + ' --WN {}'.format(WN_val)
                        for init_val in init_options:
                            command = command + ' --init {}'.format(init_val)
                            for ln_val in LN_options:
                                command = command + ' --LN {}'.format(ln_val)
                                for lr_val in lr_options:
                                    command = command +' --lr {}'.format(lr_val)

                                    print(command + '\n')
                                    commands = ''
'''
