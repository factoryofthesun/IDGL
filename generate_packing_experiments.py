import os
import torch
import argparse
from pdb import set_trace
import itertools
import random
parser = argparse.ArgumentParser()
parser.add_argument('--num_prompts', type =int,default=2, help="text prompt")
parser.add_argument('--partition', default='dreamfields-gpu', help="text prompt")  
parser.add_argument('--objects', nargs='+', help='<Required> Set flag')
opt = parser.parse_args()



conditioning_model_options = ['bert']
arch_options = ['detach_dynamic_hyper_transformer', 'dynamic_hyper_transformer']
#arch_options = [ 'hyper_transformer']
#arch_options = ['mlp']

cond_trans_opt = ['--condition_trans', '--dummy']
multiple_conditioning_transformers_opts = ['--multiple_conditioning_transformers', '--dummy']
normalization_opt = ['post_ada']
#multi_conditioning_layers_options = [True]
#ada_options = [ 'pre_ada', 'post_ada']
meta_batch_options = [2,4]

#init_options = ['extend_ortho', 'ortho', 'extend_eye', None]
init_options = [None]
experiments = [element for element in itertools.product(*[conditioning_model_options,arch_options, cond_trans_opt, multiple_conditioning_transformers_opts, normalization_opt, meta_batch_options])]

fp = open('working_prompts.txt', 'r')
prompts  = fp.readlines()
random.shuffle(prompts)
fp.close()
#prompts = [prompt.strip() for prompt in prompts]
#set_trace()
if opt.objects is not None:
    required_prompts = []
    for object_ in opt.objects:
        #new_prompts = list(filter(lambda x: sum([int(obj in x) for obj in opt.objects]) > 0, prompts ))
        new_prompts = list(filter(lambda x: sum([int(object_ in x)]) > 0, prompts ))
        required_prompts = required_prompts + new_prompts

prompts = required_prompts
#set_trace()

if len(prompts) < opt.num_prompts:
    opt.num_prompts = len(prompts) 


if opt.objects is not None:
    exp_name_suffix = '_'+'_'.join(opt.objects) + '_obj_'+ str(opt.num_prompts)
else:
    exp_name_suffix = '_all' + '_obj_'+ str(opt.num_prompts)



if not os.path.exists('{}_pack/txt/'.format(opt.num_prompts)):
    os.makedirs('{}_pack/txt/'.format(opt.num_prompts))
    os.makedirs('{}_pack/exp/'.format(opt.num_prompts))    
    start_id = 0
else:
    all_files = list(filter(lambda x: '.txt' in x, os.listdir('{}_pack/txt/'.format(opt.num_prompts))))
    if len(all_files) > 0:
        start_id = max([int(files.split('_')[0]) for files in all_files]) + 1
    else:
        start_id = 0

prompts =  random.sample(prompts,opt.num_prompts )

fp_w =  open('{}_pack/txt/{}{}.txt'.format(opt.num_prompts,start_id,exp_name_suffix),'w')
fp_w.writelines(prompts)
fp_w.close()

print(prompts)

print("correcting to  {} pack".format(opt.num_prompts))
for idx,experiment in enumerate(experiments):
    with open('{}_pack/exp/{}{}_{}_{}_{}_{}_{}_{}.sh'.format(opt.num_prompts,start_id,exp_name_suffix, experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5]), 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(". env.sh\n")
        f.write('python main.py --text {}   --iters {} -O --ckpt latest --project_name {} -O --workspace hamburger_yarn --num_layers 6 --hidden_dim 64 --lr 0.0001 --WN None --init ortho  --exp_name {}  --skip --albedo_iters 6000000 --conditioning_model {} --conditioning_mode cat --conditioning_dim 64 --pos_enc_ins 1 --arch hyper_transformer --eval_interval 10   --arch {} {} {}  --normalization {} --meta_batch_size {}'.format('{}_pack/txt/{}{}.txt'.format(opt.num_prompts,start_id,exp_name_suffix), 10000*opt.num_prompts,str(opt.num_prompts)+"_pack",'{}{}_{}_{}_{}_{}_{}'.format(start_id,exp_name_suffix,experiment[0], experiment[1].split('_')[0], experiment[2], experiment[3].split('_')[0][:5],  experiment[5]) , experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5]  ))
        #print('python main.py --text {}   --iters {} -O --ckpt latest --project_name {} -O --workspace hamburger_yarn --num_layers 6 --hidden_dim 64 --lr 0.0001 --WN None --init ortho  --exp_name {}  --skip --albedo_iters 6000000 --conditioning_model {} --conditioning_mode cat --conditioning_dim 64 --pos_enc_ins 1 --arch hyper_transformer --eval_interval 10 --multiple_conditioning_transformers  --arch {} {} {}  --normalization {} --meta_batch_size {}'.format('{}_pack/txt/{}{}.txt'.format(opt.num_prompts,start_id,exp_name_suffix), 10000*opt.num_prompts,str(opt.num_prompts)+"_pack",'{}{}_{}_{}_{}_{}_{}'.format(start_id,exp_name_suffix,experiment[0], experiment[1].split('_')[0], experiment[2], experiment[3].split('_')[0][0],  experiment[5]) , experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5]  ))
        #print('python main.py --text {}   --iters {} -O --ckpt latest --project_name {} -O --workspace hamburger_yarn --num_layers 6 --hidden_dim 64 --lr 0.0001 --WN None --init ortho  --exp_name {}  --skip --albedo_iters 6000000 --conditioning_model {} --conditioning_mode cat --conditioning_dim 64 --pos_enc_ins 1 --arch hyper_transformer --eval_interval 10 --multiple_conditioning_transformers --wandb --arch {} {} {}  --normalization {} --meta_batch_size {}'.format('{}_pack/txt/{}{}.txt'.format(opt.num_prompts,start_id,exp_name_suffix), 10000*opt.num_prompts,str(opt.num_prompts)+"_pack",'{}{}_{}_{}_{}_{}_{}_{}'.format(start_id,exp_name_suffix,experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5]) , experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5]  ))

        #print('python main.py --text {}   --iters {} -O --ckpt latest --project_name {} -O --workspace hamburger_yarn --num_layers 6 --hidden_dim 64 --lr 0.0001 --WN None --init ortho  --exp_name {}  --skip --albedo_iters 6000000 --conditioning_model {} --conditioning_mode cat --conditioning_dim 64 --pos_enc_ins 1 --arch hyper_transformer --eval_interval 10 --multiple_conditioning_transformers --wandb --arch {} {} {}  --normalization {} --meta_batch_size {}'.format('{}_pack/txt/{}{}.txt'.format(opt.num_prompts,start_id,exp_name_suffix), 10000*opt.num_prompts,str(opt.num_prompts)+"_pack",'{}{}_{}_{}_{}_{}_{}_{}'.format(start_id,exp_name_suffix,experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5]) , experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5]  ))

        f.close()

    with open('{}_pack/{}_exe.sh'.format(opt.num_prompts, start_id), 'a') as f:
        f.write('sbatch -p {} -C 48g -J {} -d singleton  {}  \n'.format(opt.partition,str(start_id)+'_'+str(idx),'{}_pack/exp/{}{}_{}_{}_{}_{}_{}_{}.sh'.format(opt.num_prompts,start_id,exp_name_suffix, experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5])))
        #print('sbatch -p {} -C 48g -J {} -d singleton  {}  \n'.format(opt.partition,str(start_id)+'_'+str(idx),'{}_pack/exp/{}{}_{}_{}_{}_{}_{}_{}.sh'.format(opt.num_prompts,start_id,exp_name_suffix, experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5])))
        #f.write('sbatch -p {} -C 48g -J {} -d singleton  {}  \n'.format(opt.partition,str(start_id)+'_'+str(idx),'{}_pack/exp/{}{}_{}_{}_{}_{}_{}_{}.sh'.format(opt.num_prompts,start_id,exp_name_suffix, experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5])))
        #f.write('sbatch -p {} -C 48g -J {} -d singleton  {}  \n'.format(opt.partition,str(start_id)+'_'+str(idx),'{}_pack/exp/{}{}_{}_{}_{}_{}_{}_{}.sh'.format(opt.num_prompts,start_id,exp_name_suffix, experiment[0], experiment[1], experiment[2], experiment[3], experiment[4], experiment[5])))
    f.close()    


