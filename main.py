import torch
import argparse
from pathlib import Path
from nerf.provider import NeRFDataset
from nerf.utils import *
from optimizer import Shampoo
import wandb
from pdb import set_trace
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# from nerf.gui import NeRFGUI

# torch.autograd.set_detect_anomaly(True)
def clear_directory(path):
    import shutil
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--engineer_prefix', type=str, default=None, help="prefix prompt")
    parser.add_argument('--style', type=str, default=None, help="style prompt")
    parser.add_argument('--object', type=str, default=None, help="object prompt")
    parser.add_argument('--engineer_suffix', type=str, default=None, help="suffix prompt")
    parser.add_argument('--poolstyle', action='store_true', help="pool the style tokens (color/obj separately)")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --fp16 --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--interp_engineer_prefix', type=str, default=None, help="text prompt")
    parser.add_argument('--interp_style', type=str, default=None, help="text prompt")
    parser.add_argument('--interp_object', type=str, default=None, help="text prompt")
    parser.add_argument('--interp_engineer_suffix', type=str, default=None, help="text prompt")
    parser.add_argument('--interp', type=str, default=None, choices={'bert', 'hyper'}, help="interpolation mode")
    parser.add_argument('--interpfreq', type=int, default=2, help="interpolation frequencies (endpoint inclusive)")
    parser.add_argument('--textindex', type=str, default=None, help="npy array containing indices to interpolate for txt file")
    parser.add_argument('--interpindex', type=str, default=None, help="npy array containing indices to interpolate for interp file")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true', help="Debugging mode")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=64, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='grid', help="nerf backbone, choose from [grid, tcnn, vanilla]")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")

    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true', help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    parser.add_argument('--lambda_entropy', type=float, default=1e-4, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")


    ### stable training options
    parser.add_argument('--clip_grad', action='store_true', help="overwrite current experiment")
    parser.add_argument('--fine_tune_conditioner', action='store_true', help="overwrite current experiment")
    parser.add_argument('--clip_grad_val', default = 1.0, type=float, help="overwrite current experiment")
    parser.add_argument('--ema_decay', default = None, type=float)
    parser.add_argument('--init', default = None)
    parser.add_argument('--normalization', type = str, default = 'No')
    parser.add_argument('--WN', type = str, default = None)
    # ### GUI options
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")

    ### Logging options
    parser.add_argument('--wandb_flag', action='store_true', help="log in wandb")
    parser.add_argument('--project_name', type=str, default='test')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--overwrite', action='store_true', help="overwrite current experiment")
    parser.add_argument('--testdir', type = str, default='inference')

    ###Network options
    parser.add_argument('--num_layers', type=int, default=3, help="render width for NeRF in training")
    parser.add_argument('--hidden_dim', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--pos_enc_ins', type=int, default=0, help="render width for NeRF in training")
    parser.add_argument('--skip', action = 'store_true')
    parser.add_argument('--bottleneck', action = 'store_true')
    parser.add_argument('--arch', type = str, default='mlp')
    ### Conditioning options
    parser.add_argument('--conditioning_model', type=str, default=None)
    parser.add_argument('--conditioning_mode', type=str, default='sum')
    parser.add_argument('--conditioning_dim', type = int, default = 0 )
    parser.add_argument('--meta_batch_size', type = int, default = 1)
    parser.add_argument('--multiple_conditioning_transformers', action = 'store_true')
    parser.add_argument('--condition_trans', action = 'store_true')
    parser.add_argument('--phrasing', action = 'store_true')
    parser.add_argument('--curricullum', action = 'store_true')

    #### Other option
    parser.add_argument('--mem', action='store_true', help="overwrite current experiment")
    parser.add_argument('--dummy', action='store_true', help="overwrite current experiment")
    # parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    # parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    # parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    # parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    # parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    opt = parser.parse_args()
    opt.textstyleidx = None
    opt.objstyleidx = None
    opt.interpstyleidx = None
    opt.interpobjidx = None

    opt.text = []
    if opt.engineer_prefix:
        with open(opt.engineer_prefix) as f:
            lines = f.readlines()
        opt.text = [" ".join(line.split()) for line in lines]

    if opt.style:
        with open(opt.style) as f:
            lines = f.readlines()

        # Save starting indices of style prompt
        if len(opt.text) == 0:
            opt.textstyleidx = [0 for _ in range(len(lines))]
        else:
            opt.textstyleidx = [len(line.split()) for line in opt.text]

        lines = [" ".join(line.split()) for line in lines]
        opt.text = [opt.text[i] + " " + " ".join(lines[i].split())  for i in range(min(len(opt.text), len(lines)))]

        # Save ending indices of style prompt
        opt.textstyleidx = [(opt.textstyleidx[i], len(opt.text[i].split())-1) for i in range(len(opt.text))]

    if opt.object:
        with open(opt.object) as f:
            lines = f.readlines()

        # Save starting indices of obj prompt
        opt.objstyleidx = [len(line.split()) for line in opt.text]

        lines = [" ".join(line.split()) for line in lines]
        opt.text = [opt.text[i] + " " + " ".join(lines[i].split())  for i in range(min(len(opt.text), len(lines)))]

        # Save ending indices of obj prompt
        opt.objstyleidx = [(opt.objstyleidx[i], len(opt.text[i].split())-1) for i in range(len(opt.text))]

    if opt.engineer_suffix:
        with open(opt.engineer_suffix) as f:
            lines = f.readlines()
        lines = [" ".join(line.split()) for line in lines]
        opt.text = [opt.text[i] + " " + " ".join(lines[i].split()) for i in range(min(len(opt.text), len(lines)))]

    opt.text = [" ".join(line.split()) for line in opt.text]
    print(opt.text)

    # Interpolation file
    interpvals = None
    if opt.interp and opt.style:
        opt.interptext = []
        if opt.interp_engineer_prefix:
            with open(opt.interp_engineer_prefix) as f:
                lines = f.readlines()
            opt.interptext = [" ".join(line.split()) for line in lines]

        if opt.interp_style:
            with open(opt.interp_style) as f:
                lines = f.readlines()

            # Save starting indices of style prompt
            opt.interpstyleidx = [len(line.split()) for line in opt.interptext]

            lines = [" ".join(line.split()) for line in lines]
            opt.interptext = [opt.interptext[i] + " " + " ".join(lines[i].split())  for i in range(min(len(opt.interptext), len(lines)))]

            # Save ending indices of style prompt
            opt.interpstyleidx = [(opt.interpstyleidx[i], len(opt.interptext[i].split())-1) for i in range(len(opt.interptext))]

        if opt.interp_object:
            with open(opt.interp_object) as f:
                lines = f.readlines()

            # Save starting indices of obj prompt
            opt.interpobjidx = [len(line.split()) for line in opt.interptext]

            lines = [" ".join(line.split()) for line in lines]
            opt.interptext = [opt.interptext[i] + " " + " ".join(lines[i].split())  for i in range(min(len(opt.interptext), len(lines)))]

            # Save ending indices of obj prompt
            opt.interpobjidx = [(opt.interpobjidx[i], len(opt.interptext[i].split())-1) for i in range(len(opt.interptext))]

        if opt.interp_engineer_suffix:
            with open(opt.interp_engineer_suffix) as f:
                lines = f.readlines()
            lines = [" ".join(line.split()) for line in lines]
            opt.interptext = [opt.interptext[i] + " " + " ".join(lines[i].split()) for i in range(min(len(opt.interptext), len(lines)))]
        opt.interptext = [" ".join(line.split()) for line in opt.interptext]

        # Duplicate text for each possible pairing and also interpolation float value
        interpvals = np.linspace(0, 1, opt.interpfreq)

        # Map each individual text line to interptext * interpfreq values
        textlen = len(opt.interptext)
        interplen = len(opt.interptext)
        opt.text = [text for text in opt.text for _ in range(interplen * opt.interpfreq)]

        opt.objstyleidx = [text for text in opt.objstyleidx for _ in range(interplen * opt.interpfreq)]
        opt.textstyleidx = [text for text in opt.textstyleidx for _ in range(interplen * opt.interpfreq)]

        opt.interptext = [text for text in opt.interptext for _ in range(opt.interpfreq)]
        opt.interptext *= textlen

        opt.interpstyleidx = [idx for idx in opt.interpstyleidx for _ in range(opt.interpfreq)]
        opt.interpstyleidx *= textlen
        opt.interpobjidx = [idx for idx in opt.interpobjidx for _ in range(opt.interpfreq)]
        opt.interpobjidx *= textlen

        # Remove duplicates
        assert len(opt.text) == len(opt.interptext), f"Expected text length {len(opt.text)} to equal interptext length {len(opt.interptext)}"
        newtext = []
        newinterp = []
        new_textstyleidx = []
        new_objstyleidx = []
        new_interpstyleidx = []
        new_interpobjidx = []
        doneprompts = set()
        for i in range(len(opt.text)):
            if opt.text[i] + opt.interptext[i] in doneprompts or opt.interptext[i] + opt.text[i] in doneprompts:
                continue
            if opt.text[i] != opt.interptext[i]:
                newtext.append(opt.text[i])
                newinterp.append(opt.interptext[i])

                new_textstyleidx.append(opt.textstyleidx[i])
                new_objstyleidx.append(opt.objstyleidx[i])

                new_interpstyleidx.append(opt.interpstyleidx[i])
                new_interpobjidx.append(opt.interpobjidx[i])
        if opt.debug:
            print(f"Found {len(opt.text) - len(newtext)} duplicate prompts between text and interp")
            promptset = [val for val in zip(newtext, newinterp)]
            print(f"Prompt set: {promptset}")
        opt.interptext = newinterp
        opt.text = newtext
        opt.textstyleidx = new_textstyleidx
        opt.objstyleidx = new_objstyleidx
        opt.interpstyleidx = new_interpstyleidx
        opt.interpobjidx = new_interpobjidx
    else:
        opt.interp = None
        opt.interpfreq = None
        opt.interptext = None
        opt.textstyleidx = None
        opt.objstyleidx = None
        opt.interpstyleidx = None
        opt.interpobjidx = None
        interpvals = None

    # Load style idx
    # if opt.interp == "bert":
    #     assert os.path.exists(opt.textindex) and os.path.exists(opt.interpindex), f"If interpolation over bert vectors, then files opt.textindex {opt.textindex} and opt.interpindex {opt.interpindex} must exist."

    if opt.textindex:
        opt.textindex = np.load(opt.textindex)
    if opt.interpindex:
        opt.interpindex = np.load(opt.interpindex)

    opt.workspace = os.path.join("outputs", opt.project_name, opt.exp_name)
    if opt.overwrite and os.path.exists(opt.workspace):
        clear_directory(opt.workspace)

    if opt.wandb_flag:
        resume_flag = opt.ckpt == 'latest'
        wandb.init(project = opt.project_name,config = opt, resume = True, name = opt.exp_name, id = opt.exp_name)
    else:
        wandb = None
    if opt.O:
        opt.fp16 = False
        opt.dir_text = True
        # use occupancy grid to prune ray sampling, faster rendering.
        opt.cuda_ray = True
        # opt.lambda_entropy = 1e-4
        # opt.lambda_opacity = 0

    elif opt.O2:
        opt.fp16 = True
        opt.dir_text = True
        opt.lambda_entropy = 1e-4 # necessary to keep non-empty
        opt.lambda_opacity = 3e-3 # no occupancy grid, so use a stronger opacity loss.

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'tcnn':
        from nerf.network_tcnn import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)

    seed_everything(opt.seed)
    if  'hyper_transformer' in opt.arch:
       from nerf.hyper_network_grid import HyperTransNeRFNetwork as NeRFNetwork

    if True:
        model = nn.DataParallel(NeRFNetwork(opt, num_layers= opt.num_layers, hidden_dim = opt.hidden_dim,wandb_obj=wandb ), device_ids = [0])
    else:
        model = NeRFNetwork(opt, num_layers= opt.num_layers, hidden_dim = opt.hidden_dim)
    #print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        size = 100
        if opt.debug:
            size = 10
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=size).dataloader()
        Path(os.path.join(opt.workspace, opt.testdir)).mkdir(exist_ok=True, parents=True)
        clear_directory(os.path.join(opt.workspace, opt.testdir))
        for idx, val in enumerate(opt.text):
            interpval = None
            if interpvals is not None:
                interpval = interpvals[idx % len(interpvals)]

            if opt.debug:
                print(f"Scene id: {idx}, Text: {val}, Interpval: {interpval}")
            name = val.lower().replace(" ", "_")
            if interpvals is not None:
                name = val.lower().replace(" ", "_") + f"_interp{interpval:0.2f}_" + opt.interptext[idx].lower().replace(" ", "_")
                if opt.phrasing:
                    name = "phrasing_" + name

            trainer.test(test_loader, name=name, save_path=os.path.join(opt.workspace, opt.testdir),
                         scene_id=idx, interpval=interpval)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)

    else:
        if opt.guidance == 'stable-diffusion':
            from nerf.sd import StableDiffusion
            guidance = StableDiffusion(device)
        elif opt.guidance == 'clip':
            from nerf.clip import CLIP
            guidance = CLIP(device)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        optimizer = lambda model: torch.optim.Adam(model.module.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        #optimizer = lambda model: Adan(model.module.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        # optimizer = lambda model: Shampoo(model.get_params(opt.lr))

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1e-3 if iter < 500 else  0.1 ** min(iter / opt.iters, 1))
        #scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter:  0.1 ** min(iter / opt.iters, 1))
        # scheduler = lambda optimizer: optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=opt.iters, pct_start=0.1)
        set_trace()
        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=opt.ema_decay, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True, wandb_obj = wandb)

        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

        trainer.train(train_loader, valid_loader,test_loader, max_epoch)

        # also test
        trainer.test(test_loader)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
