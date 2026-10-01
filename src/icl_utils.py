import csv
import json
import os
import random
import numpy as np
import torch

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device_amp():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=True
    name=torch.cuda.get_device_name(0) if device.type=='cuda' else ''
    t4='T4' in name
    bf16=device.type=='cuda' and torch.cuda.is_bf16_supported() and not t4
    dtype=torch.bfloat16 if bf16 else torch.float16
    amp=device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp and not bf16)
    return device,name,dtype,amp,scaler

def paths(cfg):
    tag=cfg.tag or f'P{cfg.P}_L{cfg.layers}_E{cfg.embd}_r{cfg.rule_frac}_i{cfg.input_frac}_c{cfg.ctx}_wd{cfg.wd}_lr{cfg.lr}_s{cfg.seed}'
    d=os.path.join(cfg.out_dir,tag)
    os.makedirs(d,exist_ok=True)
    return {
        'tag':tag,
        'run':d,
        'ckpt':os.path.join(d,'ckpt.pt'),
        'best':os.path.join(d,'best.pt'),
        'log':os.path.join(d,'log.csv'),
        'summary':os.path.join(d,'summary.json'),
        'final_curves':os.path.join(d,'final_curves.csv'),
        'curves_png':os.path.join(d,'curves.png'),
        'trajectory_png':os.path.join(d,'trajectory.png'),
        'sweep':os.path.join(cfg.out_dir,'sweep_summary.csv')
    }

def append_log(path,fields,row):
    new=not os.path.exists(path)
    with open(path,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)

def save_json(path,obj):
    with open(path,'w') as f:
        json.dump(obj,f,indent=2)

def append_summary(path,obj):
    new=not os.path.exists(path)
    with open(path,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(obj))
        if new:
            w.writeheader()
        w.writerow(obj)

def save_ckpt(path,model,opt,sched,scaler,step,meta,sig,version):
    torch.save({
        'model':model.state_dict(),
        'optimizer':opt.state_dict(),
        'scheduler':sched.state_dict(),
        'scaler':scaler.state_dict(),
        'step':step,
        'meta':meta,
        'config_sig':sig,
        'code_version':version
    },path)