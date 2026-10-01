import argparse
import os
import torch
from configs import icl_config as C
from src.icl_utils import seed_all,device_amp,paths,save_json,append_summary
from src.icl_data import ICLData
from src.icl_model import RoPETransformer
from src.icl_evaluate import Evaluator
from src.icl_train import Trainer

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--P',type=int,default=C.P)
    p.add_argument('--rule_frac',type=float,default=C.RULE_FRAC)
    p.add_argument('--input_frac',type=float,default=C.INPUT_FRAC)
    p.add_argument('--layers',type=int,default=C.N_LAYER)
    p.add_argument('--embd',type=int,default=C.N_EMBD)
    p.add_argument('--heads',type=int,default=C.N_HEAD)
    p.add_argument('--ctx',type=int,default=C.N_CTX)
    p.add_argument('--lr',type=float,default=C.LEARNING_RATE)
    p.add_argument('--wd',type=float,default=C.WEIGHT_DECAY)
    p.add_argument('--steps',type=int,default=C.TOTAL_STEPS)
    p.add_argument('--batch_groups',type=int,default=C.BATCH_GROUPS)
    p.add_argument('--eval_every',type=int,default=C.EVAL_EVERY)
    p.add_argument('--eval_samples',type=int,default=C.EVAL_SAMPLES)
    p.add_argument('--final_samples',type=int,default=C.FINAL_SAMPLES)
    p.add_argument('--conds',default='S_train_id,S_test_ood')
    p.add_argument('--seed',type=int,default=C.SEED)
    p.add_argument('--tag')
    p.add_argument('--out_dir',default='runs')
    p.add_argument('--no_compile',action='store_true')
    p.add_argument('--no_early_stop',action='store_true')
    p.add_argument('--emerge_thr',type=float,default=None)

    a=p.parse_args()
    a.emerge_thr=3/a.P if a.emerge_thr is None else a.emerge_thr
    a.active_conditions=[x.strip() for x in a.conds.split(',') if x.strip()]

    for x in ('S_train_id','S_test_ood'):
        if x not in a.active_conditions:
            a.active_conditions.append(x)

    assert a.embd%a.heads==0
    assert (a.embd//a.heads)%2==0
    assert 3*a.ctx<=C.MAX_SEQ_FOR_ROPE
    assert 1024%(a.batch_groups*4)==0

    a.accumulation_steps=1024//(a.batch_groups*4)
    a.final_seed=C.FINAL_EVAL_SEED
    return a

def main():
    cfg=parse()
    seed_all(cfg.seed)
    device,name,dtype,amp,scaler=device_amp()
    ps=paths(cfg)

    print('Run tag:',ps['tag'])
    print('Device:',device,'|',name or 'cpu')
    print(
        f'AMP: enabled={amp} dtype={dtype if amp else "n/a"} '
        f'(GradScaler={amp and dtype==torch.float16})'
    )

    data=ICLData(cfg,device)

    raw=RoPETransformer(
        cfg.P,
        cfg.embd,
        cfg.layers,
        cfg.heads,
        4*cfg.embd,
        C.MAX_SEQ_FOR_ROPE,
        C.ROPE_THETA
    ).to(device)

    model=raw

    if not cfg.no_compile and device.type=='cuda' and hasattr(torch,'compile'):
        try:
            model=torch.compile(raw)
            print('torch.compile: enabled')
        except Exception as e:
            print('torch.compile failed:',e)

    print(
        'Parameters:',
        round(sum(p.numel() for p in raw.parameters())/1e6,3),
        'M | physical seqs/step:',
        cfg.batch_groups*4,
        '| accumulation:',
        cfg.accumulation_steps
    )

    ev=Evaluator(cfg,data,model,device,dtype,amp)
    tr=Trainer(cfg,data,raw,model,ev,scaler,device,ps)

    start,meta=tr.load()
    last,status,meta,sps=tr.run(start,meta)

    print('\n===== FINAL EVAL =====')
    last_curve=ev.final()
    print('-- last model --')
    ev.show(last_curve)

    if os.path.exists(ps['best']):
        raw.load_state_dict(
            torch.load(ps['best'],map_location=device)['model']
        )

    best_curve=ev.final()
    print(f'-- best model (step {meta["best_step"]}) --')
    ev.show(best_curve)

    ev.save(
        ps,
        best_curve,
        last_curve,
        meta['best_step'],
        ps['tag'],
        ps['log']
    )

    summary={
        'tag':ps['tag'],
        'P':cfg.P,
        'layers':cfg.layers,
        'embd':cfg.embd,
        'ctx':cfg.ctx,
        'rule_frac':cfg.rule_frac,
        'input_frac':cfg.input_frac,
        'n_train_rules':len(data.train_rules),
        'n_rectangles':len(data.rectangles),
        'wd':cfg.wd,
        'lr':cfg.lr,
        'seed':cfg.seed,
        'steps_done':last,
        'status':status,
        'chance':round(1/cfg.P,4),
        'train_id_late':round(ev.late(best_curve['S_train_id']),4),
        'ood_late_best':round(ev.late(best_curve['S_test_ood']),4),
        'ood_late_last':round(ev.late(last_curve['S_test_ood']),4),
        'ood_shot0':round(float(best_curve['S_test_ood'][0]),4),
        'best_step':meta['best_step'],
        'first_cross_step':meta['first_cross'],
        'sec_per_step':round(sps,4)
    }

    save_json(ps['summary'],summary)
    append_summary(ps['sweep'],summary)

    print('\n===== SUMMARY =====')
    for k,v in summary.items():
        print(f'{k:>18}: {v}')

if __name__=='__main__':
    main()