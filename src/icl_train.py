import math
import os
import time
import torch
from torch.optim import AdamW
from configs import icl_config as C
from src.icl_utils import append_log,save_ckpt

CODE_VERSION='v3_baseline'

class Trainer:
    def __init__(self,cfg,data,raw,model,evaluator,scaler,device,paths):
        self.cfg,self.data,self.raw,self.model,self.ev,self.scaler,self.device,self.paths=cfg,data,raw,model,evaluator,scaler,device,paths
        decay=[]; no_decay=[]
        for n,p in raw.named_parameters():
            (no_decay if 'ln' in n or p.ndim==1 else decay).append(p)

        self.opt=AdamW(
            [
                {'params':decay,'weight_decay':cfg.wd},
                {'params':no_decay,'weight_decay':0}
            ],
            lr=cfg.lr,
            betas=(C.BETA1,C.BETA2),
            eps=1e-8
        )

        warm=max(1,int(cfg.steps*C.WARMUP_FRAC))

        def f(step):
            if step<warm:
                return 0.01+0.99*(step+1)/warm
            p=(step-warm)/max(1,cfg.steps-warm)
            return 0.1+0.45*(1+math.cos(math.pi*p))

        self.sched=torch.optim.lr_scheduler.LambdaLR(self.opt,f)
        self.fields=['step','loss']+[f'{c}_late' for c in cfg.active_conditions]+['S_test_ood_last_shot']

    def sig(self):
        return {
            'P':self.cfg.P,
            'rule_frac':self.cfg.rule_frac,
            'input_frac':self.cfg.input_frac,
            'ctx':self.cfg.ctx,
            'seed':self.cfg.seed,
            'embd':self.cfg.embd,
            'layers':self.cfg.layers,
            'heads':self.cfg.heads
        }

    def load(self):
        meta={'best_ood':-1.,'best_step':0,'first_cross':None,'streak':0}
        start=0

        if not os.path.exists(self.paths['ckpt']):
            return start,meta

        ck=torch.load(self.paths['ckpt'],map_location=self.device)

        if ck.get('config_sig')!=self.sig() or ck.get('code_version')!=CODE_VERSION:
            return start,meta

        self.raw.load_state_dict(ck['model'])
        self.opt.load_state_dict(ck['optimizer'])
        self.sched.load_state_dict(ck['scheduler'])
        self.scaler.load_state_dict(ck['scaler'])
        start=ck['step']
        meta=ck['meta']
        print('Resumed from step',start)
        return start,meta

    def run(self,start,meta):
        c=self.cfg
        accum=c.accumulation_steps
        status='completed'
        last=start
        n=0
        t0=time.time()

        self.model.train()
        self.opt.zero_grad(set_to_none=True)

        for step in range(start+1,c.steps+1):
            total=torch.zeros((),device=self.device)

            for _ in range(accum):
                ids=self.data.train_batch(c.batch_groups)
                with torch.amp.autocast(
                    'cuda',
                    dtype=self.ev.amp_dtype,
                    enabled=self.ev.amp
                ):
                    loss=self.data.loss(self.model(ids),ids)/accum

                self.scaler.scale(loss).backward()
                total+=loss.detach()

            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.raw.parameters(),C.GRAD_CLIP)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            last=step
            n+=1

            if step%C.PRINT_EVERY==0:
                print(
                    f'step {step:6d} | loss {total.item():.4f} | '
                    f'lr {self.opt.param_groups[0]["lr"]:.2e} | '
                    f'{(time.time()-t0)/n:.3f}s/step'
                )

            if step%c.eval_every==0 or step==c.steps:
                curves=self.ev.run(c.eval_samples,c.active_conditions)
                tr=self.ev.late(curves['S_train_id'])
                ood=self.ev.late(curves['S_test_ood'])

                print(f'\n--- eval @ step {step} (n={c.eval_samples}) ---')
                self.ev.show(curves)
                print()

                append_log(
                    self.paths['log'],
                    self.fields,
                    {
                        'step':step,
                        'loss':round(total.item(),5),
                        **{
                            f'{k}_late':round(self.ev.late(v),4)
                            for k,v in curves.items()
                        },
                        'S_test_ood_last_shot':round(float(curves['S_test_ood'][-1]),4)
                    }
                )

                if ood>meta['best_ood']:
                    meta['best_ood']=ood
                    meta['best_step']=step
                    torch.save(
                        {'model':self.raw.state_dict(),'step':step},
                        self.paths['best']
                    )

                if meta['first_cross'] is None and ood>=c.emerge_thr:
                    meta['first_cross']=step

                meta['streak']=meta['streak']+1 if ood>=.95 and tr>=.9 else 0
                save_ckpt(
                    self.paths['ckpt'],
                    self.raw,
                    self.opt,
                    self.sched,
                    self.scaler,
                    step,
                    meta,
                    self.sig(),
                    CODE_VERSION
                )

                if not c.no_early_stop:
                    if meta['streak']>=2:
                        status='emerged_stop'
                        break

                    if step>=2*c.steps//3 and total.item()>0.98*math.log(c.P):
                        status='stuck_stop'
                        break

        if last!=start or not os.path.exists(self.paths['ckpt']):
            save_ckpt(
                self.paths['ckpt'],
                self.raw,
                self.opt,
                self.sched,
                self.scaler,
                last,
                meta,
                self.sig(),
                CODE_VERSION
            )

        return last,status,meta,(time.time()-t0)/max(1,n)