import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

CONDS={
    'S_train_id':('train_rules_t','train_inputs_t'),
    'S_test_id':('train_rules_t','test_inputs_t'),
    'S_train_ood':('test_rules_t','train_inputs_t'),
    'S_test_ood':('test_rules_t','test_inputs_t'),
}

class Evaluator:
    def __init__(self,cfg,data,model,device,amp_dtype,amp):
        self.cfg,self.data,self.model,self.device=cfg,data,model,device
        self.amp_dtype,self.amp=amp_dtype,amp

    @torch.no_grad()
    def curve(self,rules,inputs,n,batch_size=128,generator=None):
        self.model.eval()
        correct=torch.zeros(self.cfg.ctx,device=self.device)
        seen=0
        while seen<n:
            bs=min(batch_size,n-seen)
            r=rules[torch.randint(rules.size(0),(bs,),device=self.device,generator=generator)]
            p=self.data.sample_inputs(inputs,bs,generator)
            x=self.data.build_sequences(r[:,None,:],p)
            with torch.amp.autocast('cuda',dtype=self.amp_dtype,enabled=self.amp):
                logits=self.model(x)
            pred=logits[:,self.data.pred_pos,:].argmax(-1)
            correct+=(pred==x[:,self.data.label_pos]).float().sum(0)
            seen+=bs
        self.model.train()
        return (correct/seen).cpu().numpy()

    def run(self,n,names=None,generator=None):
        names=names or list(CONDS)
        out={}
        for name in names:
            rattr,iattr=CONDS[name]
            out[name]=self.curve(
                getattr(self.data,rattr),
                getattr(self.data,iattr),
                n,
                generator=generator
            )
        return out

    def late(self,c):
        return float(np.mean(c[self.cfg.ctx//2:]))

    def show(self,curves):
        print(f"{'shots':>6} | "+" | ".join(f"{k:>12}" for k in curves))
        shots=sorted({s for s in (0,1,2,4,8,self.cfg.ctx//2,self.cfg.ctx-1) if s<self.cfg.ctx})
        for s in shots:
            print(f"{s:>6} | "+" | ".join(f"{curves[k][s]*100:11.2f}%" for k in curves))
        print(
            f"{'late':>6} | "+
            " | ".join(f"{self.late(curves[k])*100:11.2f}%" for k in curves)+
            f"   (chance {100/self.cfg.P:.1f}%)"
        )

    def final(self):
        g=torch.Generator(device=self.device)
        g.manual_seed(self.cfg.final_seed)
        return self.run(self.cfg.final_samples,generator=g)

    def save(self,paths,best,last,best_step,tag,log_path):
        names=list(CONDS)

        with open(paths['final_curves'],'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(
                ['shots']+
                [f'{c}_best' for c in names]+
                [f'{c}_last' for c in names]
            )
            for s in range(self.cfg.ctx):
                w.writerow(
                    [s]+
                    [best[c][s] for c in names]+
                    [last[c][s] for c in names]
                )

        plt.figure(figsize=(7,4.5))
        for c in names:
            plt.plot(range(self.cfg.ctx),best[c]*100,marker='.',label=c)
        plt.axhline(
            100/self.cfg.P,
            ls='--',
            color='gray',
            label=f'chance {100/self.cfg.P:.1f}%'
        )
        plt.xlabel('in-context examples (shots)')
        plt.ylabel('accuracy (%)')
        plt.title(f'{tag} [best @ step {best_step}]')
        plt.grid(True)
        plt.legend()
        plt.savefig(paths['curves_png'],dpi=130,bbox_inches='tight')
        plt.close()

        if os.path.exists(log_path):
            with open(log_path) as f:
                rows=list(csv.DictReader(f))
            steps=[int(r['step']) for r in rows]
            plt.figure(figsize=(7,4.5))
            for c in self.cfg.active_conditions:
                plt.plot(
                    steps,
                    [float(r[f'{c}_late'])*100 for r in rows],
                    marker='.',
                    label=f'{c} (late shots)'
                )
            plt.axhline(100/self.cfg.P,ls='--',color='gray')
            plt.xlabel('step')
            plt.ylabel('late-shot accuracy (%)')
            plt.title(f'{tag}: trajectory')
            plt.grid(True)
            plt.legend()
            plt.savefig(paths['trajectory_png'],dpi=130,bbox_inches='tight')
            plt.close()