"""Window preparation, embedding extraction, and frozen response-head training."""
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from experiments.finetune_response_windows import WindowPredictor
from experiments import additional_response_encoders as extra
from iesseeg_paper.input_studies import read_interval, prepare_encoder_input

SEED = 20260924

def prepare_windows(signal,model,seconds):
    """Match fine-tuning preprocessing, including per-window filtering for LaBraM and CBraMod."""
    samples=seconds*200
    assert signal.shape[-1]%samples==0
    if model=='biot':
        x=prepare_encoder_input(signal,model)
        windows=x.reshape(18,-1,samples).transpose(1,0,2).copy()
        windows=windows/(np.quantile(np.abs(windows),.95,axis=-1,keepdims=True)+1e-8)
    elif model in extra.NAMES:
        windows=np.stack([extra.prepare(signal[:,i:i+samples],model,seconds)
                          for i in range(0,signal.shape[-1],samples)])
    else:
        windows=np.stack([prepare_encoder_input(signal[:,i:i+samples],model).reshape(19,seconds,200)/100.
                          for i in range(0,signal.shape[-1],samples)])
    windows=windows.astype('float32')
    if not np.isfinite(windows).all():raise ValueError('Nonfinite encoder input')
    return windows

class Crops(Dataset):
    def __init__(self,frame,model,seconds):self.rows=frame.to_dict('records');self.model=model;self.seconds=seconds
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];used=1800//self.seconds*self.seconds
        raw=read_interval(Path(r['source']),float(r['start_seconds']),used)
        return prepare_windows(raw,self.model,self.seconds)


def worker_init(_):torch.set_num_threads(1)


def encode(a,frame,fold,settings):
    if a.branch=='finetuned':
        payload=torch.load(a.encoder_root/a.model/a.visit/f'fold{fold}'/'best.pt',map_location='cpu',weights_only=False)
        protocol=payload['protocol'];folds=frame.fold.to_numpy()
        for key,mask in [('train_patients',~np.isin(folds,[fold,(fold+1)%5])),('validation_patients',folds==(fold+1)%5),('test_patients',folds==fold)]:
            assert set(protocol[key])==set(frame.loc[mask,'patient_id']),key
        assert protocol['task']=='sustained' and protocol['condition']==a.visit and protocol['test_fold']==fold
    torch.manual_seed(SEED)
    model=WindowPredictor(a.model,a.workspace,a.device,settings)
    if a.branch=='finetuned':model.load_state_dict(payload['model'])
    model.eval()
    if a.branch=='finetuned':del payload
    loader=DataLoader(Crops(frame,a.model,settings['window_seconds']),batch_size=None,num_workers=a.workers,
        worker_init_fn=worker_init,multiprocessing_context='spawn' if a.workers else None,pin_memory=True)
    bank=[];logits=[];start=time.time()
    with torch.inference_mode():
        for i,windows in enumerate(loader):
            hs=[];zs=[]
            for w in windows.split(a.inference_batch_size):
                h=model.encode(w.to(a.device));hs.append(h.cpu().numpy());zs.append(model.head(h).flatten().cpu().numpy())
            bank.append(np.concatenate(hs));logits.append(np.concatenate(zs))
            if (i+1)%100==0:print('ENCODE',a.model,a.branch,a.visit,fold,i+1,len(frame),'seconds',round(time.time()-start),flush=True)
    del model;torch.cuda.empty_cache()
    x=np.stack(bank).astype('float32');z=np.stack(logits).astype('float32')
    if not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('Nonfinite model output')
    return x,z


def neural(x,y,tr,va,te,method,a,path,fold):
    torch.manual_seed(SEED+fold);rng=np.random.default_rng(SEED+fold)
    center=x[tr].mean((0,1),dtype=np.float64);scale=x[tr].std((0,1),dtype=np.float64);scale[scale<1e-6]=1
    xx=torch.as_tensor(np.clip((x-center)/scale,-10,10).astype('float32'),device=a.device)
    yy=torch.as_tensor(y,dtype=torch.float32,device=a.device)
    if method != 'window_head':
        raise ValueError('Only the benchmark window head is supported')
    net=torch.nn.Sequential(torch.nn.LayerNorm(x.shape[-1]),torch.nn.Linear(x.shape[-1],1)).to(a.device)
    def forward(ids,train=False):
        return net(xx[ids]).squeeze(-1)
    def predict(ids):
        net.eval()
        with torch.inference_mode():return np.concatenate([forward(v).cpu().numpy() for v in np.array_split(ids,math.ceil(len(ids)/64))])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-3)
    weight=torch.tensor(float((y[tr]==0).sum()/(y[tr]==1).sum()),device=a.device)
    best=-np.inf;bad=0;bestep=0;state=None
    for ep in range(1,a.epochs+1):
        net.train()
        for ids in np.array_split(rng.permutation(tr),math.ceil(len(tr)/64)):
            opt.zero_grad(set_to_none=True);z=forward(ids,True);target=yy[ids,None].expand_as(z)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target,pos_weight=weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
        v=predict(va);score=roc_auc_score(np.repeat(y[va],x.shape[1]),v.ravel())
        if score>best+1e-5:best=float(score);bad=0;bestep=ep;state={k:v.cpu().clone() for k,v in net.state_dict().items()}
        else:bad+=1
        if bad>=a.patience:break
    net.load_state_dict(state);v=predict(va);s=predict(te)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=state,center=center,scale=scale,method=method,input_dim=x.shape[-1],best_epoch=bestep,validation_auroc=best,train_indices=tr,validation_indices=va,test_indices=te),path)
    net.load_state_dict(torch.load(path,map_location=a.device,weights_only=False)['state_dict'])
    np.testing.assert_allclose(s,predict(te),rtol=1e-5,atol=1e-6)
    allz=predict(np.arange(len(x)))
    del xx,net;torch.cuda.empty_cache()
    return s,v,dict(best_epoch=bestep,stopped_epoch=ep,validation_auroc=best),allz
