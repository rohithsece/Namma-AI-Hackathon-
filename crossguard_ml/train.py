"""OPTIONAL fine-tuning. Do not use until organisers permit weight modification.
The shipped model is the unchanged COCO-pretrained checkpoint, not this output.
"""
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from crossguard.dataset import CocoRoadDataset,collate
from crossguard.model import load_detector,WEIGHTS
from evaluate import evaluate

def main():
    """Execute supervised fine-tuning pipeline on custom labeled road imagery.
    
    Workflow:
    1. Validates CLI arguments and verifies non-empty output directory.
    2. Instantiates training and validation datasets using CocoRoadDataset.
    3. Runs SHA-256 image byte hash comparison to strictly prevent train/validation data leakage.
    4. Initializes SGD optimizer with momentum and Cosine Annealing learning rate schedule.
    5. Executes training loop over specified epochs, computing training loss and validation mAP.
    6. Saves checkpoint state dicts (last.pth, best.pth), training_state.pth, and history.json.
    """
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['train-images','train-annotations','val-images','val-annotations']: p.add_argument('--'+name,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--weights',type=Path,default=WEIGHTS); a=p.parse_args()
    if a.epochs<1 or a.batch_size<2 or a.lr<=0: p.error('epochs >=1, batch-size >=2 and lr >0 required (BatchNorm in SSD head)')
    if a.output.exists() and any(a.output.iterdir()): p.error('Output folder must be empty')
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(42); random.seed(42); np.random.seed(42)
    train=CocoRoadDataset(a.train_images,a.train_annotations,augment=True)
    val=CocoRoadDataset(a.val_images,a.val_annotations)
    def hashes(ds):
        return {hashlib.sha256((ds.root/m['file_name']).read_bytes()).hexdigest() for m in ds.coco.imgs.values()}
    if hashes(train)&hashes(val): raise ValueError('Data leakage: identical image bytes across train and validation')
    if len(train)<a.batch_size: raise ValueError('Training set smaller than batch size')
    loader=torch.utils.data.DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0,collate_fn=collate,drop_last=True)
    model=load_detector(a.weights).to(a.device)
    optimizer=torch.optim.SGD([x for x in model.parameters() if x.requires_grad],lr=a.lr,momentum=.9,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs)
    history=[]; best=-float('inf')
    for epoch in range(a.epochs):
        model.train(); total=0.; n=0
        for images,targets in loader:
            images=[im.to(a.device) for im in images]
            targets=[{k:v.to(a.device) for k,v in t.items()} for t in targets]
            losses=model(images,targets); loss=sum(losses.values())
            if not torch.isfinite(loss): raise ValueError(f'Non-finite loss: {losses}')
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step(); total+=float(loss.detach()); n+=1
        scheduler.step(); metrics=evaluate(model,val,a.device)
        if metrics['mAP_50_95']<0: raise ValueError('No evaluable validation positives')
        row={'epoch':epoch+1,'mean_training_loss':total/n,**metrics}; history.append(row); print(json.dumps(row))
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        torch.save(state,a.output/'last.pth')
        if metrics['mAP_50_95']>best: best=metrics['mAP_50_95']; torch.save(state,a.output/'best.pth')
        torch.save({'epoch':epoch+1,'model':state,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},a.output/'training_state.pth')
        (a.output/'history.json').write_text(json.dumps(history,indent=2))
    (a.output/'training_config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))

if __name__=='__main__': main()
