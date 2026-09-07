"""Train a compact breast-ultrasound CNN on BUSI.

Usage:
  python cv/train_cv.py --data /path/to/Dataset_BUSI_with_GT

The script ignores BUSI segmentation masks (_mask*.png), performs a stratified
train/validation/test split, saves the model and metrics, and never uses test
images for augmentation/training.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

CLASS_NAMES = ["benign", "malignant", "normal"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SEED = 42

class BreastUltrasoundDataset(Dataset):
    def __init__(self, items, train=False, image_size=128):
        self.items = items; self.train = train; self.image_size = image_size
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        if self.train:
            if random.random() < 0.5: x = np.fliplr(x).copy()
            if random.random() < 0.25:
                x = np.clip(x * random.uniform(0.85, 1.15), 0, 1)
        x = torch.from_numpy(x).unsqueeze(0)
        x = (x - 0.5) / 0.5
        return x, label

class BreastCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(96, n_classes))
    def forward(self, x): return self.classifier(self.features(x).flatten(1))

def collect(data_root):
    root = Path(data_root)
    items=[]
    for idx, cls in enumerate(CLASS_NAMES):
        folder=root/cls
        if not folder.exists(): raise FileNotFoundError(f"Missing class folder: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS and "_mask" not in p.stem:
                items.append((str(p), idx))
    if not items: raise ValueError("No original images found. BUSI masks are ignored.")
    return items

def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            z=model(x.to(device)); p=torch.softmax(z,1).cpu().numpy()
            probs.append(p); ps.extend(p.argmax(1)); ys.extend(y.numpy())
    probs=np.concatenate(probs); acc=accuracy_score(ys,ps)
    pr,re,f1,_=precision_recall_fscore_support(ys,ps,labels=[0,1,2],zero_division=0)
    return {"accuracy":float(acc),"precision_macro":float(pr.mean()),"recall_macro":float(re.mean()),"f1_macro":float(f1.mean()),"confusion_matrix":confusion_matrix(ys,ps,labels=[0,1,2]).tolist(),"n":len(ys)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--image-size',type=int,default=128); ap.add_argument('--output',default='cv'); args=ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    items=collect(args.data); paths=[x[0] for x in items]; labels=[x[1] for x in items]
    tr, tmp = train_test_split(np.arange(len(items)), test_size=0.30, stratify=labels, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=np.array(labels)[tmp], random_state=SEED)
    train_items=[items[i] for i in tr]; val_items=[items[i] for i in va]; test_items=[items[i] for i in te]
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    for name,subset in [('train',train_items),('val',val_items),('test',test_items)]:
        (out/f'{name}_files.json').write_text(json.dumps(subset,indent=2),encoding='utf-8')
    loaders={
      'train':DataLoader(BreastUltrasoundDataset(train_items,True,args.image_size),batch_size=args.batch_size,shuffle=True,num_workers=0),
      'val':DataLoader(BreastUltrasoundDataset(val_items,False,args.image_size),batch_size=args.batch_size,shuffle=False,num_workers=0),
      'test':DataLoader(BreastUltrasoundDataset(test_items,False,args.image_size),batch_size=args.batch_size,shuffle=False,num_workers=0)}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=BreastCNN(3).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss(); best=-1; best_state=None; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0
        for x,y in loaders['train']:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); total+=float(loss.item())*len(y)
        vm=evaluate(model,loaders['val'],device); history.append({'epoch':epoch,'train_loss':total/len(train_items),**vm})
        print(f"epoch {epoch:02d}/{args.epochs} loss={total/len(train_items):.4f} val_acc={vm['accuracy']:.4f} val_f1={vm['f1_macro']:.4f}")
        if vm['f1_macro']>best: best=vm['f1_macro']; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); test_metrics=evaluate(model,loaders['test'],device)
    torch.save({'state_dict':model.state_dict(),'class_names':CLASS_NAMES,'image_size':args.image_size,'normalization':{'mean':0.5,'std':0.5},'architecture':'BreastCNN-v1'},out/'cv_model.pt')
    meta={'dataset':'BUSI Breast Ultrasound Images','total_images':len(items),'split':{'train':len(tr),'validation':len(va),'test':len(te)},'classes':CLASS_NAMES,'seed':SEED,'device':str(device),'test_metrics':test_metrics,'history':history}
    (out/'cv_metrics.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'test_metrics':test_metrics,'classes':CLASS_NAMES},indent=2))
if __name__=='__main__': main()
