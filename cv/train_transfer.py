"""Higher-quality BUSI classifier using ImageNet-pretrained ResNet18.
Run: python cv/train_transfer.py --data /path/to/Dataset_BUSI_with_GT --epochs 12
Requires internet once to download ImageNet weights unless they are already cached.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix

CLASSES=['benign','malignant','normal']; SEED=42
class DS(Dataset):
 def __init__(self,items,train=False,size=224):
  self.items=items; self.tf=transforms.Compose(([transforms.Resize((size,size)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.RandomAffine(0,translate=(.03,.03))] if train else [transforms.Resize((size,size))])+[transforms.ToTensor(),transforms.Normalize([.485]*3,[.229,.224,.225])])
 def __len__(self): return len(self.items)
 def __getitem__(self,i):
  p,y=self.items[i]; im=Image.open(p).convert('RGB'); return self.tf(im),y

def collect(root):
 out=[]
 for y,c in enumerate(CLASSES):
  d=Path(root)/c
  for p in d.iterdir():
   if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'} and '_mask' not in p.stem: out.append((str(p),y))
 return out

def ev(m,ld,dev):
 m.eval(); ys=[];ps=[]
 with torch.no_grad():
  for x,y in ld: ps.extend(m(x.to(dev)).argmax(1).cpu().numpy()); ys.extend(y.numpy())
 pr,re,f1,_=precision_recall_fscore_support(ys,ps,labels=[0,1,2],zero_division=0)
 return {'accuracy':float(accuracy_score(ys,ps)),'precision_macro':float(pr.mean()),'recall_macro':float(re.mean()),'f1_macro':float(f1.mean()),'confusion_matrix':confusion_matrix(ys,ps,labels=[0,1,2]).tolist(),'n':len(ys)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--output',default='cv');a=ap.parse_args()
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 items=collect(a.data); idx=np.arange(len(items)); y=np.array([v for _,v in items]); tr,tmp=train_test_split(idx,test_size=.30,stratify=y,random_state=SEED);va,te=train_test_split(tmp,test_size=.50,stratify=y[tmp],random_state=SEED)
 out=Path(a.output);out.mkdir(exist_ok=True);json.dump({'train':[items[i] for i in tr],'validation':[items[i] for i in va],'test':[items[i] for i in te]},open(out/'transfer_split.json','w'),indent=2)
 try: weights=ResNet18_Weights.DEFAULT; net=resnet18(weights=weights)
 except Exception as e: raise RuntimeError('ImageNet weights could not be loaded. Connect to the internet once and rerun, or pre-cache ResNet18 weights.') from e
 net.fc=nn.Linear(net.fc.in_features,3);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net.to(dev)
 loaders={k:DataLoader(DS([items[i] for i in ids],k=='train'),batch_size=a.batch_size,shuffle=k=='train',num_workers=0) for k,ids in [('train',tr),('val',va),('test',te)]}
 opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4);lossfn=nn.CrossEntropyLoss();best=-1;beststate=None;history=[]
 for ep in range(1,a.epochs+1):
  net.train();tl=0
  for x,t in loaders['train']: opt.zero_grad();z=net(x.to(dev));loss=lossfn(z,t.to(dev));loss.backward();opt.step();tl+=loss.item()*len(t)
  vm=ev(net,loaders['val'],dev);history.append({'epoch':ep,'train_loss':tl/len(tr),**vm});print(f'epoch {ep}/{a.epochs} val_acc={vm["accuracy"]:.4f} val_f1={vm["f1_macro"]:.4f}')
  if vm['f1_macro']>best:best=vm['f1_macro'];beststate={k:v.cpu().clone() for k,v in net.state_dict().items()}
 net.load_state_dict(beststate);test=ev(net,loaders['test'],dev)
 torch.save({'state_dict':net.state_dict(),'class_names':CLASSES,'image_size':224,'architecture':'resnet18_imagenet'},out/'cv_model.pt')
 json.dump({'dataset':'BUSI Breast Ultrasound Images','total_images':len(items),'split':{'train':len(tr),'validation':len(va),'test':len(te)},'classes':CLASSES,'test_metrics':test,'history':history},open(out/'cv_metrics.json','w'),indent=2)
 print(json.dumps(test,indent=2))
if __name__=='__main__':main()
