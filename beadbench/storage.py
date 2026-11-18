import io,json,os,threading,uuid,hashlib,zipfile,copy
from pathlib import Path
import numpy as np
from image_lab.validation import Invalid,Conflict
from .core import valid_image,png,validate_pattern,summary,render

class Storage:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.document=self.root/'workspace.json'
        if not self.document.exists():self.write(dict(revision=0,images=[],patterns=[],notes=''))
    def read(self):return json.loads(self.document.read_text())
    def write(self,d):
        tmp=self.document.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));os.replace(tmp,self.document)
    def path(self,key):
        if not isinstance(key,str) or len(key)!=32 or any(x not in '0123456789abcdef' for x in key):raise Invalid('无效图像ID')
        return self.root/(key+'.npy')
    def image(self,key):
        with self.lock:
            data=self.read();row=next((x for x in data['images'] if x['id']==key),None)
            if row is None:raise Invalid('没有此照片')
            raw=self.path(key).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise Invalid('照片校验失败')
            return row,valid_image(np.load(io.BytesIO(raw),allow_pickle=False))
    def save_image(self,name,a):
        a=valid_image(a);out=io.BytesIO();np.save(out,a,allow_pickle=False);raw=out.getvalue();key=uuid.uuid4().hex
        row=dict(id=key,name=str(name)[:100],width=a.shape[1],height=a.shape[0],sha256=hashlib.sha256(raw).hexdigest())
        with self.lock:
            d=self.read();self.path(key).write_bytes(raw);d['images'].append(row);d['revision']+=1;self.write(d)
        return row
    def pattern(self,key):
        with self.lock:
            row=next((p for p in self.read()['patterns'] if p['id']==key),None)
            if row is None:raise Invalid('没有此图纸')
            return copy.deepcopy(validate_pattern(row))
    def save_pattern(self,p,source):
        q=copy.deepcopy(validate_pattern(p));q.update(id=uuid.uuid4().hex,revision=0,source=source)
        with self.lock:
            d=self.read();d['patterns'].append(q);d['revision']+=1;self.write(d)
        return q
    def change(self,key,revision,callback):
        with self.lock:
            d=self.read();p=next((p for p in d['patterns'] if p['id']==key),None)
            if p is None:raise Invalid('没有此图纸')
            if revision!=p['revision']:raise Conflict('图纸已改变，请刷新后重新编辑')
            callback(p);validate_pattern(p);p['revision']+=1;d['revision']+=1;self.write(d);return copy.deepcopy(p)
    def state(self):
        with self.lock:
            d=self.read();d['patterns']=[{k:v for k,v in p.items() if k not in ('cells','locks','targets')}|dict(summary=summary(p)) for p in d['patterns']];return d
    def notes(self,text,revision):
        with self.lock:
            d=self.read()
            if revision!=d['revision']:raise Conflict('工作区已改变，请刷新后重试')
            d['notes']=str(text)[:10000];d['revision']+=1;self.write(d);return d
    def export(self,key):
        p=self.pattern(key);out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('pattern.png',render(p));z.writestr('pattern.json',json.dumps(p,ensure_ascii=False,indent=2));z.writestr('materials.json',json.dumps(summary(p),ensure_ascii=False,indent=2))
            z.writestr('README.txt','BeadBench图纸。cells为0起始色板编号；targets是原始格采样RGB目标色，后续手工上色不改目标色；locks为人工锁定格，普通重配色会保留锁格当前色。PNG为屏幕图纸，不表示物理颜色标定。演示为合成图。此版本不按库存限制分配。')
        return out.getvalue()
