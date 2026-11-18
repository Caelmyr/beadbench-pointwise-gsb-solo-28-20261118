import io,json,os,threading,uuid,hashlib,zipfile,copy
from pathlib import Path
import numpy as np
from image_lab.validation import Invalid,Conflict
from .core import valid_image,png,validate_pattern,summary,render
from .plan import validate_plan,plan_summary,render_plan

class Storage:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.document=self.root/'workspace.json'
        if not self.document.exists():self.write(dict(revision=0,images=[],patterns=[],plans=[],notes=''))
    def read(self):
        d=json.loads(self.document.read_text())
        d.setdefault('plans',[])
        return d
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
    def plan(self,key):
        with self.lock:
            row=next((p for p in self.read()['plans'] if p['id']==key),None)
            if row is None:raise Invalid('没有此排料')
            return copy.deepcopy(validate_plan(row))
    def save_plan(self,pl,source):
        q=copy.deepcopy(validate_plan(pl));q.update(id=uuid.uuid4().hex,revision=0,plan_source=source)
        with self.lock:
            d=self.read();d['plans'].append(q);d['revision']+=1;self.write(d)
        return q
    def change_plan(self,key,revision,callback):
        with self.lock:
            d=self.read();p=next((p for p in d['plans'] if p['id']==key),None)
            if p is None:raise Invalid('没有此排料')
            if revision!=p['revision']:raise Conflict('排料已改变，请刷新后重新编辑')
            callback(p);validate_plan(p);p['revision']+=1;d['revision']+=1;self.write(d);return copy.deepcopy(p),None
    def recompute_plan(self,key,revision,callback):
        """重算专用：即使回调判定不可行，也把失败尝试（含库存/方式/原因）持久化，格内容保持上次有效结果。"""
        with self.lock:
            d=self.read();p=next((p for p in d['plans'] if p['id']==key),None)
            if p is None:raise Invalid('没有此排料')
            if revision!=p['revision']:raise Conflict('排料已改变，请刷新后重新编辑')
            error=None
            try:
                callback(p)
            except Invalid as e:
                error=str(e)
            validate_plan(p);p['revision']+=1;d['revision']+=1;self.write(d)
            return copy.deepcopy(p),error
    def state(self):
        with self.lock:
            d=self.read();d['patterns']=[{k:v for k,v in p.items() if k not in ('cells','locks','targets')}|dict(summary=summary(p)) for p in d['patterns']]
            d['plans']=[{k:v for k,v in p.items() if k not in ('cells','locks','targets','original_cells')}|dict(summary=plan_summary(p)) for p in d['plans']]
            return d
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
    def export_plan(self,key):
        p=self.plan(key);s=plan_summary(p);out=io.BytesIO()
        readme=(
            'BeadBench库存排料结果。\n'
            f"来源图纸：{p['source'].get('pattern_name','')}（编号{p['source'].get('pattern_id','')}，修订{p['source'].get('pattern_revision')}）\n"
            'cells中-1表示明确留空（无豆，不计物料，不冒充任何色号）；0起始为色板编号。\n'
            'original_cells是首次选中图纸时的原配色；targets是该图纸的采样RGB目标色，重算始终以它们为准，替代色和空格不会成为新原图。\n'
            'stocks为各色库存；locks为人工锁定格，锁定即按当前色占料，空格必须先上色才能锁。\n'
            'mode=fill表示排满允许换色（未锁格在库存内最小化目标RGB平方误差）；mode=keep表示保原配色缺料留空。\n'
            'runs记录每次排料尝试的库存、方式、成败与原因。PNG为屏幕图纸，不表示物理颜色标定。\n'
        )
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('plan.png',render_plan(p))
            z.writestr('plan.json',json.dumps(p,ensure_ascii=False,indent=2))
            z.writestr('materials.json',json.dumps(s,ensure_ascii=False,indent=2))
            z.writestr('README.txt',readme)
        return out.getvalue()
