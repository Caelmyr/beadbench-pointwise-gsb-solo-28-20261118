import io
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from image_lab.validation import Invalid

DEFAULT_PALETTE=['#f4f0df','#202c46','#d64d52','#e8aa4b','#4e8d68','#5f91b5','#8b659d','#a76c4d']

def integer(value,name,low,high):
    try:x=float(value)
    except (ValueError,TypeError):raise Invalid(name+'必须为整数')
    if not np.isfinite(x) or int(x)!=x or not low<=x<=high:raise Invalid(name+f'需为{low}至{high}的整数')
    return int(x)

def number(value,name,low,high):
    try:x=float(value)
    except (ValueError,TypeError):raise Invalid(name+'必须为数值')
    if not np.isfinite(x) or not low<=x<=high:raise Invalid(name+'超出范围')
    return x

def palette(value):
    if not isinstance(value,list) or not 2<=len(value)<=8:raise Invalid('色板需为2至8个不同的#RRGGBB颜色')
    out=[]
    for c in value:
        if not isinstance(c,str) or len(c)!=7 or c[0]!='#':raise Invalid('色板颜色需为#RRGGBB')
        try:int(c[1:],16)
        except ValueError:raise Invalid('无效十六进制颜色')
        out.append(c.lower())
    if len(set(out))!=len(out):raise Invalid('色板颜色不可重复')
    return out

def palette_rgb(colors):return np.asarray([[int(c[i:i+2],16)/255 for i in (1,3,5)] for c in palette(colors)],np.float32)

def valid_image(a):
    a=np.asarray(a,dtype=np.float32)
    if a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<2 or max(a.shape[:2])>640 or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise Invalid('图像需为2至640边长的有限0至1 RGB')
    return a

def decode(raw):
    if len(raw)>8*1024*1024:raise Invalid('文件最多8MB')
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if max(im.size)>640 or min(im.size)<2:raise Invalid('图像边长需为2至640像素')
            im=ImageOps.exif_transpose(im).convert('RGBA');a=np.asarray(im,np.float32)/255
        return valid_image(a[...,:3]*a[...,3:]+1-a[...,3:])
    except Invalid:raise
    except Exception as e:raise Invalid('无法读取图像') from e

def png(a):
    out=io.BytesIO();Image.fromarray(np.rint(np.clip(a,0,1)*255).astype('uint8')).save(out,format='PNG');return out.getvalue()

def demo():
    im=Image.new('RGB',(256,256),'#f4f0df');d=ImageDraw.Draw(im)
    d.ellipse((44,30,212,208),fill='#e8aa4b');d.ellipse((62,43,195,194),fill='#d64d52')
    d.rectangle((110,155,148,236),fill='#4e8d68');d.polygon([(116,190),(54,166),(70,209),(116,221)],fill='#4e8d68')
    d.ellipse((89,91,103,105),fill='#202c46');d.ellipse((153,91,167,105),fill='#202c46');d.arc((98,96,159,143),0,180,fill='#202c46',width=5)
    d.rectangle((10,10,23,23),fill='#5f91b5');d.rectangle((233,232,246,245),fill='#8b659d')
    return np.asarray(im,np.float32)/255

def sample(image,width,height,method):
    width=integer(width,'列数',4,32);height=integer(height,'行数',4,32)
    if method not in ('area','nearest'):raise Invalid('采样需为area或nearest')
    import cv2
    return cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA if method=='area' else cv2.INTER_NEAREST).astype(np.float32)

def assign(targets,colors):
    p=palette_rgb(colors);a=np.asarray(targets,np.float32)
    return np.argmin(np.sum((a[:,:,None,:]-p[None,None,:,:])**2,axis=-1),axis=-1).astype(int)

def validate_pattern(p):
    colors=palette(p['palette']);w=integer(p['width'],'列数',4,32);h=integer(p['height'],'行数',4,32)
    a=np.asarray(p['cells']);locks=np.asarray(p['locks']);targets=np.asarray(p['targets'],np.float32)
    if a.shape!=(h,w) or locks.shape!=(h,w) or targets.shape!=(h,w,3):raise Invalid('图纸数组尺寸不一致')
    if not np.issubdtype(a.dtype,np.integer) or np.any(a<0) or np.any(a>=len(colors)):raise Invalid('格子的颜色编号不在色板内')
    if locks.dtype!=np.bool_:raise Invalid('锁格状态需为布尔值')
    if not np.isfinite(targets).all() or np.any(targets<0) or np.any(targets>1):raise Invalid('采样目标色无效')
    return p

def create_pattern(image,body):
    colors=palette(body.get('palette',DEFAULT_PALETTE));targets=sample(image,body.get('width',24),body.get('height',24),body.get('sampling','area'));h,w=targets.shape[:2]
    return validate_pattern(dict(name=str(body.get('name','新图纸'))[:100],width=w,height=h,palette=colors,targets=targets.tolist(),cells=assign(targets,colors).tolist(),locks=np.zeros((h,w),bool).tolist(),sampling=body.get('sampling','area'),notes=''))

def summary(p):
    validate_pattern(p);cells=np.asarray(p['cells']);targets=np.asarray(p['targets']);render=palette_rgb(p['palette'])[cells];counts=np.bincount(cells.ravel(),minlength=len(p['palette']))
    return dict(total=p['width']*p['height'],locked=int(np.sum(p['locks'])),colors=[dict(index=i,symbol=chr(65+i),color=c,count=int(counts[i])) for i,c in enumerate(p['palette'])],squared_rgb_error=float(np.sum((targets-render)**2)),mean_squared_rgb_error=float(np.mean((targets-render)**2)))

def render(p,cell=20,symbols=True,grid=True):
    validate_pattern(p);cell=integer(cell,'格子像素',8,40);w,h=p['width'],p['height'];im=Image.new('RGB',(w*cell,h*cell),'white');d=ImageDraw.Draw(im)
    for y in range(h):
        for x in range(w):
            idx=p['cells'][y][x];color=p['palette'][idx];x0,y0=x*cell,y*cell;d.rectangle((x0,y0,x0+cell-1,y0+cell-1),fill=color)
            if grid:d.rectangle((x0,y0,x0+cell-1,y0+cell-1),outline='#a5a5a5',width=1)
            if symbols:
                rgb=palette_rgb([color,'#000000' if color!='#000000' else '#ffffff'])[0];ink='#ffffff' if float(np.mean(rgb))<.48 else '#111111'
                d.text((x0+cell*.3,y0+cell*.2),chr(65+idx),fill=ink)
            if p['locks'][y][x]:d.rectangle((x0+2,y0+2,x0+cell-3,y0+cell-3),outline='#00ccff',width=2)
    out=io.BytesIO();im.save(out,format='PNG');return out.getvalue()

def edit_cells(p,body):
    mode=body.get('mode','paint');changes=body.get('changes')
    if mode not in ('paint','lock','unlock'):raise Invalid('编辑模式无效')
    if not isinstance(changes,list) or not 1<=len(changes)<=1024:raise Invalid('一次编辑需为1至1024格')
    index=integer(body.get('color',0),'色号',0,len(p['palette'])-1)
    coords=[]
    for c in changes:
        if not isinstance(c,list) or len(c)!=2:raise Invalid('格坐标需为[x,y]')
        coords.append((integer(c[0],'列',0,p['width']-1),integer(c[1],'行',0,p['height']-1)))
    for x,y in coords:
        if mode=='paint':p['cells'][y][x]=index
        else:p['locks'][y][x]=mode=='lock'
    return validate_pattern(p)

def transform(p,operation,body):
    import copy
    q=copy.deepcopy(p);cells=np.asarray(q['cells']);locks=np.asarray(q['locks']);targets=np.asarray(q['targets'],np.float32)
    if operation=='nearest':
        new=assign(targets,q['palette']);q['cells']=np.where(locks,cells,new).tolist()
    elif operation in ('mirror','rotate'):
        apply=(lambda a:np.flip(a,axis=1)) if operation=='mirror' else (lambda a:np.rot90(a,k=-1))
        q['cells']=apply(cells).tolist();q['locks']=apply(locks).tolist();q['targets']=apply(targets).tolist();q['height'],q['width']=np.asarray(q['cells']).shape
    elif operation in ('brightness','contrast'):
        amount=number(body.get('amount',1.1),'强度',.2,2)
        targets=np.clip(targets*amount if operation=='brightness' else (targets-.5)*amount+.5,0,1)
        q['targets']=targets.tolist();new=assign(targets,q['palette']);q['cells']=np.where(locks,cells,new).tolist()
    elif operation=='grayscale':
        gray=targets@np.array([.2126,.7152,.0722]);targets=np.repeat(gray[:,:,None],3,axis=2);q['targets']=targets.tolist();q['cells']=np.where(locks,cells,assign(targets,q['palette'])).tolist()
    elif operation=='invert':
        targets=1-targets;q['targets']=targets.tolist();q['cells']=np.where(locks,cells,assign(targets,q['palette'])).tolist()
    elif operation=='clear_locks':q['locks']=np.zeros_like(locks).tolist()
    else:raise Invalid('未知普通处理')
    q['name']=str(body.get('name',p['name']+' · '+operation))[:100];return validate_pattern(q)
