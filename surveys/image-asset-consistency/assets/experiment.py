"""Real-image encoder and local-preservation diagnostic, not generation reproduction."""
from pathlib import Path
import json,hashlib,platform,time,io
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance,ImageOps
import torch,transformers
from transformers import AutoImageProcessor,AutoModel,CLIPImageProcessor,CLIPModel

R=Path('/tmp/image-consistency-survey-work')
O=R/'experiment';O.mkdir(exist_ok=True)
torch.set_num_threads(4)
keys=['live_subject/animal/00','live_subject/animal/01','live_subject/human/00',
      'live_subject/human/01','object/00','object/01','style/00','style/01']
items=[]
for key in keys:
    path=R/'data/images'/f'{key}.jpg'
    original=Image.open(path).convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
    arr=np.asarray(original).copy()
    # A known fixed 64x64 edit rectangle; this is not a semantic segmentation mask.
    mask=np.zeros((512,512),dtype=bool);mask[224:288,224:288]=True
    edited=arr.copy();edited[mask]=np.array([240,30,30],dtype=np.uint8)
    edited_image=Image.fromarray(edited)
    jpeg=io.BytesIO();edited_image.save(jpeg,format='JPEG',quality=90)
    jpeg_image=Image.open(io.BytesIO(jpeg.getvalue())).convert('RGB')
    soft=arr.copy()
    outer=np.zeros_like(mask);outer[210:302,210:302]=True
    soft[outer]=np.round(0.95*soft[outer].astype(float)+0.05*np.array([240,30,30])).astype(np.uint8)
    soft[mask]=[240,30,30]
    wrong=Image.fromarray(np.asarray(original)[:,:,[2,1,0]].copy())
    variants={'original':original,'brightness_0.7':ImageEnhance.Brightness(original).enhance(0.7),
              'horizontal_flip':ImageOps.mirror(original),'grayscale':ImageOps.grayscale(original).convert('RGB'),
              '64px_red_patch':edited_image,'red_patch_JPEG90':jpeg_image,
              'soft_edit_92px':Image.fromarray(soft),'RGB_to_BGR':wrong}
    stem=key.replace('/','-')
    for name,img in variants.items():img.save(O/f'{stem}-{name}.png')
    outside=[]
    for name,img in variants.items():
        v=np.asarray(img).astype(np.int16);diff=np.abs(v-arr.astype(np.int16))
        outside.append({'variant':name,'outside_MAE_0_255':float(diff[~mask].mean()),
                        'outside_changed_pixel_fraction':float(np.any(diff,axis=2)[~mask].mean()),
                        'outside_max_error_0_255':int(diff[~mask].max())})
    items.append({'key':key,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'variants':variants,'outside':outside})
print('Prepared',len(items),'real references,',sum(len(x['variants']) for x in items),'actual inputs',flush=True)
results={'scope':'Actual frozen vision encoders on 8 deterministic selected DreamBench++ references; not image generation, not identity accuracy',
         'dataset_url':'https://huggingface.co/datasets/yuangpeng/dreambench_plus',
         'dataset_zip_sha256':hashlib.sha256((R/'data.zip').read_bytes()).hexdigest(),
         'resize':'All reference images resized to 512x512 using Pillow LANCZOS; may distort aspect ratio; controlled diagnostic only',
         'mask_bbox_xyxy':[224,224,288,288],'mask_pixels':4096,'outside_pixels':512*512-4096,
         'hardware':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__,
         'precision':'CPU float32, eval/no_grad, 4 torch threads','models':[],'references':[]}
for name,repo in [('clip','openai/clip-vit-base-patch32'),('dino','facebook/dino-vits16')]:
    meta=json.loads((R/f'{name}-model-meta.json').read_text());revision=meta['sha']
    print('Loading',repo,revision,flush=True)
    if name=='clip':
        proc=CLIPImageProcessor.from_pretrained(repo,revision=revision,cache_dir=str(R/'hf-cache'))
        model=CLIPModel.from_pretrained(repo,revision=revision,cache_dir=str(R/'hf-cache')).eval().float()
    else:
        proc=AutoImageProcessor.from_pretrained(repo,revision=revision,cache_dir=str(R/'hf-cache'),use_fast=False)
        model=AutoModel.from_pretrained(repo,revision=revision,cache_dir=str(R/'hf-cache')).eval().float()
    vectors=[];mapping=[]
    started=time.perf_counter()
    for item in items:
        ims=list(item['variants'].values())
        data=proc(images=ims,return_tensors='pt')
        with torch.no_grad():
            if name=='clip':v=model.get_image_features(**data)
            else:v=model(**data).last_hidden_state[:,0]
            v=torch.nn.functional.normalize(v,dim=-1)
        vectors.append(v);mapping.append({k:float((v[0]*v[i]).sum()) for i,k in enumerate(item['variants'])})
    ref=torch.stack([v[0] for v in vectors])
    matrix=(ref@ref.T).tolist()
    rec={'name':name,'repo':repo,'revision':revision,'preprocessor':proc.to_dict(),
         'similarities':mapping,'reference_pair_cosine':matrix,'elapsed_seconds':time.perf_counter()-started}
    results['models'].append(rec)
    print(name,'computed',mapping,flush=True)
    del model
for item in items:
    results['references'].append({k:v for k,v in item.items() if k!='variants'})
(O/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
for start in [0,4]:
    sheet=Image.new('RGB',(1920,1180),'#c7cbd0');draw=ImageDraw.Draw(sheet)
    for row,item in enumerate(items[start:start+4]):
        for col,(name,img) in enumerate(item['variants'].items()):
            im=img.resize((220,220),Image.Resampling.LANCZOS)
            x,y=col*240+10,row*295+38
            sheet.paste(im,(x,y));draw.text((x,y-27),item['key']+' / '+name,fill='black')
    sheet.save(O/f'variants-{start}.png')
print('Saved',O/'results.json',flush=True)
