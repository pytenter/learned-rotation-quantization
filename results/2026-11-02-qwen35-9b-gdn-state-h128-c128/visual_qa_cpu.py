import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

root=Path(sys.argv[1]);out=root/'visual_qa';out.mkdir(exist_ok=False)
records=[]
for exp in ('A','B'):
    files=sorted((root/'figures'/exp).glob('*.png'))
    assert len(files)==73
    w,h=400,210
    sheet=Image.new('RGB',(w*4,h*math.ceil(len(files)/4)),(235,235,235))
    draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(files):
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            assert im.width>=1000 and im.height>=500
            records.append({'file':str(p.relative_to(root)),'width':im.width,'height':im.height,'bitmap_decode':'PASS'})
            copy=im.convert('RGB');copy.thumbnail((w-8,h-28))
            x=(i%4)*w;y=(i//4)*h
            sheet.paste(copy,(x+(w-copy.width)//2,y+24))
            draw.text((x+5,y+5),exp+'/'+p.name,fill='black')
    sheet.save(out/f'{exp}_all73_contact_sheet.png')
with (out/'BITMAP_CHECKS.json').open('x') as f:
    json.dump({'status':'PASS_BITMAP_DECODE_AND_DIMENSIONS','files':records,
               'scope':'all146 PNGs decoded; contact sheets for subsequent visual inspection'},f,indent=2)
print('146_BITMAPS_VERIFIED_CONTACT_SHEETS_READY',flush=True)
