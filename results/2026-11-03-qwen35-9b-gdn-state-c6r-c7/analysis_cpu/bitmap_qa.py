import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,jd,sha

def main():
    out=ROOT/'visual_qa';out.mkdir(exist_ok=False)
    files=sorted((ROOT/'figures').rglob('*.png'));assert len(files)==100
    checks=[]
    for p in files:
        with Image.open(p) as im:
            im.load();width,height=im.size
            assert width>=1800 and height>=600,(p,width,height)
            assert im.mode in ['RGB','RGBA']
        checks.append({'path':str(p.relative_to(ROOT)),'width':width,'height':height,'sha256':sha(p)})
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    except OSError:font=ImageFont.load_default()
    for exp in ['A','B']:
        paths=sorted((ROOT/'figures'/exp).glob('*.png'));assert len(paths)==50
        contact=Image.new('RGB',(2000,2800),'white');draw=ImageDraw.Draw(contact)
        for i,path in enumerate(paths):
            x=(i%5)*400;y=(i//5)*280
            with Image.open(path) as image:
                thumb=image.convert('RGB');thumb.thumbnail((390,246))
                contact.paste(thumb,(x+(400-thumb.width)//2,y+25+(248-thumb.height)//2))
            draw.text((x+5,y+5),exp+' '+path.name,fill='black',font=font)
        contact.save(out/f'{exp}_all50_contact_sheet.png')
    jd(out/'BITMAP_CHECKS.json',{'status':'PASS','PNG_count':100,'files':checks,
        'checks':'all files fully decoded; dimensions/mode valid; contact sheets generated; visual layout still independently inspected'})
    print('BITMAP_QA_PASS_100_PNG',flush=True)

if __name__=='__main__':main()
