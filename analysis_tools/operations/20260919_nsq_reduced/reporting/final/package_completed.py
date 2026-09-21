"""Embed report figures and pre-rendered formula SVGs for offline sharing."""
from pathlib import Path
from lxml import html, etree
from PIL import Image
import ast, base64, hashlib, json, mimetypes, argparse

def main(out):
    out=Path(out).resolve()
    old=Path('/Users/heejoon/Documents/ChatGPT/OptiQ/reports/20260918_offline_reports/build_offline.py')
    styles={}
    for node in ast.parse(old.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['CSS','JS'] for t in node.targets):
            styles[node.targets[0].id]=ast.literal_eval(node.value)
    dom=html.fragment_fromstring((out/'report_fragment.html').read_text(),create_parent='div')
    def uri(p):
        return 'data:'+(mimetypes.guess_type(p.name)[0] or 'application/octet-stream')+';base64,'+base64.b64encode(p.read_bytes()).decode()
    images=[]
    for img in dom.xpath('.//img'):
        p=out/img.get('src'); assert p.is_file(),p
        img.set('src',uri(p));img.set('loading','lazy');img.set('decoding','async')
        with Image.open(p) as im:img.set('width',str(im.width));img.set('height',str(im.height))
        images.append({'path':str(p.relative_to(out)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for a in dom.xpath('.//a[@href]'):
        href=a.get('href')
        if href.startswith(('http:','https:','#','data:')):continue
        p=out/href; assert p.is_file(),p
        a.set('href',uri(p));a.set('download',p.name)
    toc=[]
    for i,h in enumerate(dom.xpath('.//h2')):
        h.set('id',f'section-{i}');toc.append(f'<li><a href="#section-{i}">{h.text_content()}</a></li>')
    for table in dom.xpath('.//table'):
        parent=table.getparent();idx=parent.index(table);parent.remove(table)
        wrapper=etree.Element('div',{'class':'table-wrap'});wrapper.append(table);parent.insert(idx,wrapper)
    content=etree.tostring(dom,encoding='unicode',method='html')
    doc='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Non-stationary Q · 완료된 1,823개 실험</title><style>'+styles['CSS']+'details>summary{cursor:pointer;font-weight:600;padding:14px;background:#f1f5fa;border-radius:6px;margin:10px 0}</style></head><body><main><p class="offline-note">그림·수식·연결된 표가 모두 이 파일에 포함된 오프라인 보고서입니다. 그림을 클릭하면 확대됩니다. 2026-09-19 범위 축소 시점에 완료된 1,823개 비교 실험의 분석입니다.</p><nav class="toc"><strong>목차</strong><ol>'+''.join(toc)+'</ol></nav>'+content+'</main><dialog id="figure-viewer"><button onclick="closeFigure()">닫기 ×</button><img alt="" onclick="closeFigure()"></dialog><script>'+styles['JS']+'</script></body></html>'
    (out/'report.html').write_text(doc)
    manifest={'images':images,'image_count':len(images),'html_bytes':len(doc.encode()),'html_sha256':hashlib.sha256(doc.encode()).hexdigest(),'formula_count':doc.count('<mjx-container')}
    (out/'OFFLINE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='images'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');main(p.parse_args().out)
