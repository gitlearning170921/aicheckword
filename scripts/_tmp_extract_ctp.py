import os, zipfile

d = r'C:\Users\yuwell\Downloads'
ref = None
for n in os.listdir(d):
    p = os.path.join(d, n)
    if p.endswith('.zip') and os.path.getsize(p) > 200000 and 'draft_886' not in n:
        ref = p
        break
print('using ref:', ref)
out_dir = os.path.join(d, '_extract_ctp')
os.makedirs(out_dir, exist_ok=True)
keys = [
    'CTP-001 呼吸数据管理软件与呼吸机兼容性测试方案',
    'CTP-001 呼吸数据管理软件与呼吸机兼容性测试用例表',
    'CTR-001 呼吸数据管理软件与呼吸机兼容性测试执行表',
]
files = []
with zipfile.ZipFile(ref) as zf:
    for info in zf.infolist():
        for s in keys:
            if s in info.filename:
                outp = os.path.join(out_dir, info.filename.replace('/', '__'))
                with zf.open(info) as src, open(outp, 'wb') as dst:
                    dst.write(src.read())
                files.append(outp)
                print('extracted', info.filename, info.file_size)
print('saved:', files)
