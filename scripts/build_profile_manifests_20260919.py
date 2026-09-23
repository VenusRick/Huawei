# -*- coding: utf-8 -*-
import json, random, re
from pathlib import Path
from collections import Counter, defaultdict

OUT=Path('output/final_profile_runtime_20260919')
SEED=20260919

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')

# ---------------- ISCXVPN VPN behavior ----------------
roots=[
 Path('/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01'),
 Path('/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02'),
]
def beh(name):
    s=name.lower()
    if 'chat' in s: return 'chat'
    if 'audio' in s or 'voipbuster' in s: return 'audio'
    if any(x in s for x in ['video','vimeo','youtube','netflix']): return 'video'
    return None

files=defaultdict(list)
for root in roots:
    for p in sorted(root.glob('*')):
        if p.suffix.lower() not in ('.pcap','.pcapng'): continue
        b=beh(p.name)
        if b: files[b].append(p)
print('VPN behavior files',{k:len(v) for k,v in files.items()})
brows=[]
for label in ['chat','audio','video']:
    fs=files[label][:]
    rng=random.Random(f'{SEED}:vpn:{label}');rng.shuffle(fs)
    nv=max(1,round(len(fs)*.30))
    # only train/validation; no new test is claimed in runtime integration
    val=set(fs[:nv])
    for p in fs:
        sp='validation' if p in val else 'train'
        # Dataset file is a controlled behavior capture; cover the full file.
        brows.append({
            'capture_id':p.stem,
            'trial_id':p.stem,
            'platform':'pc',
            'device_id':'iscxvpn-official',
            'app':'VPN',
            'behavior':label,
            'pcap':str(p),
            'label_source':'dataset_official',
            'behavior_start':0.0,
            'behavior_end':1.0e9,
            'network_env':'ISCXVPN2016',
            'terminal_ip':'',
            'split':sp,
        })
write_jsonl(OUT/'behavior/manifest.jsonl',brows)
print('behavior manifest',len(brows),Counter(x['split'] for x in brows),
      Counter((x['behavior'],x['split']) for x in brows))

# ---------------- ISCXTor tool binary ----------------
troot=Path('/workspace/datasets/ISCXTor2016/pcap')
trows=[]
for label,sub in [('Tor','Tor'),('NonTor','NonTor')]:
    fs=sorted([p for p in (troot/sub).glob('*') if p.suffix.lower() in ('.pcap','.pcapng')])
    rng=random.Random(f'{SEED}:tor:{label}');rng.shuffle(fs)
    nv=max(1,round(len(fs)*.30))
    val=set(fs[:nv])
    for p in fs:
        sp='validation' if p in val else 'train'
        trows.append({
            'capture_id':p.stem,
            'trial_id':p.stem,
            'platform':'pc',
            'device_id':'iscxtor-official',
            'app':label,
            'behavior':'anonymous' if label=='Tor' else 'normal',
            'pcap':str(p),
            'label_source':'dataset_official',
            'network_env':'ISCXTor2016',
            'terminal_ip':'',
            'split':sp,
        })
write_jsonl(OUT/'tunnel/manifest.jsonl',trows)
print('tunnel manifest',len(trows),Counter(x['split'] for x in trows),
      Counter((x['app'],x['split']) for x in trows))
