"""检查采集视频: ten diverse current-expert episodes and front-side follow videos.

CPU only by default. Reuses completed traces in the same fingerprinted output;
never drops failures or resamples according to task outcomes.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import run_bottle as B
from audit_visual_case import audit_case
from collect_visual_cohort import EXPERT_RUN_OPTIONS, STRATUM_ORDER, expert_config
from probe_visual_expert import atomic_json
from run_visual_bottle import run_case
from visual_follow_video import render_case, heading
from visual_sampling import DEFAULT_PROFILE, PROFILES, profile_edges, sample_reset
from visual_scene import BASE_SCENE, STRATA, make_sim, scene_directory
from visual_sensor import BottleProxySensor

ROOT=Path(__file__).resolve().parents[3]
FEATURES=('bottle_along_m','inboard_m','spawn_edge_m','spawn_lat_m','yaw_error_deg')
PLAN_VERSION='coverage_bins5_farthest_v1'


def fingerprint(profile):
    folder=Path(__file__).parent
    files=set(folder.glob('visual_*.py'))-{folder/'visual_follow_video.py'}
    files.update(folder/n for n in ('run_visual_bottle.py','collect_visual_cohort.py','run_bottle.py',
        'run_spike.py','run_bend.py','bfm_policy.py','bottle_probe.py','probe_visual_expert.py','audit_visual_case.py'))
    files.update(Path(p) for p in (B.R.CKPT,B.R.FK_XML,B.R.MODE_TABLE,B.R.META))
    files.update(BASE_SCENE/n for n in ('scene_locomani.xml','scene_meta.json'))
    tree=ET.parse(BASE_SCENE/'scene_locomani.xml');compiler=tree.find('compiler')
    for element in tree.iter():
        if not element.get('file'):continue
        file=Path(element.get('file'))
        if not file.is_absolute():
            attribute='meshdir' if element.tag=='mesh' else 'texturedir'
            prefix=compiler.get(attribute,'') if compiler is not None else ''
            file=BASE_SCENE/prefix/file
        if file.is_file():files.add(file.resolve())
    hashes={str(p.resolve()):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sorted(files)}
    spec=dict(scene_profile=profile,expert_config=asdict(expert_config()),run_options=EXPERT_RUN_OPTIONS,
              source_sha256=hashes,torch=torch.__version__,strata=STRATA,plan_version=PLAN_VERSION,
              scene_mode=B.SCENE_MODE,policy_device='cpu',episodes=10,policy_seed=17)
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    return digest,spec


def choose_cases(profile, candidate_count=6000):
    """Select by reset parameters only; no model, rendering or physics calls."""
    meta=json.loads((BASE_SCENE/'scene_meta.json').read_text())
    source=np.array(meta['objects']['water_bottle']['init_pos'],dtype=float)
    B.STAND_D=(expert_config().stand_distance,)*2
    edges=profile_edges(profile)
    candidates={};raws={};normalised={}
    for s,stratum in enumerate(STRATUM_ORDER):
        records=[];features=[]
        for j in range(candidate_count):
            seed=100000+s*candidate_count+j;edge=edges[j%len(edges)]
            os.environ['BFM_EDGE']=edge;B.EDGE_INBOARD=STRATA[stratum][1]
            bottle0=source.copy();bottle0[2]+=STRATA[stratum][0]-meta['surfaces']['A']['top']
            sim=SimpleNamespace(crate0=bottle0,smeta=meta)
            bottle,_,spawn,yaw,scene=sample_reset(B,sim,np.random.default_rng(seed),edge,stratum,profile)
            n=np.zeros(2);axis=0 if edge[0]=='x' else 1;n[axis]=1 if edge[1]=='-' else -1
            left=np.array([-n[1],n[0]])
            delta=bottle[:2]-spawn
            yawerr=np.arctan2(np.sin(yaw-np.arctan2(delta[1],delta[0])),np.cos(yaw-np.arctan2(delta[1],delta[0])))
            v=[float((bottle[:2]-meta['surfaces']['A']['xy'])@left),scene['inboard'],scene['spawn_edge'],scene['spawn_lat'],float(np.degrees(yawerr))]
            records.append(dict(seed=seed,edge=edge,stratum=stratum,parameters=dict(zip(FEATURES,v)),
                                bottle_xyz=bottle.tolist(),spawn_xy=spawn.tolist(),spawn_yaw=yaw))
            features.append(v)
        raw=np.array(features);lo=raw.min(0);hi=raw.max(0)
        candidates[stratum]=records;raws[stratum]=raw
        normalised[stratum]=(raw-lo)/np.maximum(hi-lo,1e-9)
    selected=[];norms=[];used={s:set() for s in STRATUM_ORDER};seen=set()
    # First three are near the middle of each stratum; remaining seven add
    # missing marginal bins and distant combinations (four high, three each low).
    for i in range(10):
        stratum=STRATUM_ORDER[i%len(STRATUM_ORDER)];x=normalised[stratum]
        bins=np.minimum((x*5).astype(int),4)
        if i<3:score=-np.sum((x-.5)**2,axis=1)
        else:
            novelty=np.array([sum((k,int(b)) not in seen for k,b in enumerate(row)) for row in bins])
            dist=np.min(np.sum((x[:,None,:]-np.asarray(norms)[None,:,:])**2,axis=2),axis=1)
            score=novelty*10+dist
        for j in used[stratum]:score[j]=-np.inf
        j=int(np.argmax(score));used[stratum].add(j)
        selected.append(candidates[stratum][j]);norms.append(x[j])
        seen.update((k,int(b)) for k,b in enumerate(bins[j]))
    coverage={s:{name:dict(candidate_min=float(raws[s][:,k].min()),candidate_max=float(raws[s][:,k].max()),
        selected_min=min(c['parameters'][name] for c in selected if c['stratum']==s),
        selected_max=max(c['parameters'][name] for c in selected if c['stratum']==s)) for k,name in enumerate(FEATURES)} for s in STRATUM_ORDER}
    return dict(version=PLAN_VERSION,candidate_resets=len(STRATUM_ORDER)*candidate_count,
                physics_calls_during_selection=0,policy_calls_during_selection=0,cases=selected,coverage=coverage,
                marginal_bins_covered=len(seen),marginal_bins_total=5*len(FEATURES),
                note='Coverage demonstration, not iid evaluation or a success-rate estimate; no recovery interventions.')


def verify_encoded_exports(reports):
    """Create the review images by decoding the actual MP4, not render buffers."""
    for r in reports:
        if r.get('encoded_snapshots_checked') and all(Path(p).exists() for p in r['snapshots']):
            continue
        video=Path(r['video'])
        checkpoints=sorted(set([0,r['frames']//3,2*r['frames']//3,r['frames']-1]))
        select='+'.join(f'eq(n\\,{i})' for i in checkpoints)
        pattern=video.with_name(video.stem+'_decoded%02d.jpg')
        subprocess.run(['ffmpeg','-v','error','-y','-threads','1','-i',str(video),
                        '-vf',f'select={select}','-fps_mode','vfr','-frames:v',str(len(checkpoints)),
                        '-threads','1',str(pattern)],check=True)
        snapshots=[str(video.with_name(video.stem+f'_decoded{i:02d}.jpg')) for i in range(1,len(checkpoints)+1)]
        for p in snapshots:
            with Image.open(p) as img:
                if img.size!=(800,696):raise ValueError(f'Bad decoded review frame: {p}')
        r.update(snapshots=snapshots,encoded_snapshots_checked=True)
        atomic_json(video.with_suffix('.json'),r)


def gallery(out,reports,plan):
    verify_encoded_exports(reports)
    manifest=json.loads((out/'manifest.json').read_text())
    manifest.update(videos=reports,encoded_representative_frames_checked=True)
    atomic_json(out/'manifest.json',manifest)
    font=ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',18)
    sheet=Image.new('RGB',(1000,5*285),'#f2f5fa');draw=ImageDraw.Draw(sheet)
    cards=[]
    for i,r in enumerate(reports):
        # Last frame is useful for posture/outcome; full 4-frame strips are separate.
        pic=Image.open(r['snapshots'][-1]);pic.thumbnail((480,235))
        x=(i%2)*500+10;y=(i//2)*285
        sheet.paste(pic,(x,y));draw.text((x,y+239),f"{i+1:02d} {r['stratum']} | {'成功' if r['success'] else '失败'}",font=font,fill='#21354d')
        cards.append(f'<article><h2>{i+1:02d} · {html.escape(r["stratum"])} · {"成功" if r["success"] else "失败"}</h2>'
            f'<video controls preload="metadata" poster="{Path(r["snapshots"][-1]).name}" src="{Path(r["video"]).name}"></video>'
            f'<p>seed {r["seed"]} | {r["duration_s"]:.2f}s</p><pre>{html.escape(json.dumps(r["parameters"],ensure_ascii=False,indent=2))}</pre>'
            f'<p>{html.escape(r["why"])}</p><a href="{Path(r["video"]).name}">下载本集</a></article>')
    sheet.save(out/'overview.jpg',quality=91)
    # Two detailed sheets, each five episodes x four times, for visual QA.
    for group in range(2):
        contact=Image.new('RGB',(1600,5*368),'#f2f5fa')
        draw=ImageDraw.Draw(contact)
        for row,r in enumerate(reports[group*5:group*5+5]):
            for col,p in enumerate(r['snapshots']):
                with Image.open(p) as source:
                    thumb=source.resize((400,348))
                    contact.paste(thumb,(col*400,row*368))
            draw.text((8,row*368+348),f"{group*5+row+1:02d} {r['stratum']} seed {r['seed']}",font=font,fill='#21354d')
        contact.save(out/f'review_{group+1}.jpg',quality=92)
    page='<!doctype html><meta charset="utf-8"><title>检查采集视频 · 10集</title><style>body{font:16px system-ui;background:#eef3f8;color:#18334f;margin:30px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px}article{background:white;padding:20px;border-radius:14px}video{width:100%}pre{font-size:13px}</style>'
    page+='<h1>检查采集视频 · 10集随机化覆盖</h1><p>当前专家；覆盖式挑选，包含失败，不能将成功比例当作随机测试成功率。全部视频采用侧前方全身跟随镜头。</p>'
    page+=f'<p>覆盖归一化参数区间：{plan["marginal_bins_covered"]}/{plan["marginal_bins_total"]} 个边际分箱。</p><main>'+''.join(cards)+'</main>'
    (out/'index.html').write_text(page)
    with zipfile.ZipFile(out/'videos_10.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for r in reports:
            archive.write(r['video'],Path(r['video']).name)
            archive.write(Path(r['video']).with_suffix('.json'),Path(r['video']).with_suffix('.json').name)
            for p in r['snapshots']:archive.write(p,Path(p).name)
        for name in ('index.html','overview.jpg','review_1.jpg','review_2.jpg','plan.json','manifest.json'):
            archive.write(out/name,name)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--scene-profile',choices=PROFILES,default=DEFAULT_PROFILE)
    p.add_argument('--out',type=Path,help='Optional explicit output directory; compatible completed cases are reused')
    p.add_argument('--plan-only',action='store_true',help='Show the ten resets; no physics or video')
    args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(17)
    digest,spec=fingerprint(args.scene_profile)
    out=args.out or ROOT/'outputs/videos'/f'scalebfm_check10_{args.scene_profile}_{digest}'
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This video check is already running')
    saved=out/'config.json'
    if saved.exists() and json.loads(saved.read_text())!=json.loads(json.dumps(spec)):
        raise SystemExit('Existing check belongs to different code/assets/settings; use the automatic new directory')
    atomic_json(saved,spec)
    if (out/'plan.json').exists():plan=json.loads((out/'plan.json').read_text())
    else:
        plan=choose_cases(args.scene_profile);atomic_json(out/'plan.json',plan)
    print(f'OUTPUT {out}',flush=True)
    print(f'Coverage {plan["marginal_bins_covered"]}/{plan["marginal_bins_total"]}; candidates={plan["candidate_resets"]}, physics=0',flush=True)
    if args.plan_only:return
    (out/'source').mkdir(exist_ok=True)
    for name in spec['source_sha256']:
        source=Path(name)
        if source.suffix=='.py':shutil.copy2(source,out/'source'/source.name)
    for source in (Path(__file__),Path(__file__).with_name('visual_follow_video.py')):
        shutil.copy2(source,out/'source'/source.name)
    B.R.GRIP_KP,B.R.GRIP_KD=1.,.025
    policy=None;envs={};reports=[]
    try:
        for i,c in enumerate(plan['cases'],1):
            case=out/'episodes'/f"seed{c['seed']}_{c['edge']}";case.mkdir(parents=True,exist_ok=True)
            if not (case/'result.json').exists():
                if (case/'simulation_result.json').exists():
                    result=json.loads((case/'simulation_result.json').read_text())
                    directory=scene_directory(out/'scene_assets',c['stratum'])
                else:
                    if any(case.iterdir()):raise RuntimeError(f'Interrupted episode requires inspection, will not silently retry: {case}')
                    atomic_json(case/'attempt.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),case=c))
                    if policy is None:policy=B.Policy(device='cpu')
                    if c['stratum'] not in envs:
                        directory=scene_directory(out/'scene_assets',c['stratum']);sim=make_sim(policy,directory)
                        envs[c['stratum']]=(sim,BottleProxySensor(sim),directory)
                    sim,sensor,directory=envs[c['stratum']]
                    B.EDGE_INBOARD=STRATA[c['stratum']][1]
                    print(f"COLLECT {i}/10 {c['stratum']} seed={c['seed']}",flush=True)
                    result=run_case(sim,policy,sensor,c['seed'],c['edge'],case,expert_config(),
                        **EXPERT_RUN_OPTIONS,scene_profile=args.scene_profile,stratum=c['stratum'])
                result.update(stratum=c['stratum'],scene_directory=str(directory.resolve()))
                atomic_json(case/'result.json',result)
            else:result=json.loads((case/'result.json').read_text())
            if not (case/'audit.json').exists():atomic_json(case/'audit.json',audit_case(case))
            video=out/f"{i:02d}_{c['stratum']}_seed{c['seed']}.mp4"
            print(f"RENDER {i}/10 task={'OK' if result['ok'] else result['why']}",flush=True)
            report=render_case(case,video,i,c['parameters']);reports.append(report)
            atomic_json(out/'progress.json',dict(completed_videos=len(reports),total=10,reports=reports))
            print(f'DONE {i}/10 {video.name}',flush=True)
    finally:
        for _,sensor,_ in envs.values():sensor.close()
    manifest=dict(status='complete',profile=args.scene_profile,code_fingerprint=digest,episodes=10,
        successes=sum(r['success'] for r in reports),videos=reports,
        scope='10 current-expert coverage demonstrations, not a VLA evaluation or training dataset; original Goal remains stopped.',
        camera='right-front follow; all-frame full-body projected bounds checked; representative encoded frames should be visually reviewed')
    atomic_json(out/'manifest.json',manifest);gallery(out,reports,plan)
    print(f'COMPLETE {out}/index.html\nZIP {out}/videos_10.zip',flush=True)


if __name__=='__main__':main()
