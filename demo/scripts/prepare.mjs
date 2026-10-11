import fs from 'node:fs';
import path from 'node:path';
import {DEMO, ensureDir, saveJSON, exec, ffmpeg, probe} from './lib.mjs';

export async function prepare() {
  const takeIndex=process.argv.indexOf('--take');
  const latest=takeIndex>=0?{evidence:'evidence/runs/'+process.argv[takeIndex+1]}:JSON.parse(fs.readFileSync(path.join(DEMO,'evidence/latest.json'),'utf8'));
  const evidence=path.join(DEMO,latest.evidence);
  const report=JSON.parse(fs.readFileSync(path.join(evidence,'capture.json'),'utf8'));
  if(report.failure) throw new Error('Latest capture failed. Retake before rendering. '+report.failure.split('\n')[0]);
  if(report.errors.length || report.cleanupPreset!=='NORMAL' || !report.blackout?.unchangedPose) throw new Error('Capture evidence failed verification.');
  const required=['NORMAL','DEGRADED','SEVERE','BLACKOUT','RESTORE'];
  if(required.some((p,i)=>report.confirmations[i]?.preset!==p)) throw new Error('Missing real confirmed preset sequence.');
  const raw=path.join(DEMO,report.rawBrowser);
  const rawProbe=await probe(raw);
  const stream=rawProbe.streams.find(s=>s.codec_type==='video');
  if(stream.width!==1920 || stream.height!==1080) throw new Error('Raw browser resolution incorrect.');
  ensureDir(path.join(DEMO,'public/captures'));
  ensureDir(path.join(DEMO,'public/onboard'));
  const props={missionId:report.missionId,scenes:[],duration:0};
  let cursor=12*30;
  const maximum={initialization:14,perception:18,mapping:18,telemetry:13,network:24,blackout:20,recovery:22,history:12,unified:10};
  const trims=[];
  const supplemental=path.join(evidence,'native-approved.json');
  const nativeSource=fs.existsSync(supplemental)?JSON.parse(fs.readFileSync(supplemental,'utf8')).file:null;
  for(const source of report.scenes) {
    const seconds=Math.min(source.end-source.start,maximum[source.id]);
    const frames=Math.floor(seconds*30);
    const file=`captures/${report.take}-${source.id}.mp4`;
    const target=path.join(DEMO,'public',file);
    if(!fs.existsSync(target)) await exec(ffmpeg,['-hide_banner','-loglevel','error','-ss',source.start.toFixed(3),'-i',raw,'-t',(frames/30).toFixed(3),'-an','-vf','fps=30','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',target]);
    const clipProbe=await probe(target);
    props.scenes.push({...source,file,frames,startFrame:cursor});
    trims.push({id:source.id,source:path.relative(DEMO,raw),startS:source.start,durationS:frames/30,speed:1,encodedDurationS:Number(clipProbe.format.duration)});
    cursor+=frames;
    if(source.id==='perception') {
      if(nativeSource) {
        const file=`captures/${report.take}-simulator-cropped.mp4`;
        if(!fs.existsSync(path.join(DEMO,'public',file))) await exec(ffmpeg,['-hide_banner','-loglevel','error','-ss','8','-i',path.join(DEMO,nativeSource),'-t','4','-an','-vf','crop=1646:964:274:116,scale=1920:1080,fps=30','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',path.join(DEMO,'public',file)]);
        props.scenes.push({id:'simulator',title:'Onboard Systems in Webots',subtitle:'Actual simulator workspace · sensor and mapping panels · recorded excerpt',file,frames:120,startFrame:cursor});
        trims.push({id:'simulator',source:nativeSource,startS:8,durationS:4,speed:1,crop:'1646:964:274:116 (simulator viewport only)'});
        cursor+=120;
      }
      const tag=fs.existsSync(path.join(evidence,'panels/t0075_ai.png'))?'t0075':'t0035';
      for(const key of ['rgb','ai','depth','map']) {
        const from=path.join(evidence,'panels',`${tag}_${key}.png`);
        if(!fs.existsSync(from)) throw new Error('Missing actual onboard panel: '+from);
        const file=`onboard/${report.take}-${tag}_${key}.png`;
        fs.copyFileSync(from,path.join(DEMO,'public',file));
        props[key]=file;
      }
      props.panelTime=`≥ ${Number(tag.slice(1))} s`;
      props.scenes.push({id:'sensors',title:'Onboard perception',subtitle:'Actual synchronized capture',file:'',frames:120,startFrame:cursor});
      cursor+=120;
    }
  }
  props.duration=cursor+6*30;
  if(props.duration/30<150 || props.duration/30>185) throw new Error('Unexpected final duration: '+props.duration/30);
  saveJSON(path.join(DEMO,'public/timeline.json'),props);
  saveJSON(path.join(evidence,'edit-manifest.json'),{take:report.take,missionId:report.missionId,fps:30,outputSeconds:props.duration/30,trims,notes:['1× source speed. Browser raw timestamps retained.','Navigation between focused pages is edited out.','Blackout/recovery source is temporally contiguous; chapter boundary is not a fabricated state.','Onboard stills are labelled with their simulation-time trigger, not presented as live downlink.','No synthetic telemetry, audio, music or inference results.']});
  console.log('Prepared',props.duration/30,'seconds from actual mission',report.missionId);
  return {props,report,evidence};
}
