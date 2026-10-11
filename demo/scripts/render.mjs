import fs from 'node:fs';
import path from 'node:path';
import {bundle} from '@remotion/bundler';
import {selectComposition,renderMedia,renderStill} from '@remotion/renderer';
import {DEMO, EDGE, ensureDir, saveJSON} from './lib.mjs';
import {prepare} from './prepare.mjs';

const {props,report,evidence}=await prepare();
const serveUrl=await bundle({entryPoint:path.join(DEMO,'src/index.tsx'),rootDir:DEMO,publicDir:path.join(DEMO,'public'),outDir:path.join(DEMO,'temp/bundle')});
const composition=await selectComposition({serveUrl,id:'ArtemisDemo',inputProps:props,browserExecutable:EDGE});
const qa=path.join(DEMO,'output/qa');
ensureDir(qa);
const samples=[{id:'intro',frame:180},...props.scenes.map(s=>({id:s.id,frame:s.startFrame+Math.min(s.frames-1,Math.floor(s.frames*.68))})),{id:'ending',frame:props.duration-80}];
for(const s of samples) {
  await renderStill({serveUrl,composition,inputProps:props,browserExecutable:EDGE,frame:s.frame,output:path.join(qa,`${s.id}.png`),imageFormat:'png'});
  console.log('Preview:',s.id);
}
saveJSON(path.join(qa,'samples.json'),samples);
if(process.argv.includes('--preview')) process.exit(0);
const output=path.join(DEMO,'output/artemis-x-demo-final.mp4');
if(fs.existsSync(output)) fs.renameSync(output,path.join(DEMO,`output/artemis-x-demo-final.previous-${Date.now()}.mp4`));
let previous=-1;
await renderMedia({serveUrl,composition,inputProps:props,browserExecutable:EDGE,codec:'h264',outputLocation:output,crf:18,pixelFormat:'yuv420p',concurrency:4,muted:true,imageFormat:'jpeg',jpegQuality:95,onProgress:({progress})=>{const percent=Math.floor(progress*100/5)*5;if(percent!==previous){previous=percent; console.log(`Render ${percent}%`);}}});
saveJSON(path.join(evidence,'render.json'),{output:path.relative(DEMO,output),renderedUTC:new Date().toISOString(),durationFrames:props.duration,fps:30,resolution:'1920x1080',codec:'H.264',audio:'none — captioned, music-free edition',missionId:report.missionId});
console.log('Final MP4:',output);
