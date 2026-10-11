// Cinematic V2 render. Reuses the recorded chapters and timeline of the V1 film WITHOUT re-running prepare(): the
// original MP4, captures, timeline and evidence are read only. Generates its own sound design (synthesised here, no
// third-party music) and two real frames for the before/after comparison.
//   node scripts/render-v2.mjs            full render → output/artemis-x-demo-cinematic-v2.mp4
//   node scripts/render-v2.mjs --preview  key-frame stills only (output/qa-v2/)
//   node scripts/render-v2.mjs --frames 2600,3400   specific stills
import fs from 'node:fs';
import path from 'node:path';
import {bundle} from '@remotion/bundler';
import {renderMedia, renderStill, selectComposition} from '@remotion/renderer';
import {DEMO, EDGE, ensureDir, exec, ffmpeg, saveJSON} from './lib.mjs';

const timeline = JSON.parse(fs.readFileSync(path.join(DEMO, 'public/timeline.json'), 'utf8'));
const scene = (id) => timeline.scenes.find((s) => s.id === id);
const pub = (p) => path.join(DEMO, 'public', p);
ensureDir(pub('v2/audio'));
ensureDir(pub('v2/stills'));

// ---- sound design (lavfi synthesis; restrained, secondary to the picture) -------------------------------------------
const seconds = timeline.duration / 30 + 2;
const audio = {
  bed: 'v2/audio/bed.wav', whoosh: 'v2/audio/whoosh.wav', drone: 'v2/audio/drone.wav', thud: 'v2/audio/thud.wav', chime: 'v2/audio/chime.wav',
};
const lavfi = async (out, inputs, filter) => {
  if (fs.existsSync(pub(out))) return;
  await exec(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', ...inputs.flatMap((i) => ['-f', 'lavfi', '-i', i]),
    '-filter_complex', filter, '-ac', '2', '-ar', '48000', pub(out)]);
};
await lavfi(audio.bed, [`sine=f=55:d=${seconds}`, `sine=f=82.41:d=${seconds}`, `sine=f=110.6:d=${seconds}`, `anoisesrc=c=pink:a=0.05:d=${seconds}`],
  `[0]volume=0.45[a];[1]volume=0.32,tremolo=f=0.1:d=0.6[b];[2]volume=0.16,tremolo=f=0.13:d=0.7[c];[3]lowpass=f=380,volume=0.7[d];` +
  `[a][b][c][d]amix=inputs=4:normalize=0,lowpass=f=900,volume=1.6,afade=t=in:d=4,afade=t=out:st=${seconds - 6}:d=6`);
await lavfi(audio.whoosh, ['anoisesrc=c=pink:a=0.6:d=1.2'],
  '[0]highpass=f=500,lowpass=f=3200,afade=t=in:d=0.45:curve=qsin,afade=t=out:st=0.45:d=0.75:curve=exp,volume=0.55');
await lavfi(audio.drone, ['sine=f=41.2:d=26', 'sine=f=61.7:d=26', 'anoisesrc=c=brown:a=0.15:d=26'],
  '[0]volume=0.7[a];[1]volume=0.35,tremolo=f=0.8:d=0.5[b];[2]lowpass=f=160[c];[a][b][c]amix=inputs=3:normalize=0,lowpass=f=220,afade=t=in:d=2,afade=t=out:st=23:d=3');
await lavfi(audio.thud, ['sine=f=48:d=1.4', 'anoisesrc=c=brown:a=0.4:d=1.4'],
  '[0]volume=0.9[a];[1]lowpass=f=120[b];[a][b]amix=inputs=2:normalize=0,afade=t=out:st=0.05:d=1.3:curve=exp');
await lavfi(audio.chime, ['sine=f=523.25:d=4', 'sine=f=659.25:d=4', 'sine=f=783.99:d=4'],
  '[0]afade=t=out:d=3.8:curve=exp,volume=0.3[a];[1]adelay=110|110,afade=t=out:d=3.6:curve=exp,volume=0.25[b];' +
  '[2]adelay=220|220,afade=t=out:d=3.4:curve=exp,volume=0.22[c];[a][b][c]amix=inputs=3:normalize=0,aecho=0.6:0.5:180:0.25');

// ---- real before/after frames (recorded footage; cropped to the map + rover panel) ------------------------------------
const stills = {stale: 'v2/stills/blackout-stale.png', live: 'v2/stills/recovery-live.png'};
const grab = async (file, t, out) => {
  if (fs.existsSync(pub(out))) return;
  await exec(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-ss', String(t), '-i', pub(file), '-frames:v', '1',
    '-vf', 'crop=1446:777:340:295', pub(out)]);
};
await grab(scene('blackout').file, 19.4, stills.stale);
await grab(scene('recovery').file, 13.0, stills.live);

const props = {...timeline, audio, stills};
const serveUrl = await bundle({entryPoint: path.join(DEMO, 'src/index.tsx'), rootDir: DEMO, publicDir: path.join(DEMO, 'public'),
  outDir: path.join(DEMO, 'temp/bundle-v2')});
const composition = await selectComposition({serveUrl, id: 'ArtemisDemoV2', inputProps: props, browserExecutable: EDGE});
const qa = path.join(DEMO, 'output/qa-v2');
ensureDir(qa);

const fi = process.argv.indexOf('--frames');
const net = scene('network').startFrame, bo = scene('blackout').startFrame, rc = scene('recovery').startFrame;
const key = fi >= 0 ? process.argv[fi + 1].split(',').map((f) => ({id: `f${f}`, frame: Number(f)})) : [
  {id: 'intro', frame: 150}, {id: 'overview-push', frame: scene('initialization').startFrame + 170},
  {id: 'detection-zoom', frame: scene('perception').startFrame + 150}, {id: 'mapping', frame: scene('mapping').startFrame + 110},
  {id: 'telemetry', frame: scene('telemetry').startFrame + 150},
  {id: 'normal', frame: net + 80}, {id: 'degraded-control', frame: net + 150}, {id: 'degraded', frame: net + 290},
  {id: 'severe', frame: net + 540}, {id: 'blackout', frame: bo + 80}, {id: 'blackout-map', frame: bo + 260},
  {id: 'blackout-age', frame: bo + 420}, {id: 'restore-cyan', frame: rc + 36}, {id: 'restore-green', frame: rc + 70},
  {id: 'recovery-map', frame: rc + 190}, {id: 'recovery-dtn', frame: rc + 330}, {id: 'before-after', frame: rc + 500},
  {id: 'history', frame: scene('history').startFrame + 130}, {id: 'unified', frame: scene('unified').startFrame + 150},
  {id: 'transition', frame: scene('mapping').startFrame - 8}, {id: 'ending', frame: timeline.duration - 70},
];
for (const k of key) {
  await renderStill({serveUrl, composition, inputProps: props, browserExecutable: EDGE, frame: k.frame, output: path.join(qa, `${k.id}.png`), imageFormat: 'png'});
  console.log('still', k.id, k.frame);
}
if (process.argv.includes('--preview') || fi >= 0) process.exit(0);

const output = path.join(DEMO, 'output/artemis-x-demo-cinematic-v2.mp4');
if (fs.existsSync(output)) fs.renameSync(output, path.join(DEMO, `output/artemis-x-demo-cinematic-v2.previous-${Date.now()}.mp4`));
let last = -1;
await renderMedia({serveUrl, composition, inputProps: props, browserExecutable: EDGE, codec: 'h264', audioCodec: 'aac', audioBitrate: '192k',
  outputLocation: output, crf: 17, pixelFormat: 'yuv420p', concurrency: 4, imageFormat: 'jpeg', jpegQuality: 95,
  onProgress: ({progress}) => { const p = Math.floor((progress * 100) / 5) * 5; if (p !== last) { last = p; console.log(`render ${p}%`); } }});
saveJSON(path.join(qa, 'render.json'), {output: path.relative(DEMO, output), renderedUTC: new Date().toISOString(), durationFrames: timeline.duration,
  fps: 30, resolution: '1920x1080', codec: 'H.264 + AAC', sourceTimeline: 'public/timeline.json (V1, read only)', missionId: timeline.missionId});
console.log('V2:', output);
