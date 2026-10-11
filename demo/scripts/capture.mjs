import fs from 'node:fs';
import path from 'node:path';
import {chromium} from 'playwright';
import {DEMO, ROOT, API, DASH, EDGE, Stack, sleep, waitFor, getAPI, presetAPI, ensureDir, saveJSON, exec, ffmpeg} from './lib.mjs';

const take = new Date().toISOString().replace(/[:.]/g, '-');
const evidence = path.join(DEMO, 'evidence/runs', take);
const rawDir = path.join(DEMO, 'recordings', take);
ensureDir(rawDir);
const stack = new Stack(evidence);
const report = {take, startedUTC:new Date().toISOString(), viewport:{width:1920,height:1080}, scenes:[], confirmations:[], websocket:[], screenshots:[], errors:[], limitations:[]};
let browser, context, page, rawPath, epoch;
let desktopRecorder = null;
const now = () => (Date.now()-epoch)/1000;
const uiFresh = (state) => page.locator('.simulation-status').waitFor({state:'visible'}).then(() =>
  waitFor(async () => (await page.locator('.simulation-status').getAttribute('class'))?.includes('is-'+state.toLowerCase()), 'UI '+state, 60000));
const current = () => getAPI('/missions/'+report.missionId+'/overview');
async function nav(label, id, missionId=report.missionId) {
  await page.locator('#mission-navigation').getByRole('link',{name:label,exact:true}).click();
  await page.locator(`[data-page="${id}"]`).waitFor();
  await page.locator(`[data-page="${id}"] .cw-eyebrow`).filter({hasText:missionId}).waitFor();
  await page.mouse.move(1890,1060);
}
async function shot(name) {
  const file=path.join(evidence,name+'.png');
  await page.screenshot({path:file});
  report.screenshots.push({name,t:now(),file:path.relative(DEMO,file)});
}
async function clip(id, title, subtitle, seconds, action, extra={}) {
  const start=now();
  if (action) await action();
  const elapsed=now()-start;
  await sleep(Math.max(0,seconds-elapsed)*1000);
  await shot(id);
  report.scenes.push({id,title,subtitle,start,end:now(),...extra});
  console.log('Captured:',id, 'at',now().toFixed(1),'s');
  saveJSON(path.join(evidence,'capture.json'),report);
}
async function apply(preset) {
  const before=now();
  const response=page.waitForResponse((r) => r.url().endsWith('/api/v1/network/preset') && r.request().method()==='POST');
  await page.getByRole('group',{name:'Lunar link presets'}).getByRole('button',{name:new RegExp('^'+preset+'\\b')}).click();
  const posted=await response;
  if (!posted.ok()) throw new Error(`${preset} POST failed: ${posted.status()}`);
  const status=await waitFor(async () => {const st=await getAPI('/network/status'); return st.preset===preset && st;},preset+' backend confirmation');
  await waitFor(async () => (await page.locator('.network-control-feedback').textContent())?.includes(preset+' confirmed'),preset+' UI confirmation');
  const confirmation={preset,clickAt:before,confirmedAt:now(),utc:new Date().toISOString(),status};
  report.confirmations.push(confirmation);
  console.log('Confirmed:',preset);
  return confirmation;
}
async function startDesktopCapture() {
  try {
    const title = (await exec('powershell.exe',['-NoProfile','-Command',"Get-Process -Name webots,webots-bin -ErrorAction SilentlyContinue | Where-Object MainWindowTitle | Select-Object -First 1 -ExpandProperty MainWindowTitle"])).trim();
    if (!title) throw new Error('Webots window has no visible title');
    await exec('powershell.exe',['-NoProfile','-File',path.join(DEMO,'scripts/focus-webots.ps1')]);
    const {spawn}=await import('node:child_process');
    const fd=fs.openSync(path.join(evidence,'desktop-capture.log'),'a');
    const file=path.join(rawDir,'webots-native.mkv');
    desktopRecorder=spawn(ffmpeg,['-hide_banner','-f','gdigrab','-framerate','30','-draw_mouse','0','-video_size','1920x1080','-i','desktop','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-preset','ultrafast','-crf','20','-pix_fmt','yuv420p',file],{windowsHide:true,stdio:['pipe','ignore',fd]});
    report.desktop={file:path.relative(DEMO,file),start:now(),title,method:'foreground desktop capture'};
    desktopRecorder.once('exit',code=>{report.desktop.exitCode=code;});
    desktopRecorder.once('error',error=>{report.limitations.push('Native capture: '+error.message);});
  } catch(error) {report.limitations.push('Native Webots capture unavailable: '+error.message);}
}

try {
  await stack.startServices();
  const oldMissions=await getAPI('/missions?limit=30');
  const oldIds=new Set(oldMissions.map(m=>m.mission_id));
  report.historicalMission=oldMissions.find(m=>(m.counters?.pose??0)>20)?.mission_id??null;
  browser=await chromium.launch({executablePath:EDGE,headless:true});
  context=await browser.newContext({viewport:report.viewport,deviceScaleFactor:1,recordVideo:{dir:rawDir,size:report.viewport}});
  epoch=Date.now();
  report.browserEpochUTC=new Date(epoch).toISOString();
  page=await context.newPage();
  rawPath=await page.video().path();
  page.on('pageerror',e=>report.errors.push({t:now(),kind:'pageerror',message:e.message}));
  page.on('websocket',ws=>{
    ws.on('framereceived',({payload})=>{
      try {const event=JSON.parse(String(payload)); report.websocket.push({at:now(),...event});} catch {}
    });
  });
  await page.goto(DASH+'/settings');
  await page.getByRole('switch',{name:'Unified Mission Control',exact:true}).waitFor();
  if (await page.getByRole('switch',{name:'Unified Mission Control',exact:true}).isChecked()) await page.getByRole('switch',{name:'Unified Mission Control',exact:true}).click();
  if (!await page.getByRole('switch',{name:'Follow the newest mission',exact:true}).isChecked()) await page.getByRole('switch',{name:'Follow the newest mission',exact:true}).click();
  stack.startRover();
  const mission=await waitFor(async()=> (await getAPI('/missions?limit=10')).find(m=>!oldIds.has(m.mission_id) && m.counters.pose>3),'new genuine rover mission',120000);
  report.missionId=mission.mission_id;
  await nav('Overview','overview');
  await uiFresh('LIVE');
  await waitFor(async()=> (await getAPI('/missions/'+report.missionId+'/obstacles?limit=100')).some(o=>o.source==='yolo+depth'),'YOLO + depth obstacles',90000);
  await startDesktopCapture();
  await clip('initialization','Real-time Ground Station Monitoring','A live mission. Two clocks. One received picture.',16);
  await nav('Detection','detection');
  await clip('perception','Edge AI Hazard Detection','Local YOLO + aligned depth. Confidence is not geometric accuracy.',20,async()=>{
    await page.locator('.data-table-interactive-row').filter({hasText:'YOLO + depth'}).first().click();
    await page.locator('.detection-sheet[data-slot="sheet-content"]').waitFor();
    await sleep(7000);
    await page.getByRole('tab',{name:'Evidence',exact:true}).click();
    await sleep(5000);
    await page.getByRole('tab',{name:'Overview',exact:true}).click();
  });
  await page.keyboard.press('Escape');
  await nav('Mapping','map');
  await clip('mapping','Spatial Awareness from Received Telemetry','Odom frame · observed ≠ free · dashed rings show uncertainty.',20,async()=>{
    const nearest=page.locator('.cw-list button').first();
    if (await nearest.count()) await nearest.click();
    await sleep(5000);
    await page.getByRole('button',{name:'Zoom in',exact:true}).click();
    await sleep(4000);
    await page.getByRole('button',{name:'Zoom out',exact:true}).click();
  });
  await nav('Telemetry','telemetry');
  await clip('telemetry','Compact Binary Telemetry Over UDP','Received position, heading, uncertainty and packet evidence.',15);
  await nav('Communication','network');
  await clip('network','Operating Under Lunar Communication Constraints','Confirmed presets. Measured losses. A real constrained link.',24,async()=>{
    await apply('NORMAL'); await sleep(3000);
    await apply('DEGRADED'); await sleep(7000);
    await apply('SEVERE'); await sleep(9000);
  });
  await clip('blackout','Communication Lost. Mission Continues.','Last received map retained · reception age increases.',20,async()=>{
    await apply('BLACKOUT');
    await uiFresh('STALE');
    await nav('Mapping','map');
    const before=await current();
    await sleep(5000);
    const after=await current();
    const tBefore=before.latest_position?.t_obs_s;
    const tAfter=after.latest_position?.t_obs_s;
    report.blackout={before,after,unchangedPose: tBefore!==undefined && tBefore===tAfter};
    if (!report.blackout.unchangedPose) throw new Error('Blackout pose freeze could not be proven.');
    await page.locator('.cw-stale').waitFor();
  });
  await clip('recovery','Store-and-Forward Resilience','Reception resumes. Retained observations are reconciled; delivery is not guaranteed.',22,async()=>{
    await nav('Communication','network');
    const restore=await apply('RESTORE');
    await uiFresh('LIVE');
    report.recovery={liveAt:now(),delayS:now()-restore.confirmedAt,overview:await current()};
    await nav('Mapping','map');
  });
  await nav('Overview','overview');
  await page.getByRole('link',{name:'Open mission history'}).click();
  await page.locator('[data-page="missions"]').waitFor();
  await clip('history','Persistent Mission Intelligence','Historical mission · MongoDB-backed records, isolated from the live session.',12,async()=>{
    if (!report.historicalMission) {report.limitations.push('No older stored mission available; archive directory only.'); return;}
    const row=page.locator('.data-table-interactive-row').filter({hasText:'#'+report.historicalMission});
    await row.first().getByRole('button',{name:'Use mission',exact:true}).click();
    await waitFor(()=>page.url().includes('mission='+report.historicalMission),'historical mission pinned');
    await nav('Mapping','map',report.historicalMission);
  },{historical:true,missionId:report.historicalMission});
  await nav('Settings','settings',report.historicalMission??report.missionId);
  if (!await page.getByRole('switch',{name:'Follow the newest mission',exact:true}).isChecked()) await page.getByRole('switch',{name:'Follow the newest mission',exact:true}).click();
  await waitFor(async()=> (await page.locator('.cw-eyebrow').textContent()).includes(report.missionId),'return to live mission');
  await page.getByRole('switch',{name:'Unified Mission Control',exact:true}).click();
  await page.locator('.brand-lockup').click();
  await page.locator('.unified-command-center, .mission-grid, #surface').first().waitFor();
  await uiFresh('LIVE');
  await clip('unified','Unified Mission Control','One integrated operator view. The same genuine received mission.',12);
  if (report.errors.length) throw new Error('Browser errors were recorded; review before rendering.');
  report.rawBrowser=path.relative(DEMO,rawPath);
  report.recordingEndedUTC=new Date().toISOString();
} catch(error) {
  report.failure=error.stack;
  if (page) try {await shot('failure'); report.failureDOM=await page.locator('body').innerText();} catch {}
  console.error(error);
  process.exitCode=1;
} finally {
  if(stack.owned.some(x=>x.name==='link')) {
    try {const restored=await presetAPI('NORMAL'); report.cleanupPreset=restored.preset;} catch(error) {report.cleanupError=error.message;}
  } else report.cleanupSkipped='No emulator was started by this capture; no existing control plane was changed.';
  if (desktopRecorder?.exitCode===null) {
    const done=new Promise(resolve=>desktopRecorder.once('close',resolve));
    desktopRecorder.stdin.write('q\n');
    await Promise.race([done,sleep(5000)]);
  }
  if (context) await context.close();
  if (browser) await browser.close();
  saveJSON(path.join(evidence,'capture.json'),report);
  // After the recording, allow the already-running tour to finish for its DTN report.
  if (!report.failure) {
    console.log('Capture complete. Waiting for the rover final report without starting another mission…');
    try {await waitFor(()=>fs.existsSync(path.join(evidence,'tour/summary.json')),'tour final evidence',420000);} catch(error) {report.limitations.push(error.message);}
  }
  await stack.close();
  report.finishedUTC=new Date().toISOString();
  if (fs.existsSync(path.join(evidence,'tour/summary.json'))) report.roverSummary=JSON.parse(fs.readFileSync(path.join(evidence,'tour/summary.json'),'utf8'));
  saveJSON(path.join(evidence,'capture.json'),report);
  saveJSON(path.join(DEMO,'evidence/latest.json'),{take,evidence:path.relative(DEMO,evidence),rawDir:path.relative(DEMO,rawDir)});
  console.log('Evidence:',evidence);
}
