import {spawn, spawnSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import {fileURLToPath} from 'node:url';
import ffmpeg from 'ffmpeg-static';
import ffprobe from 'ffprobe-static';

export const DEMO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const ROOT = path.dirname(DEMO);
export const API = 'http://127.0.0.1:8000';
export const DASH = 'http://localhost:3000';
export const EDGE = process.env.ARTEMIS_EDGE_EXE ?? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
export {ffmpeg};
export const ensureDir = (p) => fs.mkdirSync(p, {recursive: true});
export const saveJSON = (p, v) => {ensureDir(path.dirname(p)); fs.writeFileSync(p, JSON.stringify(v, null, 2));};
export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
export async function waitFor(fn, label, timeout = 60000) {
  const deadline = Date.now() + timeout;
  let last;
  while (Date.now() < deadline) {
    try {const result = await fn(); if (result) return result;} catch (error) {last = error;}
    await sleep(350);
  }
  throw new Error(`Timed out: ${label}. ${last?.message ?? ''}`);
}
export async function getAPI(route) {
  const r = await fetch(API + '/api/v1' + route, {signal: AbortSignal.timeout(5000)});
  if (!r.ok) throw new Error(`${route}: HTTP ${r.status}`);
  return (await r.json()).data;
}
export async function presetAPI(preset) {
  const r = await fetch(API + '/api/v1/network/preset', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({preset}),
    signal: AbortSignal.timeout(5000),
  });
  if (!r.ok) throw new Error(`Preset ${preset}: HTTP ${r.status}`);
  return (await r.json()).data;
}
export const tcpOpen = (port) => new Promise((resolve) => {
  const s = net.createConnection({host: '127.0.0.1', port});
  s.setTimeout(800);
  s.once('connect', () => {s.destroy(); resolve(true);});
  s.once('error', () => resolve(false));
  s.once('timeout', () => {s.destroy(); resolve(false);});
});
export function exec(executable, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(executable, args, {cwd: DEMO, windowsHide: true, ...opts});
    let out = '', err = '';
    p.stdout?.on('data', (b) => {out += b;});
    p.stderr?.on('data', (b) => {err += b;});
    p.once('error', reject);
    p.once('close', (code) => code === 0 ? resolve(out) : reject(new Error(`${path.basename(executable)} exited ${code}: ${err.slice(-3000)}`)));
  });
}
export async function probe(file) {
  return JSON.parse(await exec(ffprobe.path, ['-v', 'error', '-show_format', '-show_streams', '-of', 'json', file]));
}
export class Stack {
  constructor(dir) {this.dir = dir; this.owned = [];}
  launch(name, exe, args, extra = {}, visible = false) {
    ensureDir(this.dir);
    const fd = fs.openSync(path.join(this.dir, name + '.log'), 'a');
    const env = {...process.env, ...extra};
    delete env.VIRTUAL_ENV;
    for (const [key, value] of Object.entries(env)) if (value === null) delete env[key];
    const p = spawn(exe, args, {cwd: ROOT, env, stdio: ['pipe', fd, fd], windowsHide: !visible});
    p.once('error', (error) => console.error(`${name}: ${error.message}`));
    this.owned.push({name, p, fd});
    return p;
  }
  async startServices() {
    for (const port of [8000, 5761]) if (await tcpOpen(port)) throw new Error(`Port ${port} already occupied. Refusing to stop or replace a pre-existing service.`);
    const runningWebots=(await exec('powershell.exe',['-NoProfile','-Command',"@(Get-Process -Name webots,webots-bin -ErrorAction SilentlyContinue).Count"])).trim();
    if(Number(runningWebots)>0) throw new Error('A pre-existing Webots process is running. Refusing to replace it.');
    const occupiedUDP=(await exec('powershell.exe',['-NoProfile','-Command',"Get-NetUDPEndpoint -ErrorAction SilentlyContinue | Where-Object LocalPort -in @(5790,5791,5792,5793,5794) | Select-Object -ExpandProperty LocalPort"])).trim();
    if(occupiedUDP) throw new Error('Demo UDP ports are already occupied: '+occupiedUDP.replaceAll('\r\n',', '));
    const db = spawnSync('docker', ['ps', '--filter', 'name=artemis-mongodb', '--format', '{{.Status}}'], {encoding:'utf8', windowsHide:true});
    if (!db.stdout.includes('healthy')) throw new Error('MongoDB must already be healthy; start it using the demo guide.');
    if (!await tcpOpen(3000)) {
      this.launch('frontend', process.execPath, [path.join(ROOT,'frontend/apps/web/node_modules/next/dist/bin/next'), 'dev', '--hostname','127.0.0.1', '--port','3000', path.join(ROOT,'frontend/apps/web')], {NEXT_PUBLIC_GROUND_API:API,NEXT_PUBLIC_GROUND_WS:API.replace('http','ws')});
      await waitFor(async()=>{const r=await fetch(DASH+'/settings'); return r.ok;},'dashboard startup',90000);
    }
    this.launch('ground', path.join(ROOT, 'ground_station/.venv/Scripts/python.exe'),
      ['-m', 'uvicorn', 'artemis_ground.main:app', '--app-dir', 'ground_station', '--host', '127.0.0.1', '--port', '8000'], {
        ARTEMIS_GROUND_RECEIVER_ENABLED:'true', ARTEMIS_GROUND_UDP_HOST:'127.0.0.1', ARTEMIS_GROUND_UDP_PORT:'5790',
        ARTEMIS_GROUND_ACK_TO:'127.0.0.1:5794', ARTEMIS_GROUND_EMULATOR_CONTROL:'127.0.0.1:5792',
        ARTEMIS_GROUND_NETWORK_CONTROL_ENABLED:'true', ARTEMIS_GROUND_LINK_METRICS_PERIOD_S:'1.0',
      });
    await waitFor(async () => {const h=await getAPI('/health'); return h.mongodb==='up' && h.receiver.enabled;}, 'ground health');
    this.launch('link', path.join(ROOT, 'backend/.venv/Scripts/python.exe'),
      ['scripts/lunar_link.py', 'serve', '--reverse', '--preset', 'NORMAL', '--log', path.join(this.dir, 'link.jsonl'), '--stats', path.join(this.dir, 'link-stats.json')]);
    await waitFor(async () => (await getAPI('/network/status')).preset === 'NORMAL', 'emulator NORMAL');
    this.launch('inference', path.join(ROOT, 'backend/.venv/Scripts/python.exe'), ['scripts/inference_server.py', '--port', '5761']);
    await waitFor(() => tcpOpen(5761), 'YOLO warmup / port 5761', 120000);
  }
  startRover() {
    this.launch('webots', path.join(process.env.WEBOTS_HOME ?? 'C:/Program Files/Webots', 'msys64/mingw64/bin/webots.exe'),
      ['--batch', '--mode=realtime', '--stdout', '--stderr', path.join(ROOT, 'artemis-webots/worlds/artemis_r4_monitor.wbt')], {
        ARTEMIS_MODE:'demo', ARTEMIS_INFERENCE:'1', ARTEMIS_INFER_PORT:'5761', ARTEMIS_MAPPING:'1',
        ARTEMIS_MONITOR:'1', ARTEMIS_MONITOR_PANELS:'rgb,ai,depth,map', ARTEMIS_SENSOR_MS:'64',
        ARTEMIS_TELEMETRY:'1', ARTEMIS_TELEMETRY_HOST:'127.0.0.1', ARTEMIS_TELEMETRY_PORT:'5791',
        ARTEMIS_TELEMETRY_DTN:'1', ARTEMIS_TELEMETRY_ACK_PORT:'5793', ARTEMIS_TELEMETRY_SESSION:null,
        ARTEMIS_WEBOTS_OUT:path.join(this.dir,'tour'), ARTEMIS_WEBOTS_QUIT:'1', ARTEMIS_FAULTS:null,
        ARTEMIS_MAP_RESET_AT:null, ARTEMIS_DTN_MAX_ENTRIES:null, ARTEMIS_DTN_MAX_BYTES:null,
        ARTEMIS_MONITOR_SAVE:path.join(this.dir,'panels'), ARTEMIS_MONITOR_SAVE_AT:'20,35,50,75,100,130,160,190,220,250,300',
        ARTEMIS_PROFILE:null, ARTEMIS_MEMTRACE:null,
      }, true);
  }
  async close() {
    // Only processes launched by this exact Stack instance may be terminated.
    const link = this.owned.find((x) => x.name === 'link');
    if (link?.p.exitCode === null) {link.p.stdin.write('QUIT\n'); await sleep(500);}
    for (const {p, fd} of this.owned.reverse()) {
      if (p.exitCode === null && p.pid) spawnSync('taskkill', ['/PID', String(p.pid), '/T', '/F'], {windowsHide:true, stdio:'ignore'});
      try {fs.closeSync(fd);} catch {}
    }
  }
}
