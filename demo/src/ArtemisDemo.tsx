import React from 'react';
import {AbsoluteFill, Easing, Img, interpolate, OffthreadVideo, Sequence, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';

export type Scene = {id:string; title:string; subtitle:string; file:string; frames:number; startFrame:number; historical?:boolean; missionId?:string};
export type DemoProps = {missionId:string; scenes:Scene[]; rgb?:string; ai?:string; depth?:string; map?:string; panelTime?:string; native?:string; nativeFrames?:number; nativeStart?:number; duration:number};
const ink='#050b14', blue='#55baff', muted='#8ba3bd';
const ease=Easing.bezier(.16,1,.3,1);
const enter=(frame:number,delay=0)=>interpolate(frame,[delay,delay+20],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp',easing:ease});

function Logo({size=100}:{size?:number}) {return <svg width={size} height={size*66/76} viewBox="0 0 76 66"><defs><linearGradient id="metal" x2="0" y2="1"><stop stopColor="#fff"/><stop offset="1" stopColor="#95bee8"/></linearGradient><linearGradient id="orbit"><stop stopColor="#9cedff"/><stop offset=".55" stopColor="#65b8ff"/><stop offset="1" stopColor="#316cf5"/></linearGradient></defs><path d="M14 43C3 14 29 6 40 10M45 10C70 12 73 31 64 42" stroke="#1745a0" strokeWidth="1.2" fill="none"/><path d="M38 1 61 40 49 34 38 17 27 35 15 41 38 1Z" fill="url(#metal)"/><path d="M38 20 43 34 38 31 33 35 38 20Z" fill="#091321"/><path d="M2 61C20 35 54 34 74 61" stroke="url(#orbit)" strokeWidth="3.5" fill="none"/><path d="M11 62C25 45 51 44 66 62" stroke="#68b9ff" strokeWidth="2.6" fill="none"/><path d="M21 62C30 53 46 52 55 62" stroke="#225df2" strokeWidth="1.7" fill="none"/></svg>}
const Backdrop=()=> <AbsoluteFill style={{background:`radial-gradient(ellipse at 78% 18%, #102841 0%, ${ink} 60%)`}}><AbsoluteFill style={{opacity:.18,backgroundImage:'linear-gradient(#23415c 1px, transparent 1px),linear-gradient(90deg,#23415c 1px,transparent 1px)',backgroundSize:'96px 96px',maskImage:'linear-gradient(to bottom, transparent, black)'}}/></AbsoluteFill>;
function Intro({rgb,panelTime}:DemoProps) {
  const f=useCurrentFrame();
  const a=enter(f,10),b=enter(f,28),c=enter(f,46);
  return <AbsoluteFill><Backdrop/>
    <div style={{position:'absolute',left:100,top:90,display:'flex',alignItems:'center',gap:22,opacity:a,transform:`translateY(${(1-a)*18}px)`}}><Logo size={94}/><div style={{letterSpacing:8,fontSize:25,fontWeight:600}}>ARTEMIS-X<div style={{fontSize:13,letterSpacing:5,color:muted,marginTop:9}}>MISSION INTELLIGENCE</div></div></div>
    <div style={{position:'absolute',left:100,top:360,width:950,opacity:b,transform:`translateY(${(1-b)*24}px)`}}><div style={{fontSize:15,color:blue,letterSpacing:5,marginBottom:28}}>PROJECT C.R.A.T.E.R · ATEC 2026</div><h1 style={{fontSize:78,fontWeight:600,letterSpacing:-3,lineHeight:1.09,margin:0}}>AI-Powered Lunar<br/>Reconnaissance</h1><p style={{color:muted,fontSize:25,lineHeight:1.6,maxWidth:850,marginTop:35}}>Perceive. Localize. Transmit. Visualize.<br/><span style={{fontSize:21}}>Real simulation. Actual mission evidence.</span></p></div>
    {rgb&&<div style={{position:'absolute',right:80,top:278,width:720,border:'1px solid #29455e',borderRadius:16,overflow:'hidden',opacity:c,transform:`translateY(${(1-c)*18}px)`}}><Img src={staticFile(rgb)} style={{width:'100%',display:'block'}}/><div style={{padding:'18px 22px',color:muted,fontSize:16,background:'#07111e'}}>ONBOARD RGB CAPTURE · {panelTime} SIM · NOT A LIVE FEED</div></div>}
    <div style={{position:'absolute',left:100,right:100,bottom:92,display:'flex',justifyContent:'space-between',borderTop:'1px solid #254057',paddingTop:24,opacity:c}}><span style={{color:muted,fontSize:18,letterSpacing:3}}>EDGE AI / MAPPING / BINARY UDP / STORE & FORWARD</span><span style={{color:blue,fontSize:17}}>REAL SYSTEM CAPTURE</span></div>
  </AbsoluteFill>;
}
function UIShot({scene}: {scene:Scene}) {
  const f=useCurrentFrame();
  const a=enter(f,0);
  const number=scene.id==='blackout'?'06':scene.id==='recovery'?'07':scene.id==='history'?'08':scene.id==='unified'?'09':{initialization:'01',perception:'02',mapping:'03',telemetry:'04',network:'05',simulator:'02'}[scene.id]??'';
  return <AbsoluteFill><Backdrop/>
    <div style={{position:'absolute',left:80,top:16,width:1760,height:990,border:'1px solid #2c4962',borderRadius:12,overflow:'hidden',boxShadow:'0 15px 65px #0008'}}><OffthreadVideo src={staticFile(scene.file)} muted style={{width:'100%',height:'100%',transform:scene.id==='mapping'?`scale(${interpolate(f,[0,scene.frames-1],[1,1.008])})`:'none',transformOrigin:'55% 55%'}}/></div>
    <div style={{position:'absolute',left:80,bottom:15,right:80,display:'flex',gap:24,alignItems:'center',opacity:a,transform:`translateY(${(1-a)*10}px)`}}><div style={{color:blue,fontSize:23,fontVariantNumeric:'tabular-nums',fontWeight:600,borderRight:'1px solid #304c66',paddingRight:24}}>{number}</div><div><div style={{fontSize:23,fontWeight:600,letterSpacing:-.3}}>{scene.title}</div><div style={{fontSize:16,color:scene.historical?'#e3b269':muted,marginTop:4}}>{scene.subtitle}</div></div><div style={{marginLeft:'auto',fontSize:12,letterSpacing:2,color:muted}}>ARTEMIS-X / SYSTEM DEMO</div></div>
  </AbsoluteFill>;
}
function SensorBridge(props:DemoProps) {
  const f=useCurrentFrame();
  const panels=[['RGB CAMERA',props.rgb],['LOCAL YOLO DETECTION',props.ai],['ALIGNED DEPTH',props.depth],['ONBOARD LOCAL MAP',props.map]];
  return <AbsoluteFill><Backdrop/><div style={{position:'absolute',left:80,top:45,right:80,display:'flex',justifyContent:'space-between',alignItems:'center'}}><div><div style={{fontSize:14,color:blue,letterSpacing:4}}>ONBOARD PERCEPTION PIPELINE</div><h2 style={{fontSize:35,margin:'10px 0 0',fontWeight:500}}>From sensor evidence to spatial awareness.</h2></div><div style={{color:muted,fontSize:16,textAlign:'right'}}>ACTUAL CAPTURE · {props.panelTime} SIM<br/>Synchronized stills, not live dashboard streams</div></div>
    <div style={{position:'absolute',left:160,right:160,top:165,display:'grid',gridTemplateColumns:'1fr 1fr',gap:24}}>{panels.map(([label,file],i)=>{const a=enter(f,i*5);return <div key={label} style={{border:'1px solid #2c4962',background:'#081320',borderRadius:10,overflow:'hidden',opacity:a,transform:`translateY(${(1-a)*10}px)`}}><div style={{padding:'10px 18px',fontSize:14,letterSpacing:2,color:blue}}>{label}</div>{file&&<Img src={staticFile(file)} style={{display:'block',width:'100%',height:370,objectFit:'contain',background:'#000'}}/>}</div>})}</div>
    <div style={{position:'absolute',left:160,bottom:30,fontSize:17,color:muted}}>Frozen YOLO11n · local inference · depth fusion · wheel-odometry frame</div>
  </AbsoluteFill>;
}
function Ending() {
  const f=useCurrentFrame(),a=enter(f,8),b=enter(f,24);
  return <AbsoluteFill><Backdrop/><div style={{position:'absolute',inset:0,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',opacity:a,transform:`translateY(${(1-a)*16}px)`}}><Logo size={135}/><div style={{fontSize:55,letterSpacing:12,marginTop:28}}>ARTEMIS-X</div><div style={{fontSize:26,color:muted,marginTop:25,opacity:b}}>Autonomous Perception. Resilient Communication. Mission Intelligence.</div><div style={{fontSize:16,letterSpacing:4,color:blue,marginTop:60,opacity:b}}>PROJECT C.R.A.T.E.R · ATEC 2026</div></div></AbsoluteFill>;
}
export function ArtemisDemo(props:DemoProps) {
  const {fps}=useVideoConfig();
  const frame=useCurrentFrame();
  const layers=[{id:'intro',start:0,frames:12*fps,node:<Intro {...props}/>},...props.scenes.map(scene=>({id:scene.id,start:scene.startFrame,frames:scene.frames,node:scene.id==='sensors'?<SensorBridge {...props}/>:<UIShot scene={scene}/>})),{id:'ending',start:props.duration-6*fps,frames:6*fps,node:<Ending/>}];
  return <AbsoluteFill style={{background:ink,color:'#eef6ff',fontFamily:'Segoe UI, Arial, sans-serif'}}>
    {layers.map((layer,index)=>{
      const next=layers[index+1];
      const fade=next?interpolate(frame,[next.start,next.start+12],[1,0],{extrapolateLeft:'clamp',extrapolateRight:'clamp'}):1;
      return <Sequence key={layer.id} from={layer.start} durationInFrames={Math.min(layer.frames+(next?12:0),props.duration-layer.start)}><AbsoluteFill style={{opacity:fade,zIndex:layers.length-index}}>{layer.node}</AbsoluteFill></Sequence>;
    })}
    <AbsoluteFill style={{background:'#000',opacity:interpolate(frame,[props.duration-25,props.duration-1],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp',easing:Easing.inOut(Easing.cubic)}),zIndex:100}}/>
  </AbsoluteFill>;
}
