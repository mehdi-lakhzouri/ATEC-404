import React from 'react';
import {Composition, registerRoot} from 'remotion';
import {ArtemisDemo, type DemoProps} from './ArtemisDemo';
import {ArtemisDemoV2, type V2Props} from './v2/ArtemisDemoV2';
const defaults:DemoProps={missionId:'',scenes:[],duration:5400};
const Root=()=> <>
  <Composition id="ArtemisDemo" component={ArtemisDemo} width={1920} height={1080} fps={30} durationInFrames={5400} defaultProps={defaults} calculateMetadata={({props})=>({durationInFrames:props.duration})}/>
  <Composition id="ArtemisDemoV2" component={ArtemisDemoV2} width={1920} height={1080} fps={30} durationInFrames={5400} defaultProps={defaults as V2Props} calculateMetadata={({props})=>({durationInFrames:props.duration})}/>
</>;
registerRoot(Root);
