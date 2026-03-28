var J={backbone:{dim:3072,n_layers:26,head_dim:128,hidden_dim:9216,n_heads:32,n_kv_heads:8,vocab_size:131072,rope_theta:1e6,norm_eps:1e-5},fm:{input_dim:3072,dim:3072,n_layers:3,head_dim:128,hidden_dim:9216,n_heads:32,n_kv_heads:8,nfe:8,cfg_alpha:1.2,rope_theta:1e4,sigma:1e-5,sigma_max:1,n_acoustic_out:36,semantic_vocab:8320},codec:{dim:1024,hidden_dim:4096,head_dim:128,n_heads:8,n_kv_heads:8,semantic_codebook_size:8192,semantic_dim:256,n_acoustic_codebook:36,acoustic_codebook_size:21,sampling_rate:24e3,frame_rate:12.5,patch_size:240,decoder_stages:4,decoder_layers_per_stage:2,decoder_conv_strides:[1,2,2,2],decoder_conv_kernels:[3,4,4,4],attn_sliding_window:16,norm_eps:.01,qk_norm_eps:1e-6,qk_norm:!0,layer_scale:!0,weight_norm_conv:!0}};var ke=`
struct Params {
  M: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_SIZE: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.M) { return; }
  let tid = lid.x;
  var sum: f32 = 0.0;
  let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;
  for (var t: u32 = 0u; t < num_tiles; t++) {
    let k = t * TILE_SIZE + tid;
    if (k < params.K) {
      sum += matrix[row * params.K + k] * vector[k];
    }
  }
  sdata[tid] = sum;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
    workgroupBarrier();
  }
  if (tid == 0u) { output[row] = sdata[0]; }
}
`;var ya=5,xa=50;function Ua(){let u=J;return[{name:"backbone.qkv_proj",M:u.backbone.n_heads*u.backbone.head_dim+2*u.backbone.n_kv_heads*u.backbone.head_dim,K:u.backbone.dim,N:1,count:u.backbone.n_layers,component:"backbone"},{name:"backbone.o_proj",M:u.backbone.dim,K:u.backbone.n_heads*u.backbone.head_dim,N:1,count:u.backbone.n_layers,component:"backbone"},{name:"backbone.ffn_gate_up",M:u.backbone.hidden_dim*2,K:u.backbone.dim,N:1,count:u.backbone.n_layers,component:"backbone"},{name:"backbone.ffn_down",M:u.backbone.dim,K:u.backbone.hidden_dim,N:1,count:u.backbone.n_layers,component:"backbone"},{name:"backbone.lm_head",M:u.backbone.vocab_size,K:u.backbone.dim,N:1,count:1,component:"backbone"},{name:"fm.qkv_proj",M:u.fm.n_heads*u.fm.head_dim+2*u.fm.n_kv_heads*u.fm.head_dim,K:u.fm.dim,N:1,count:u.fm.n_layers*u.fm.nfe*2,component:"fm"},{name:"fm.o_proj",M:u.fm.dim,K:u.fm.n_heads*u.fm.head_dim,N:1,count:u.fm.n_layers*u.fm.nfe*2,component:"fm"},{name:"fm.ffn_gate_up",M:u.fm.hidden_dim*2,K:u.fm.dim,N:1,count:u.fm.n_layers*u.fm.nfe*2,component:"fm"},{name:"fm.ffn_down",M:u.fm.dim,K:u.fm.hidden_dim,N:1,count:u.fm.n_layers*u.fm.nfe*2,component:"fm"},{name:"codec.qkv_proj",M:u.codec.n_heads*u.codec.head_dim+2*u.codec.n_kv_heads*u.codec.head_dim,K:u.codec.dim,N:1,count:u.codec.decoder_stages*u.codec.decoder_layers_per_stage,component:"codec"},{name:"codec.ffn_gate_up",M:u.codec.hidden_dim*2,K:u.codec.dim,N:1,count:u.codec.decoder_stages*u.codec.decoder_layers_per_stage,component:"codec"},{name:"codec.ffn_down",M:u.codec.dim,K:u.codec.hidden_dim,N:1,count:u.codec.decoder_stages*u.codec.decoder_layers_per_stage,component:"codec"}]}async function Ba(){if(!navigator.gpu)throw new Error("WebGPU not supported in this browser");let u=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!u)throw new Error("No WebGPU adapter found");let r=u.features.has("shader-f16"),o=u.features.has("timestamp-query"),t=[];return r&&t.push("shader-f16"),o&&t.push("timestamp-query"),await u.requestDevice({requiredFeatures:t,requiredLimits:{maxBufferSize:1024*1024*1024,maxStorageBufferBindingSize:512*1024*1024}})}function Ca(u,r){let o=u.createShaderModule({code:r});return u.createComputePipeline({layout:"auto",compute:{module:o,entryPoint:"main"}})}async function qa(u,r,o,t,e){let n=e?2:4,s=o*t*n;if(s>u.limits.maxStorageBufferBindingSize)return[-1];let i=u.createBuffer({size:s,usage:GPUBufferUsage.STORAGE,mappedAtCreation:!0});if(e){let c=new Uint16Array(i.getMappedRange());for(let d=0;d<c.length;d++)c[d]=we((Math.random()-.5)*.1)}else{let c=new Float32Array(i.getMappedRange());for(let d=0;d<c.length;d++)c[d]=(Math.random()-.5)*.1}i.unmap();let a=u.createBuffer({size:t*n,usage:GPUBufferUsage.STORAGE,mappedAtCreation:!0});if(e){let c=new Uint16Array(a.getMappedRange());for(let d=0;d<c.length;d++)c[d]=we((Math.random()-.5)*.1)}else{let c=new Float32Array(a.getMappedRange());for(let d=0;d<c.length;d++)c[d]=(Math.random()-.5)*.1}a.unmap();let _=u.createBuffer({size:o*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),m=u.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:!0});new Uint32Array(m.getMappedRange()).set([o,t]),m.unmap();let w=u.createBindGroup({layout:r.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:_}},{binding:3,resource:{buffer:m}}]}),v=[];for(let c=0;c<ya;c++){let d=u.createCommandEncoder(),f=d.beginComputePass();f.setPipeline(r),f.setBindGroup(0,w),f.dispatchWorkgroups(o),f.end(),u.queue.submit([d.finish()])}await u.queue.onSubmittedWorkDone();for(let c=0;c<xa;c++){let d=u.createCommandEncoder(),f=d.beginComputePass();f.setPipeline(r),f.setBindGroup(0,w),f.dispatchWorkgroups(o),f.end();let k=performance.now();u.queue.submit([d.finish()]),await u.queue.onSubmittedWorkDone();let q=performance.now();v.push(q-k)}return i.destroy(),a.destroy(),_.destroy(),m.destroy(),v}function we(u){let r=new ArrayBuffer(4);new Float32Array(r)[0]=u;let o=new Uint32Array(r)[0],t=o>>16&32768,e=(o>>23&255)-127+15,n=o>>13&1023;return e<=0?t:e>=31?t|31744:t|e<<10|n}function Ga(u){let r=[...u].sort((t,e)=>t-e),o=Math.floor(r.length/2);return r.length%2?r[o]:(r[o-1]+r[o])/2}async function Aa(u){let r=u||console.log;r("Initializing WebGPU...");let o=await Ba(),t=o.adapterInfo,e=t?`${t.vendor} ${t.architecture} ${t.device}`:"Unknown GPU",n=o.features.has("shader-f16"),s=o.features.has("timestamp-query");r(`GPU: ${e}`),r(`F16 support: ${n}`),r(`Timestamp queries: ${s}`);let i=Ca(o,ke),a=Ua(),_=[];for(let l of a){r(`
Benchmarking ${l.name} [${l.M} \xD7 ${l.K}] \xD7 ${l.count}...`);let p=await qa(o,i,l.M,l.K,!1);if(p[0]===-1){r(`  SKIPPED \u2014 buffer too large (${(l.M*l.K*4/1e9).toFixed(2)} GB)`),_.push({name:l.name,M:l.M,K:l.K,avgMs:-1,medianMs:-1,minMs:-1,count:l.count,totalMs:-1,component:l.component,gflops:0});continue}let h=p.reduce((y,U)=>y+U,0)/p.length,P=Ga(p),C=Math.min(...p),G=2*l.M*l.K/(P/1e3)/1e9,x={name:l.name,M:l.M,K:l.K,avgMs:h,medianMs:P,minMs:C,count:l.count,totalMs:P*l.count,component:l.component,gflops:G};_.push(x),r(`  median: ${P.toFixed(3)}ms | total (\xD7${l.count}): ${x.totalMs.toFixed(2)}ms | ${G.toFixed(1)} GFLOPS`)}let m=_.filter(l=>l.component==="backbone").reduce((l,p)=>l+p.totalMs,0),w=_.filter(l=>l.component==="fm").reduce((l,p)=>l+p.totalMs,0),v=_.filter(l=>l.component==="codec").reduce((l,p)=>l+p.totalMs,0),c=m+w+v,d=80,f=c<d,k=d/c,q={backbone_total_ms:m,fm_total_ms:w,codec_total_ms:v,total_per_frame_ms:c,target_ms:d,feasible:f,realtime_factor:k};return r(`
========================================`),r("VOXTRAL TTS \u2014 PHASE 0a GO/NO-GO RESULTS"),r("========================================"),r(`Backbone (26 layers):  ${m.toFixed(2)}ms`),r(`FM (3 layers \xD7 16):   ${w.toFixed(2)}ms`),r(`Codec (8 layers):     ${v.toFixed(2)}ms`),r("\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"),r(`Total per frame:      ${c.toFixed(2)}ms`),r(`Target (12.5 fps):    ${d}ms`),r(`Realtime factor:      ${k.toFixed(2)}x`),r(`
VERDICT: ${f?"\u2705 GO \u2014 real-time TTS is feasible!":"\u274C NO-GO \u2014 too slow for real-time"}`),f||(r(`
Note: These are matmul-only times. Real inference adds ~30-50% overhead`),r("for norms, activations, attention, sampling, etc."),r(`
For feasibility, we need matmul total < ~55ms (leaving ~25ms for overhead).`)),o.destroy(),{device:e,hasF16:n,hasTimestamp:s,results:_,summary:q}}async function Pe(u){let o=await(await fetch(u,{headers:{Range:"bytes=0-7"}})).arrayBuffer(),t=Number(new DataView(o).getBigUint64(0,!0)),n=await(await fetch(u,{headers:{Range:`bytes=8-${8+t-1}`}})).text();return{header:JSON.parse(n),dataOffset:8+t}}function fe(u){let r=new Uint16Array(u),o=new Uint16Array(r.length);for(let t=0;t<r.length;t++){let e=r[t],n=e>>15&1,s=e>>7&255,i=e&127;if(s===255)o[t]=n<<15|31744|(i?512:0);else if(s===0)o[t]=n<<15;else{let a=s-127;if(a>15)o[t]=n<<15|31744;else if(a<-14){let _=-14-a;if(_>10)o[t]=n<<15;else{let m=(128|i<<1)>>_>>1;o[t]=n<<15|m&1023}}else{let _=a+15,m=i<<3;o[t]=n<<15|_<<10|m&1023}}}return o.buffer}async function pe(u){let r=await fetch(`${u}/manifest.json`);if(!r.ok)throw new Error(`Failed to load manifest: ${r.status}`);return r.json()}async function ye(u,r,o){let t=r.tensors[o];if(!t)throw new Error(`Tensor not found: ${o}`);let e=`${u}/${t.file}`,s=await(await fetch(e,{headers:{Range:`bytes=${t.offset}-${t.offset+t.size-1}`}})).arrayBuffer();return t.dtype==="f16"?new Uint16Array(s):new Float32Array(s)}async function Sa(u,r,o,t){let e=r[t];if(!e||!("data_offsets"in e))throw new Error(`Tensor not found in safetensors: ${t}`);let[n,s]=e.data_offsets,i=o+n,a=o+s-1,m=await(await fetch(u,{headers:{Range:`bytes=${i}-${a}`}})).arrayBuffer();if(e.dtype==="BF16"){let w=fe(m);return new Uint16Array(w)}else{if(e.dtype==="F16")return new Uint16Array(m);if(e.dtype==="F32"){let w=new Float32Array(m),v=new Uint16Array(w.length);for(let c=0;c<w.length;c++)v[c]=xe(w[c]);return v}else throw new Error(`Unsupported dtype: ${e.dtype}`)}}function xe(u){let r=new ArrayBuffer(4);new Float32Array(r)[0]=u;let o=new Uint32Array(r)[0],t=o>>16&32768,e=(o>>23&255)-127+15,n=o>>13&1023;return e<=0?t:e>=31?t|31744:t|e<<10|n}function _e(u,r,o){let t=r.byteLength,e=Math.ceil(t/4)*4,n=u.createBuffer({size:e,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:o,mappedAtCreation:!0});return new Uint16Array(n.getMappedRange(0,r.byteLength)).set(r),n.unmap(),n}async function Fa(u,r,o,t,e){let n=Object.entries(o.tensors).filter(([_,m])=>m.component===t).map(([_,m])=>_),s=new Map,i=new Map,a=n.length;for(let _=0;_<n.length;_++){let m=n[_],w=o.tensors[m];e&&e({loaded:_,total:a,component:t,tensor:m});let v=await ye(r,o,m),c=_e(u,v,m);s.set(m,c),i.set(m,{shape:w.shape,buffer:c})}return e&&e({loaded:a,total:a,component:t,tensor:"done"}),{buffers:s,tensors:i}}async function ee(u,r,o,t,e){let n=Object.entries(o.tensors).filter(([v,c])=>c.component===t).map(([v,c])=>v),i=o.tensors[n[0]].file;e&&e({loaded:0,total:n.length,component:t,tensor:`downloading ${i}...`});let a=await fetch(`${r}/${i}`);if(!a.ok)throw new Error(`Failed to download ${i}: ${a.status}`);let _=await a.arrayBuffer(),m=new Map,w=new Map;for(let v=0;v<n.length;v++){let c=n[v],d=o.tensors[c],f=new Uint16Array(_,d.offset,d.size/2),k=_e(u,f,c);m.set(c,k),w.set(c,{shape:d.shape,buffer:k}),e&&(v%20===0||v===n.length-1)&&e({loaded:v+1,total:n.length,component:t,tensor:c})}return{buffers:m,tensors:w}}var $a="voxtral-weights",Ma=1,L="tensors";function Ue(){return new Promise((u,r)=>{let o=indexedDB.open($a,Ma);o.onupgradeneeded=()=>{let t=o.result;t.objectStoreNames.contains(L)||t.createObjectStore(L)},o.onsuccess=()=>u(o.result),o.onerror=()=>r(o.error)})}async function Ta(u,r){return new Promise((o,t)=>{let s=u.transaction(L,"readonly").objectStore(L).get(r);s.onsuccess=()=>o(s.result??null),s.onerror=()=>t(s.error)})}async function za(u,r,o){return new Promise((t,e)=>{let i=u.transaction(L,"readwrite").objectStore(L).put(o,r);i.onsuccess=()=>t(),i.onerror=()=>e(i.error)})}async function Ea(u){return new Promise((r,o)=>{let n=u.transaction(L,"readonly").objectStore(L).count();n.onsuccess=()=>r(n.result),n.onerror=()=>o(n.error)})}async function Oa(){let u=await Ue();return new Promise((r,o)=>{let n=u.transaction(L,"readwrite").objectStore(L).clear();n.onsuccess=()=>{u.close(),r()},n.onerror=()=>{u.close(),o(n.error)}})}var ie="https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/consolidated.safetensors";function Ia(u){return u.startsWith("acoustic_transformer.")?"fm":u.startsWith("audio_tokenizer.")?"codec":u.startsWith("layers.")||u.startsWith("norm.")||u.startsWith("mm_audio_embeddings.")?"backbone":"other"}async function le(u,r=ie,o){let{header:t,dataOffset:e}=await Pe(r),n=await Ue(),s=await Ea(n),i=[];for(let[c,d]of Object.entries(t)){if(c==="__metadata__")continue;let f=d;if(!f.data_offsets)continue;let k=Ia(c);k!=="other"&&i.push({name:c,entry:f,component:k})}let a={backbone:0,fm:1,codec:2};i.sort((c,d)=>(a[c.component]??9)-(a[d.component]??9));let _=i.length,m=0;o&&o({loaded:0,total:_,component:"init",tensor:s>0?`${s} tensors cached in IndexedDB`:"Starting fresh download...",cached:!1,bytesDownloaded:0});let w={backbone:{buffers:new Map,tensors:new Map},fm:{buffers:new Map,tensors:new Map},codec:{buffers:new Map,tensors:new Map}},v=`v1:${e}:${_}`;for(let c=0;c<i.length;c++){let{name:d,entry:f,component:k}=i[c],q=`${v}:${d}`,l,p=!1,h=await Ta(n,q);if(h)l=new Uint16Array(h),p=!0;else{let[b,G]=f.data_offsets,x=e+b,y=e+G-1,U=G-b,S=await fetch(r,{headers:{Range:`bytes=${x}-${y}`}});if(!S.ok&&S.status!==206)throw new Error(`Failed to fetch tensor ${d}: HTTP ${S.status}`);let M=await S.arrayBuffer();if(m+=U,f.dtype==="BF16"){let $=fe(M);l=new Uint16Array($)}else if(f.dtype==="F16")l=new Uint16Array(M);else if(f.dtype==="F32"){let $=new Float32Array(M),z=new Uint16Array($.length);for(let T=0;T<$.length;T++)z[T]=xe($[T]);l=z}else throw new Error(`Unsupported dtype for ${d}: ${f.dtype}`);await za(n,q,l.buffer)}let P=_e(u,l,d),C=w[k];C.buffers.set(d,P),C.tensors.set(d,{shape:f.shape,buffer:P}),l=null,o&&o({loaded:c+1,total:_,component:k,tensor:d,cached:p,bytesDownloaded:m})}return n.close(),{backbone:w.backbone,fm:w.fm,codec:w.codec}}var K={UNK:0,BOS:1,EOS:2,INST:3,INST_END:4,AUDIO:24,BEGIN_AUDIO:25,OUTPUT_AUDIO:26,AUDIO_TO_TEXT:35,TEXT_TO_AUDIO:36,PAD:11},he=class u{vocab=new Map;specialTokens=new Map;pattern;voiceNumTokens=new Map;constructor(r){for(let t of r.vocab){let e=atob(t.token_bytes);this.vocab.set(e,t.rank)}let o=r.config.default_num_special_tokens;for(let t of r.special_tokens)this.specialTokens.set(t.token_str,t.rank);try{this.pattern=new RegExp(r.config.pattern,"gu")}catch{this.pattern=/\S+|\s+/gu}if(r.audio?.voice_num_audio_tokens)for(let[t,e]of Object.entries(r.audio.voice_num_audio_tokens))this.voiceNumTokens.set(t,e)}static async load(r){let t=await(await fetch(r)).json();return new u(t)}getVoiceNumTokens(r){let o=this.voiceNumTokens.get(r);if(o===void 0)throw new Error(`Unknown voice: ${r}. Available: ${[...this.voiceNumTokens.keys()].join(", ")}`);return o}buildTTSPrompt(r,o){let t=this.getVoiceNumTokens(o),e=[];e.push(K.BOS),e.push(K.BEGIN_AUDIO);let n=e.length;for(let i=0;i<t;i++)e.push(K.AUDIO);e.push(K.TEXT_TO_AUDIO);let s=this.encode(r);return e.push(...s),e.push(K.AUDIO_TO_TEXT),e.push(K.BEGIN_AUDIO),{tokens:e,audioTokenStart:n,audioTokenCount:t}}encode(r){let o=[],t=r.matchAll(this.pattern);for(let e of t){let n=e[0],s=this.vocab.get(n);if(s!==void 0){o.push(s+1e3);continue}let a=new TextEncoder().encode(n);for(let _ of a){let m=String.fromCharCode(_),w=this.vocab.get(m);w!==void 0?o.push(w+1e3):o.push(K.UNK)}}return o}get voices(){return[...this.voiceNumTokens.keys()]}};var Be=`
struct Params {
  M: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;   // packed f16 [M, K/2]
@group(0) @binding(1) var<storage, read> vector: array<f32>;   // [K]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [M]
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.M) { return; }
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  // Each thread accumulates over a strided portion of K
  var pk = tid;
  let rowBase = row * K_packed;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * vector[k0] + f32(pair.y) * vector[k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  // Tree reduction
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[row] = sdata[0];
  }
}
`,Ce=`
struct Params {
  M: u32,
  K: u32,
  row_offset: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;   // packed f16 [total_M, K/2]
@group(0) @binding(1) var<storage, read> vector: array<f32>;   // [K]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [total_M]
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let localRow = wid.x;
  if (localRow >= params.M) { return; }
  let globalRow = localRow + params.row_offset;
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  var pk = tid;
  let rowBase = globalRow * K_packed;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * vector[k0] + f32(pair.y) * vector[k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[globalRow] = sdata[0];
  }
}
`,qe=`
struct Params {
  dim: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;  // packed f16
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  // Compute sum of squares
  var ss: f32 = 0.0;
  var i = tid;
  while (i < params.dim) {
    let v = input[i];
    ss += v * v;
    i += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(params.dim) + params.eps);

  // Scale and write output
  i = tid;
  while (i < params.dim) {
    let wPacked = weight[i / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (i & 1u) == 1u);
    output[i] = input[i] * rms * w;
    i += WG;
  }
}
`,Ge=`
struct Params {
  token_id: u32,
  dim: u32,
}

@group(0) @binding(0) var<storage, read> table: array<u32>;  // packed f16 [vocab, dim/2]
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let packedIdx = params.token_id * (params.dim / 2u) + i / 2u;
  let packed = table[packedIdx];
  let pair = unpack2x16float(packed);
  output[i] = select(f32(pair.x), f32(pair.y), (i & 1u) == 1u);
}
`,Ae=`
struct Params {
  dim: u32,       // head_dim (128)
  pos: u32,       // sequence position
  n_heads: u32,   // number of heads to process
  theta: f32,     // rope_theta (1e6 for backbone, 1e4 for FM)
}

@group(0) @binding(0) var<storage, read_write> qk: array<f32>;  // [n_heads, dim]
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_pairs = params.n_heads * (params.dim / 2u);
  if (idx >= total_pairs) { return; }

  let head = idx / (params.dim / 2u);
  let pair = idx % (params.dim / 2u);

  let freq = 1.0 / pow(params.theta, f32(pair * 2u) / f32(params.dim));
  let angle = f32(params.pos) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let base = head * params.dim + pair * 2u;
  let x0 = qk[base];
  let x1 = qk[base + 1u];

  qk[base]     = x0 * cos_a - x1 * sin_a;
  qk[base + 1u] = x0 * sin_a + x1 * cos_a;
}
`,Se=`
struct Params {
  dim: u32,       // head_dim (128)
  pos: u32,       // sequence position for RoPE angle computation
  n_heads: u32,   // number of heads to process
  theta: f32,     // rope_theta
  offset: u32,    // element offset into qk buffer
}

@group(0) @binding(0) var<storage, read_write> qk: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_pairs = params.n_heads * (params.dim / 2u);
  if (idx >= total_pairs) { return; }

  let head = idx / (params.dim / 2u);
  let pair = idx % (params.dim / 2u);

  let freq = 1.0 / pow(params.theta, f32(pair * 2u) / f32(params.dim));
  let angle = f32(params.pos) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let base = params.offset + head * params.dim + pair * 2u;
  let x0 = qk[base];
  let x1 = qk[base + 1u];

  qk[base]     = x0 * cos_a - x1 * sin_a;
  qk[base + 1u] = x0 * sin_a + x1 * cos_a;
}
`,Fe=`
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,      // current position + 1
  kv_repeat: u32,    // n_heads / n_kv_heads
}

@group(0) @binding(0) var<storage, read> q: array<f32>;         // [n_heads, head_dim]
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;   // [max_seq, n_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>; // [n_heads, seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let p = idx % params.seq_len;

  let kv_h = h / params.kv_repeat;
  let scale = 1.0 / sqrt(f32(params.head_dim));

  var dot: f32 = 0.0;
  let qBase = h * params.head_dim;
  let kBase = p * params.n_kv_heads * params.head_dim + kv_h * params.head_dim;

  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[qBase + d] * k_cache[kBase + d];
  }

  scores[h * params.seq_len + p] = dot * scale;
}
`,$e=`
struct Params {
  n_heads: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>; // [n_heads, seq_len]
@group(0) @binding(1) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let h = wid.x;
  if (h >= params.n_heads) { return; }
  let tid = lid.x;
  let base = h * params.seq_len;

  // Find max
  var maxVal: f32 = -1e30;
  var i = tid;
  while (i < params.seq_len) {
    maxVal = max(maxVal, scores[base + i]);
    i += WG;
  }
  sdata[tid] = maxVal;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] = max(sdata[tid], sdata[tid + s]); }
    workgroupBarrier();
  }
  let globalMax = sdata[0];

  // Compute exp and sum
  var expSum: f32 = 0.0;
  i = tid;
  while (i < params.seq_len) {
    let e = exp(scores[base + i] - globalMax);
    scores[base + i] = e;
    expSum += e;
    i += WG;
  }
  sdata[tid] = expSum;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }
  let totalSum = sdata[0];

  // Normalize
  i = tid;
  while (i < params.seq_len) {
    scores[base + i] /= totalSum;
    i += WG;
  }
}
`,Me=`
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,
  kv_repeat: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;     // [n_heads, seq_len]
@group(0) @binding(1) var<storage, read> v_cache: array<f32>;    // [max_seq, n_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [n_heads * head_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.head_dim;
  if (idx >= total) { return; }

  let h = idx / params.head_dim;
  let d = idx % params.head_dim;
  let kv_h = h / params.kv_repeat;

  var sum: f32 = 0.0;
  for (var p: u32 = 0u; p < params.seq_len; p++) {
    let score = scores[h * params.seq_len + p];
    let vIdx = p * params.n_kv_heads * params.head_dim + kv_h * params.head_dim + d;
    sum += score * v_cache[vIdx];
  }

  output[idx] = sum;
}
`,Te=`
struct Params {
  pos: u32,
  kv_dim: u32,  // n_kv_heads * head_dim
}

@group(0) @binding(0) var<storage, read> k_new: array<f32>;    // [kv_dim]
@group(0) @binding(1) var<storage, read> v_new: array<f32>;    // [kv_dim]
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>; // [max_seq, kv_dim]
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.kv_dim) { return; }

  let cacheIdx = params.pos * params.kv_dim + i;
  k_cache[cacheIdx] = k_new[i];
  v_cache[cacheIdx] = v_new[i];
}
`,ze=`
struct Params {
  dim: u32,
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;   // [hidden_dim] from w1 (in-place output)
@group(0) @binding(1) var<storage, read> up: array<f32>;     // [hidden_dim] from w3
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let g = gate[i];
  let silu = g / (1.0 + exp(-g));
  gate[i] = silu * up[i];
}
`,Ee=`
struct Params {
  dim: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  output[i] = a[i] + b[i];
}
`,Oe=`
struct Params {
  dim: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  a[i] = a[i] + b[i];
}
`,Ie=`
struct Params {
  size: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.size) { return; }
  dst[i] = src[i];
}
`,Le=`
struct Params {
  dim: u32,
  t: f32,  // timestep value [0, 1]
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;  // [dim]
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let half_dim = params.dim / 2u;
  if (i >= half_dim) { return; }

  // Match vLLM-Omni/MLX: freq = exp(-log(10000) * i / half_dim), (cos, sin) order
  let freq = exp(-9.210340371976184 * f32(i) / f32(half_dim));
  let angle = params.t * freq;

  // Output layout: [cos_0, cos_1, ..., cos_{n-1}, sin_0, sin_1, ..., sin_{n-1}]
  output[i] = cos(angle);
  output[half_dim + i] = sin(angle);
}
`,We=`
struct Params {
  dim: u32,
  dt: f32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;     // [dim] in-place
@group(0) @binding(1) var<storage, read> velocity: array<f32>;     // [dim]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  x[i] = x[i] + velocity[i] * params.dt;
}
`,Ne=`
struct Params {
  dim: u32,
  alpha: f32,
}

@group(0) @binding(0) var<storage, read_write> v_cond: array<f32>;  // in-place output
@group(0) @binding(1) var<storage, read> v_uncond: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  v_cond[i] = params.alpha * v_cond[i] + (1.0 - params.alpha) * v_uncond[i];
}
`,De=`
struct Params {
  dim: u32,
  levels: u32,      // 21
  offset: u32,      // 2 (special tokens)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;       // [dim] continuous
@group(0) @binding(1) var<storage, read_write> output: array<u32>; // [dim] quantized codes
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let clamped = clamp(input[i], -1.0, 1.0);
  let scaled = (clamped + 1.0) * 0.5 * f32(params.levels - 1u);
  output[i] = u32(round(scaled)) + params.offset;
}
`,Re=`
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,     // 3 for FM
  kv_repeat: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;   // [seq_len * n_heads * head_dim]
@group(0) @binding(1) var<storage, read> k: array<f32>;   // [seq_len * n_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>; // [n_heads * seq_len * seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / (params.seq_len * params.seq_len);
  let rem = idx % (params.seq_len * params.seq_len);
  let qi = rem / params.seq_len;
  let ki = rem % params.seq_len;
  let kv_h = h / params.kv_repeat;

  let scale = 1.0 / sqrt(f32(params.head_dim));
  var dot: f32 = 0.0;

  let qBase = qi * params.n_heads * params.head_dim + h * params.head_dim;
  let kBase = ki * params.n_kv_heads * params.head_dim + kv_h * params.head_dim;

  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[qBase + d] * k[kBase + d];
  }

  scores[idx] = dot * scale;
}
`,je=`
struct Params {
  n_heads: u32,
  seq_len: u32,  // 3
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let qi = idx % params.seq_len;
  let base = h * params.seq_len * params.seq_len + qi * params.seq_len;

  // Find max
  var maxVal: f32 = -1e30;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    maxVal = max(maxVal, scores[base + j]);
  }

  // Exp and sum
  var expSum: f32 = 0.0;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    let e = exp(scores[base + j] - maxVal);
    scores[base + j] = e;
    expSum += e;
  }

  // Normalize
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    scores[base + j] /= expSum;
  }
}
`,Ve=`
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,
  kv_repeat: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.seq_len * params.n_heads * params.head_dim;
  if (idx >= total) { return; }

  let qi = idx / (params.n_heads * params.head_dim);
  let rem = idx % (params.n_heads * params.head_dim);
  let h = rem / params.head_dim;
  let d = rem % params.head_dim;
  let kv_h = h / params.kv_repeat;

  var sum: f32 = 0.0;
  let scoreBase = h * params.seq_len * params.seq_len + qi * params.seq_len;
  for (var ki: u32 = 0u; ki < params.seq_len; ki++) {
    let score = scores[scoreBase + ki];
    let vIdx = ki * params.n_kv_heads * params.head_dim + kv_h * params.head_dim + d;
    sum += score * v[vIdx];
  }

  output[idx] = sum;
}
`,Ke=`
struct Params {
  n_frames: u32,
  codebook_dim: u32,  // 256
}

@group(0) @binding(0) var<storage, read> codes: array<u32>;     // [n_frames] semantic codes
@group(0) @binding(1) var<storage, read> codebook: array<u32>;  // packed f16 [8192, 256/2]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [n_frames, 256]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_frames * params.codebook_dim;
  if (idx >= total) { return; }

  let t = idx / params.codebook_dim;
  let d = idx % params.codebook_dim;

  let code = codes[t];
  let packedIdx = code * (params.codebook_dim / 2u) + d / 2u;
  let packed = codebook[packedIdx];
  let pair = unpack2x16float(packed);
  output[idx] = select(f32(pair.x), f32(pair.y), (d & 1u) == 1u);
}
`,He=`
struct Params {
  n_entries: u32,
  dim: u32,  // 256
  epsilon: f32,
}

@group(0) @binding(0) var<storage, read_write> codebook: array<u32>;  // packed f16 [n_entries, dim/2]
@group(0) @binding(1) var<storage, read> usage: array<u32>;           // packed f16 [n_entries/2]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_entries * (params.dim / 2u);
  if (idx >= total) { return; }

  let entry = idx / (params.dim / 2u);

  // Read usage for this entry (packed f16)
  let usage_pair_idx = entry / 2u;
  let usage_packed = usage[usage_pair_idx];
  let usage_pair = unpack2x16float(usage_packed);
  let usage_val = max(select(f32(usage_pair.x), f32(usage_pair.y), (entry & 1u) == 1u), params.epsilon);

  // Read codebook pair, divide by usage, write back
  let packed = codebook[idx];
  let pair = unpack2x16float(packed);
  let x = f32(pair.x) / usage_val;
  let y = f32(pair.y) / usage_val;
  codebook[idx] = pack2x16float(vec2<f32>(x, y));
}
`,Ye=`
struct Params {
  n_frames: u32,
  n_codebook: u32,   // 36
  levels: u32,        // 21
  offset: u32,        // 2 (special tokens to subtract)
}

@group(0) @binding(0) var<storage, read> codes: array<u32>;       // [n_frames, 36]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [n_frames, 36]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_frames * params.n_codebook;
  if (idx >= total) { return; }

  let code = codes[idx] - params.offset;
  output[idx] = f32(code) * 2.0 / f32(params.levels - 1u) - 1.0;
}
`,Qe=`
struct Params {
  c_in: u32,
  c_out: u32,
  kernel: u32,
  n_frames: u32,
  stride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;    // [n_frames_in, c_in] (time-first)
@group(0) @binding(1) var<storage, read> weight: array<u32>;   // packed f16 [c_out, c_in, kernel]
@group(0) @binding(2) var<storage, read> g: array<u32>;        // packed f16 [c_out, 1, 1] (weight norm scale)
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [n_frames_out, c_out] (time-first)
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let n_frames_out = params.n_frames;
  let total = params.c_out * n_frames_out;
  if (idx >= total) { return; }

  let co = idx / n_frames_out;
  let t_out = idx % n_frames_out;

  // Get weight norm scale
  let gPacked = g[co / 2u];
  let gPair = unpack2x16float(gPacked);
  let gVal = select(f32(gPair.x), f32(gPair.y), (co & 1u) == 1u);

  // Compute weight L2 norm for this output channel
  var wNorm: f32 = 0.0;
  let wBase = co * params.c_in * params.kernel;
  let wSize = params.c_in * params.kernel;
  let wPacked = wSize / 2u;
  for (var p: u32 = 0u; p < wPacked; p++) {
    let packed = weight[wBase / 2u + p];
    let pair = unpack2x16float(packed);
    wNorm += f32(pair.x) * f32(pair.x) + f32(pair.y) * f32(pair.y);
  }
  wNorm = 1.0 / sqrt(wNorm + 1e-12);

  // Convolution with causal padding \u2014 input is [T, C] layout
  var sum: f32 = 0.0;
  let pad = params.kernel - 1u;
  let n_frames_in = params.n_frames * params.stride;

  for (var ci: u32 = 0u; ci < params.c_in; ci++) {
    for (var k: u32 = 0u; k < params.kernel; k++) {
      let t_in = i32(t_out * params.stride) - i32(pad) + i32(k);
      if (t_in >= 0 && u32(t_in) < n_frames_in) {
        let wIdx = (co * params.c_in + ci) * params.kernel + k;
        let wPacked2 = weight[wIdx / 2u];
        let wPair2 = unpack2x16float(wPacked2);
        let w = select(f32(wPair2.x), f32(wPair2.y), (wIdx & 1u) == 1u);
        // Time-first input: input[t, ci] = input[t * c_in + ci]
        let x = input[u32(t_in) * params.c_in + ci];
        sum += (w * wNorm * gVal) * x;
      }
    }
  }

  // Time-first output: output[t_out, co] = output[t_out * c_out + co]
  output[t_out * params.c_out + co] = sum;
}
`,Xe=`
struct Params {
  c_in: u32,
  c_out: u32,
  kernel: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<u32>;   // packed f16 [c_in, c_out, kernel]
@group(0) @binding(1) var<storage, read> g: array<u32>;        // packed f16 [c_in]
@group(0) @binding(2) var<storage, read_write> scale: array<f32>; // [c_in] output
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let ci = wid.x;
  if (ci >= params.c_in) { return; }
  let tid = lid.x;

  // Compute ||v[ci, :, :]||^2 with parallel reduction
  let total_elems = params.c_out * params.kernel;
  let base = ci * params.c_out * params.kernel;
  var ss: f32 = 0.0;
  var idx = tid;
  while (idx < total_elems) {
    let wIdx = base + idx;
    let packed = weight[wIdx / 2u];
    let pair = unpack2x16float(packed);
    let w = select(f32(pair.x), f32(pair.y), (wIdx & 1u) == 1u);
    ss += w * w;
    idx += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let normInv = 1.0 / sqrt(sdata[0] + 1e-12);
    // Read g[ci]
    let gPacked = g[ci / 2u];
    let gPair = unpack2x16float(gPacked);
    let gVal = select(f32(gPair.x), f32(gPair.y), (ci & 1u) == 1u);
    scale[ci] = gVal * normInv;
  }
}
`,Ze=`
struct Params {
  c_in: u32,
  c_out: u32,
  kernel: u32,
  n_frames_out: u32,  // T_in * stride
  stride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;    // [T_in, c_in]
@group(0) @binding(1) var<storage, read> weight: array<u32>;   // packed f16 [c_in, c_out, kernel]
@group(0) @binding(2) var<storage, read> scale: array<f32>;    // [c_in] precomputed g[ci]/||v[ci,:,:]||
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [n_frames_out, c_out]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.c_out * params.n_frames_out;
  if (idx >= total) { return; }

  let co = idx / params.n_frames_out;
  let t_out = idx % params.n_frames_out;

  // Transposed convolution with per-c_in weight normalization
  var sum: f32 = 0.0;
  let n_frames_in = params.n_frames_out / params.stride;

  for (var k: u32 = 0u; k < params.kernel; k++) {
    let diff = i32(t_out) - i32(k);
    if (diff >= 0 && (u32(diff) % params.stride) == 0u) {
      let t_in = u32(diff) / params.stride;
      if (t_in < n_frames_in) {
        for (var ci: u32 = 0u; ci < params.c_in; ci++) {
          let wIdx = (ci * params.c_out + co) * params.kernel + k;
          let wPacked = weight[wIdx / 2u];
          let wPair = unpack2x16float(wPacked);
          let w = select(f32(wPair.x), f32(wPair.y), (wIdx & 1u) == 1u);
          let x = input[t_in * params.c_in + ci];
          sum += (w * scale[ci]) * x;
        }
      }
    }
  }

  output[t_out * params.c_out + co] = sum;
}
`,Je=`
struct Params {
  dim: u32,
  n_frames: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<u32>;  // packed f16 [dim]
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.dim * params.n_frames;
  if (idx >= total) { return; }

  let d = idx % params.dim;
  let sPacked = scale[d / 2u];
  let sPair = unpack2x16float(sPacked);
  let s = select(f32(sPair.x), f32(sPair.y), (d & 1u) == 1u);

  output[idx] = input[idx] * s + residual[idx];
}
`,ea=`
struct Params {
  dim: u32,
}

struct MaxResult {
  value: f32,
  index: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;  // [1] index
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;

struct SharedEntry {
  value: f32,
  index: u32,
}
var<workgroup> sdata: array<SharedEntry, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  // Each thread finds max over its portion
  var bestVal: f32 = -1e30;
  var bestIdx: u32 = 0u;

  var i = tid;
  while (i < params.dim) {
    let v = input[i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
    i += WG;
  }

  sdata[tid] = SharedEntry(bestVal, bestIdx);
  workgroupBarrier();

  // Reduction
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      if (sdata[tid + s].value > sdata[tid].value) {
        sdata[tid] = sdata[tid + s];
      }
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    result[0] = sdata[0].index;
  }
}
`,aa=`
struct Params {
  M: u32,
  K: u32,
  src_offset: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.M) { return; }
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  var pk = tid;
  let rowBase = row * K_packed;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * vector[params.src_offset + k0]
         + f32(pair.y) * vector[params.src_offset + k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[params.dst_offset + row] = sdata[0];
  }
}
`,ta=`
struct Params {
  dim: u32,
  eps: f32,
  src_offset: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  var ss: f32 = 0.0;
  var i = tid;
  while (i < params.dim) {
    let v = input[params.src_offset + i];
    ss += v * v;
    i += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(params.dim) + params.eps);

  i = tid;
  while (i < params.dim) {
    let wPacked = weight[i / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (i & 1u) == 1u);
    output[params.dst_offset + i] = input[params.src_offset + i] * rms * w;
    i += WG;
  }
}
`,ra=`
struct Params {
  dim: u32,
  off_a: u32,
  off_b: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  output[params.dst_offset + i] = a[params.off_a + i] + b[params.off_b + i];
}
`,sa=`
struct Params {
  dim: u32,
  off_a: u32,
  off_b: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  a[params.off_a + i] = a[params.off_a + i] + b[params.off_b + i];
}
`,oa=`
struct Params {
  dim: u32,
  off_gate: u32,
  off_up: u32,
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let g = gate[params.off_gate + i];
  let silu = g / (1.0 + exp(-g));
  gate[params.off_gate + i] = silu * up[params.off_up + i];
}
`,ia=`
struct Params {
  size: u32,
  src_offset: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.size) { return; }
  dst[params.dst_offset + i] = src[params.src_offset + i];
}
`,na=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  window: u32,      // sliding window size (0 = full attention)
}

@group(0) @binding(0) var<storage, read> q: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(1) var<storage, read> k: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;  // [n_heads, seq_len, seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // gid.x = ki_group (within seq_len), gid.y = qi, gid.z = head
  let ki = gid.x;
  let qi = gid.y;
  let h = gid.z;
  if (ki >= params.seq_len || qi >= params.seq_len || h >= params.n_heads) { return; }
  let idx = h * params.seq_len * params.seq_len + qi * params.seq_len + ki;

  // Causal mask: ki > qi means future \u2014 mask out
  if (ki > qi) {
    scores[idx] = -1e30;
    return;
  }

  // Sliding window mask
  if (params.window > 0u && qi - ki >= params.window) {
    scores[idx] = -1e30;
    return;
  }

  let scale = 1.0 / sqrt(f32(params.head_dim));

  // ALiBi slope: 2^(-8*h/n_heads) \u2014 geometric series from 2^(-8/n) to 2^(-8)
  let slope = pow(2.0, -8.0 * f32(h + 1u) / f32(params.n_heads));
  let alibi = -slope * f32(qi - ki);

  var dot: f32 = 0.0;
  let qBase = qi * params.n_heads * params.head_dim + h * params.head_dim;
  let kBase = ki * params.n_heads * params.head_dim + h * params.head_dim;

  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[qBase + d] * k[kBase + d];
  }

  scores[idx] = dot * scale + alibi;
}
`,da=`
struct Params {
  n_heads: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let qi = idx % params.seq_len;
  let base = h * params.seq_len * params.seq_len + qi * params.seq_len;

  var maxVal: f32 = -1e30;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    maxVal = max(maxVal, scores[base + j]);
  }

  var expSum: f32 = 0.0;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    let e = exp(scores[base + j] - maxVal);
    scores[base + j] = e;
    expSum += e;
  }

  for (var j: u32 = 0u; j < params.seq_len; j++) {
    scores[base + j] /= expSum;
  }
}
`,ca=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;   // [n_heads, seq_len, seq_len]
@group(0) @binding(1) var<storage, read> v: array<f32>;        // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [seq_len, n_heads, head_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.seq_len * params.n_heads * params.head_dim;
  if (idx >= total) { return; }

  let qi = idx / (params.n_heads * params.head_dim);
  let rem = idx % (params.n_heads * params.head_dim);
  let h = rem / params.head_dim;
  let d = rem % params.head_dim;

  var sum: f32 = 0.0;
  let scoreBase = h * params.seq_len * params.seq_len + qi * params.seq_len;
  for (var ki: u32 = 0u; ki < params.seq_len; ki++) {
    let score = scores[scoreBase + ki];
    let vIdx = ki * params.n_heads * params.head_dim + h * params.head_dim + d;
    sum += score * v[vIdx];
  }

  output[idx] = sum;
}
`,ua=`
struct Params {
  M: u32,
  K: u32,
  T: u32,  // number of frames
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  let frame = wid.y;
  if (row >= params.M || frame >= params.T) { return; }
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  var pk = tid;
  let rowBase = row * K_packed;
  let inputBase = frame * params.K;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * input[inputBase + k0]
         + f32(pair.y) * input[inputBase + k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[frame * params.M + row] = sdata[0];
  }
}
`,ma=`
struct Params {
  dim: u32,
  eps: f32,
  T: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let frame = wid.x;
  if (frame >= params.T) { return; }
  let tid = lid.x;
  let base = frame * params.dim;

  var ss: f32 = 0.0;
  var i = tid;
  while (i < params.dim) {
    let v = input[base + i];
    ss += v * v;
    i += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(params.dim) + params.eps);

  i = tid;
  while (i < params.dim) {
    let wPacked = weight[i / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (i & 1u) == 1u);
    output[base + i] = input[base + i] * rms * w;
    i += WG;
  }
}
`,fa=`
struct Params {
  total: u32,  // T * hidden_dim
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;  // in-place output
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  let g = gate[i];
  let silu = g / (1.0 + exp(-g));
  gate[i] = silu * up[i];
}
`,pa=`
struct Params {
  total: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  output[i] = a[i] + b[i];
}
`,_a=`
struct Params {
  total: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  dst[i] = src[i];
}
`,la=`
struct Params {
  dim: u32,
  total: u32,  // T * dim
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<u32>;  // packed f16 [dim]
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total) { return; }

  let d = idx % params.dim;
  let sPacked = scale[d / 2u];
  let sPair = unpack2x16float(sPacked);
  let s = select(f32(sPair.x), f32(sPair.y), (d & 1u) == 1u);

  output[idx] = input[idx] * s + residual[idx];
}
`,ha=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;  // [seq_len, n_heads * head_dim]
@group(0) @binding(1) var<storage, read> weight: array<u32>;  // packed f16 [n_heads * head_dim]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.seq_len * params.n_heads;
  if (idx >= total) { return; }

  let base = idx * params.head_dim;
  // Head index for weight lookup: weight is [n_heads * head_dim], each head has head_dim elements
  let h = idx % params.n_heads;
  let wOffset = h * params.head_dim;

  // Compute RMS
  var ss: f32 = 0.0;
  for (var d: u32 = 0u; d < params.head_dim; d++) {
    let v = data[base + d];
    ss += v * v;
  }
  let rms = 1.0 / sqrt(ss / f32(params.head_dim) + params.eps);

  // Normalize with per-head weight (in-place)
  for (var d: u32 = 0u; d < params.head_dim; d++) {
    let wIdx = wOffset + d;
    let wPacked = weight[wIdx / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (wIdx & 1u) == 1u);
    data[base + d] = data[base + d] * rms * w;
  }
}
`,ga=`
struct Params {
  T: u32,
  sem_dim: u32,   // 256
  ac_dim: u32,    // 36
}

@group(0) @binding(0) var<storage, read> semantic: array<f32>;  // [T, 256]
@group(0) @binding(1) var<storage, read> acoustic: array<f32>;  // [T, 36]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [T, 292]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let out_dim = params.sem_dim + params.ac_dim;
  let total = params.T * out_dim;
  if (idx >= total) { return; }

  let t = idx / out_dim;
  let d = idx % out_dim;

  if (d < params.sem_dim) {
    output[idx] = semantic[t * params.sem_dim + d];
  } else {
    output[idx] = acoustic[t * params.ac_dim + (d - params.sem_dim)];
  }
}
`,ba=`
struct Params {
  dim: u32,
  acoustic_base: u32,    // 8194 (semantic_codebook_size + 2 specials)
  acoustic_stride: u32,  // 23 (acoustic_codebook_size + 2 specials)
  n_acoustic: u32,       // 36
}

@group(0) @binding(0) var<storage, read> table: array<u32>;           // packed f16 [vocab, dim/2]
@group(0) @binding(1) var<storage, read> semantic_code: array<u32>;   // [1] semantic argmax
@group(0) @binding(2) var<storage, read> acoustic_codes: array<u32>;  // [36] FSQ codes with +2 offset
@group(0) @binding(3) var<storage, read_write> output: array<f32>;    // [dim]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let half_i = i / 2u;
  let is_odd = (i & 1u) == 1u;
  var sum: f32 = 0.0;

  // Semantic codebook (offset 0)
  let sem_row = semantic_code[0];
  let sem_packed = table[sem_row * (params.dim / 2u) + half_i];
  let sem_pair = unpack2x16float(sem_packed);
  sum += select(f32(sem_pair.x), f32(sem_pair.y), is_odd);

  // 36 acoustic codebooks
  for (var k: u32 = 0u; k < params.n_acoustic; k++) {
    let ac_row = params.acoustic_base + k * params.acoustic_stride + acoustic_codes[k];
    let ac_packed = table[ac_row * (params.dim / 2u) + half_i];
    let ac_pair = unpack2x16float(ac_packed);
    sum += select(f32(ac_pair.x), f32(ac_pair.y), is_odd);
  }

  output[i] = sum;
}
`,va=`
struct Params {
  total: u32,
}

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  buf[i] = 0.0;
}
`;function g(u,r){return Math.ceil(u/r)}function Wa(u,r,o){let t=u.length,e=new Float32Array(t);for(let c=0;c<t;c++)e[c]=u[c]/o;let n=-1/0;for(let c=0;c<t;c++)e[c]>n&&(n=e[c]);let s=0;for(let c=0;c<t;c++)e[c]=Math.exp(e[c]-n),s+=e[c];for(let c=0;c<t;c++)e[c]/=s;let i=Array.from({length:t},(c,d)=>d);i.sort((c,d)=>e[d]-e[c]);let a=0,_=t;for(let c=0;c<t;c++)if(a+=e[i[c]],a>=r){_=c+1;break}let m=0;for(let c=0;c<_;c++)m+=e[i[c]];let w=Math.random()*m,v=0;for(let c=0;c<_;c++)if(v+=e[i[c]],v>=w)return i[c];return i[0]}var ge=class{device=null;config;maxSeqLen;modelBuffers=null;workBuffers=null;pipelines=null;kvCaches=[];position=0;constructor(r={}){this.config=r.config||J,this.maxSeqLen=r.maxSeqLen||4096}async init(){if(!navigator.gpu)throw new Error("WebGPU not supported");let r=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!r)throw new Error("No WebGPU adapter");let o=[];r.features.has("shader-f16")&&o.push("shader-f16");let t=2*1024*1024*1024,e=r.limits.maxBufferSize,n=r.limits.maxStorageBufferBindingSize,s=e>=t&&n>=t;this.device=await r.requestDevice({requiredFeatures:o,requiredLimits:{maxBufferSize:s?t:e,maxStorageBufferBindingSize:s?t:n}}),this.createWorkBuffers(),this.createKVCaches(),this.createPipelines()}createPipeline(r,o){let t=this.device,e=t.createShaderModule({code:r,label:o});return t.createComputePipeline({layout:"auto",compute:{module:e,entryPoint:"main"},label:o})}createPipelines(){let r=(o,t)=>this.createPipeline(o,t);this.pipelines={matvecF16:r(Be,"matvecF16"),matvecF16Chunked:r(Ce,"matvecF16Chunked"),matvecF16Offset:r(aa,"matvecF16Offset"),rmsNorm:r(qe,"rmsNorm"),rmsNormOffset:r(ta,"rmsNormOffset"),embeddingLookup:r(Ge,"embeddingLookup"),rope:r(Ae,"rope"),ropeOffset:r(Se,"ropeOffset"),attnScore:r(Fe,"attnScore"),softmax:r($e,"softmax"),attnValue:r(Me,"attnValue"),kvCacheWrite:r(Te,"kvCacheWrite"),swiGLU:r(ze,"swiGLU"),addVectors:r(Ee,"addVectors"),addVectorsOffset:r(ra,"addVectorsOffset"),addInPlace:r(Oe,"addInPlace"),addInPlaceOffset:r(sa,"addInPlaceOffset"),copyBuffer:r(Ie,"copyBuffer"),copyBufferOffset:r(ia,"copyBufferOffset"),timeEmbedding:r(Le,"timeEmbedding"),eulerStep:r(We,"eulerStep"),cfgCombine:r(Ne,"cfgCombine"),fsqQuantize:r(De,"fsqQuantize"),biAttnScore:r(Re,"biAttnScore"),biSoftmax:r(je,"biSoftmax"),biAttnValue:r(Ve,"biAttnValue"),swiGLUOffset:r(oa,"swiGLUOffset"),zeroFill:r(va,"zeroFill"),multiCodebookEmbed:r(ba,"multiCodebookEmbed"),vqLookup:r(Ke,"vqLookup"),fsqDequant:r(Ye,"fsqDequant"),causalConv1d:r(Qe,"causalConv1d"),causalConvTranspose1d:r(Ze,"causalConvTranspose1d"),convTransposeNormScale:r(Xe,"convTransposeNormScale"),layerScale:r(Je,"layerScale"),alibiAttnScore:r(na,"alibiAttnScore"),codecSoftmax:r(da,"codecSoftmax"),codecAttnValue:r(ca,"codecAttnValue"),batchedMatvecF16:r(ua,"batchedMatvecF16"),batchedRmsNorm:r(ma,"batchedRmsNorm"),batchedSwiGLU:r(fa,"batchedSwiGLU"),batchedAdd:r(pa,"batchedAdd"),batchedCopy:r(_a,"batchedCopy"),batchedLayerScale:r(la,"batchedLayerScale"),qkNorm:r(ha,"qkNorm"),concatCodecInput:r(ga,"concatCodecInput"),argmax:r(ea,"argmax"),normalizeCodebook:r(He,"normalizeCodebook")}}createUniform(r){let o=this.device,t=Math.ceil(r.byteLength/16)*16,e=o.createBuffer({size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(e.getMappedRange()).set(new Uint8Array(r)),e.unmap(),e}packUniform(r){let o=new ArrayBuffer(r.length*4),t=new Uint32Array(o),e=new Float32Array(o);for(let n=0;n<r.length;n++){let s=r[n];s.u!==void 0?t[n]=s.u:s.f!==void 0&&(e[n]=s.f)}return this.createUniform(o)}createWorkBuffers(){let r=this.device,o=this.config.backbone,t=this.config.fm,e=GPUBufferUsage.STORAGE,n=GPUBufferUsage.COPY_SRC,s=GPUBufferUsage.COPY_DST,i=(a,_,m=0)=>r.createBuffer({size:a,usage:e|n|s|m,label:_});this.workBuffers={hidden:i(o.dim*4,"hidden"),residual:i(o.dim*4,"residual"),normed:i(o.dim*4,"normed"),q:i(o.n_heads*o.head_dim*4,"q"),k:i(o.n_kv_heads*o.head_dim*4,"k"),v:i(o.n_kv_heads*o.head_dim*4,"v"),attn_out:i(o.n_heads*o.head_dim*4,"attn_out"),scores:i(o.n_heads*this.maxSeqLen*4,"scores"),gate:i(o.hidden_dim*4,"gate"),up:i(o.hidden_dim*4,"up"),down:i(o.dim*4,"down"),x_t:i(t.n_acoustic_out*4,"x_t"),velocity:i(t.n_acoustic_out*4,"velocity"),v_uncond:i(t.n_acoustic_out*4,"v_uncond"),time_embed:i(t.dim*4,"time_embed"),time_proj:i(t.dim*4,"time_proj"),x_t_proj:i(t.dim*4,"x_t_proj"),fm_hidden:i(t.dim*4,"fm_hidden"),fm_residual:i(t.dim*4,"fm_residual"),fm_normed:i(t.dim*4,"fm_normed"),fm_q:i(3*t.n_heads*t.head_dim*4,"fm_q"),fm_k:i(3*t.n_kv_heads*t.head_dim*4,"fm_k"),fm_v:i(3*t.n_kv_heads*t.head_dim*4,"fm_v"),fm_attn_out:i(3*t.n_heads*t.head_dim*4,"fm_attn_out"),fm_scores:i(t.n_heads*3*3*4,"fm_scores"),fm_seq:i(3*t.dim*4,"fm_seq"),fm_gate:i(3*t.hidden_dim*4,"fm_gate"),fm_up:i(3*t.hidden_dim*4,"fm_up"),fm_down:i(3*t.dim*4,"fm_down"),semantic_logits:i(t.semantic_vocab*4,"semantic_logits"),semantic_argmax:i(4,"semantic_argmax"),acoustic_out:i(t.n_acoustic_out*4,"acoustic_out"),acoustic_codes:i(t.n_acoustic_out*4,"acoustic_codes"),logits:i(o.vocab_size*4,"logits"),argmax_result:i(4,"argmax_result")}}createKVCaches(){let r=this.device,o=this.config.backbone,t=o.n_kv_heads*o.head_dim;this.kvCaches=[];for(let e=0;e<o.n_layers;e++)this.kvCaches.push({k:r.createBuffer({size:this.maxSeqLen*t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:`kv_cache.${e}.k`}),v:r.createBuffer({size:this.maxSeqLen*t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:`kv_cache.${e}.v`})})}async loadWeights(r,o){let t=this.device,e=await pe(r),n=o||(()=>{});n({loaded:0,total:3,component:"all",tensor:"Loading backbone..."});let s=await ee(t,r,e,"backbone",o);n({loaded:1,total:3,component:"all",tensor:"Loading FM transformer..."});let i=await ee(t,r,e,"fm",o);n({loaded:2,total:3,component:"all",tensor:"Loading codec decoder..."});let a=await ee(t,r,e,"codec",o);this.modelBuffers=this.organizeWeights(s,i,a),n({loaded:3,total:3,component:"all",tensor:"Done!"})}async loadWeightsFromHF(r=ie,o){let t=this.device,{backbone:e,fm:n,codec:s}=await le(t,r,o);this.modelBuffers=this.organizeWeights(e,n,s),await this.normalizeVQCodebook(),await this.precomputeConvTransposeScales()}async normalizeVQCodebook(){let r=this.device,o=this.pipelines,t=this.modelBuffers,e=this.config.codec,n=this.packUniform([{u:e.semantic_codebook_size},{u:e.semantic_dim},{f:1e-5}]),s=r.createCommandEncoder({label:"normalize_codebook"}),i=s.beginComputePass({label:"normalize_codebook"});this.dispatch(i,o.normalizeCodebook,[t.codec_semantic_codebook,t.codec_cluster_usage,n],[g(e.semantic_codebook_size*e.semantic_dim/2,128)]),i.end(),r.queue.submit([s.finish()]),await r.queue.onSubmittedWorkDone()}async precomputeConvTransposeScales(){let r=this.device,o=this.pipelines,t=this.modelBuffers,e=this.config.codec,n=r.createCommandEncoder({label:"precompute_conv_transpose_scales"});for(let s=0;s<e.decoder_stages;s++){let i=t.codec_stages[s];if(!i.conv_w||!i.conv_g)continue;let a=r.createBuffer({size:e.dim*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,label:`codec_conv_transpose_scale_s${s}`});i.conv_scale=a;let m=this.packUniform([{u:e.dim},{u:e.dim},{u:4}]),w=n.beginComputePass({label:`conv_transpose_norm_scale_s${s}`});this.dispatch(w,o.convTransposeNormScale,[i.conv_w,i.conv_g,a,m],[e.dim]),w.end()}r.queue.submit([n.finish()]),await r.queue.onSubmittedWorkDone()}organizeWeights(r,o,t){let e=(a,_)=>{let m=a.buffers.get(_);if(!m)throw new Error(`Missing weight: ${_}`);return m},n=[];for(let a=0;a<this.config.backbone.n_layers;a++)n.push({attn_norm:e(r,`layers.${a}.attention_norm.weight`),wq:e(r,`layers.${a}.attention.wq.weight`),wk:e(r,`layers.${a}.attention.wk.weight`),wv:e(r,`layers.${a}.attention.wv.weight`),wo:e(r,`layers.${a}.attention.wo.weight`),ffn_norm:e(r,`layers.${a}.ffn_norm.weight`),w1:e(r,`layers.${a}.feed_forward.w1.weight`),w2:e(r,`layers.${a}.feed_forward.w2.weight`),w3:e(r,`layers.${a}.feed_forward.w3.weight`)});let s=[];for(let a=0;a<this.config.fm.n_layers;a++)s.push({attn_norm:e(o,`acoustic_transformer.layers.${a}.attention_norm.weight`),wq:e(o,`acoustic_transformer.layers.${a}.attention.wq.weight`),wk:e(o,`acoustic_transformer.layers.${a}.attention.wk.weight`),wv:e(o,`acoustic_transformer.layers.${a}.attention.wv.weight`),wo:e(o,`acoustic_transformer.layers.${a}.attention.wo.weight`),ffn_norm:e(o,`acoustic_transformer.layers.${a}.ffn_norm.weight`),w1:e(o,`acoustic_transformer.layers.${a}.feed_forward.w1.weight`),w2:e(o,`acoustic_transformer.layers.${a}.feed_forward.w2.weight`),w3:e(o,`acoustic_transformer.layers.${a}.feed_forward.w3.weight`)});let i=[];for(let a=0;a<4;a++){let _=1+a*2,m=2+a*2,w=a<3;i.push({transformer_layers:this.getCodecTransformerLayers(t,_),...w?{conv_w:e(t,`audio_tokenizer.decoder_blocks.${m}.conv.parametrizations.weight.original1`),conv_g:e(t,`audio_tokenizer.decoder_blocks.${m}.conv.parametrizations.weight.original0`)}:{}})}return{tok_embeddings:e(r,"mm_audio_embeddings.tok_embeddings.weight"),audio_embeddings:e(r,"mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"),backbone_layers:n,final_norm:e(r,"norm.weight"),fm_input_proj:e(o,"acoustic_transformer.input_projection.weight"),fm_llm_proj:e(o,"acoustic_transformer.llm_projection.weight"),fm_time_proj:e(o,"acoustic_transformer.time_projection.weight"),fm_layers:s,fm_norm:e(o,"acoustic_transformer.norm.weight"),fm_semantic_out:e(o,"acoustic_transformer.semantic_codebook_output.weight"),fm_acoustic_out:e(o,"acoustic_transformer.acoustic_codebook_output.weight"),codec_input_conv_w:e(t,"audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1"),codec_input_conv_g:e(t,"audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0"),codec_stages:i,codec_output_conv_w:e(t,"audio_tokenizer.output_proj.conv.parametrizations.weight.original1"),codec_output_conv_g:e(t,"audio_tokenizer.output_proj.conv.parametrizations.weight.original0"),codec_semantic_codebook:e(t,"audio_tokenizer.quantizer.semantic_codebook.embedding_sum"),codec_cluster_usage:e(t,"audio_tokenizer.quantizer.semantic_codebook.cluster_usage")}}getCodecTransformerLayers(r,o){let t=n=>{let s=r.buffers.get(n);if(!s)throw new Error(`Missing codec weight: ${n}`);return s},e=[];for(let n=0;n<2;n++){let s=`audio_tokenizer.decoder_blocks.${o}.layers.${n}`;e.push({attn_norm:t(`${s}.attention_norm.weight`),q_norm:t(`${s}.attention.q_norm.weight`),k_norm:t(`${s}.attention.k_norm.weight`),wq:t(`${s}.attention.wq.weight`),wk:t(`${s}.attention.wk.weight`),wv:t(`${s}.attention.wv.weight`),wo:t(`${s}.attention.wo.weight`),attn_scale:t(`${s}.attention_scale`),ffn_norm:t(`${s}.ffn_norm.weight`),w1:t(`${s}.feed_forward.w1.weight`),w2:t(`${s}.feed_forward.w2.weight`),w3:t(`${s}.feed_forward.w3.weight`),ffn_scale:t(`${s}.ffn_scale`)})}return e}dispatch(r,o,t,e){let n=t.map((i,a)=>({binding:a,resource:{buffer:i}})),s=this.device.createBindGroup({layout:o.getBindGroupLayout(0),entries:n});r.setPipeline(o),r.setBindGroup(0,s),r.dispatchWorkgroups(...e)}backboneStep(r,o,t=!1,e){let n=this.pipelines,s=this.workBuffers,i=this.modelBuffers,a=this.config.backbone,_=this.position,m;m=r.beginComputePass({label:`embed_pos${_}`});let w=this.packUniform([{u:o},{u:a.dim}]),v=t?i.audio_embeddings:i.tok_embeddings;if(this.dispatch(m,n.embeddingLookup,[v,s.hidden,w],[g(a.dim,256)]),m.end(),e){m=r.beginComputePass({label:`voice_embed_pos${_}`});let f=this.packUniform([{u:a.dim}]);this.dispatch(m,n.copyBuffer,[e,s.hidden,f],[g(a.dim,256)]),m.end()}for(let f=0;f<a.n_layers;f++){let k=i.backbone_layers[f],q=this.kvCaches[f];m=r.beginComputePass({label:`layer${f}_attn`});let l=this.packUniform([{u:a.dim}]);this.dispatch(m,n.copyBuffer,[s.hidden,s.residual,l],[g(a.dim,256)]);let p=this.packUniform([{u:a.dim},{f:a.norm_eps}]);this.dispatch(m,n.rmsNorm,[s.hidden,k.attn_norm,s.normed,p],[1]),m.end(),m=r.beginComputePass({label:`layer${f}_qkv`});let h=this.packUniform([{u:a.n_heads*a.head_dim},{u:a.dim}]);this.dispatch(m,n.matvecF16,[k.wq,s.normed,s.q,h],[a.n_heads*a.head_dim]);let P=this.packUniform([{u:a.n_kv_heads*a.head_dim},{u:a.dim}]);this.dispatch(m,n.matvecF16,[k.wk,s.normed,s.k,P],[a.n_kv_heads*a.head_dim]),this.dispatch(m,n.matvecF16,[k.wv,s.normed,s.v,P],[a.n_kv_heads*a.head_dim]),m.end(),m=r.beginComputePass({label:`layer${f}_rope_attn`});let C=this.packUniform([{u:a.head_dim},{u:_},{u:a.n_heads},{f:a.rope_theta}]);this.dispatch(m,n.rope,[s.q,C],[g(a.n_heads*a.head_dim/2,64)]);let b=this.packUniform([{u:a.head_dim},{u:_},{u:a.n_kv_heads},{f:a.rope_theta}]);this.dispatch(m,n.rope,[s.k,b],[g(a.n_kv_heads*a.head_dim/2,64)]);let G=this.packUniform([{u:_},{u:a.n_kv_heads*a.head_dim}]);this.dispatch(m,n.kvCacheWrite,[s.k,s.v,q.k,q.v,G],[g(a.n_kv_heads*a.head_dim,256)]);let x=_+1,y=a.n_heads/a.n_kv_heads,U=this.packUniform([{u:a.n_heads},{u:a.n_kv_heads},{u:a.head_dim},{u:x},{u:y}]);this.dispatch(m,n.attnScore,[s.q,q.k,s.scores,U],[g(a.n_heads*x,64)]),m.end(),m=r.beginComputePass({label:`layer${f}_attn_out`});let S=this.packUniform([{u:a.n_heads},{u:x}]);this.dispatch(m,n.softmax,[s.scores,S],[a.n_heads]);let M=this.packUniform([{u:a.n_heads},{u:a.n_kv_heads},{u:a.head_dim},{u:x},{u:y}]);this.dispatch(m,n.attnValue,[s.scores,q.v,s.attn_out,M],[g(a.n_heads*a.head_dim,128)]),m.end(),m=r.beginComputePass({label:`layer${f}_wo_res`});let $=this.packUniform([{u:a.dim},{u:a.n_heads*a.head_dim}]);this.dispatch(m,n.matvecF16,[k.wo,s.attn_out,s.hidden,$],[a.dim]),m.end(),m=r.beginComputePass({label:`layer${f}_res1`});let z=this.packUniform([{u:a.dim}]);this.dispatch(m,n.addInPlace,[s.hidden,s.residual,z],[g(a.dim,256)]),this.dispatch(m,n.copyBuffer,[s.hidden,s.residual,l],[g(a.dim,256)]);let T=this.packUniform([{u:a.dim},{f:a.norm_eps}]);this.dispatch(m,n.rmsNorm,[s.hidden,k.ffn_norm,s.normed,T],[1]),m.end(),m=r.beginComputePass({label:`layer${f}_ffn`});let I=this.packUniform([{u:a.hidden_dim},{u:a.dim}]);this.dispatch(m,n.matvecF16,[k.w1,s.normed,s.gate,I],[a.hidden_dim]),this.dispatch(m,n.matvecF16,[k.w3,s.normed,s.up,I],[a.hidden_dim]),m.end(),m=r.beginComputePass({label:`layer${f}_ffn_out`});let B=this.packUniform([{u:a.hidden_dim}]);this.dispatch(m,n.swiGLU,[s.gate,s.up,B],[g(a.hidden_dim,256)]);let E=this.packUniform([{u:a.dim},{u:a.hidden_dim}]);this.dispatch(m,n.matvecF16,[k.w2,s.gate,s.hidden,E],[a.dim]),m.end(),m=r.beginComputePass({label:`layer${f}_res2`}),this.dispatch(m,n.addInPlace,[s.hidden,s.residual,z],[g(a.dim,256)]),m.end()}m=r.beginComputePass({label:"final_norm"});let c=this.packUniform([{u:a.dim},{f:a.norm_eps}]);this.dispatch(m,n.rmsNorm,[s.hidden,i.final_norm,s.normed,c],[1]),m.end(),m=r.beginComputePass({label:"lm_head"});for(let k=0;k<a.vocab_size;k+=65535){let q=Math.min(65535,a.vocab_size-k),l=this.packUniform([{u:q},{u:a.dim},{u:k}]);this.dispatch(m,n.matvecF16Chunked,[i.tok_embeddings,s.normed,s.logits,l],[q])}m.end(),m=r.beginComputePass({label:"argmax"});let d=this.packUniform([{u:a.vocab_size}]);this.dispatch(m,n.argmax,[s.logits,s.argmax_result,d],[1]),m.end()}fmTransformerPass(r,o){let t=this.pipelines,e=this.workBuffers,n=this.modelBuffers,s=this.config.fm,i=s.dim,a=3,_=s.n_heads*s.head_dim,m=s.n_kv_heads*s.head_dim,w=s.n_heads/s.n_kv_heads;for(let v=0;v<s.n_layers;v++){let c=n.fm_layers[v],d;d=r.beginComputePass({label:`fm_l${v}_attn_prep`});let f=this.packUniform([{u:a*i},{u:0},{u:0}]);this.dispatch(d,t.copyBufferOffset,[e.fm_seq,e.fm_down,f],[g(a*i,256)]);for(let p=0;p<a;p++){let h=p*i,P=this.packUniform([{u:i},{f:1e-5},{u:h},{u:h}]);this.dispatch(d,t.rmsNormOffset,[e.fm_seq,c.attn_norm,e.fm_gate,P],[1])}d.end(),d=r.beginComputePass({label:`fm_l${v}_qkv`});for(let p=0;p<a;p++){let h=p*i,P=p*_,C=p*m,b=this.packUniform([{u:_},{u:i},{u:h},{u:P}]);this.dispatch(d,t.matvecF16Offset,[c.wq,e.fm_gate,e.fm_q,b],[_]);let G=this.packUniform([{u:m},{u:i},{u:h},{u:C}]);this.dispatch(d,t.matvecF16Offset,[c.wk,e.fm_gate,e.fm_k,G],[m]),this.dispatch(d,t.matvecF16Offset,[c.wv,e.fm_gate,e.fm_v,G],[m])}d.end(),d=r.beginComputePass({label:`fm_l${v}_attn`});let k=this.packUniform([{u:s.n_heads},{u:s.n_kv_heads},{u:s.head_dim},{u:a},{u:w}]);this.dispatch(d,t.biAttnScore,[e.fm_q,e.fm_k,e.fm_scores,k],[g(s.n_heads*a*a,64)]),d.end(),d=r.beginComputePass({label:`fm_l${v}_attn_val`});let q=this.packUniform([{u:s.n_heads},{u:a}]);this.dispatch(d,t.biSoftmax,[e.fm_scores,q],[g(s.n_heads*a,64)]);let l=this.packUniform([{u:s.n_heads},{u:s.n_kv_heads},{u:s.head_dim},{u:a},{u:w}]);this.dispatch(d,t.biAttnValue,[e.fm_scores,e.fm_v,e.fm_attn_out,l],[g(a*s.n_heads*s.head_dim,64)]),d.end(),d=r.beginComputePass({label:`fm_l${v}_wo_res`});for(let p=0;p<a;p++){let h=p*_,P=p*i,C=this.packUniform([{u:i},{u:_},{u:h},{u:P}]);this.dispatch(d,t.matvecF16Offset,[c.wo,e.fm_attn_out,e.fm_seq,C],[i])}d.end(),d=r.beginComputePass({label:`fm_l${v}_res1`});for(let p=0;p<a;p++){let h=p*i,P=this.packUniform([{u:i},{u:h},{u:h}]);this.dispatch(d,t.addInPlaceOffset,[e.fm_seq,e.fm_down,P],[g(i,256)])}this.dispatch(d,t.copyBufferOffset,[e.fm_seq,e.fm_down,f],[g(a*i,256)]),d.end(),d=r.beginComputePass({label:`fm_l${v}_ffn`});for(let p=0;p<a;p++){let h=p*i,P=p*s.hidden_dim,C=this.packUniform([{u:i},{f:1e-5},{u:h},{u:0}]);this.dispatch(d,t.rmsNormOffset,[e.fm_seq,c.ffn_norm,e.fm_normed,C],[1]);let b=this.packUniform([{u:s.hidden_dim},{u:i},{u:0},{u:P}]);this.dispatch(d,t.matvecF16Offset,[c.w1,e.fm_normed,e.fm_gate,b],[s.hidden_dim]),this.dispatch(d,t.matvecF16Offset,[c.w3,e.fm_normed,e.fm_up,b],[s.hidden_dim])}d.end(),d=r.beginComputePass({label:`fm_l${v}_ffn_act`});for(let p=0;p<a;p++){let h=p*s.hidden_dim,P=this.packUniform([{u:s.hidden_dim},{u:h},{u:h}]);this.dispatch(d,t.swiGLUOffset,[e.fm_gate,e.fm_up,P],[g(s.hidden_dim,256)])}d.end(),d=r.beginComputePass({label:`fm_l${v}_ffn_down`});for(let p=0;p<a;p++){let h=p*s.hidden_dim,P=p*i,C=this.packUniform([{u:i},{u:s.hidden_dim},{u:h},{u:P}]);this.dispatch(d,t.matvecF16Offset,[c.w2,e.fm_gate,e.fm_seq,C],[i])}d.end(),d=r.beginComputePass({label:`fm_l${v}_res2`});for(let p=0;p<a;p++){let h=p*i,P=this.packUniform([{u:i},{u:h},{u:h}]);this.dispatch(d,t.addInPlaceOffset,[e.fm_seq,e.fm_down,P],[g(i,256)])}d.end()}{let v=r.beginComputePass({label:"fm_final_norm_vel"}),c=this.packUniform([{u:i},{f:1e-5},{u:0},{u:0}]);this.dispatch(v,t.rmsNormOffset,[e.fm_seq,n.fm_norm,e.fm_normed,c],[1]);let d=this.packUniform([{u:s.n_acoustic_out},{u:i}]);this.dispatch(v,t.matvecF16,[n.fm_acoustic_out,e.fm_normed,o,d],[s.n_acoustic_out]),v.end()}}fmForward(r,o){let t=this.pipelines,e=this.workBuffers,n=this.modelBuffers,s=this.config.fm,i=s.dim,a;a=r.beginComputePass({label:"fm_init"});let _=this.packUniform([{u:s.semantic_vocab},{u:i}]);this.dispatch(a,t.matvecF16,[n.fm_semantic_out,e.normed,e.semantic_logits,_],[s.semantic_vocab]);let m=this.packUniform([{u:i},{u:i}]);this.dispatch(a,t.matvecF16,[n.fm_llm_proj,e.normed,e.fm_hidden,m],[i]);{let c=o??new Float32Array(s.n_acoustic_out);if(!o)for(let d=0;d<s.n_acoustic_out;d++){let f=Math.random(),k=Math.random();c[d]=Math.sqrt(-2*Math.log(f))*Math.cos(2*Math.PI*k)}this.device.queue.writeBuffer(e.x_t,0,c)}a.end(),a=r.beginComputePass({label:"fm_semantic_argmax"});let w=this.packUniform([{u:s.semantic_vocab}]);this.dispatch(a,t.argmax,[e.semantic_logits,e.semantic_argmax,w],[1]),a.end();for(let c=0;c<s.nfe-1;c++){let d=c/(s.nfe-1),f=1/(s.nfe-1);a=r.beginComputePass({label:`fm_step${c}_prep`});let k=this.packUniform([{u:i},{f:d}]);this.dispatch(a,t.timeEmbedding,[e.time_embed,k],[g(i/2,256)]),a.end(),a=r.beginComputePass({label:`fm_step${c}_proj`});let q=this.packUniform([{u:i},{u:i}]);this.dispatch(a,t.matvecF16,[n.fm_time_proj,e.time_embed,e.time_proj,q],[i]);let l=this.packUniform([{u:i},{u:s.n_acoustic_out}]);this.dispatch(a,t.matvecF16,[n.fm_input_proj,e.x_t,e.x_t_proj,l],[i]),a.end(),a=r.beginComputePass({label:`fm_step${c}_assemble`});let p=this.packUniform([{u:i},{u:0},{u:0}]);this.dispatch(a,t.copyBufferOffset,[e.x_t_proj,e.fm_seq,p],[g(i,256)]);let h=this.packUniform([{u:i},{u:0},{u:i}]);this.dispatch(a,t.copyBufferOffset,[e.time_proj,e.fm_seq,h],[g(i,256)]);let P=this.packUniform([{u:i},{u:0},{u:2*i}]);this.dispatch(a,t.copyBufferOffset,[e.fm_hidden,e.fm_seq,P],[g(i,256)]),a.end(),this.fmTransformerPass(r,e.velocity),a=r.beginComputePass({label:`fm_step${c}_uncond`}),this.dispatch(a,t.copyBufferOffset,[e.x_t_proj,e.fm_seq,p],[g(i,256)]),this.dispatch(a,t.copyBufferOffset,[e.time_proj,e.fm_seq,h],[g(i,256)]);let C=this.packUniform([{u:i}]);this.dispatch(a,t.zeroFill,[e.fm_residual,C],[g(i,256)]),this.dispatch(a,t.copyBufferOffset,[e.fm_residual,e.fm_seq,P],[g(i,256)]),a.end(),this.fmTransformerPass(r,e.v_uncond),a=r.beginComputePass({label:`fm_step${c}_euler`});let b=this.packUniform([{u:s.n_acoustic_out},{f:s.cfg_alpha}]);this.dispatch(a,t.cfgCombine,[e.velocity,e.v_uncond,b],[g(s.n_acoustic_out,64)]);let G=this.packUniform([{u:s.n_acoustic_out},{f}]);this.dispatch(a,t.eulerStep,[e.x_t,e.velocity,G],[g(s.n_acoustic_out,64)]),a.end()}a=r.beginComputePass({label:"fm_fsq"});let v=this.packUniform([{u:s.n_acoustic_out},{u:this.config.codec.acoustic_codebook_size},{u:2}]);this.dispatch(a,t.fsqQuantize,[e.x_t,e.acoustic_codes,v],[g(s.n_acoustic_out,64)]),a.end()}async codecDecode(r,o){let t=this.device,e=this.pipelines,n=this.modelBuffers,s=this.config.codec,i=r.length,a=s.dim,_=this.uploadArray(r),m=this.uploadArray(o),w=this.createGPUBuffer(i*s.semantic_dim*4,"codec_sem_embed"),v=this.createGPUBuffer(i*s.n_acoustic_codebook*4,"codec_ac_float"),c=this.createGPUBuffer(i*292*4,"codec_concat"),d=i,f=this.createGPUBuffer(d*a*4,"codec_cur"),k=this.createGPUBuffer(d*a*4,"codec_tmp"),q=t.createCommandEncoder({label:"codec_decode"}),l=[],p=(B,E,A,F)=>{let O=q.beginComputePass({label:F});this.dispatch(O,B,E,A),O.end()},h=this.packUniform([{u:i},{u:s.semantic_dim}]);p(e.vqLookup,[_,n.codec_semantic_codebook,w,h],[g(i*s.semantic_dim,128)],"codec_vq");let P=this.packUniform([{u:i},{u:s.n_acoustic_codebook},{u:s.acoustic_codebook_size},{u:2}]);p(e.fsqDequant,[m,v,P],[g(i*s.n_acoustic_codebook,64)],"codec_fsq");let C=this.packUniform([{u:i},{u:s.semantic_dim},{u:s.n_acoustic_codebook}]);p(e.concatCodecInput,[w,v,c,C],[g(i*292,256)],"codec_concat");let b=this.packUniform([{u:292},{u:a},{u:3},{u:d},{u:1}]);p(e.causalConv1d,[c,n.codec_input_conv_w,n.codec_input_conv_g,f,b],[g(a*d,64)],"codec_input_conv");let G=[2,2,2,1],x=[4,4,4,3],y=[2,4,8,16];for(let B=0;B<s.decoder_stages;B++){let E=n.codec_stages[B];for(let A=0;A<s.decoder_layers_per_stage;A++){let F=E.transformer_layers[A],O=d*a*4;k.size<O&&(l.push(k),k=this.createGPUBuffer(O,"codec_tmp"));let W=k,D=d*a,ae=this.packUniform([{u:D}]);p(e.batchedCopy,[f,W,ae],[g(D,256)],`codec_s${B}_l${A}_copy_res`);let Z=this.packUniform([{u:a},{f:s.norm_eps},{u:d}]),H=this.createGPUBuffer(O,"codec_attn_normed");p(e.batchedRmsNorm,[f,F.attn_norm,H,Z],[d],`codec_s${B}_l${A}_attn_norm`);let R=this.createGPUBuffer(d*a*4,"codec_q"),Y=this.createGPUBuffer(d*a*4,"codec_k"),j=this.createGPUBuffer(d*a*4,"codec_v"),N=this.packUniform([{u:a},{u:a},{u:d}]);p(e.batchedMatvecF16,[F.wq,H,R,N],[a,d],`codec_s${B}_l${A}_qproj`),p(e.batchedMatvecF16,[F.wk,H,Y,N],[a,d],`codec_s${B}_l${A}_kproj`),p(e.batchedMatvecF16,[F.wv,H,j,N],[a,d],`codec_s${B}_l${A}_vproj`);let Q=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:d},{f:s.qk_norm_eps}]);p(e.qkNorm,[R,F.q_norm,Q],[g(d*s.n_heads,128)],`codec_s${B}_l${A}_qnorm`),p(e.qkNorm,[Y,F.k_norm,Q],[g(d*s.n_heads,128)],`codec_s${B}_l${A}_knorm`);let V=this.createGPUBuffer(s.n_heads*d*d*4,"codec_scores"),te=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:d},{u:y[B]}]);p(e.alibiAttnScore,[R,Y,V,te],[g(d,64),d,s.n_heads],`codec_s${B}_l${A}_attn_score`);let ne=this.packUniform([{u:s.n_heads},{u:d}]);p(e.codecSoftmax,[V,ne],[g(s.n_heads*d,64)],`codec_s${B}_l${A}_softmax`);let X=this.createGPUBuffer(d*a*4,"codec_attn_out"),de=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:d}]);p(e.codecAttnValue,[V,j,X,de],[g(d*s.n_heads*s.head_dim,64)],`codec_s${B}_l${A}_attn_val`);let ce=this.createGPUBuffer(O,"codec_wo_out");p(e.batchedMatvecF16,[F.wo,X,ce,N],[a,d],`codec_s${B}_l${A}_wo`);let be=this.packUniform([{u:a},{u:D}]);p(e.batchedLayerScale,[ce,F.attn_scale,W,f,be],[g(D,256)],`codec_s${B}_l${A}_attn_res`),p(e.batchedCopy,[f,W,ae],[g(D,256)],`codec_s${B}_l${A}_copy_ffn_res`);let re=this.createGPUBuffer(O,"codec_ffn_normed");p(e.batchedRmsNorm,[f,F.ffn_norm,re,Z],[d],`codec_s${B}_l${A}_ffn_norm`);let se=d*s.hidden_dim,oe=this.createGPUBuffer(se*4,"codec_gate"),ue=this.createGPUBuffer(se*4,"codec_up"),ve=this.packUniform([{u:s.hidden_dim},{u:a},{u:d}]);p(e.batchedMatvecF16,[F.w1,re,oe,ve],[s.hidden_dim,d],`codec_s${B}_l${A}_gate`),p(e.batchedMatvecF16,[F.w3,re,ue,ve],[s.hidden_dim,d],`codec_s${B}_l${A}_up`);let wa=this.packUniform([{u:se}]);p(e.batchedSwiGLU,[oe,ue,wa],[g(se,256)],`codec_s${B}_l${A}_swiglu`);let me=this.createGPUBuffer(O,"codec_down"),Pa=this.packUniform([{u:a},{u:s.hidden_dim},{u:d}]);p(e.batchedMatvecF16,[F.w2,oe,me,Pa],[a,d],`codec_s${B}_l${A}_down`),p(e.batchedLayerScale,[me,F.ffn_scale,W,f,be],[g(D,256)],`codec_s${B}_l${A}_ffn_res`),l.push(H,R,Y,j,V,X,ce,re,oe,ue,me)}if(E.conv_w&&E.conv_scale&&G[B]>1){let A=d*G[B],F=this.createGPUBuffer(A*a*4,"codec_upsampled"),O=this.packUniform([{u:a},{u:a},{u:x[B]},{u:A},{u:G[B]}]);p(e.causalConvTranspose1d,[f,E.conv_w,E.conv_scale,F,O],[g(a*A,64)],`codec_s${B}_conv_up`),l.push(f),f=F,d=A}}let U=d,S=this.createGPUBuffer(U*s.patch_size*4,"codec_output"),M=this.packUniform([{u:a},{u:s.patch_size},{u:7},{u:U},{u:1}]);p(e.causalConv1d,[f,n.codec_output_conv_w,n.codec_output_conv_g,S,M],[g(s.patch_size*U,64)],"codec_output_conv"),t.pushErrorScope("validation"),t.queue.submit([q.finish()]),await t.queue.onSubmittedWorkDone();let $=await t.popErrorScope();$&&(globalThis.__codecError=$.message);let z=U*s.patch_size,T=await this.readF32Array(S,z),I=0;for(let B=0;B<Math.min(T.length,1e3);B++)T[B]!==0&&I++;globalThis.__codecDebug={outT:U,patchSize:s.patch_size,totalSamples:z,nonZero:I,first5:Array.from(T.slice(0,5)),curT:d};for(let B of l)B.destroy();return _.destroy(),m.destroy(),w.destroy(),v.destroy(),c.destroy(),f.destroy(),k.destroy(),S.destroy(),T}async debugCodecDecode(r,o){let t=this.device,e=this.pipelines,n=this.modelBuffers,s=this.config.codec,i=r.length,a=s.dim,_={},m=this.uploadArray(r),w=this.uploadArray(o),v=this.createGPUBuffer(i*s.semantic_dim*4,"codec_sem_embed"),c=this.createGPUBuffer(i*s.n_acoustic_codebook*4,"codec_ac_float"),d=this.createGPUBuffer(i*292*4,"codec_concat"),f=i,k=this.createGPUBuffer(f*a*4,"codec_cur"),q=this.createGPUBuffer(f*a*4,"codec_tmp"),l=[],p=(b,G,x,y,U)=>{let S=b.beginComputePass({label:U});this.dispatch(S,G,x,y),S.end()};{let b=t.createCommandEncoder({label:"codec_phase1"}),G=this.packUniform([{u:i},{u:s.semantic_dim}]);p(b,e.vqLookup,[m,n.codec_semantic_codebook,v,G],[g(i*s.semantic_dim,128)],"codec_vq");let x=this.packUniform([{u:i},{u:s.n_acoustic_codebook},{u:s.acoustic_codebook_size},{u:2}]);p(b,e.fsqDequant,[w,c,x],[g(i*s.n_acoustic_codebook,64)],"codec_fsq");let y=this.packUniform([{u:i},{u:s.semantic_dim},{u:s.n_acoustic_codebook}]);p(b,e.concatCodecInput,[v,c,d,y],[g(i*292,256)],"codec_concat");let U=this.packUniform([{u:292},{u:a},{u:3},{u:f},{u:1}]);p(b,e.causalConv1d,[d,n.codec_input_conv_w,n.codec_input_conv_g,k,U],[g(a*f,64)],"codec_input_conv"),t.queue.submit([b.finish()]),await t.queue.onSubmittedWorkDone()}_.vq_embed=await this.readF32Array(v,i*s.semantic_dim),_.fsq_dequant=await this.readF32Array(c,i*s.n_acoustic_codebook),_.concat=await this.readF32Array(d,i*292),_.after_input_conv=await this.readF32Array(k,f*a);let h=[2,2,2,1],P=[4,4,4,3],C=[2,4,8,16];for(let b=0;b<s.decoder_stages;b++){let G=n.codec_stages[b],x=t.createCommandEncoder({label:`codec_stage${b}`});for(let y=0;y<s.decoder_layers_per_stage;y++){let U=G.transformer_layers[y],S=f*a*4;q.size<S&&(l.push(q),q=this.createGPUBuffer(S,"codec_tmp"));let M=q,$=f*a,z=this.packUniform([{u:$}]);p(x,e.batchedCopy,[k,M,z],[g($,256)],`codec_s${b}_l${y}_copy_res`);let T=this.packUniform([{u:a},{f:s.norm_eps},{u:f}]),I=this.createGPUBuffer(S,"codec_attn_normed");p(x,e.batchedRmsNorm,[k,U.attn_norm,I,T],[f],`codec_s${b}_l${y}_attn_norm`);let B=this.createGPUBuffer(f*a*4,"codec_q"),E=this.createGPUBuffer(f*a*4,"codec_k"),A=this.createGPUBuffer(f*a*4,"codec_v"),F=this.packUniform([{u:a},{u:a},{u:f}]);p(x,e.batchedMatvecF16,[U.wq,I,B,F],[a,f],`codec_s${b}_l${y}_qproj`),p(x,e.batchedMatvecF16,[U.wk,I,E,F],[a,f],`codec_s${b}_l${y}_kproj`),p(x,e.batchedMatvecF16,[U.wv,I,A,F],[a,f],`codec_s${b}_l${y}_vproj`);let O=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:f},{f:s.qk_norm_eps}]);p(x,e.qkNorm,[B,U.q_norm,O],[g(f*s.n_heads,128)],`codec_s${b}_l${y}_qnorm`),p(x,e.qkNorm,[E,U.k_norm,O],[g(f*s.n_heads,128)],`codec_s${b}_l${y}_knorm`);let W=this.createGPUBuffer(s.n_heads*f*f*4,"codec_scores"),D=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:f},{u:C[b]}]);p(x,e.alibiAttnScore,[B,E,W,D],[g(f,64),f,s.n_heads],`codec_s${b}_l${y}_attn_score`);let ae=this.packUniform([{u:s.n_heads},{u:f}]);p(x,e.codecSoftmax,[W,ae],[g(s.n_heads*f,64)],`codec_s${b}_l${y}_softmax`);let Z=this.createGPUBuffer(f*a*4,"codec_attn_out"),H=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:f}]);p(x,e.codecAttnValue,[W,A,Z,H],[g(f*s.n_heads*s.head_dim,64)],`codec_s${b}_l${y}_attn_val`);let R=this.createGPUBuffer(S,"codec_wo_out");p(x,e.batchedMatvecF16,[U.wo,Z,R,F],[a,f],`codec_s${b}_l${y}_wo`);let Y=this.packUniform([{u:a},{u:$}]);p(x,e.batchedLayerScale,[R,U.attn_scale,M,k,Y],[g($,256)],`codec_s${b}_l${y}_attn_res`),p(x,e.batchedCopy,[k,M,z],[g($,256)],`codec_s${b}_l${y}_copy_ffn_res`);let j=this.createGPUBuffer(S,"codec_ffn_normed");p(x,e.batchedRmsNorm,[k,U.ffn_norm,j,T],[f],`codec_s${b}_l${y}_ffn_norm`);let N=f*s.hidden_dim,Q=this.createGPUBuffer(N*4,"codec_gate"),V=this.createGPUBuffer(N*4,"codec_up"),te=this.packUniform([{u:s.hidden_dim},{u:a},{u:f}]);p(x,e.batchedMatvecF16,[U.w1,j,Q,te],[s.hidden_dim,f],`codec_s${b}_l${y}_gate`),p(x,e.batchedMatvecF16,[U.w3,j,V,te],[s.hidden_dim,f],`codec_s${b}_l${y}_up`);let ne=this.packUniform([{u:N}]);p(x,e.batchedSwiGLU,[Q,V,ne],[g(N,256)],`codec_s${b}_l${y}_swiglu`);let X=this.createGPUBuffer(S,"codec_down"),de=this.packUniform([{u:a},{u:s.hidden_dim},{u:f}]);p(x,e.batchedMatvecF16,[U.w2,Q,X,de],[a,f],`codec_s${b}_l${y}_down`),p(x,e.batchedLayerScale,[X,U.ffn_scale,M,k,Y],[g($,256)],`codec_s${b}_l${y}_ffn_res`),l.push(I,B,E,A,W,Z,R,j,Q,V,X)}if(G.conv_w&&G.conv_scale&&h[b]>1){let y=f*h[b],U=this.createGPUBuffer(y*a*4,"codec_upsampled"),S=this.packUniform([{u:a},{u:a},{u:P[b]},{u:y},{u:h[b]}]);p(x,e.causalConvTranspose1d,[k,G.conv_w,G.conv_scale,U,S],[g(a*y,64)],`codec_s${b}_conv_up`),t.queue.submit([x.finish()]),await t.queue.onSubmittedWorkDone(),_[`after_stage${b}_transformer`]=await this.readF32Array(k,f*a),l.push(k),k=U,f=y,_[`after_stage${b}_conv_up`]=await this.readF32Array(k,f*a)}else t.queue.submit([x.finish()]),await t.queue.onSubmittedWorkDone(),_[`after_stage${b}_transformer`]=await this.readF32Array(k,f*a)}{let b=f,G=this.createGPUBuffer(b*s.patch_size*4,"codec_output"),x=this.packUniform([{u:a},{u:s.patch_size},{u:7},{u:b},{u:1}]),y=t.createCommandEncoder({label:"codec_output"});p(y,e.causalConv1d,[k,n.codec_output_conv_w,n.codec_output_conv_g,G,x],[g(s.patch_size*b,64)],"codec_output_conv"),t.queue.submit([y.finish()]),await t.queue.onSubmittedWorkDone(),_.after_output_conv=await this.readF32Array(G,b*s.patch_size),_.audio=_.after_output_conv,l.push(G)}for(let b of l)b.destroy();return m.destroy(),w.destroy(),v.destroy(),c.destroy(),d.destroy(),k.destroy(),q.destroy(),_}uploadArray(r){let o=this.device.createBuffer({size:r.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!0});return r instanceof Uint32Array?new Uint32Array(o.getMappedRange()).set(r):new Float32Array(o.getMappedRange()).set(r),o.unmap(),o}createGPUBuffer(r,o){return this.device.createBuffer({size:Math.max(r,4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,label:o})}async readBuffer(r,o){let t=this.device,e=t.createBuffer({size:o,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),n=t.createCommandEncoder();n.copyBufferToBuffer(r,0,e,0,o),t.queue.submit([n.finish()]),await e.mapAsync(GPUMapMode.READ);let s=e.getMappedRange().slice(0);return e.unmap(),e.destroy(),s}async readU32(r){let o=await this.readBuffer(r,4);return new Uint32Array(o)[0]}async readF32Array(r,o){let t=await this.readBuffer(r,o*4);return new Float32Array(t)}async readU32Array(r,o){let t=await this.readBuffer(r,o*4);return new Uint32Array(t)}get isReady(){return this.device!==null&&this.modelBuffers!==null&&this.pipelines!==null}async debugRead(r,o=16){let t=this.workBuffers,e=t[r];if(!e)throw new Error(`Unknown buffer: ${r}. Available: ${Object.keys(t).join(", ")}`);return this.readF32Array(e,o)}async debugBackboneStep(r){let o=this.device.createCommandEncoder();this.backboneStep(o,r),this.device.queue.submit([o.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++;let t=this.workBuffers,e=await this.readF32Array(t.hidden,16),n=await this.readF32Array(t.normed,16),s=await this.readF32Array(t.logits,16),i=await this.readU32(t.argmax_result),a=await this.readF32Array(t.logits,1024),_=-1/0;for(let m=0;m<a.length;m++)a[m]>_&&(_=a[m]);return{hidden:e,normed:n,logits_first16:s,logits_max:_,argmax:i}}async debugBackboneLayerByLayer(r){let o=this.pipelines,t=this.workBuffers,e=this.modelBuffers,n=this.config.backbone,s=this.position,i=n.dim,a;{let c=this.device.createCommandEncoder();a=c.beginComputePass({label:"debug_embed"});let d=this.packUniform([{u:r},{u:i}]);this.dispatch(a,o.embeddingLookup,[e.tok_embeddings,t.hidden,d],[g(i,256)]),a.end(),this.device.queue.submit([c.finish()]),await this.device.queue.onSubmittedWorkDone()}let _=await this.readF32Array(t.hidden,i),m=[];for(let c=0;c<n.n_layers;c++){let d=e.backbone_layers[c],f=this.kvCaches[c];{let h=this.device.createCommandEncoder();a=h.beginComputePass({label:`debug_l${c}_attn_prep`});let P=this.packUniform([{u:i}]);this.dispatch(a,o.copyBuffer,[t.hidden,t.residual,P],[g(i,256)]);let C=this.packUniform([{u:i},{f:n.norm_eps}]);this.dispatch(a,o.rmsNorm,[t.hidden,d.attn_norm,t.normed,C],[1]),a.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let k=await this.readF32Array(t.normed,i);{let h=this.device.createCommandEncoder();a=h.beginComputePass({label:`debug_l${c}_qkv`});let P=this.packUniform([{u:n.n_heads*n.head_dim},{u:i}]);this.dispatch(a,o.matvecF16,[d.wq,t.normed,t.q,P],[n.n_heads*n.head_dim]);let C=this.packUniform([{u:n.n_kv_heads*n.head_dim},{u:i}]);this.dispatch(a,o.matvecF16,[d.wk,t.normed,t.k,C],[n.n_kv_heads*n.head_dim]),this.dispatch(a,o.matvecF16,[d.wv,t.normed,t.v,C],[n.n_kv_heads*n.head_dim]),a.end(),a=h.beginComputePass({label:`debug_l${c}_rope_attn`});let b=this.packUniform([{u:n.head_dim},{u:s},{u:n.n_heads},{f:n.rope_theta}]);this.dispatch(a,o.rope,[t.q,b],[g(n.n_heads*n.head_dim/2,64)]);let G=this.packUniform([{u:n.head_dim},{u:s},{u:n.n_kv_heads},{f:n.rope_theta}]);this.dispatch(a,o.rope,[t.k,G],[g(n.n_kv_heads*n.head_dim/2,64)]);let x=this.packUniform([{u:s},{u:n.n_kv_heads*n.head_dim}]);this.dispatch(a,o.kvCacheWrite,[t.k,t.v,f.k,f.v,x],[g(n.n_kv_heads*n.head_dim,256)]);let y=s+1,U=n.n_heads/n.n_kv_heads,S=this.packUniform([{u:n.n_heads},{u:n.n_kv_heads},{u:n.head_dim},{u:y},{u:U}]);this.dispatch(a,o.attnScore,[t.q,f.k,t.scores,S],[g(n.n_heads*y,64)]),a.end(),a=h.beginComputePass({label:`debug_l${c}_attn_out`});let M=this.packUniform([{u:n.n_heads},{u:y}]);this.dispatch(a,o.softmax,[t.scores,M],[n.n_heads]);let $=this.packUniform([{u:n.n_heads},{u:n.n_kv_heads},{u:n.head_dim},{u:y},{u:U}]);this.dispatch(a,o.attnValue,[t.scores,f.v,t.attn_out,$],[g(n.n_heads*n.head_dim,128)]),a.end(),a=h.beginComputePass({label:`debug_l${c}_wo`});let z=this.packUniform([{u:i},{u:n.n_heads*n.head_dim}]);this.dispatch(a,o.matvecF16,[d.wo,t.attn_out,t.hidden,z],[i]),a.end(),a=h.beginComputePass({label:`debug_l${c}_res1`});let T=this.packUniform([{u:i}]);this.dispatch(a,o.addInPlace,[t.hidden,t.residual,T],[g(i,256)]),a.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let q=await this.readF32Array(t.hidden,i);{let h=this.device.createCommandEncoder();a=h.beginComputePass({label:`debug_l${c}_ffn_prep`});let P=this.packUniform([{u:i}]);this.dispatch(a,o.copyBuffer,[t.hidden,t.residual,P],[g(i,256)]);let C=this.packUniform([{u:i},{f:n.norm_eps}]);this.dispatch(a,o.rmsNorm,[t.hidden,d.ffn_norm,t.normed,C],[1]),a.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let l=await this.readF32Array(t.normed,i);{let h=this.device.createCommandEncoder();a=h.beginComputePass({label:`debug_l${c}_ffn`});let P=this.packUniform([{u:n.hidden_dim},{u:i}]);this.dispatch(a,o.matvecF16,[d.w1,t.normed,t.gate,P],[n.hidden_dim]),this.dispatch(a,o.matvecF16,[d.w3,t.normed,t.up,P],[n.hidden_dim]),a.end(),a=h.beginComputePass({label:`debug_l${c}_ffn_out`});let C=this.packUniform([{u:n.hidden_dim}]);this.dispatch(a,o.swiGLU,[t.gate,t.up,C],[g(n.hidden_dim,256)]);let b=this.packUniform([{u:i},{u:n.hidden_dim}]);this.dispatch(a,o.matvecF16,[d.w2,t.gate,t.hidden,b],[i]),a.end(),a=h.beginComputePass({label:`debug_l${c}_res2`});let G=this.packUniform([{u:i}]);this.dispatch(a,o.addInPlace,[t.hidden,t.residual,G],[g(i,256)]),a.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let p=await this.readF32Array(t.hidden,i);m.push({attn_norm:k,attn_out:q,ffn_norm:l,ffn_out:p})}{let c=this.device.createCommandEncoder();a=c.beginComputePass({label:"debug_final_norm"});let d=this.packUniform([{u:i},{f:n.norm_eps}]);this.dispatch(a,o.rmsNorm,[t.hidden,e.final_norm,t.normed,d],[1]),a.end(),this.device.queue.submit([c.finish()]),await this.device.queue.onSubmittedWorkDone()}let w=await this.readF32Array(t.normed,i),v=await this.readF32Array(t.hidden,i);return this.position++,{embed:_,layers:m,final_norm:w,hidden:v}}async debugFMForward(r=42){let o=this.config.fm,t=this.device.createCommandEncoder();this.fmForward(t),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let e=this.workBuffers,n=await this.readF32Array(e.semantic_logits,o.semantic_vocab),s=await this.readU32Array(e.acoustic_codes,o.n_acoustic_out),i=await this.readF32Array(e.x_t,o.n_acoustic_out);return{semantic_logits:n,velocities:[],acoustic_codes:s,x_final:i}}reset(){this.position=0}async backboneStepAndRead(r,o=!1){let t=this.device.createCommandEncoder();this.backboneStep(t,r,o),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let e=await this.readU32(this.workBuffers.argmax_result);return this.position++,e}async debugBackboneStepFull(r,o=!1){let t=this.device.createCommandEncoder();this.backboneStep(t,r,o),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let e=await this.readF32Array(this.workBuffers.normed,this.config.backbone.dim);return this.position++,e}async debugFMStep(r){let o=this.config.fm,t=this.device.createCommandEncoder();this.fmForward(t,r),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let e=this.workBuffers;return{semantic_logits:await this.readF32Array(e.semantic_logits,o.semantic_vocab),acoustic_codes:await this.readU32Array(e.acoustic_codes,o.n_acoustic_out),x_final:await this.readF32Array(e.x_t,o.n_acoustic_out)}}async fmStepAndRead(){let r=this.device.createCommandEncoder();return this.fmForward(r),this.device.queue.submit([r.finish()]),await this.device.queue.onSubmittedWorkDone(),this.readU32Array(this.workBuffers.acoustic_codes,this.config.fm.n_acoustic_out)}async generate(r,o,t,e,n=500,s){if(!this.isReady)throw new Error("Engine not initialized. Call init() and loadWeights() first.");this.reset();let i=performance.now(),a=[];if(e&&t>0){let h=this.config.backbone.dim;for(let P=0;P<t;P++){let C=this.device.createBuffer({size:h*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!0});new Float32Array(C.getMappedRange()).set(e.subarray(P*h,(P+1)*h)),C.unmap(),a.push(C)}}for(let h=0;h<r.length;h++){let P=r[h],C=this.device.createCommandEncoder();if(h>=o&&h<o+t&&a.length>0){let b=h-o;this.backboneStep(C,P,!1,a[b])}else this.backboneStep(C,P);this.device.queue.submit([C.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let _=performance.now();{let h=this.device.createCommandEncoder();this.backboneStep(h,24,!1),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let m=[],w=[],v=this.config.backbone,c=this.config.fm,d=this.pipelines,f=this.workBuffers,k=this.modelBuffers;for(let h=0;h<n;h++){if(h>0){let U=this.device.createCommandEncoder(),S=U.beginComputePass({label:`multiCBEmbed_frame${h}`}),M=this.packUniform([{u:v.dim},{u:8194},{u:23},{u:36}]);this.dispatch(S,d.multiCodebookEmbed,[k.audio_embeddings,f.semantic_argmax,f.acoustic_codes,f.hidden,M],[g(v.dim,256)]),S.end();let $=U.beginComputePass({label:`mcb_copy_frame${h}`}),z=this.packUniform([{u:v.dim}]);this.dispatch($,d.copyBuffer,[f.hidden,f.fm_gate,z],[g(v.dim,256)]),$.end(),this.backboneStep(U,0,!1,f.fm_gate),this.device.queue.submit([U.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let P=this.device.createCommandEncoder();this.fmForward(P),this.device.queue.submit([P.finish()]),await this.device.queue.onSubmittedWorkDone();let C=await this.readF32Array(f.semantic_logits,c.semantic_vocab);C[0]=-1/0;let b=8194;for(let U=b;U<C.length;U++)C[U]=-1/0;let G=Wa(C,.9,.8);if(G<=1)break;m.push(G);let x=new Uint32Array([G]);this.device.queue.writeBuffer(f.semantic_argmax,0,x);let y=await this.readU32Array(f.acoustic_codes,c.n_acoustic_out);w.push(Array.from(y)),s?.(h,G,y)}let q=performance.now(),l;if(m.length>0){let h=new Uint32Array(m),P=new Uint32Array(w.flat());l=await this.codecDecode(h,P)}else l=new Float32Array(0);let p=performance.now();return{semanticCodes:m,acousticCodes:w,audio:l,stats:{backboneMs:_-i,fmMs:q-_,codecMs:p-q,totalMs:p-i,framesGenerated:m.length}}}destroy(){if(this.workBuffers)for(let r of Object.values(this.workBuffers))r.destroy();for(let r of this.kvCaches)r.k.destroy(),r.v.destroy();this.device?.destroy()}};function ka(u){let r=new DataView(u),o=new Uint8Array(u);if(o[0]!==147||o[1]!==78||o[2]!==85||o[3]!==77||o[4]!==80||o[5]!==89)throw new Error("Not a valid .npy file");let t=o[6],e=o[7],n,s;if(t===1)n=r.getUint16(8,!0),s=10;else if(t===2)n=r.getUint32(8,!0),s=12;else throw new Error(`Unsupported npy version: ${t}.${e}`);let i=new TextDecoder().decode(o.slice(s,s+n)),a=i.match(/'descr'\s*:\s*'([^']+)'/),_=i.match(/'shape'\s*:\s*\(([^)]*)\)/),m=i.match(/'fortran_order'\s*:\s*(True|False)/);if(!a||!_)throw new Error(`Cannot parse npy header: ${i}`);let w=a[1],v=_[1].trim(),c=v===""?[]:v.split(",").filter(l=>l.trim()!=="").map(l=>parseInt(l.trim()));if(m?m[1]==="True":!1)throw new Error("Fortran-order arrays not supported");let f=s+n,k=u.slice(f),q;switch(w){case"<f4":case"=f4":q=new Float32Array(k);break;case"<f8":case"=f8":{let l=new Float64Array(k);q=new Float32Array(l.length);for(let p=0;p<l.length;p++)q[p]=l[p];break}case"<i4":case"=i4":q=new Int32Array(k);break;case"<u4":case"=u4":q=new Uint32Array(k);break;case"<i8":case"=i8":{let l=new BigInt64Array(k);q=new Int32Array(l.length);for(let p=0;p<l.length;p++)q[p]=Number(l[p]);break}default:throw new Error(`Unsupported dtype: ${w}`)}return{dtype:w,shape:c,data:q}}async function Na(u){let r=await fetch(u);if(!r.ok)throw new Error(`Failed to fetch ${u}: ${r.status}`);let o=await r.arrayBuffer();return ka(o)}function Da(u,r,o=.01,t=.01){if(u.length!==r.length)return{passed:!1,maxAbsDiff:1/0,maxRelDiff:1/0,mismatchCount:u.length,totalCount:r.length};let e=0,n=0,s=0;for(let i=0;i<u.length;i++){let a=u[i],_=r[i],m=Math.abs(a-_),w=Math.abs(_)>1e-8?m/Math.abs(_):0;e=Math.max(e,m),n=Math.max(n,w),m>o+t*Math.abs(_)&&s++}return{passed:s===0,maxAbsDiff:e,maxRelDiff:n,mismatchCount:s,totalCount:u.length}}export{ie as HF_VOXTRAL_URL,K as TOKENS,he as TekkenTokenizer,ge as VoxtralEngine,Da as allclose,Oa as clearWeightCache,fe as convertBF16toF16,J as defaultConfig,ee as loadComponentBulk,Fa as loadComponentWeights,pe as loadManifest,Na as loadNpy,ye as loadTensorFromManifest,Sa as loadTensorFromSafetensors,le as loadWeightsFromHF,ka as parseNpy,Pe as parseSafetensorsHeader,Aa as runBenchmark};
