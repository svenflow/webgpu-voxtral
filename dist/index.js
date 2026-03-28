var ve={backbone:{dim:3072,n_layers:26,head_dim:128,hidden_dim:9216,n_heads:32,n_kv_heads:8,vocab_size:131072,rope_theta:1e6,norm_eps:1e-5},fm:{input_dim:3072,dim:3072,n_layers:3,head_dim:128,hidden_dim:9216,n_heads:32,n_kv_heads:8,nfe:8,cfg_alpha:1.2,rope_theta:1e4,sigma:1e-5,sigma_max:1,n_acoustic_out:36,semantic_vocab:8320},codec:{dim:1024,hidden_dim:4096,head_dim:128,n_heads:8,n_kv_heads:8,semantic_codebook_size:8192,semantic_dim:256,n_acoustic_codebook:36,acoustic_codebook_size:21,sampling_rate:24e3,frame_rate:12.5,patch_size:240,decoder_stages:4,decoder_layers_per_stage:2,decoder_conv_strides:[1,2,2,2],decoder_conv_kernels:[3,4,4,4],attn_sliding_window:16,norm_eps:.01,qk_norm_eps:1e-6,qk_norm:!0,layer_scale:!0,weight_norm_conv:!0}};async function ka(x){let o=await(await fetch(x,{headers:{Range:"bytes=0-7"}})).arrayBuffer(),t=Number(new DataView(o).getBigUint64(0,!0)),n=await(await fetch(x,{headers:{Range:`bytes=8-${8+t-1}`}})).text();return{header:JSON.parse(n),dataOffset:8+t}}function wa(x){let s=new Uint16Array(x),o=new Uint16Array(s.length);for(let t=0;t<s.length;t++){let a=s[t],n=a>>15&1,r=a>>7&255,i=a&127;if(r===255)o[t]=n<<15|31744|(i?512:0);else if(r===0)o[t]=n<<15;else{let e=r-127;if(e>15)o[t]=n<<15|31744;else if(e<-14){let g=-14-e;if(g>10)o[t]=n<<15;else{let u=(128|i<<1)>>g>>1;o[t]=n<<15|u&1023}}else{let g=e+15,u=i<<3;o[t]=n<<15|g<<10|u&1023}}}return o.buffer}async function ke(x){let s=await fetch(`${x}/manifest.json`);if(!s.ok)throw new Error(`Failed to load manifest: ${s.status}`);return s.json()}function Pa(x){let s=new ArrayBuffer(4);new Float32Array(s)[0]=x;let o=new Uint32Array(s)[0],t=o>>16&32768,a=(o>>23&255)-127+15,n=o>>13&1023;return a<=0?t:a>=31?t|31744:t|a<<10|n}function we(x,s,o){let t=s.byteLength,a=Math.ceil(t/4)*4,n=x.createBuffer({size:a,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:o,mappedAtCreation:!0});return new Uint16Array(n.getMappedRange(0,s.byteLength)).set(s),n.unmap(),n}async function me(x,s,o,t,a){let n=Object.entries(o.tensors).filter(([k,c])=>c.component===t).map(([k,c])=>k),i=o.tensors[n[0]].file;a&&a({loaded:0,total:n.length,component:t,tensor:`downloading ${i}...`});let e=await fetch(`${s}/${i}`);if(!e.ok)throw new Error(`Failed to download ${i}: ${e.status}`);let g=await e.arrayBuffer(),u=new Map,B=new Map;for(let k=0;k<n.length;k++){let c=n[k],d=o.tensors[c],f=new Uint16Array(g,d.offset,d.size/2),b=we(x,f,c);u.set(c,b),B.set(c,{shape:d.shape,buffer:b}),a&&(k%20===0||k===n.length-1)&&a({loaded:k+1,total:n.length,component:t,tensor:c})}return{buffers:u,tensors:B}}var ya="voxtral-weights",xa=1,L="tensors";function he(){return new Promise((x,s)=>{let o=indexedDB.open(ya,xa);o.onupgradeneeded=()=>{let t=o.result;t.objectStoreNames.contains(L)||t.createObjectStore(L)},o.onsuccess=()=>x(o.result),o.onerror=()=>s(o.error)})}async function Ua(x,s){return new Promise((o,t)=>{let r=x.transaction(L,"readonly").objectStore(L).get(s);r.onsuccess=()=>o(r.result??null),r.onerror=()=>t(r.error)})}async function Ba(x,s,o){return new Promise((t,a)=>{let i=x.transaction(L,"readwrite").objectStore(L).put(o,s);i.onsuccess=()=>t(),i.onerror=()=>a(i.error)})}async function qa(x){return new Promise((s,o)=>{let n=x.transaction(L,"readonly").objectStore(L).count();n.onsuccess=()=>s(n.result),n.onerror=()=>o(n.error)})}async function Ca(){let x=await he();return new Promise((s,o)=>{let n=x.transaction(L,"readwrite").objectStore(L).clear();n.onsuccess=()=>{x.close(),s()},n.onerror=()=>{x.close(),o(n.error)}})}async function Ga(){let x=await he();return new Promise((s,o)=>{let n=x.transaction(L,"readonly").objectStore(L).openCursor(),r=0,i=0;n.onsuccess=()=>{let e=n.result;if(e){r++;let g=e.value;(g instanceof ArrayBuffer||g&&g.byteLength!==void 0)&&(i+=g.byteLength),e.continue()}else x.close(),s({count:r,sizeBytes:i})},n.onerror=()=>{x.close(),o(n.error)}})}var J="https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/consolidated.safetensors";function Sa(x){return x.startsWith("acoustic_transformer.")?"fm":x.startsWith("audio_tokenizer.")?"codec":x.startsWith("layers.")||x.startsWith("norm.")||x.startsWith("mm_audio_embeddings.")?"backbone":"other"}async function Pe(x,s=J,o){let{header:t,dataOffset:a}=await ka(s),n=await he(),r=await qa(n),i=[];for(let[h,l]of Object.entries(t)){if(h==="__metadata__")continue;let v=l;if(!v.data_offsets)continue;let p=Sa(h);p!=="other"&&i.push({name:h,entry:v,component:p})}let e={backbone:0,fm:1,codec:2};i.sort((h,l)=>(e[h.component]??9)-(e[l.component]??9));let g=i.length,u=0;o&&o({loaded:0,total:g,component:"init",tensor:r>0?`${r} tensors cached in IndexedDB`:"Starting fresh download...",cached:!1,bytesDownloaded:0});let B={backbone:{buffers:new Map,tensors:new Map},fm:{buffers:new Map,tensors:new Map},codec:{buffers:new Map,tensors:new Map}},k=`v1:${a}:${g}`,c=6,d=0;async function f(h,l,v){let p=await Ua(n,v);if(p)return{f16Data:new Uint16Array(p),fromCache:!0,fetchedBytes:0};let[U,P]=l.data_offsets,w=a+U,q=a+P-1,S=P-U,M=await fetch(s,{headers:{Range:`bytes=${w}-${q}`}});if(!M.ok&&M.status!==206)throw new Error(`Failed to fetch tensor ${h}: HTTP ${M.status}`);let $=await M.arrayBuffer(),T;if(l.dtype==="BF16"){let z=wa($);T=new Uint16Array(z)}else if(l.dtype==="F16")T=new Uint16Array($);else if(l.dtype==="F32"){let z=new Float32Array($),O=new Uint16Array(z.length);for(let y=0;y<z.length;y++)O[y]=Pa(z[y]);T=O}else throw new Error(`Unsupported dtype for ${h}: ${l.dtype}`);return await Ba(n,v,T.buffer),{f16Data:T,fromCache:!1,fetchedBytes:S}}let b=new Map,F=0,C=0,m=new Map;for(;F<i.length&&b.size<c;){let h=F++,{name:l,entry:v}=i[h],p=`${k}:${l}`;b.set(h,f(l,v,p).then(U=>({idx:h,...U})))}for(;C<i.length;){if(m.has(C)){let l=m.get(C);m.delete(C);let{name:v,entry:p,component:U}=i[C];u+=l.fetchedBytes;let P=we(x,l.f16Data,v),w=B[U];w.buffers.set(v,P),w.tensors.set(v,{shape:p.shape,buffer:P}),d++,o&&o({loaded:d,total:g,component:U,tensor:v,cached:l.fromCache,bytesDownloaded:u}),C++;continue}let h=await Promise.race(b.values());if(b.delete(h.idx),m.set(h.idx,{f16Data:h.f16Data,fromCache:h.fromCache,fetchedBytes:h.fetchedBytes}),F<i.length){let l=F++,{name:v,entry:p}=i[l],U=`${k}:${v}`;b.set(l,f(v,p,U).then(P=>({idx:l,...P})))}}return n.close(),{backbone:B.backbone,fm:B.fm,codec:B.codec}}var ye=`
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
`,xe=`
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
`,Ue=`
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
`,Be=`
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
`,qe=`
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
`,Ce=`
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
`,Ge=`
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
`,Se=`
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
`,Fe=`
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
`,Ae=`
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
`,$e=`
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
`,Te=`
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
`,ze=`
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
`,Me=`
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
`,Oe=`
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
`,Ee=`
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
`,Le=`
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
  let scaled = round((clamped + 1.0) * 0.5 * f32(params.levels - 1u));
  output[i] = u32(clamp(scaled, 0.0, f32(params.levels - 1u))) + params.offset;
}
`,De=`
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
`,Ie=`
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
`,Re=`
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
`,Ne=`
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
`,Ve=`
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
`,je=`
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
`,Ke=`
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
  // 2D dispatch: gid.x = t_out (within workgroups), gid.y = co
  let t_out = gid.x;
  let co = gid.y;
  let n_frames_out = params.n_frames;
  if (t_out >= n_frames_out || co >= params.c_out) { return; }

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
`,He=`
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
`,Ye=`
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
  // 2D dispatch: gid.x = t_out (within workgroups), gid.y = co
  let t_out = gid.x;
  let co = gid.y;
  if (t_out >= params.n_frames_out || co >= params.c_out) { return; }

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
`,Qe=`
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
`,Xe=`
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
`,Ze=`
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
`,Je=`
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
`,ea=`
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
`,aa=`
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
`,ta=`
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
`,ra=`
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
`,sa=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  window: u32,      // sliding window size (must be > 0)
}

@group(0) @binding(0) var<storage, read> q: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(1) var<storage, read> k: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;  // [n_heads, seq_len, window]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // gid.x = slot (within window), gid.y = qi, gid.z = head
  let slot = gid.x;
  let qi = gid.y;
  let h = gid.z;
  let W = params.window;
  if (slot >= W || qi >= params.seq_len || h >= params.n_heads) { return; }

  let idx = h * params.seq_len * W + qi * W + slot;

  // Map slot to actual key position
  // ki_start = max(0, qi - W + 1), ki = ki_start + slot
  var ki_start: u32 = 0u;
  if (qi + 1u > W) { ki_start = qi - W + 1u; }
  let ki = ki_start + slot;

  // Inactive slot (beyond causal boundary or past qi)
  if (ki > qi) {
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
`,ia=`
struct Params {
  n_heads: u32,
  seq_len: u32,
  window: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;  // [n_heads, seq_len, window]
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let qi = idx % params.seq_len;
  let W = params.window;
  let base = h * params.seq_len * W + qi * W;

  var maxVal: f32 = -1e30;
  for (var j: u32 = 0u; j < W; j++) {
    maxVal = max(maxVal, scores[base + j]);
  }

  var expSum: f32 = 0.0;
  for (var j: u32 = 0u; j < W; j++) {
    let e = exp(scores[base + j] - maxVal);
    scores[base + j] = e;
    expSum += e;
  }

  for (var j: u32 = 0u; j < W; j++) {
    scores[base + j] /= expSum;
  }
}
`,oa=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  window: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;   // [n_heads, seq_len, window]
@group(0) @binding(1) var<storage, read> v: array<f32>;        // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [seq_len, n_heads, head_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 2D dispatch: gid.x = within (n_heads * head_dim), gid.y = qi
  let hd_idx = gid.x;
  let qi = gid.y;
  let nhd = params.n_heads * params.head_dim;
  if (hd_idx >= nhd || qi >= params.seq_len) { return; }

  let h = hd_idx / params.head_dim;
  let d = hd_idx % params.head_dim;
  let W = params.window;

  // ki_start = max(0, qi - W + 1)
  var ki_start: u32 = 0u;
  if (qi + 1u > W) { ki_start = qi - W + 1u; }

  var sum: f32 = 0.0;
  let scoreBase = h * params.seq_len * W + qi * W;
  for (var slot: u32 = 0u; slot < W; slot++) {
    let ki = ki_start + slot;
    if (ki > qi) { break; }
    let score = scores[scoreBase + slot];
    let vIdx = ki * params.n_heads * params.head_dim + h * params.head_dim + d;
    sum += score * v[vIdx];
  }

  output[qi * nhd + hd_idx] = sum;
}
`,na=`
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
`,da=`
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
`,ca=`
struct Params {
  total: u32,    // T * hidden_dim
  grid_x: u32,   // number of workgroups in x dimension (for 2D dispatch)
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;  // in-place output
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * params.grid_x * 256u + gid.x;
  if (i >= params.total) { return; }
  let g = gate[i];
  let silu = g / (1.0 + exp(-g));
  gate[i] = silu * up[i];
}
`,ua=`
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
`,ma=`
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
`,fa=`
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
`,pa=`
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
`,_a=`
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
`,la=`
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
`,ha=`
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
`;function _(x,s){return Math.ceil(x/s)}function Aa(x,s,o){let t=x.length,a=new Float32Array(t);for(let c=0;c<t;c++)a[c]=x[c]/o;let n=-1/0;for(let c=0;c<t;c++)a[c]>n&&(n=a[c]);let r=0;for(let c=0;c<t;c++)a[c]=Math.exp(a[c]-n),r+=a[c];for(let c=0;c<t;c++)a[c]/=r;let i=Array.from({length:t},(c,d)=>d);i.sort((c,d)=>a[d]-a[c]);let e=0,g=t;for(let c=0;c<t;c++)if(e+=a[i[c]],e>=s){g=c+1;break}let u=0;for(let c=0;c<g;c++)u+=a[i[c]];let B=Math.random()*u,k=0;for(let c=0;c<g;c++)if(k+=a[i[c]],k>=B)return i[c];return i[0]}var re=class{device=null;config;maxSeqLen;modelBuffers=null;workBuffers=null;pipelines=null;kvCaches=[];position=0;constructor(s={}){this.config=s.config||ve,this.maxSeqLen=s.maxSeqLen||4096}async init(){if(!navigator.gpu)throw new Error("WebGPU not supported");let s=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!s)throw new Error("No WebGPU adapter");let o=[];s.features.has("shader-f16")&&o.push("shader-f16");let t=2*1024*1024*1024,a=s.limits.maxBufferSize,n=s.limits.maxStorageBufferBindingSize,r=a>=t&&n>=t;this.device=await s.requestDevice({requiredFeatures:o,requiredLimits:{maxBufferSize:r?t:a,maxStorageBufferBindingSize:r?t:n}}),this.createWorkBuffers(),this.createKVCaches(),this.createPipelines()}createPipeline(s,o){let t=this.device,a=t.createShaderModule({code:s,label:o});return t.createComputePipeline({layout:"auto",compute:{module:a,entryPoint:"main"},label:o})}createPipelines(){let s=(o,t)=>this.createPipeline(o,t);this.pipelines={matvecF16:s(ye,"matvecF16"),matvecF16Chunked:s(xe,"matvecF16Chunked"),matvecF16Offset:s(Ze,"matvecF16Offset"),rmsNorm:s(Ue,"rmsNorm"),rmsNormOffset:s(Je,"rmsNormOffset"),embeddingLookup:s(Be,"embeddingLookup"),rope:s(qe,"rope"),ropeOffset:s(Ce,"ropeOffset"),attnScore:s(Ge,"attnScore"),softmax:s(Se,"softmax"),attnValue:s(Fe,"attnValue"),kvCacheWrite:s(Ae,"kvCacheWrite"),swiGLU:s($e,"swiGLU"),addVectors:s(Te,"addVectors"),addVectorsOffset:s(ea,"addVectorsOffset"),addInPlace:s(ze,"addInPlace"),addInPlaceOffset:s(aa,"addInPlaceOffset"),copyBuffer:s(Me,"copyBuffer"),copyBufferOffset:s(ra,"copyBufferOffset"),timeEmbedding:s(Oe,"timeEmbedding"),eulerStep:s(We,"eulerStep"),cfgCombine:s(Ee,"cfgCombine"),fsqQuantize:s(Le,"fsqQuantize"),biAttnScore:s(De,"biAttnScore"),biSoftmax:s(Ie,"biSoftmax"),biAttnValue:s(Re,"biAttnValue"),swiGLUOffset:s(ta,"swiGLUOffset"),zeroFill:s(ha,"zeroFill"),multiCodebookEmbed:s(la,"multiCodebookEmbed"),vqLookup:s(Ne,"vqLookup"),fsqDequant:s(je,"fsqDequant"),causalConv1d:s(Ke,"causalConv1d"),causalConvTranspose1d:s(Ye,"causalConvTranspose1d"),convTransposeNormScale:s(He,"convTransposeNormScale"),layerScale:s(Qe,"layerScale"),alibiAttnScore:s(sa,"alibiAttnScore"),codecSoftmax:s(ia,"codecSoftmax"),codecAttnValue:s(oa,"codecAttnValue"),batchedMatvecF16:s(na,"batchedMatvecF16"),batchedRmsNorm:s(da,"batchedRmsNorm"),batchedSwiGLU:s(ca,"batchedSwiGLU"),batchedAdd:s(ua,"batchedAdd"),batchedCopy:s(ma,"batchedCopy"),batchedLayerScale:s(fa,"batchedLayerScale"),qkNorm:s(pa,"qkNorm"),concatCodecInput:s(_a,"concatCodecInput"),argmax:s(Xe,"argmax"),normalizeCodebook:s(Ve,"normalizeCodebook")}}createUniform(s){let o=this.device,t=Math.ceil(s.byteLength/16)*16,a=o.createBuffer({size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(a.getMappedRange()).set(new Uint8Array(s)),a.unmap(),a}packUniform(s){let o=new ArrayBuffer(s.length*4),t=new Uint32Array(o),a=new Float32Array(o);for(let n=0;n<s.length;n++){let r=s[n];r.u!==void 0?t[n]=r.u:r.f!==void 0&&(a[n]=r.f)}return this.createUniform(o)}createWorkBuffers(){let s=this.device,o=this.config.backbone,t=this.config.fm,a=GPUBufferUsage.STORAGE,n=GPUBufferUsage.COPY_SRC,r=GPUBufferUsage.COPY_DST,i=(e,g,u=0)=>s.createBuffer({size:e,usage:a|n|r|u,label:g});this.workBuffers={hidden:i(o.dim*4,"hidden"),residual:i(o.dim*4,"residual"),normed:i(o.dim*4,"normed"),q:i(o.n_heads*o.head_dim*4,"q"),k:i(o.n_kv_heads*o.head_dim*4,"k"),v:i(o.n_kv_heads*o.head_dim*4,"v"),attn_out:i(o.n_heads*o.head_dim*4,"attn_out"),scores:i(o.n_heads*this.maxSeqLen*4,"scores"),gate:i(o.hidden_dim*4,"gate"),up:i(o.hidden_dim*4,"up"),down:i(o.dim*4,"down"),x_t:i(t.n_acoustic_out*4,"x_t"),velocity:i(t.n_acoustic_out*4,"velocity"),v_uncond:i(t.n_acoustic_out*4,"v_uncond"),time_embed:i(t.dim*4,"time_embed"),time_proj:i(t.dim*4,"time_proj"),x_t_proj:i(t.dim*4,"x_t_proj"),fm_hidden:i(t.dim*4,"fm_hidden"),fm_residual:i(t.dim*4,"fm_residual"),fm_normed:i(t.dim*4,"fm_normed"),fm_q:i(3*t.n_heads*t.head_dim*4,"fm_q"),fm_k:i(3*t.n_kv_heads*t.head_dim*4,"fm_k"),fm_v:i(3*t.n_kv_heads*t.head_dim*4,"fm_v"),fm_attn_out:i(3*t.n_heads*t.head_dim*4,"fm_attn_out"),fm_scores:i(t.n_heads*3*3*4,"fm_scores"),fm_seq:i(3*t.dim*4,"fm_seq"),fm_gate:i(3*t.hidden_dim*4,"fm_gate"),fm_up:i(3*t.hidden_dim*4,"fm_up"),fm_down:i(3*t.dim*4,"fm_down"),semantic_logits:i(t.semantic_vocab*4,"semantic_logits"),semantic_argmax:i(4,"semantic_argmax"),acoustic_out:i(t.n_acoustic_out*4,"acoustic_out"),acoustic_codes:i(t.n_acoustic_out*4,"acoustic_codes"),logits:i(o.vocab_size*4,"logits"),argmax_result:i(4,"argmax_result")}}createKVCaches(){let s=this.device,o=this.config.backbone,t=o.n_kv_heads*o.head_dim;this.kvCaches=[];for(let a=0;a<o.n_layers;a++)this.kvCaches.push({k:s.createBuffer({size:this.maxSeqLen*t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:`kv_cache.${a}.k`}),v:s.createBuffer({size:this.maxSeqLen*t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:`kv_cache.${a}.v`})})}async loadWeights(s,o){let t=this.device,a=await ke(s),n=o||(()=>{});n({loaded:0,total:3,component:"all",tensor:"Loading backbone..."});let r=await me(t,s,a,"backbone",o);n({loaded:1,total:3,component:"all",tensor:"Loading FM transformer..."});let i=await me(t,s,a,"fm",o);n({loaded:2,total:3,component:"all",tensor:"Loading codec decoder..."});let e=await me(t,s,a,"codec",o);this.modelBuffers=this.organizeWeights(r,i,e),n({loaded:3,total:3,component:"all",tensor:"Done!"})}async loadWeightsFromHF(s=J,o){let t=this.device,{backbone:a,fm:n,codec:r}=await Pe(t,s,o);this.modelBuffers=this.organizeWeights(a,n,r),await this.normalizeVQCodebook(),await this.precomputeConvTransposeScales()}async normalizeVQCodebook(){let s=this.device,o=this.pipelines,t=this.modelBuffers,a=this.config.codec,n=this.packUniform([{u:a.semantic_codebook_size},{u:a.semantic_dim},{f:1e-5}]),r=s.createCommandEncoder({label:"normalize_codebook"}),i=r.beginComputePass({label:"normalize_codebook"});this.dispatch(i,o.normalizeCodebook,[t.codec_semantic_codebook,t.codec_cluster_usage,n],[_(a.semantic_codebook_size*a.semantic_dim/2,128)]),i.end(),s.queue.submit([r.finish()]),await s.queue.onSubmittedWorkDone()}async precomputeConvTransposeScales(){let s=this.device,o=this.pipelines,t=this.modelBuffers,a=this.config.codec,n=s.createCommandEncoder({label:"precompute_conv_transpose_scales"});for(let r=0;r<a.decoder_stages;r++){let i=t.codec_stages[r];if(!i.conv_w||!i.conv_g)continue;let e=s.createBuffer({size:a.dim*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,label:`codec_conv_transpose_scale_s${r}`});i.conv_scale=e;let u=this.packUniform([{u:a.dim},{u:a.dim},{u:4}]),B=n.beginComputePass({label:`conv_transpose_norm_scale_s${r}`});this.dispatch(B,o.convTransposeNormScale,[i.conv_w,i.conv_g,e,u],[a.dim]),B.end()}s.queue.submit([n.finish()]),await s.queue.onSubmittedWorkDone()}organizeWeights(s,o,t){let a=(e,g)=>{let u=e.buffers.get(g);if(!u)throw new Error(`Missing weight: ${g}`);return u},n=[];for(let e=0;e<this.config.backbone.n_layers;e++)n.push({attn_norm:a(s,`layers.${e}.attention_norm.weight`),wq:a(s,`layers.${e}.attention.wq.weight`),wk:a(s,`layers.${e}.attention.wk.weight`),wv:a(s,`layers.${e}.attention.wv.weight`),wo:a(s,`layers.${e}.attention.wo.weight`),ffn_norm:a(s,`layers.${e}.ffn_norm.weight`),w1:a(s,`layers.${e}.feed_forward.w1.weight`),w2:a(s,`layers.${e}.feed_forward.w2.weight`),w3:a(s,`layers.${e}.feed_forward.w3.weight`)});let r=[];for(let e=0;e<this.config.fm.n_layers;e++)r.push({attn_norm:a(o,`acoustic_transformer.layers.${e}.attention_norm.weight`),wq:a(o,`acoustic_transformer.layers.${e}.attention.wq.weight`),wk:a(o,`acoustic_transformer.layers.${e}.attention.wk.weight`),wv:a(o,`acoustic_transformer.layers.${e}.attention.wv.weight`),wo:a(o,`acoustic_transformer.layers.${e}.attention.wo.weight`),ffn_norm:a(o,`acoustic_transformer.layers.${e}.ffn_norm.weight`),w1:a(o,`acoustic_transformer.layers.${e}.feed_forward.w1.weight`),w2:a(o,`acoustic_transformer.layers.${e}.feed_forward.w2.weight`),w3:a(o,`acoustic_transformer.layers.${e}.feed_forward.w3.weight`)});let i=[];for(let e=0;e<4;e++){let g=1+e*2,u=2+e*2,B=e<3;i.push({transformer_layers:this.getCodecTransformerLayers(t,g),...B?{conv_w:a(t,`audio_tokenizer.decoder_blocks.${u}.conv.parametrizations.weight.original1`),conv_g:a(t,`audio_tokenizer.decoder_blocks.${u}.conv.parametrizations.weight.original0`)}:{}})}return{tok_embeddings:a(s,"mm_audio_embeddings.tok_embeddings.weight"),audio_embeddings:a(s,"mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"),backbone_layers:n,final_norm:a(s,"norm.weight"),fm_input_proj:a(o,"acoustic_transformer.input_projection.weight"),fm_llm_proj:a(o,"acoustic_transformer.llm_projection.weight"),fm_time_proj:a(o,"acoustic_transformer.time_projection.weight"),fm_layers:r,fm_norm:a(o,"acoustic_transformer.norm.weight"),fm_semantic_out:a(o,"acoustic_transformer.semantic_codebook_output.weight"),fm_acoustic_out:a(o,"acoustic_transformer.acoustic_codebook_output.weight"),codec_input_conv_w:a(t,"audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1"),codec_input_conv_g:a(t,"audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0"),codec_stages:i,codec_output_conv_w:a(t,"audio_tokenizer.output_proj.conv.parametrizations.weight.original1"),codec_output_conv_g:a(t,"audio_tokenizer.output_proj.conv.parametrizations.weight.original0"),codec_semantic_codebook:a(t,"audio_tokenizer.quantizer.semantic_codebook.embedding_sum"),codec_cluster_usage:a(t,"audio_tokenizer.quantizer.semantic_codebook.cluster_usage")}}getCodecTransformerLayers(s,o){let t=n=>{let r=s.buffers.get(n);if(!r)throw new Error(`Missing codec weight: ${n}`);return r},a=[];for(let n=0;n<2;n++){let r=`audio_tokenizer.decoder_blocks.${o}.layers.${n}`;a.push({attn_norm:t(`${r}.attention_norm.weight`),q_norm:t(`${r}.attention.q_norm.weight`),k_norm:t(`${r}.attention.k_norm.weight`),wq:t(`${r}.attention.wq.weight`),wk:t(`${r}.attention.wk.weight`),wv:t(`${r}.attention.wv.weight`),wo:t(`${r}.attention.wo.weight`),attn_scale:t(`${r}.attention_scale`),ffn_norm:t(`${r}.ffn_norm.weight`),w1:t(`${r}.feed_forward.w1.weight`),w2:t(`${r}.feed_forward.w2.weight`),w3:t(`${r}.feed_forward.w3.weight`),ffn_scale:t(`${r}.ffn_scale`)})}return a}dispatch(s,o,t,a){let n=t.map((i,e)=>({binding:e,resource:{buffer:i}})),r=this.device.createBindGroup({layout:o.getBindGroupLayout(0),entries:n});s.setPipeline(o),s.setBindGroup(0,r),s.dispatchWorkgroups(...a)}backboneStep(s,o,t=!1,a){let n=this.pipelines,r=this.workBuffers,i=this.modelBuffers,e=this.config.backbone,g=this.position,u;u=s.beginComputePass({label:`embed_pos${g}`});let B=this.packUniform([{u:o},{u:e.dim}]),k=t?i.audio_embeddings:i.tok_embeddings;if(this.dispatch(u,n.embeddingLookup,[k,r.hidden,B],[_(e.dim,256)]),u.end(),a){u=s.beginComputePass({label:`voice_embed_pos${g}`});let f=this.packUniform([{u:e.dim}]);this.dispatch(u,n.copyBuffer,[a,r.hidden,f],[_(e.dim,256)]),u.end()}for(let f=0;f<e.n_layers;f++){let b=i.backbone_layers[f],F=this.kvCaches[f];u=s.beginComputePass({label:`layer${f}_attn`});let C=this.packUniform([{u:e.dim}]);this.dispatch(u,n.copyBuffer,[r.hidden,r.residual,C],[_(e.dim,256)]);let m=this.packUniform([{u:e.dim},{f:e.norm_eps}]);this.dispatch(u,n.rmsNorm,[r.hidden,b.attn_norm,r.normed,m],[1]),u.end(),u=s.beginComputePass({label:`layer${f}_qkv`});let h=this.packUniform([{u:e.n_heads*e.head_dim},{u:e.dim}]);this.dispatch(u,n.matvecF16,[b.wq,r.normed,r.q,h],[e.n_heads*e.head_dim]);let l=this.packUniform([{u:e.n_kv_heads*e.head_dim},{u:e.dim}]);this.dispatch(u,n.matvecF16,[b.wk,r.normed,r.k,l],[e.n_kv_heads*e.head_dim]),this.dispatch(u,n.matvecF16,[b.wv,r.normed,r.v,l],[e.n_kv_heads*e.head_dim]),u.end(),u=s.beginComputePass({label:`layer${f}_rope_attn`});let v=this.packUniform([{u:e.head_dim},{u:g},{u:e.n_heads},{f:e.rope_theta}]);this.dispatch(u,n.rope,[r.q,v],[_(e.n_heads*e.head_dim/2,64)]);let p=this.packUniform([{u:e.head_dim},{u:g},{u:e.n_kv_heads},{f:e.rope_theta}]);this.dispatch(u,n.rope,[r.k,p],[_(e.n_kv_heads*e.head_dim/2,64)]);let U=this.packUniform([{u:g},{u:e.n_kv_heads*e.head_dim}]);this.dispatch(u,n.kvCacheWrite,[r.k,r.v,F.k,F.v,U],[_(e.n_kv_heads*e.head_dim,256)]);let P=g+1,w=e.n_heads/e.n_kv_heads,q=this.packUniform([{u:e.n_heads},{u:e.n_kv_heads},{u:e.head_dim},{u:P},{u:w}]);this.dispatch(u,n.attnScore,[r.q,F.k,r.scores,q],[_(e.n_heads*P,64)]),u.end(),u=s.beginComputePass({label:`layer${f}_attn_out`});let S=this.packUniform([{u:e.n_heads},{u:P}]);this.dispatch(u,n.softmax,[r.scores,S],[e.n_heads]);let M=this.packUniform([{u:e.n_heads},{u:e.n_kv_heads},{u:e.head_dim},{u:P},{u:w}]);this.dispatch(u,n.attnValue,[r.scores,F.v,r.attn_out,M],[_(e.n_heads*e.head_dim,128)]),u.end(),u=s.beginComputePass({label:`layer${f}_wo_res`});let $=this.packUniform([{u:e.dim},{u:e.n_heads*e.head_dim}]);this.dispatch(u,n.matvecF16,[b.wo,r.attn_out,r.hidden,$],[e.dim]),u.end(),u=s.beginComputePass({label:`layer${f}_res1`});let T=this.packUniform([{u:e.dim}]);this.dispatch(u,n.addInPlace,[r.hidden,r.residual,T],[_(e.dim,256)]),this.dispatch(u,n.copyBuffer,[r.hidden,r.residual,C],[_(e.dim,256)]);let z=this.packUniform([{u:e.dim},{f:e.norm_eps}]);this.dispatch(u,n.rmsNorm,[r.hidden,b.ffn_norm,r.normed,z],[1]),u.end(),u=s.beginComputePass({label:`layer${f}_ffn`});let O=this.packUniform([{u:e.hidden_dim},{u:e.dim}]);this.dispatch(u,n.matvecF16,[b.w1,r.normed,r.gate,O],[e.hidden_dim]),this.dispatch(u,n.matvecF16,[b.w3,r.normed,r.up,O],[e.hidden_dim]),u.end(),u=s.beginComputePass({label:`layer${f}_ffn_out`});let y=this.packUniform([{u:e.hidden_dim}]);this.dispatch(u,n.swiGLU,[r.gate,r.up,y],[_(e.hidden_dim,256)]);let W=this.packUniform([{u:e.dim},{u:e.hidden_dim}]);this.dispatch(u,n.matvecF16,[b.w2,r.gate,r.hidden,W],[e.dim]),u.end(),u=s.beginComputePass({label:`layer${f}_res2`}),this.dispatch(u,n.addInPlace,[r.hidden,r.residual,T],[_(e.dim,256)]),u.end()}u=s.beginComputePass({label:"final_norm"});let c=this.packUniform([{u:e.dim},{f:e.norm_eps}]);this.dispatch(u,n.rmsNorm,[r.hidden,i.final_norm,r.normed,c],[1]),u.end(),u=s.beginComputePass({label:"lm_head"});for(let b=0;b<e.vocab_size;b+=65535){let F=Math.min(65535,e.vocab_size-b),C=this.packUniform([{u:F},{u:e.dim},{u:b}]);this.dispatch(u,n.matvecF16Chunked,[i.tok_embeddings,r.normed,r.logits,C],[F])}u.end(),u=s.beginComputePass({label:"argmax"});let d=this.packUniform([{u:e.vocab_size}]);this.dispatch(u,n.argmax,[r.logits,r.argmax_result,d],[1]),u.end()}fmTransformerPass(s,o){let t=this.pipelines,a=this.workBuffers,n=this.modelBuffers,r=this.config.fm,i=r.dim,e=3,g=r.n_heads*r.head_dim,u=r.n_kv_heads*r.head_dim,B=r.n_heads/r.n_kv_heads;for(let k=0;k<r.n_layers;k++){let c=n.fm_layers[k],d;d=s.beginComputePass({label:`fm_l${k}_attn_prep`});let f=this.packUniform([{u:e*i},{u:0},{u:0}]);this.dispatch(d,t.copyBufferOffset,[a.fm_seq,a.fm_down,f],[_(e*i,256)]);for(let m=0;m<e;m++){let h=m*i,l=this.packUniform([{u:i},{f:1e-5},{u:h},{u:h}]);this.dispatch(d,t.rmsNormOffset,[a.fm_seq,c.attn_norm,a.fm_gate,l],[1])}d.end(),d=s.beginComputePass({label:`fm_l${k}_qkv`});for(let m=0;m<e;m++){let h=m*i,l=m*g,v=m*u,p=this.packUniform([{u:g},{u:i},{u:h},{u:l}]);this.dispatch(d,t.matvecF16Offset,[c.wq,a.fm_gate,a.fm_q,p],[g]);let U=this.packUniform([{u},{u:i},{u:h},{u:v}]);this.dispatch(d,t.matvecF16Offset,[c.wk,a.fm_gate,a.fm_k,U],[u]),this.dispatch(d,t.matvecF16Offset,[c.wv,a.fm_gate,a.fm_v,U],[u])}d.end(),d=s.beginComputePass({label:`fm_l${k}_attn`});let b=this.packUniform([{u:r.n_heads},{u:r.n_kv_heads},{u:r.head_dim},{u:e},{u:B}]);this.dispatch(d,t.biAttnScore,[a.fm_q,a.fm_k,a.fm_scores,b],[_(r.n_heads*e*e,64)]),d.end(),d=s.beginComputePass({label:`fm_l${k}_attn_val`});let F=this.packUniform([{u:r.n_heads},{u:e}]);this.dispatch(d,t.biSoftmax,[a.fm_scores,F],[_(r.n_heads*e,64)]);let C=this.packUniform([{u:r.n_heads},{u:r.n_kv_heads},{u:r.head_dim},{u:e},{u:B}]);this.dispatch(d,t.biAttnValue,[a.fm_scores,a.fm_v,a.fm_attn_out,C],[_(e*r.n_heads*r.head_dim,64)]),d.end(),d=s.beginComputePass({label:`fm_l${k}_wo_res`});for(let m=0;m<e;m++){let h=m*g,l=m*i,v=this.packUniform([{u:i},{u:g},{u:h},{u:l}]);this.dispatch(d,t.matvecF16Offset,[c.wo,a.fm_attn_out,a.fm_seq,v],[i])}d.end(),d=s.beginComputePass({label:`fm_l${k}_res1`});for(let m=0;m<e;m++){let h=m*i,l=this.packUniform([{u:i},{u:h},{u:h}]);this.dispatch(d,t.addInPlaceOffset,[a.fm_seq,a.fm_down,l],[_(i,256)])}this.dispatch(d,t.copyBufferOffset,[a.fm_seq,a.fm_down,f],[_(e*i,256)]),d.end(),d=s.beginComputePass({label:`fm_l${k}_ffn`});for(let m=0;m<e;m++){let h=m*i,l=m*r.hidden_dim,v=this.packUniform([{u:i},{f:1e-5},{u:h},{u:0}]);this.dispatch(d,t.rmsNormOffset,[a.fm_seq,c.ffn_norm,a.fm_normed,v],[1]);let p=this.packUniform([{u:r.hidden_dim},{u:i},{u:0},{u:l}]);this.dispatch(d,t.matvecF16Offset,[c.w1,a.fm_normed,a.fm_gate,p],[r.hidden_dim]),this.dispatch(d,t.matvecF16Offset,[c.w3,a.fm_normed,a.fm_up,p],[r.hidden_dim])}d.end(),d=s.beginComputePass({label:`fm_l${k}_ffn_act`});for(let m=0;m<e;m++){let h=m*r.hidden_dim,l=this.packUniform([{u:r.hidden_dim},{u:h},{u:h}]);this.dispatch(d,t.swiGLUOffset,[a.fm_gate,a.fm_up,l],[_(r.hidden_dim,256)])}d.end(),d=s.beginComputePass({label:`fm_l${k}_ffn_down`});for(let m=0;m<e;m++){let h=m*r.hidden_dim,l=m*i,v=this.packUniform([{u:i},{u:r.hidden_dim},{u:h},{u:l}]);this.dispatch(d,t.matvecF16Offset,[c.w2,a.fm_gate,a.fm_seq,v],[i])}d.end(),d=s.beginComputePass({label:`fm_l${k}_res2`});for(let m=0;m<e;m++){let h=m*i,l=this.packUniform([{u:i},{u:h},{u:h}]);this.dispatch(d,t.addInPlaceOffset,[a.fm_seq,a.fm_down,l],[_(i,256)])}d.end()}{let k=s.beginComputePass({label:"fm_final_norm_vel"}),c=this.packUniform([{u:i},{f:1e-5},{u:0},{u:0}]);this.dispatch(k,t.rmsNormOffset,[a.fm_seq,n.fm_norm,a.fm_normed,c],[1]);let d=this.packUniform([{u:r.n_acoustic_out},{u:i}]);this.dispatch(k,t.matvecF16,[n.fm_acoustic_out,a.fm_normed,o,d],[r.n_acoustic_out]),k.end()}}fmForward(s,o){let t=this.pipelines,a=this.workBuffers,n=this.modelBuffers,r=this.config.fm,i=r.dim,e;e=s.beginComputePass({label:"fm_init"});let g=this.packUniform([{u:r.semantic_vocab},{u:i}]);this.dispatch(e,t.matvecF16,[n.fm_semantic_out,a.normed,a.semantic_logits,g],[r.semantic_vocab]);let u=this.packUniform([{u:i},{u:i}]);this.dispatch(e,t.matvecF16,[n.fm_llm_proj,a.normed,a.fm_hidden,u],[i]);{let c=o??new Float32Array(r.n_acoustic_out);if(!o)for(let d=0;d<r.n_acoustic_out;d++){let f=Math.random(),b=Math.random();c[d]=Math.sqrt(-2*Math.log(f))*Math.cos(2*Math.PI*b)}this.device.queue.writeBuffer(a.x_t,0,c)}e.end(),e=s.beginComputePass({label:"fm_semantic_argmax"});let B=this.packUniform([{u:r.semantic_vocab}]);this.dispatch(e,t.argmax,[a.semantic_logits,a.semantic_argmax,B],[1]),e.end();for(let c=0;c<r.nfe-1;c++){let d=c/(r.nfe-1),f=1/(r.nfe-1);e=s.beginComputePass({label:`fm_step${c}_prep`});let b=this.packUniform([{u:i},{f:d}]);this.dispatch(e,t.timeEmbedding,[a.time_embed,b],[_(i/2,256)]),e.end(),e=s.beginComputePass({label:`fm_step${c}_proj`});let F=this.packUniform([{u:i},{u:i}]);this.dispatch(e,t.matvecF16,[n.fm_time_proj,a.time_embed,a.time_proj,F],[i]);let C=this.packUniform([{u:i},{u:r.n_acoustic_out}]);this.dispatch(e,t.matvecF16,[n.fm_input_proj,a.x_t,a.x_t_proj,C],[i]),e.end(),e=s.beginComputePass({label:`fm_step${c}_assemble`});let m=this.packUniform([{u:i},{u:0},{u:0}]);this.dispatch(e,t.copyBufferOffset,[a.x_t_proj,a.fm_seq,m],[_(i,256)]);let h=this.packUniform([{u:i},{u:0},{u:i}]);this.dispatch(e,t.copyBufferOffset,[a.time_proj,a.fm_seq,h],[_(i,256)]);let l=this.packUniform([{u:i},{u:0},{u:2*i}]);this.dispatch(e,t.copyBufferOffset,[a.fm_hidden,a.fm_seq,l],[_(i,256)]),e.end(),this.fmTransformerPass(s,a.velocity),e=s.beginComputePass({label:`fm_step${c}_uncond`}),this.dispatch(e,t.copyBufferOffset,[a.x_t_proj,a.fm_seq,m],[_(i,256)]),this.dispatch(e,t.copyBufferOffset,[a.time_proj,a.fm_seq,h],[_(i,256)]);let v=this.packUniform([{u:i}]);this.dispatch(e,t.zeroFill,[a.fm_residual,v],[_(i,256)]),this.dispatch(e,t.copyBufferOffset,[a.fm_residual,a.fm_seq,l],[_(i,256)]),e.end(),this.fmTransformerPass(s,a.v_uncond),e=s.beginComputePass({label:`fm_step${c}_euler`});let p=this.packUniform([{u:r.n_acoustic_out},{f:r.cfg_alpha}]);this.dispatch(e,t.cfgCombine,[a.velocity,a.v_uncond,p],[_(r.n_acoustic_out,64)]);let U=this.packUniform([{u:r.n_acoustic_out},{f}]);this.dispatch(e,t.eulerStep,[a.x_t,a.velocity,U],[_(r.n_acoustic_out,64)]),e.end()}e=s.beginComputePass({label:"fm_fsq"});let k=this.packUniform([{u:r.n_acoustic_out},{u:this.config.codec.acoustic_codebook_size},{u:2}]);this.dispatch(e,t.fsqQuantize,[a.x_t,a.acoustic_codes,k],[_(r.n_acoustic_out,64)]),e.end()}async codecDecode(s,o){let t=this.device,a=this.pipelines,n=this.modelBuffers,r=this.config.codec,i=s.length,e=r.dim,g=this.uploadArray(s),u=this.uploadArray(o),B=this.createGPUBuffer(i*r.semantic_dim*4,"codec_sem_embed"),k=this.createGPUBuffer(i*r.n_acoustic_codebook*4,"codec_ac_float"),c=this.createGPUBuffer(i*292*4,"codec_concat"),d=i,f=this.createGPUBuffer(d*e*4,"codec_cur"),b=this.createGPUBuffer(d*e*4,"codec_tmp"),F=t.createCommandEncoder({label:"codec_decode"}),C=[],m=(y,W,G,A)=>{let E=F.beginComputePass({label:A});this.dispatch(E,y,W,G),E.end()},h=this.packUniform([{u:i},{u:r.semantic_dim}]);m(a.vqLookup,[g,n.codec_semantic_codebook,B,h],[_(i*r.semantic_dim,128)],"codec_vq");let l=this.packUniform([{u:i},{u:r.n_acoustic_codebook},{u:r.acoustic_codebook_size},{u:2}]);m(a.fsqDequant,[u,k,l],[_(i*r.n_acoustic_codebook,64)],"codec_fsq");let v=this.packUniform([{u:i},{u:r.semantic_dim},{u:r.n_acoustic_codebook}]);m(a.concatCodecInput,[B,k,c,v],[_(i*292,256)],"codec_concat");let p=this.packUniform([{u:292},{u:e},{u:3},{u:d},{u:1}]);m(a.causalConv1d,[c,n.codec_input_conv_w,n.codec_input_conv_g,f,p],[_(d,64),e],"codec_input_conv");let U=[2,2,2,1],P=[4,4,4,3],w=[2,4,8,16];for(let y=0;y<r.decoder_stages;y++){let W=n.codec_stages[y];for(let G=0;G<r.decoder_layers_per_stage;G++){let A=W.transformer_layers[G],E=d*e*4;b.size<E&&(C.push(b),b=this.createGPUBuffer(E,"codec_tmp"));let I=b,D=d*e,ie=this.packUniform([{u:D}]);m(a.batchedCopy,[f,I,ie],[_(D,256)],`codec_s${y}_l${G}_copy_res`);let oe=this.packUniform([{u:e},{f:r.norm_eps},{u:d}]),N=this.createGPUBuffer(E,"codec_attn_normed");m(a.batchedRmsNorm,[f,A.attn_norm,N,oe],[d],`codec_s${y}_l${G}_attn_norm`);let X=this.createGPUBuffer(d*e*4,"codec_q"),K=this.createGPUBuffer(d*e*4,"codec_k"),Z=this.createGPUBuffer(d*e*4,"codec_v"),V=this.packUniform([{u:e},{u:e},{u:d}]);m(a.batchedMatvecF16,[A.wq,N,X,V],[e,d],`codec_s${y}_l${G}_qproj`),m(a.batchedMatvecF16,[A.wk,N,K,V],[e,d],`codec_s${y}_l${G}_kproj`),m(a.batchedMatvecF16,[A.wv,N,Z,V],[e,d],`codec_s${y}_l${G}_vproj`);let H=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:d},{f:r.qk_norm_eps}]);m(a.qkNorm,[X,A.q_norm,H],[_(d*r.n_heads,128)],`codec_s${y}_l${G}_qnorm`),m(a.qkNorm,[K,A.k_norm,H],[_(d*r.n_heads,128)],`codec_s${y}_l${G}_knorm`);let R=w[y],Y=this.createGPUBuffer(r.n_heads*d*R*4,"codec_scores"),ne=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:d},{u:R}]);m(a.alibiAttnScore,[X,K,Y,ne],[_(R,64),d,r.n_heads],`codec_s${y}_l${G}_attn_score`);let ee=this.packUniform([{u:r.n_heads},{u:d},{u:R}]);m(a.codecSoftmax,[Y,ee],[_(r.n_heads*d,64)],`codec_s${y}_l${G}_softmax`);let ae=this.createGPUBuffer(d*e*4,"codec_attn_out"),fe=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:d},{u:R}]);m(a.codecAttnValue,[Y,Z,ae,fe],[_(r.n_heads*r.head_dim,64),d],`codec_s${y}_l${G}_attn_val`);let Q=this.createGPUBuffer(E,"codec_wo_out");m(a.batchedMatvecF16,[A.wo,ae,Q,V],[e,d],`codec_s${y}_l${G}_wo`);let de=this.packUniform([{u:e},{u:D}]);m(a.batchedLayerScale,[Q,A.attn_scale,I,f,de],[_(D,256)],`codec_s${y}_l${G}_attn_res`),m(a.batchedCopy,[f,I,ie],[_(D,256)],`codec_s${y}_l${G}_copy_ffn_res`);let ce=this.createGPUBuffer(E,"codec_ffn_normed");m(a.batchedRmsNorm,[f,A.ffn_norm,ce,oe],[d],`codec_s${y}_l${G}_ffn_norm`);let te=d*r.hidden_dim,ue=this.createGPUBuffer(te*4,"codec_gate"),pe=this.createGPUBuffer(te*4,"codec_up"),be=this.packUniform([{u:r.hidden_dim},{u:e},{u:d}]);m(a.batchedMatvecF16,[A.w1,ce,ue,be],[r.hidden_dim,d],`codec_s${y}_l${G}_gate`),m(a.batchedMatvecF16,[A.w3,ce,pe,be],[r.hidden_dim,d],`codec_s${y}_l${G}_up`);let _e=Math.min(_(te,256),65535),ga=_(_(te,256),_e),ba=this.packUniform([{u:te},{u:_e}]);m(a.batchedSwiGLU,[ue,pe,ba],[_e,ga],`codec_s${y}_l${G}_swiglu`);let le=this.createGPUBuffer(E,"codec_down"),va=this.packUniform([{u:e},{u:r.hidden_dim},{u:d}]);m(a.batchedMatvecF16,[A.w2,ue,le,va],[e,d],`codec_s${y}_l${G}_down`),m(a.batchedLayerScale,[le,A.ffn_scale,I,f,de],[_(D,256)],`codec_s${y}_l${G}_ffn_res`),C.push(N,X,K,Z,Y,ae,Q,ce,ue,pe,le)}if(W.conv_w&&W.conv_scale&&U[y]>1){let G=d*U[y],A=this.createGPUBuffer(G*e*4,"codec_upsampled"),E=this.packUniform([{u:e},{u:e},{u:P[y]},{u:G},{u:U[y]}]);m(a.causalConvTranspose1d,[f,W.conv_w,W.conv_scale,A,E],[_(G,64),e],`codec_s${y}_conv_up`),C.push(f),f=A,d=G}}let q=d,S=this.createGPUBuffer(q*r.patch_size*4,"codec_output"),M=this.packUniform([{u:e},{u:r.patch_size},{u:7},{u:q},{u:1}]);m(a.causalConv1d,[f,n.codec_output_conv_w,n.codec_output_conv_g,S,M],[_(q,64),r.patch_size],"codec_output_conv"),t.pushErrorScope("validation"),t.queue.submit([F.finish()]),await t.queue.onSubmittedWorkDone();let $=await t.popErrorScope();$&&(globalThis.__codecError=$.message);let T=q*r.patch_size,z=await this.readF32Array(S,T),O=0;for(let y=0;y<Math.min(z.length,1e3);y++)z[y]!==0&&O++;globalThis.__codecDebug={outT:q,patchSize:r.patch_size,totalSamples:T,nonZero:O,first5:Array.from(z.slice(0,5)),curT:d};for(let y of C)y.destroy();return g.destroy(),u.destroy(),B.destroy(),k.destroy(),c.destroy(),f.destroy(),b.destroy(),S.destroy(),z}async debugCodecDecode(s,o){let t=this.device,a=this.pipelines,n=this.modelBuffers,r=this.config.codec,i=s.length,e=r.dim,g={},u=this.uploadArray(s),B=this.uploadArray(o),k=this.createGPUBuffer(i*r.semantic_dim*4,"codec_sem_embed"),c=this.createGPUBuffer(i*r.n_acoustic_codebook*4,"codec_ac_float"),d=this.createGPUBuffer(i*292*4,"codec_concat"),f=i,b=this.createGPUBuffer(f*e*4,"codec_cur"),F=this.createGPUBuffer(f*e*4,"codec_tmp"),C=[],m=(p,U,P,w,q)=>{let S=p.beginComputePass({label:q});this.dispatch(S,U,P,w),S.end()};{let p=t.createCommandEncoder({label:"codec_phase1"}),U=this.packUniform([{u:i},{u:r.semantic_dim}]);m(p,a.vqLookup,[u,n.codec_semantic_codebook,k,U],[_(i*r.semantic_dim,128)],"codec_vq");let P=this.packUniform([{u:i},{u:r.n_acoustic_codebook},{u:r.acoustic_codebook_size},{u:2}]);m(p,a.fsqDequant,[B,c,P],[_(i*r.n_acoustic_codebook,64)],"codec_fsq");let w=this.packUniform([{u:i},{u:r.semantic_dim},{u:r.n_acoustic_codebook}]);m(p,a.concatCodecInput,[k,c,d,w],[_(i*292,256)],"codec_concat");let q=this.packUniform([{u:292},{u:e},{u:3},{u:f},{u:1}]);m(p,a.causalConv1d,[d,n.codec_input_conv_w,n.codec_input_conv_g,b,q],[_(f,64),e],"codec_input_conv"),t.queue.submit([p.finish()]),await t.queue.onSubmittedWorkDone()}g.vq_embed=await this.readF32Array(k,i*r.semantic_dim),g.fsq_dequant=await this.readF32Array(c,i*r.n_acoustic_codebook),g.concat=await this.readF32Array(d,i*292),g.after_input_conv=await this.readF32Array(b,f*e);let h=[2,2,2,1],l=[4,4,4,3],v=[2,4,8,16];for(let p=0;p<r.decoder_stages;p++){let U=n.codec_stages[p],P=t.createCommandEncoder({label:`codec_stage${p}`});for(let w=0;w<r.decoder_layers_per_stage;w++){let q=U.transformer_layers[w],S=f*e*4;F.size<S&&(C.push(F),F=this.createGPUBuffer(S,"codec_tmp"));let M=F,$=f*e,T=this.packUniform([{u:$}]);m(P,a.batchedCopy,[b,M,T],[_($,256)],`codec_s${p}_l${w}_copy_res`);let z=this.packUniform([{u:e},{f:r.norm_eps},{u:f}]),O=this.createGPUBuffer(S,"codec_attn_normed");m(P,a.batchedRmsNorm,[b,q.attn_norm,O,z],[f],`codec_s${p}_l${w}_attn_norm`);let y=this.createGPUBuffer(f*e*4,"codec_q"),W=this.createGPUBuffer(f*e*4,"codec_k"),G=this.createGPUBuffer(f*e*4,"codec_v"),A=this.packUniform([{u:e},{u:e},{u:f}]);m(P,a.batchedMatvecF16,[q.wq,O,y,A],[e,f],`codec_s${p}_l${w}_qproj`),m(P,a.batchedMatvecF16,[q.wk,O,W,A],[e,f],`codec_s${p}_l${w}_kproj`),m(P,a.batchedMatvecF16,[q.wv,O,G,A],[e,f],`codec_s${p}_l${w}_vproj`);let E=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:f},{f:r.qk_norm_eps}]);m(P,a.qkNorm,[y,q.q_norm,E],[_(f*r.n_heads,128)],`codec_s${p}_l${w}_qnorm`),m(P,a.qkNorm,[W,q.k_norm,E],[_(f*r.n_heads,128)],`codec_s${p}_l${w}_knorm`);let I=v[p],D=this.createGPUBuffer(r.n_heads*f*I*4,"codec_scores"),ie=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:f},{u:I}]);m(P,a.alibiAttnScore,[y,W,D,ie],[_(I,64),f,r.n_heads],`codec_s${p}_l${w}_attn_score`);let oe=this.packUniform([{u:r.n_heads},{u:f},{u:I}]);m(P,a.codecSoftmax,[D,oe],[_(r.n_heads*f,64)],`codec_s${p}_l${w}_softmax`);let N=this.createGPUBuffer(f*e*4,"codec_attn_out"),X=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:f},{u:I}]);m(P,a.codecAttnValue,[D,G,N,X],[_(r.n_heads*r.head_dim,64),f],`codec_s${p}_l${w}_attn_val`);let K=this.createGPUBuffer(S,"codec_wo_out");m(P,a.batchedMatvecF16,[q.wo,N,K,A],[e,f],`codec_s${p}_l${w}_wo`);let Z=this.packUniform([{u:e},{u:$}]);m(P,a.batchedLayerScale,[K,q.attn_scale,M,b,Z],[_($,256)],`codec_s${p}_l${w}_attn_res`),m(P,a.batchedCopy,[b,M,T],[_($,256)],`codec_s${p}_l${w}_copy_ffn_res`);let V=this.createGPUBuffer(S,"codec_ffn_normed");m(P,a.batchedRmsNorm,[b,q.ffn_norm,V,z],[f],`codec_s${p}_l${w}_ffn_norm`);let H=f*r.hidden_dim,R=this.createGPUBuffer(H*4,"codec_gate"),Y=this.createGPUBuffer(H*4,"codec_up"),ne=this.packUniform([{u:r.hidden_dim},{u:e},{u:f}]);m(P,a.batchedMatvecF16,[q.w1,V,R,ne],[r.hidden_dim,f],`codec_s${p}_l${w}_gate`),m(P,a.batchedMatvecF16,[q.w3,V,Y,ne],[r.hidden_dim,f],`codec_s${p}_l${w}_up`);let ee=Math.min(_(H,256),65535),ae=_(_(H,256),ee),fe=this.packUniform([{u:H},{u:ee}]);m(P,a.batchedSwiGLU,[R,Y,fe],[ee,ae],`codec_s${p}_l${w}_swiglu`);let Q=this.createGPUBuffer(S,"codec_down"),de=this.packUniform([{u:e},{u:r.hidden_dim},{u:f}]);m(P,a.batchedMatvecF16,[q.w2,R,Q,de],[e,f],`codec_s${p}_l${w}_down`),m(P,a.batchedLayerScale,[Q,q.ffn_scale,M,b,Z],[_($,256)],`codec_s${p}_l${w}_ffn_res`),C.push(O,y,W,G,D,N,K,V,R,Y,Q)}if(U.conv_w&&U.conv_scale&&h[p]>1){let w=f*h[p],q=this.createGPUBuffer(w*e*4,"codec_upsampled"),S=this.packUniform([{u:e},{u:e},{u:l[p]},{u:w},{u:h[p]}]);m(P,a.causalConvTranspose1d,[b,U.conv_w,U.conv_scale,q,S],[_(w,64),e],`codec_s${p}_conv_up`),t.queue.submit([P.finish()]),await t.queue.onSubmittedWorkDone(),g[`after_stage${p}_transformer`]=await this.readF32Array(b,f*e),C.push(b),b=q,f=w,g[`after_stage${p}_conv_up`]=await this.readF32Array(b,f*e)}else t.queue.submit([P.finish()]),await t.queue.onSubmittedWorkDone(),g[`after_stage${p}_transformer`]=await this.readF32Array(b,f*e)}{let p=f,U=this.createGPUBuffer(p*r.patch_size*4,"codec_output"),P=this.packUniform([{u:e},{u:r.patch_size},{u:7},{u:p},{u:1}]),w=t.createCommandEncoder({label:"codec_output"});m(w,a.causalConv1d,[b,n.codec_output_conv_w,n.codec_output_conv_g,U,P],[_(p,64),r.patch_size],"codec_output_conv"),t.queue.submit([w.finish()]),await t.queue.onSubmittedWorkDone(),g.after_output_conv=await this.readF32Array(U,p*r.patch_size),g.audio=g.after_output_conv,C.push(U)}for(let p of C)p.destroy();return u.destroy(),B.destroy(),k.destroy(),c.destroy(),d.destroy(),b.destroy(),F.destroy(),g}uploadArray(s){let o=this.device.createBuffer({size:s.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!0});return s instanceof Uint32Array?new Uint32Array(o.getMappedRange()).set(s):new Float32Array(o.getMappedRange()).set(s),o.unmap(),o}createGPUBuffer(s,o){return this.device.createBuffer({size:Math.max(s,4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,label:o})}async readBuffer(s,o){let t=this.device,a=t.createBuffer({size:o,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),n=t.createCommandEncoder();n.copyBufferToBuffer(s,0,a,0,o),t.queue.submit([n.finish()]),await a.mapAsync(GPUMapMode.READ);let r=a.getMappedRange().slice(0);return a.unmap(),a.destroy(),r}async readU32(s){let o=await this.readBuffer(s,4);return new Uint32Array(o)[0]}async readF32Array(s,o){let t=await this.readBuffer(s,o*4);return new Float32Array(t)}async readU32Array(s,o){let t=await this.readBuffer(s,o*4);return new Uint32Array(t)}get isReady(){return this.device!==null&&this.modelBuffers!==null&&this.pipelines!==null}async debugRead(s,o=16){let t=this.workBuffers,a=t[s];if(!a)throw new Error(`Unknown buffer: ${s}. Available: ${Object.keys(t).join(", ")}`);return this.readF32Array(a,o)}async debugBackboneStep(s){let o=this.device.createCommandEncoder();this.backboneStep(o,s),this.device.queue.submit([o.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++;let t=this.workBuffers,a=await this.readF32Array(t.hidden,16),n=await this.readF32Array(t.normed,16),r=await this.readF32Array(t.logits,16),i=await this.readU32(t.argmax_result),e=await this.readF32Array(t.logits,1024),g=-1/0;for(let u=0;u<e.length;u++)e[u]>g&&(g=e[u]);return{hidden:a,normed:n,logits_first16:r,logits_max:g,argmax:i}}async debugBackboneLayerByLayer(s){let o=this.pipelines,t=this.workBuffers,a=this.modelBuffers,n=this.config.backbone,r=this.position,i=n.dim,e;{let c=this.device.createCommandEncoder();e=c.beginComputePass({label:"debug_embed"});let d=this.packUniform([{u:s},{u:i}]);this.dispatch(e,o.embeddingLookup,[a.tok_embeddings,t.hidden,d],[_(i,256)]),e.end(),this.device.queue.submit([c.finish()]),await this.device.queue.onSubmittedWorkDone()}let g=await this.readF32Array(t.hidden,i),u=[];for(let c=0;c<n.n_layers;c++){let d=a.backbone_layers[c],f=this.kvCaches[c];{let h=this.device.createCommandEncoder();e=h.beginComputePass({label:`debug_l${c}_attn_prep`});let l=this.packUniform([{u:i}]);this.dispatch(e,o.copyBuffer,[t.hidden,t.residual,l],[_(i,256)]);let v=this.packUniform([{u:i},{f:n.norm_eps}]);this.dispatch(e,o.rmsNorm,[t.hidden,d.attn_norm,t.normed,v],[1]),e.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let b=await this.readF32Array(t.normed,i);{let h=this.device.createCommandEncoder();e=h.beginComputePass({label:`debug_l${c}_qkv`});let l=this.packUniform([{u:n.n_heads*n.head_dim},{u:i}]);this.dispatch(e,o.matvecF16,[d.wq,t.normed,t.q,l],[n.n_heads*n.head_dim]);let v=this.packUniform([{u:n.n_kv_heads*n.head_dim},{u:i}]);this.dispatch(e,o.matvecF16,[d.wk,t.normed,t.k,v],[n.n_kv_heads*n.head_dim]),this.dispatch(e,o.matvecF16,[d.wv,t.normed,t.v,v],[n.n_kv_heads*n.head_dim]),e.end(),e=h.beginComputePass({label:`debug_l${c}_rope_attn`});let p=this.packUniform([{u:n.head_dim},{u:r},{u:n.n_heads},{f:n.rope_theta}]);this.dispatch(e,o.rope,[t.q,p],[_(n.n_heads*n.head_dim/2,64)]);let U=this.packUniform([{u:n.head_dim},{u:r},{u:n.n_kv_heads},{f:n.rope_theta}]);this.dispatch(e,o.rope,[t.k,U],[_(n.n_kv_heads*n.head_dim/2,64)]);let P=this.packUniform([{u:r},{u:n.n_kv_heads*n.head_dim}]);this.dispatch(e,o.kvCacheWrite,[t.k,t.v,f.k,f.v,P],[_(n.n_kv_heads*n.head_dim,256)]);let w=r+1,q=n.n_heads/n.n_kv_heads,S=this.packUniform([{u:n.n_heads},{u:n.n_kv_heads},{u:n.head_dim},{u:w},{u:q}]);this.dispatch(e,o.attnScore,[t.q,f.k,t.scores,S],[_(n.n_heads*w,64)]),e.end(),e=h.beginComputePass({label:`debug_l${c}_attn_out`});let M=this.packUniform([{u:n.n_heads},{u:w}]);this.dispatch(e,o.softmax,[t.scores,M],[n.n_heads]);let $=this.packUniform([{u:n.n_heads},{u:n.n_kv_heads},{u:n.head_dim},{u:w},{u:q}]);this.dispatch(e,o.attnValue,[t.scores,f.v,t.attn_out,$],[_(n.n_heads*n.head_dim,128)]),e.end(),e=h.beginComputePass({label:`debug_l${c}_wo`});let T=this.packUniform([{u:i},{u:n.n_heads*n.head_dim}]);this.dispatch(e,o.matvecF16,[d.wo,t.attn_out,t.hidden,T],[i]),e.end(),e=h.beginComputePass({label:`debug_l${c}_res1`});let z=this.packUniform([{u:i}]);this.dispatch(e,o.addInPlace,[t.hidden,t.residual,z],[_(i,256)]),e.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let F=await this.readF32Array(t.hidden,i);{let h=this.device.createCommandEncoder();e=h.beginComputePass({label:`debug_l${c}_ffn_prep`});let l=this.packUniform([{u:i}]);this.dispatch(e,o.copyBuffer,[t.hidden,t.residual,l],[_(i,256)]);let v=this.packUniform([{u:i},{f:n.norm_eps}]);this.dispatch(e,o.rmsNorm,[t.hidden,d.ffn_norm,t.normed,v],[1]),e.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let C=await this.readF32Array(t.normed,i);{let h=this.device.createCommandEncoder();e=h.beginComputePass({label:`debug_l${c}_ffn`});let l=this.packUniform([{u:n.hidden_dim},{u:i}]);this.dispatch(e,o.matvecF16,[d.w1,t.normed,t.gate,l],[n.hidden_dim]),this.dispatch(e,o.matvecF16,[d.w3,t.normed,t.up,l],[n.hidden_dim]),e.end(),e=h.beginComputePass({label:`debug_l${c}_ffn_out`});let v=this.packUniform([{u:n.hidden_dim}]);this.dispatch(e,o.swiGLU,[t.gate,t.up,v],[_(n.hidden_dim,256)]);let p=this.packUniform([{u:i},{u:n.hidden_dim}]);this.dispatch(e,o.matvecF16,[d.w2,t.gate,t.hidden,p],[i]),e.end(),e=h.beginComputePass({label:`debug_l${c}_res2`});let U=this.packUniform([{u:i}]);this.dispatch(e,o.addInPlace,[t.hidden,t.residual,U],[_(i,256)]),e.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let m=await this.readF32Array(t.hidden,i);u.push({attn_norm:b,attn_out:F,ffn_norm:C,ffn_out:m})}{let c=this.device.createCommandEncoder();e=c.beginComputePass({label:"debug_final_norm"});let d=this.packUniform([{u:i},{f:n.norm_eps}]);this.dispatch(e,o.rmsNorm,[t.hidden,a.final_norm,t.normed,d],[1]),e.end(),this.device.queue.submit([c.finish()]),await this.device.queue.onSubmittedWorkDone()}let B=await this.readF32Array(t.normed,i),k=await this.readF32Array(t.hidden,i);return this.position++,{embed:g,layers:u,final_norm:B,hidden:k}}async debugFMForward(s=42){let o=this.config.fm,t=this.device.createCommandEncoder();this.fmForward(t),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let a=this.workBuffers,n=await this.readF32Array(a.semantic_logits,o.semantic_vocab),r=await this.readU32Array(a.acoustic_codes,o.n_acoustic_out),i=await this.readF32Array(a.x_t,o.n_acoustic_out);return{semantic_logits:n,velocities:[],acoustic_codes:r,x_final:i}}reset(){this.position=0}async backboneStepAndRead(s,o=!1){let t=this.device.createCommandEncoder();this.backboneStep(t,s,o),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let a=await this.readU32(this.workBuffers.argmax_result);return this.position++,a}async debugBackboneStepFull(s,o=!1){let t=this.device.createCommandEncoder();this.backboneStep(t,s,o),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let a=await this.readF32Array(this.workBuffers.normed,this.config.backbone.dim);return this.position++,a}async debugFMStep(s){let o=this.config.fm,t=this.device.createCommandEncoder();this.fmForward(t,s),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let a=this.workBuffers;return{semantic_logits:await this.readF32Array(a.semantic_logits,o.semantic_vocab),acoustic_codes:await this.readU32Array(a.acoustic_codes,o.n_acoustic_out),x_final:await this.readF32Array(a.x_t,o.n_acoustic_out)}}async fmStepAndRead(){let s=this.device.createCommandEncoder();return this.fmForward(s),this.device.queue.submit([s.finish()]),await this.device.queue.onSubmittedWorkDone(),this.readU32Array(this.workBuffers.acoustic_codes,this.config.fm.n_acoustic_out)}async generate(s,o,t,a,n=500,r,i){if(!this.isReady)throw new Error("Engine not initialized. Call init() and loadWeights() first.");this.reset();let e=performance.now();i?.("backbone");let g=[];if(a&&t>0){let l=this.config.backbone.dim;for(let v=0;v<t;v++){let p=this.device.createBuffer({size:l*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!0});new Float32Array(p.getMappedRange()).set(a.subarray(v*l,(v+1)*l)),p.unmap(),g.push(p)}}for(let l=0;l<s.length;l++){let v=s[l],p=this.device.createCommandEncoder();if(l>=o&&l<o+t&&g.length>0){let U=l-o;this.backboneStep(p,v,!1,g[U])}else this.backboneStep(p,v);this.device.queue.submit([p.finish()]),this.position++}await this.device.queue.onSubmittedWorkDone();let u=performance.now();i?.("backbone",u-e),i?.("fm");{let l=this.device.createCommandEncoder();this.backboneStep(l,24,!1),this.device.queue.submit([l.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let B=[],k=[],c=this.config.backbone,d=this.config.fm,f=this.pipelines,b=this.workBuffers,F=this.modelBuffers;for(let l=0;l<n;l++){if(l>0){let S=this.device.createCommandEncoder(),M=S.beginComputePass({label:`multiCBEmbed_frame${l}`}),$=this.packUniform([{u:c.dim},{u:8194},{u:23},{u:36}]);this.dispatch(M,f.multiCodebookEmbed,[F.audio_embeddings,b.semantic_argmax,b.acoustic_codes,b.hidden,$],[_(c.dim,256)]),M.end();let T=S.beginComputePass({label:`mcb_copy_frame${l}`}),z=this.packUniform([{u:c.dim}]);this.dispatch(T,f.copyBuffer,[b.hidden,b.fm_gate,z],[_(c.dim,256)]),T.end(),this.backboneStep(S,0,!1,b.fm_gate),this.device.queue.submit([S.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let v=this.device.createCommandEncoder();this.fmForward(v),this.device.queue.submit([v.finish()]),await this.device.queue.onSubmittedWorkDone();let p=await this.readF32Array(b.semantic_logits,d.semantic_vocab);p[0]=-1/0;let U=8194;for(let S=U;S<p.length;S++)p[S]=-1/0;let P=Aa(p,.9,.8);if(P<=1)break;B.push(P);let w=new Uint32Array([P]);this.device.queue.writeBuffer(b.semantic_argmax,0,w);let q=await this.readU32Array(b.acoustic_codes,d.n_acoustic_out);k.push(Array.from(q)),r?.(l,P,q)}let C=performance.now();i?.("fm",C-u),i?.("codec");let m;if(B.length>0){let l=new Uint32Array(B),v=new Uint32Array(k.flat());m=await this.codecDecode(l,v)}else m=new Float32Array(0);let h=performance.now();return i?.("codec",h-C),{semanticCodes:B,acousticCodes:k,audio:m,stats:{backboneMs:u-e,fmMs:C-u,codecMs:h-C,totalMs:h-e,framesGenerated:B.length}}}destroy(){if(this.workBuffers)for(let s of Object.values(this.workBuffers))s.destroy();for(let s of this.kvCaches)s.k.destroy(),s.v.destroy();this.device?.destroy()}};var j={UNK:0,BOS:1,EOS:2,INST:3,INST_END:4,AUDIO:24,BEGIN_AUDIO:25,OUTPUT_AUDIO:26,AUDIO_TO_TEXT:35,TEXT_TO_AUDIO:36,PAD:11},se=class x{bytesToRank=new Map;specialTokens=new Map;pattern;voiceNumTokens=new Map;numSpecialTokens;constructor(s){this.numSpecialTokens=s.config.default_num_special_tokens;for(let o of s.vocab){let t=atob(o.token_bytes);this.bytesToRank.set(t,o.rank)}for(let o of s.special_tokens)this.specialTokens.set(o.token_str,o.rank);try{this.pattern=new RegExp(s.config.pattern,"gu")}catch{this.pattern=/\S+|\s+/gu}if(s.audio?.voice_num_audio_tokens)for(let[o,t]of Object.entries(s.audio.voice_num_audio_tokens))this.voiceNumTokens.set(o,t)}static async load(s){let t=await(await fetch(s)).json();return new x(t)}getVoiceNumTokens(s){let o=this.voiceNumTokens.get(s);if(o===void 0)throw new Error(`Unknown voice: ${s}. Available: ${[...this.voiceNumTokens.keys()].join(", ")}`);return o}buildTTSPrompt(s,o){let t=this.getVoiceNumTokens(o),a=[];a.push(j.BOS),a.push(j.BEGIN_AUDIO);let n=a.length;for(let i=0;i<t;i++)a.push(j.AUDIO);a.push(j.TEXT_TO_AUDIO);let r=this.encode(s);return a.push(...r),a.push(j.AUDIO_TO_TEXT),a.push(j.BEGIN_AUDIO),{tokens:a,audioTokenStart:n,audioTokenCount:t}}encode(s){let o=[],t=s.matchAll(this.pattern);for(let a of t){let n=a[0],r=this.bpeEncode(n);o.push(...r)}return o}bpeEncode(s){let t=new TextEncoder().encode(s),a=[];for(let n of t)a.push(String.fromCharCode(n));if(a.length<=1){let n=this.bytesToRank.get(a[0]);return[n!==void 0?n+this.numSpecialTokens:j.UNK]}for(;a.length>1;){let n=1/0,r=-1;for(let e=0;e<a.length-1;e++){let g=a[e]+a[e+1],u=this.bytesToRank.get(g);u!==void 0&&u<n&&(n=u,r=e)}if(r===-1)break;let i=[];for(let e=0;e<a.length;e++)e===r?(i.push(a[e]+a[e+1]),e++):i.push(a[e]);a=i}return a.map(n=>{let r=this.bytesToRank.get(n);return r!==void 0?r+this.numSpecialTokens:j.UNK})}get voices(){return[...this.voiceNumTokens.keys()]}};var $a="https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main",ge=class x{engine;tokenizer;modelsUrl;voiceCache=new Map;constructor(s,o,t){this.engine=s,this.tokenizer=o,this.modelsUrl=t}static async load(s={}){let o=s.modelsUrl??$a,t=new re({maxSeqLen:s.maxSeqLen});await t.init();let[,a]=await Promise.all([t.loadWeightsFromHF(s.weightsUrl??J,s.onProgress),se.load(`${o}/tekken.json`)]);return new x(t,a,o)}get voices(){return this.tokenizer.voices}async speak(s,o="casual_female",t={}){let{tokens:a,audioTokenStart:n,audioTokenCount:r}=this.tokenizer.buildTTSPrompt(s,o),i=this.voiceCache.get(o);if(!i){let e=await fetch(`${this.modelsUrl}/voice_embedding_f32/${o}.bin`);if(!e.ok)throw new Error(`Failed to load voice "${o}": ${e.status} ${e.statusText}`);i=new Float32Array(await e.arrayBuffer()),this.voiceCache.set(o,i)}return this.engine.generate(a,n,r,i,t.maxFrames??500,t.onFrame,t.onStage)}destroy(){this.engine.destroy(),this.voiceCache.clear()}};export{J as HF_VOXTRAL_URL,j as TOKENS,se as TekkenTokenizer,ge as Voxtral,re as VoxtralEngine,Ca as clearWeightCache,Ga as getWeightCacheInfo};
