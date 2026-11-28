"use strict";
// Only connect to an already-running shared server. Never resume a second copy or guess a thread.
const {spawn} = require("node:child_process");
const {createInterface} = require("node:readline");
const {createHash} = require("node:crypto");
const fs = require("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8").replace(/^\uFEFF/,""));
let transport, end, seq = 0, submitted = false;
const pending = new Map();
function message(line) {
  let m;
  try { m=JSON.parse(line); } catch { return; }
  if (!Object.hasOwn(m,"id") || !pending.has(m.id)) return;
  const {resolve,reject} = pending.get(m.id);
  pending.delete(m.id);
  if (m.error) reject(Error("protocol")); else resolve(m.result);
}
function closed() {
  for (const {reject} of pending.values()) reject(Error("transport"));
  pending.clear();
}
async function connect() {
  const url=process.env.CLASSPAGE_CODEX_WS;
  if (url) {
    const u=new URL(url);
    if (u.protocol!=="ws:" || !["127.0.0.1","[::1]"].includes(u.hostname) || u.username || u.password || u.search || u.hash)
      throw Error("仅支持明确配置的本机共享 WebSocket 服务");
    const ws = new WebSocket(u.href);
    end=()=>ws.close();
    ws.addEventListener("message",e=>message(String(e.data)));
    ws.addEventListener("close",closed);
    await new Promise((ok,bad)=>{ws.addEventListener("open",ok,{once:true});ws.addEventListener("error",bad,{once:true})});
    transport=m=>ws.send(JSON.stringify(m));
  } else {
    const args=["app-server","proxy"];
    if(process.env.CLASSPAGE_CODEX_SOCKET)args.push("--sock",process.env.CLASSPAGE_CODEX_SOCKET);
    const p=spawn(process.platform==="win32"?"codex.exe":"codex",args,{windowsHide:true,stdio:["pipe","pipe","ignore"]});
    end=()=>{p.stdin.end();p.kill()};
    p.on("error",closed);p.on("exit",closed);p.stdin.on("error",closed);
    createInterface({input:p.stdout}).on("line",message);
    transport=m=>p.stdin.write(JSON.stringify(m)+"\n");
  }
}
function rpc(method,params) {
  return new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});transport({id,method,params})});
}
function hasPairingReceipt(turns) {
  for(const turn of turns) for(const item of turn.items || []) {
    if(item.type!=="userMessage")continue;
    for(const block of item.content || []) {
      if(block.type!=="text" || !block.text.includes("studio_session_pair"))continue;
      const codes=block.text.toUpperCase().match(/\bCP-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}\b/g)||[];
      if(codes.some(c=>createHash("sha256").update(c).digest("hex")===input.proof))return true;
    }
  }
  return false;
}
async function run() {
  if(!/^[0-9a-f-]{36}$/i.test(input.thread))return "no_thread";
  await connect();
  await rpc("initialize",{clientInfo:{name:"classpage-studio",version:"1.0"},capabilities:{experimentalApi:true}});
  transport({method:"initialized"});
  const r=await rpc("thread/read",{threadId:input.thread,includeTurns:false});
  if(r.thread.id!==input.thread || !["active","idle"].includes(r.thread.status.type))return "not_loaded";
  if(input.action==="verify") {
    const page=await rpc("thread/turns/list",{threadId:input.thread,limit:12,sortDirection:"desc",itemsView:"full"});
    return hasPairingReceipt(page.data)?"verified":"no_proof";
  }
  if(input.action!=="notify")throw Error("action");
  if(r.thread.status.type!=="idle")return "busy";
  submitted=true;
  await rpc("turn/start",{threadId:input.thread,input:[{type:"text",text:"课页 Studio：你绑定的对话框有一条新留言。请使用本聊天已有的凭据调用 studio_session_receive、确认并处理；如反馈已完成，无需重复编辑。",text_elements:[]}]});
  return "sent";
}
let done=false;
function finish(status) {
  if(done)return;
  done=true;
  process.stdout.write(JSON.stringify({status})+"\n");
  try {end?.()}catch{}
  setTimeout(()=>process.exit(0),20);
}
const timer=setTimeout(()=>finish(submitted?"uncertain":"unavailable"),6500);
run().then(finish,()=>finish(submitted?"uncertain":"unavailable")).finally(()=>clearTimeout(timer));
