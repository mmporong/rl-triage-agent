"use strict";
const $ = (q) => document.querySelector(q);
const steps = Array.from(document.querySelectorAll("[data-step]"));
const labels = {"Train/mean_reward":"Training reward","Train/mean_episode_length":"Episode length (logged)","Metrics/base_velocity/error_vel_xy":"Linear tracking error","Metrics/base_velocity/error_vel_yaw":"Turning error"};
const checks = {recorded_http_200:"Recorded API request returned HTTP 200",exact_model_identity:"Exact Token Factory model ID matches",citations_in_supplied_facts:"Every citation exists in the supplied evidence",allowed_action_and_target:"Action and inspection target are allowed",original_core_accepts_decision:"Original investigation functions accepted this reply",no_execution_authority:"No approval or GPU execution authority is granted"};
let evidence, active=0, stage=0, timer=null;
const escapeHTML = (v) => String(v).replace(/[&<>"']/g,(c)=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const showJSON = (x) => escapeHTML(JSON.stringify(x,null,2));
const canonical = (x) => JSON.stringify(normalize(x));
function normalize(x){if(Array.isArray(x))return x.map(normalize);if(x&&typeof x==="object")return Object.fromEntries(Object.keys(x).sort().map((k)=>[k,normalize(x[k])]));return x;}
const digest = async (raw) => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",raw))).map((x)=>x.toString(16).padStart(2,"0")).join("");
function requireValue(ok,message){if(!ok)throw new Error(message);}
function caption(id){const p=id.split(":");return (p[1]==="case"?"Candidate ":"Reference ")+(labels[p.slice(2).join(":")]||p.slice(2).join(":"));}
function factTable(row){const metrics=Array.from(new Set(Object.keys(row.facts).map((k)=>k.split(":").slice(2).join(":"))));return '<table><thead><tr><th scope="col">Supplied measurement</th><th scope="col">Candidate</th><th scope="col">Reference</th></tr></thead><tbody>'+metrics.map((m)=>'<tr><td>'+escapeHTML(labels[m]||m)+'</td>'+["case","reference"].map((role)=>{const key="telemetry:"+role+":"+m;const value=row.facts[key];return '<td class="'+(row.decision.fact_ids.includes(key)?"cited":"")+'">'+(typeof value==="number"?new Intl.NumberFormat("en-US",{maximumSignificantDigits:5}).format(value):"Not supplied")+'</td>';}).join("")+'</tr>').join("")+'</tbody></table>';}
function stop(){if(timer)clearInterval(timer);timer=null;$("#play").textContent="Play walkthrough";$("#play").setAttribute("aria-pressed","false");}
function render(){const row=evidence.records[active];steps.forEach((b,i)=>{if(i===stage)b.setAttribute("aria-current","step");else b.removeAttribute("aria-current");});let html="";
if(stage===0){html='<p class="eyebrow">1 · Observe the training record</p><h2>'+(row.control?"A matching-reference control":"A candidate that tracks poorly")+'</h2><p>'+(row.data_policy==="eager"?"This recorded request supplied all eight training facts.":"This recorded request supplied two reward facts and offered a tool to read the remaining training facts.")+' Highlighted values were cited by the model.</p>'+factTable(row);}
if(stage===1){html='<p class="eyebrow">2 · Read the actual model decision</p><h2>'+(row.decision.action==="hold"?"The model returned HOLD":"Inspect "+(row.decision.target==="P_reward"?"the reward mechanism":"episode termination"))+'</h2><span class="badge '+(row.control?"hold":"pending")+'">'+(row.control?"No inspection requested":"Inspection proposal")+'</span><p>This is the stored Token Factory response, with its original cited facts.</p><ul class="fact-list">'+row.decision.fact_ids.map((id)=>'<li>'+escapeHTML(caption(id))+'</li>').join("")+'</ul><details><summary>Read the exact structured reply</summary><pre>'+showJSON(row.decision)+'</pre></details>';}
if(stage===2){html='<p class="eyebrow">3 · Validate before taking action</p><h2>Evidence and decision checks passed</h2><p>The CPU preparation ran the original investigation functions against the recorded request and reply.</p><ul class="checks">'+Object.keys(checks).map((key)=>'<li>'+checks[key]+'</li>').join("")+'</ul>';}
if(stage===3){html='<p class="eyebrow">4 · Review the next action</p><h2>'+(row.review_request?"Ready for an approval review":"HOLD: no inspection to dispatch")+'</h2><span class="badge '+(row.review_request?"pending":"hold")+'">'+(row.review_request?"Execution is not authorized":"No GPU action")+'</span><p>'+(row.review_request?"The proposed inspection is "+escapeHTML(row.decision.target)+". Running it requires a fresh execution registration, approval and resource lease.":"The model returned HOLD for this identical-reference control. This replay has no inspection request to export.")+'</p><button id="download" class="primary" '+(row.review_request?"":"disabled")+'>Save inspection review JSON</button><span id="saved-status" role="status"></span>'+(row.review_request?'<details id="review-detail"><summary>Inspect the review request</summary><pre id="review-json">'+showJSON(row.review_request)+'</pre></details>':"")+'<details><summary>Inspect request and response provenance</summary><p>Model: <code>'+escapeHTML(row.model)+'</code></p><p>Request SHA-256: <code>'+row.request_sha256+'</code></p><p>Response SHA-256: <code>'+row.response_sha256+'</code></p><a href="'+escapeHTML(row.request_file)+'">Recorded request JSON</a> · <a href="'+escapeHTML(row.response_file)+'">Recorded response JSON</a></details>';}
$("#panel").innerHTML=html;
if($("#download"))$("#download").addEventListener("click",()=>{const bytes=JSON.stringify(row.review_request,null,2)+"\n";const url=URL.createObjectURL(new Blob([bytes],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="inspection-review-"+row.id+".json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$("#saved-status").textContent=" Review JSON prepared; no inspection was executed.";});
window.judgeReplay={verified:true,record:row.id,stage,playing:timer!==null,action:row.decision.action,execution_allowed:false};}
function setStage(next){stage=next;render();}
async function initialize(){
 const manifestResponse=await fetch("WEB_INDEX.json",{cache:"no-store"});
 requireValue(manifestResponse.ok,"Run prepare_demo.py first, then serve this bundle on localhost.");
 const manifest=await manifestResponse.json();
 requireValue(manifest.schema==="tokenfactory_judge_web_manifest_v1","Unexpected evidence manifest");
 const files={};
 for(const [name,expected] of Object.entries(manifest.files)){
  requireValue(/^[a-zA-Z0-9_./-]+$/.test(name)&&!name.includes("..")&&!name.startsWith("/"),"Unsafe evidence path");
  const response=await fetch(name,{cache:"no-store"});
  requireValue(response.ok,"Evidence file is unavailable: "+name);
  const raw=await response.arrayBuffer();
  requireValue(await digest(raw)===expected,"Evidence file hash differs: "+name);
  files[name]=name.endsWith(".json")?JSON.parse(new TextDecoder().decode(raw)):new TextDecoder().decode(raw);
 }
 evidence=files["WEB_EVIDENCE.json"];
 const index=files["PUBLIC_INDEX.json"],training=files["evidence/training_source.json"];
 const model="nvidia/Nemotron-3.5-Super-VL";
 requireValue(evidence.schema==="tokenfactory_judge_view_v1"&&evidence.records.length===8
  &&evidence.new_HTTP_or_GPU_calls===0&&evidence.approval_granted===false&&evidence.execution_allowed===false,"Unexpected replay scope");
 requireValue(index.core_model===model&&index.approval_granted===false&&index.execution_allowed===false,"Recorded index authority or model differs");
 requireValue(Array.isArray(index.core_records)&&index.core_records.length===8,"Recorded index count differs");
 const originalIDs=index.core_records.map((r)=>r.episode_id),viewIDs=evidence.records.map((r)=>r.id);
 requireValue(new Set(originalIDs).size===8&&new Set(viewIDs).size===8
  &&canonical(originalIDs.slice().sort())===canonical(viewIDs.slice().sort()),"Replay episode IDs differ or repeat");
 let holds=0,proposals=0;
 for(const row of evidence.records){
  const r=index.core_records.find((x)=>x.episode_id===row.id);
  requireValue(r.approval_granted===false&&r.execution_allowed===false,"Recorded episode authority differs");
  requireValue(row.request_file===r.request&&row.response_file===r.response,"Provenance link differs");
  requireValue(row.control===(r.training_arms.case==="NONE")&&row.data_policy===r.data_policy
   &&row.trial===r.trial+1,"Displayed record metadata differs");
  const request=files[r.request],response=files[r.response],meta=files[r.metadata];
  const decision=JSON.parse(response.choices[0].message.content);
  let text=request.messages[1].content;if(Array.isArray(text))text=text[0].text;
  const view=JSON.parse(text);
  requireValue(canonical(decision)===canonical(row.decision)&&canonical(decision)===canonical(r.decision),"Decision differs from recorded response");
  requireValue(canonical(view.evidence_facts)===canonical(row.facts),"Supplied facts differ from request");
  requireValue(meta.http_status===200&&request.model===model&&response.model===model&&row.model===model
   &&row.response_id===response.id&&canonical(row.usage)===canonical(response.usage),"Provider evidence differs");
  requireValue(row.request_sha256===manifest.files[r.request]&&row.response_sha256===manifest.files[r.response],"Provenance hashes differ");
  requireValue(Object.keys(row.checks).length===Object.keys(checks).length&&Object.keys(checks).every((k)=>row.checks[k]===true),"A host validation check failed");
  requireValue(decision.fact_ids.length>0&&new Set(decision.fact_ids).size===decision.fact_ids.length
   &&decision.fact_ids.every((k)=>Object.hasOwn(view.evidence_facts,k)),"Unknown or duplicate citation");
  requireValue(["hold","run_probe"].includes(decision.action)&&view.available_actions[decision.action].includes(decision.target),"Unregistered action or target");
  for(const [id,value] of Object.entries(view.evidence_facts)){
   const parts=id.split(":"),role=parts[1],metric=parts.slice(2).join(":");
   requireValue(parts[0]==="telemetry"&&value===training.arms[r.training_arms[role]].telemetry_final_20pct[metric],"Fact differs from published training");
  }
  for(const role of ["case","reference"])
   requireValue(r.checkpoint_sha256[role]===training.arms[r.training_arms[role]].checkpoint.sha256,"Checkpoint source differs");
  if(decision.action==="hold"){
   holds++;requireValue(decision.target===null&&row.review_request===null,"HOLD cannot contain an inspection request");
  }else{
   proposals++;
   requireValue(r.probe_proposal.probe===decision.target&&r.probe_proposal.requires_new_authorization===true,"Original inspection proposal differs");
   const expected={schema:"tokenfactory_recorded_inspection_review_v1",episode_id:r.episode_id,
    provider:"Nebius Token Factory",model,probe:decision.target,fact_ids:decision.fact_ids,
    request_sha256:manifest.files[r.request],response_sha256:manifest.files[r.response],
    checkpoint_sha256:r.checkpoint_sha256,requires_new_registration_approval_and_lease:true,
    approval_granted:false,execution_allowed:false,
    scope:"Recorded inspection proposal for review; not an executable authorization."};
   requireValue(canonical(row.review_request)===canonical(expected),"Inspection review is not bound to its recorded evidence");
  }
 }
 requireValue(holds===4&&proposals===4,"Recorded totals differ");
 $("#record").innerHTML=evidence.records.map((r,i)=>'<option value="'+i+'">'+(r.control?"Identical-reference control":"Candidate training run")+' · '+(r.data_policy==="eager"?"full context":"on-demand context")+' · trial '+r.trial+'</option>').join("");
 $("#record").disabled=false;$("#play").disabled=false;
 steps.forEach((b)=>{b.disabled=false;b.addEventListener("click",()=>{stop();setStage(Number(b.dataset.step));});});
 $("#record").addEventListener("change",()=>{stop();active=Number($("#record").value);setStage(0);});
 $("#play").addEventListener("click",()=>{
  if(timer){stop();render();return;}
  stage=0;timer=setInterval(()=>{if(stage<3)setStage(stage+1);else{stop();render();}},5000);
  $("#play").textContent="Pause walkthrough";$("#play").setAttribute("aria-pressed","true");render();
 });
 requireValue(canonical(window.NimEvidence.fromSources(files,manifest.files))===canonical(files["NEMOTRON_EVIDENCE.json"]),"Historical NIM evidence differs from its accepted source or scope");
 window.NimEvidence.install(files["NEMOTRON_EVIDENCE.json"],()=>{stop();render();});
 $("#verification").textContent="Evidence files verified";render();
 if(new URLSearchParams(location.search).get("demo")==="1"&&!matchMedia("(prefers-reduced-motion: reduce)").matches)$("#play").click();
}
initialize().catch((e)=>{stop();evidence=null;window.judgeReplay={verified:false,error:e.message};$("#verification").textContent="Evidence verification failed";$("#verification").className="error";$("#panel").className="error-panel";$("#panel").innerHTML='<h2>The replay could not verify its evidence</h2><p>'+escapeHTML(e.message)+'</p><p>Run the CPU preparation from the README and use a localhost server. Do not approve an inspection from an unverified record.</p>';$("#record").disabled=true;$("#play").disabled=true;steps.forEach((b)=>b.disabled=true);if($("#evidence-toggle"))$("#evidence-toggle").disabled=true;});
