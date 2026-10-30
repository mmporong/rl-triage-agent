"use strict";
window.NimEvidence=(()=>{
 const commit="b29a791db9f6397c4b04698b349ef68606273ae5";
 const anchorSHA="2ca099294618dcb50c10d444816433e0e46131fd419211b467ebaa4e5c07b7ee";
 const names=["latency_analysis.json","latency_pairs.csv","model_quality.csv","vl_observations_18.json","vl_aux_observations_3.json","vl_selection_and_render.json","summary.json","architecture_comparison.json","api_reference_evidence.json","sources.json","recovery_summary.json"];
 const el=(q)=>document.querySelector(q);
 const esc=(s)=>String(s).replace(/[&<>"']/g,(c)=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
 const fmt=(x,d=3)=>Number(x).toFixed(d);
 function fromSources(files,hashes){
  const anchor=files["NIM_SOURCE_ANCHOR.json"];
  if(hashes["NIM_SOURCE_ANCHOR.json"]!==anchorSHA||!anchor||anchor.source_commit!==commit||Object.keys(anchor.files).length!==names.length||!names.every(n=>anchor.files["nim-evidence/"+n]===hashes["nim-evidence/"+n]))throw Error("Accepted NIM source anchor differs");
  const latency=files["nim-evidence/latency_analysis.json"],summary=files["nim-evidence/summary.json"],recovery=files["nim-evidence/recovery_summary.json"],h=summary.headline;
  return {schema:"nemotron_evidence_view_v1",provider:"NVIDIA hosted NIM",phase:"historical_development_measurements",latency_measurement_date:"2026-10-16",source_commit:commit,latency_unit:"API request wall-clock seconds",latency:latency.stats,quality:latency.quality,
   vision:{valid:h.VL_valid_observations,classification:h.VL_fixed_eval_class_correspondence,auxiliary:h.VL_s2080_aux_observations},loop:h.fresh_multimodal_diagnosis,
   evaluation:{ratios:recovery.fixed_evaluation.ratios,criteria:recovery.fixed_evaluation.registered_criteria,outcome:recovery.outcome,iterations:recovery.training.iterations,conditions:recovery.fixed_evaluation.CORRECTED.condition_count},
   boundaries:{token_factory_latency_comparison:false,whole_workflow_speedup:false,human_labeled_video_accuracy:false,image_selection_benefit:false,new_3_5_recovery:false,overall_quality_acceptance_met:false,grants_execution_authority:false},
   input_sha256:Object.fromEntries(names.map(n=>["nim-evidence/"+n,hashes["nim-evidence/"+n]]))};
 }
 function install(data,stopWalkthrough){
  let topic="latency",opened=false;
  const panel=el("#nemotron-panel"),button=el("#evidence-toggle"),workspace=el(".workspace"),steps=el("#steps"),title=el("h1"),mode=el(".mode");
  const original={title:title.textContent,mode:mode.textContent,pageTitle:document.title};
  function source(name,label){return '<a href="nim-evidence/'+esc(name)+'">'+esc(label)+'</a>';}
  function render(){
   let body='';
   if(topic==="latency"){
    const q=(stage,arm)=>data.quality.find(r=>r.stage===stage&&r.model_arm===arm);
    const v1=data.latency.B1.both_contract_valid,v2=data.latency.B2.both_contract_valid;
    const rows=["B1","B2"].map(k=>{const r=data.latency[k].all_matched_requests;return '<tr><th scope="row">'+k+'</th><td>'+fmt(r.old_mean_wall_s)+'</td><td>'+fmt(r.new_mean_wall_s)+'</td><td>'+fmt(r.mean_wait_reduction_percent,2)+'% shorter</td><td>'+r.paired_requests+' pairs / '+r.unique_inputs+' inputs × 3</td></tr>';}).join('');
    body='<h2>Shorter observed API response waits</h2><p class="evidence-scope">NVIDIA-hosted NIM · 16 October 2026 · same-input development requests · seconds</p><table id="nim-latency-table"><thead><tr><th>Task</th><th>3 Super mean (s)</th><th>3.5 mean (s)</th><th>Mean wait</th><th>Sample unit</th></tr></thead><tbody>'+rows+'</tbody></table><p>Only the model ID changed within each paired request. These are API-call wall times, excluding validation, GPU inspections and training. They are not Token Factory latency measurements.</p><details><summary>Both-valid pairs and mixed quality results</summary><p>The post-hoc subset with two valid replies was '+fmt(v1.mean_wait_reduction_percent,2)+'% shorter for B1 ('+v1.paired_requests+' pairs) and '+fmt(v2.mean_wait_reduction_percent,2)+'% for B2 ('+v2.paired_requests+' pairs).</p><table><thead><tr><th>Development quality check</th><th>3 Super</th><th>3.5</th></tr></thead><tbody><tr><td>B1 contract-invalid replies</td><td>'+q('B1','old').contract_invalid+'/'+q('B1','old').requests+'</td><td>'+q('B1','new').contract_invalid+'/'+q('B1','new').requests+'</td></tr><tr><td>B2 citation errors</td><td>'+q('B2','old').evidence_citation_error+'/'+q('B2','old').requests+'</td><td>'+q('B2','new').evidence_citation_error+'/'+q('B2','new').requests+'</td></tr><tr><td>B2 contract-valid correct top-1</td><td>'+q('B2','old').contract_valid_true_cause_top1+'/'+q('B2','old').requests+'</td><td>'+q('B2','new').contract_valid_true_cause_top1+'/'+q('B2','new').requests+'</td></tr><tr><td>B2 parsed top-1, ignoring contract</td><td>'+q('B2','old').true_cause_top1_ignoring_contract+'/'+q('B2','old').requests+'</td><td>'+q('B2','new').true_cause_top1_ignoring_contract+'/'+q('B2','new').requests+'</td></tr></tbody></table><p>Quality was mixed; the full preregistered acceptance criteria were not met. Repeated requests are not 57 independent new faults. Server hardware, precision, queueing and enabled optimizations were not isolated.</p></details><p class="source-links">'+source('latency_analysis.json','Accepted analysis')+' · '+source('latency_pairs.csv','57 paired requests')+' · '+source('model_quality.csv','Quality counts')+'</p>';
   }else if(topic==="vision"){
    const v=data.vision;
    body='<h2>Policy-frame observations, checked separately</h2><p class="evidence-scope">NVIDIA-hosted NIM · historical development observation cohort</p><table id="nim-vl-table"><thead><tr><th>Observation check</th><th>Recorded result</th></tr></thead><tbody><tr><td>Valid structured replies</td><td>'+v.valid+'/18</td></tr><tr><td>Correspondence with fixed-evaluation classes</td><td>'+v.classification.matched+'/'+v.classification.total+'</td></tr><tr><td>Normal-policy group</td><td>'+v.classification.normal.matched+'/'+v.classification.normal.total+'</td></tr><tr><td>Failed-policy group</td><td>'+v.classification.failed.matched+'/'+v.classification.failed.total+'</td></tr></tbody></table><p>This compares walking observations with fixed-evaluation groups. It is not human-labeled video accuracy or cause-diagnosis accuracy. The videos were not all covered by the later render audit.</p><p>A separate seed-2080 cohort had '+v.auxiliary.matched+'/'+v.auxiliary.total+' corresponding auxiliary observations. They were not used for decisions; 3 Super made that demonstration’s diagnosis and correction decisions.</p><p class="source-links">'+source('vl_observations_18.json','18 observation records')+' · '+source('vl_aux_observations_3.json','Separate auxiliary cohort')+'</p>';
   }else{
    const l=data.loop,e=data.evaluation;
    body='<h2>Model proposal to an actual inspection</h2><p class="evidence-scope">Separate NVIDIA-hosted NIM development run · diagnosis seed '+l.seed+' · read-only execution evidence</p><table id="nim-loop-table"><tbody><tr><th>Input and citation</th><td>'+l.available_frames+' frames plus training facts; '+l.cited_telemetry_facts+' telemetry citations, '+l.cited_frames+' frame citations</td></tr><tr><th>Model-selected inspection</th><td>'+esc(l.selected_probe)+'; actual normal/abnormal separation</td></tr><tr><th>Follow-up proposal</th><td>'+esc(l.actual_correction_proposal)+'; cited the abnormal inspection outcome</td></tr><tr><th>Registered retraining</th><td>Fresh seed '+l.fresh_restart_seed+', '+e.iterations+' updates</td></tr><tr><th>Fixed evaluation</th><td>'+e.conditions+' conditions; <strong>'+esc(e.outcome)+'</strong></td></tr><tr><th>Yaw error ratio</th><td>'+fmt(e.ratios.err_yaw_ratio,5)+' &gt; '+e.criteria.yaw_rate_rmse_ratio_max+' acceptance limit</td></tr></tbody></table><p>The inspection and correction workflow ran, but the new policy did not meet the recovery criteria. Frames were available; their contribution to choosing a better inspection was not established. This is separate from the Token Factory replay and the earlier 3 Super recovery.</p><p class="source-links">'+source('recovery_summary.json','Actual recovery evaluation')+' · '+source('summary.json','Consolidated scope')+' · '+source('vl_selection_and_render.json','Frame and selection evidence')+'</p>';
   }
   panel.innerHTML='<p class="eyebrow">Measured evidence · providers and phases kept separate</p><nav class="evidence-topics" aria-label="Nemotron evidence topics"><button data-nim-topic="latency" aria-pressed="'+(topic==="latency")+'">API response time</button><button data-nim-topic="vision" aria-pressed="'+(topic==="vision")+'">VL observations</button><button data-nim-topic="loop" aria-pressed="'+(topic==="loop")+'">Actual inspection loop</button></nav><article>'+body+'</article><p class="scope evidence-footer">Source commit '+commit.slice(0,7)+'. Read-only historical evidence; this view grants no execution authority. '+source('sources.json','Source inventory and SHA-256')+'</p>';
   panel.querySelectorAll('[data-nim-topic]').forEach(b=>b.addEventListener('click',()=>{topic=b.dataset.nimTopic;render();}));
   window.nimEvidenceState={open:opened,topic,provider:data.provider,phase:data.phase,unit:topic==="latency"?data.latency_unit:(topic==="vision"?"observation counts":"registered inspection/evaluation records"),execution_allowed:false};
  }
  button.disabled=false;
  button.addEventListener('click',()=>{
   stopWalkthrough();opened=!opened;button.setAttribute('aria-expanded',String(opened));button.textContent=opened?'Back to TF investigation':'Nemotron evidence';panel.hidden=!opened;workspace.hidden=opened;steps.hidden=opened;el('#record').disabled=opened;el('#play').disabled=opened;
   title.textContent=opened?'Nemotron 3.5: measured capabilities and boundaries':original.title;mode.textContent=opened?'NVIDIA-hosted NIM · historical evidence':original.mode;document.title=opened?'RL Triage Agent · Nemotron evidence':original.pageTitle;
   if(opened)render();else window.nimEvidenceState={open:false,topic,execution_allowed:false};
  });
  window.nimEvidenceState={open:false,topic,execution_allowed:false};
 }
 return {fromSources,install};
})();
