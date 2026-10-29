'use strict';
(async function () {
  const el = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const manifest = await fetch('source-map.json').then(r=>r.json());
  const data = {};
  try {
    for (const f of manifest.files.filter(f=>f.path.startsWith('data/'))) {
      const bytes = await fetch(f.path).then(r=>{if(!r.ok)throw Error(r.status);return r.arrayBuffer()});
      const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(n=>n.toString(16).padStart(2,'0')).join('');
      if (digest !== f.sha256) throw Error('Source hash mismatch: '+f.path);
      data[f.id] = JSON.parse(new TextDecoder().decode(bytes));
    }
    const s=data.r1_summary,p=data.r1_probes,m=data.r1_model,t=data.r1_training,r=data.residual;
    if(s.outcome!=='recovered'||p.probes.length!==6||m.diagnosis.decision.probe!=='P_reward') throw Error('Evidence contract mismatch');
    window.replayEvidence={verified:true,sourceCount:Object.keys(data).length,sourceMap:manifest,scene:0};
    el('integrity').textContent=Object.keys(data).length+' evidence files / SHA-256 verified';
    const table=(heads,rows)=>'<table><thead><tr>'+heads.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
    const metric=(title,n,unit,cls='')=>'<section class="panel"><h2>'+title+'</h2><div class="metric '+cls+'">'+n+'</div><p class="sub">'+unit+'</p></section>';
    const scenes = [
      {label:'1 · Failure',sources:['r1_summary','r1_training'],html:()=>'<span class="pill">R1 / recorded development run / source seed '+s.source_training_seed+'</span><h1>Surviving is not the same as following commands.</h1><p>All three policies survive 20 seconds with zero falls. Tracking reveals the failure.</p><div class="row">'+metric('Normal control',s.fixed_evaluation_metrics.NONE.lin_vel_rmse_mps.toFixed(3),'linear velocity RMSE · m/s','good')+metric('Faulty PPO epochs',s.fixed_evaluation_metrics.R1.lin_vel_rmse_mps.toFixed(3),'linear velocity RMSE · m/s','bad')+'</div><p class="callout">Task: Isaac Lab / Unitree Go2 velocity tracking. Known defect; this is a development case.</p>'},
      {label:'2 · Inspect',sources:['r1_model','r1_probes'],html:()=>'<span class="pill">Recorded Nemotron 3 Super response</span><h1>A model proposes a check. Evidence can disagree.</h1><div class="row"><section class="panel"><h2>First choice: '+esc(m.diagnosis.decision.probe)+'</h2><p>Reward ranked first; optimizer ranked fourth.</p><p class="mono">'+esc(m.diagnosis.call.provider_model)+'</p><p class="small">The model selected the first probe. The next probes followed the registered sequence.</p></section><section class="panel">'+table(['PROBE','RECORDED OUTCOME'],p.probes.slice(0,5).map(x=>[esc(x.probe),'<span class="good">'+esc(x.outcome)+'</span>']))+'</section></div>'},
      {label:'3 · Confirm',sources:['r1_probes','r1_training'],html:()=>'<span class="pill">P_optimizer_updates / fresh three-update inspection</span><h1>Count what the optimizer actually did.</h1><p>Configured epochs × minibatches: 5 × 4 = 20 steps per update.</p><div class="row">'+metric('Normal control',p.probes[5].measurement.control.optimizer_steps.join(' / '),'optimizer steps · updates 0, 1, 2','good')+metric('R1 case',p.probes[5].measurement.case.optimizer_steps.join(' / '),'optimizer steps · updates 0, 1, 2','bad')+'</div><p class="callout">The inspection starts fresh with the same arm configuration. It does not inspect the trained checkpoint.</p>'},
      {label:'4 · Correct',sources:['r1_probes','r1_training','r1_model'],html:()=>'<span class="pill">Recorded approval and training artifacts</span><h1>Restore the registered setting, then retrain.</h1><div class="row"><section class="panel"><h2>'+esc(p.correction.correction_id)+'</h2><p>Recovery mode: <b>'+esc(p.correction.recovery_mode)+'</b></p><p>Configured → faulty → restored epochs</p><div class="metric">'+s.diagnostic_sequence[2].epoch_transition.join(' → ')+'</div><p class="small">Source seed '+s.source_training_seed+' → restart seed '+s.corrected_restart_seed+'</p></section><section class="panel"><h2>Approval event sequence</h2><pre>'+p.correction.approval_events.map((x,i)=>(i+1)+'. '+esc(x)).join('\n')+'</pre><p class="small">Recorded single consumption; this replay issues no approval or execution request.</p></section></div>'},
      {label:'5 · Evaluate',sources:['r1_summary','r1_eval'],html:()=>'<span class="pill">Fixed evaluation / 26 conditions / 1,040 environments</span><h1>Recovery is a separate behavioral test.</h1><div class="panel">'+table(['POLICY','LINEAR RMSE m/s','YAW RMSE rad/s','FALL RATE'],Object.entries(s.fixed_evaluation_metrics).map(([k,v])=>[esc(k),v.lin_vel_rmse_mps.toFixed(3),v.yaw_rate_rmse_radps.toFixed(3),v.fall_rate.toFixed(2)]))+'</div><p class="callout"><b class="good">'+esc(s.outcome.toUpperCase())+'</b> · Corrected checkpoint '+s.recovery.checkpoint_sha256.slice(0,16)+'…<br><span class="small">One registered R1 development run. No aggregate success-rate or model-superiority claim.</span></p>'},
      {label:'6 · Residual',sources:['residual','u1'],html:()=>'<span class="pill">Separate compound-defect development runs / v2</span><h1>One correction can leave another failure.</h1><div class="panel">'+table(['SEED','FIRST CORRECTION','FIRST RESULT','AFTER BOTH'],r.cases.map(c=>[c.seed,esc(c.correction_order[0]),'<span class="bad">'+esc(c.repair_verdicts[0])+'</span>','<span class="good">'+esc(c.repair_verdicts[1])+'</span>']))+'</div><p>Seeds 2120 and 2121 needed two corrections in opposite orders.</p><p class="sub">Developed after prior F1/R1 results. Final policies reproduce same-seed normal controls. U1 seed2062 is a separate expanded-scope confirmation with no model calls.</p>'}
    ];
    el('nav').innerHTML=scenes.map((x,i)=>'<button data-scene="'+i+'">'+x.label+'</button>').join('');
    function show(i) {
      window.replayEvidence.scene=i; el('stage').innerHTML=scenes[i].html();
      document.querySelectorAll('nav button').forEach((b,j)=>b.setAttribute('aria-current',String(i===j)));
      el('source').innerHTML=scenes[i].sources.map(id=>{const f=manifest.files.find(x=>x.id===id);return '<a href="'+f.path+'">'+esc(f.original)+'</a> · SHA '+f.sha256.slice(0,16)+'…'}).join('<br>');
    }
    document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>show(Number(b.dataset.scene)));
    show(0);
    if(new URLSearchParams(location.search).has('demo')){
      const start=performance.now();
      setInterval(()=>{const sec=(performance.now()-start)/1000;el('progress').value=Math.min(sec,108);const i=Math.min(5,Math.floor(sec/18));if(i!==window.replayEvidence.scene)show(i);window.replayEvidence.elapsed=sec;},100);
    }
  } catch(e) {document.body.dataset.error='true';el('integrity').textContent='VERIFICATION FAILED';el('stage').textContent=e.message;window.replayEvidence={verified:false,error:e.message};throw e;}
})();
