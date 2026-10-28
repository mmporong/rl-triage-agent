"""CPU fixture만으로 v1 prefix·v2 runner·readonly source object 검증을 확인한다."""
from __future__ import annotations

import hashlib
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"evals"), str(ROOT/"src")]
import run_agentic_development_v2 as D
import test_selection_comparison_controller_v1 as F
from test_agentic_investigation_evidence_v1 import case  # noqa: F401


def response(action, target, fact="config:action_scale"):
    return {"provider_model": D.CALL.O.MODEL, "finish_reason": "stop", "wall_s": 0.01,
            "content": json.dumps({"action": action, "target": target, "fact_ids": [fact]})}


@pytest.fixture
def plan(case):  # noqa: F811
    root = case["root"]
    index_path = root/"index.json"
    D.J._write_new(index_path, case["index"])
    audit = root/"prior-audit.json"
    D.J._write_new(audit, {"known_pending": 0})
    cases = [{"id": f"cpu_dev_{n:02d}", "evidence_id": "evidence-01", "initial_facts": "config_only",
              "frame_ids": ["f01", "f02"]} for n in range(1,4)]
    old = {"schema": D.V1.SCHEMA, "roots": {k:str(v) for k,v in case["roots"].items()},
        "index": {"path": str(index_path), "sha256": D.J.file_sha(index_path)},
        "index_canonical_sha256": D.E.index_sha256(case["index"]), "output": str(root/"legacy-run"),
        "source_anchor": "a"*40, "goal": "등록 자료로 다음 검사를 고른다", "max_decisions": 4,"api_cap":12,"cases": cases,
        "budget": {"prior_audit": D._ref(audit),"original_core_allocation_cap":120,"new_phase_cap":12,"combined_allocation_cap":132}}
    protocol = {"schema":"agentic_investigation_development_protocol_v1","cases":cases,"goal":old["goal"],
        "model":D.CALL.PARAMS,"input_index_raw_sha256":old["index"]["sha256"],
        "input_index_canonical_sha256":old["index_canonical_sha256"],"budget":{"combined_allocation_cap":132},
        "api_cap":12,"max_decisions_per_case":4,"sources_lf":{}}
    pp = root/"protocol-v1.json";D.J._write_new(pp,protocol);old["protocol"]=D._ref(pp)
    for c in cases:
        ctx=D.A.prepare(F.snapshot(root,"case","model_adaptive"),goal="CPU fixture prior invalid")
        capsule=D.CALL.V1.freeze(ctx,Path(old["output"])/c["id"]/"decision-01",source_anchor="a"*40,protocol_sha256=old["protocol"]["sha256"])
        D.CALL.V1.advance(capsule,Path(old["output"])/"api",lambda *_:response("read_evidence",None))
        D.J._write_new(Path(old["output"])/c["id"]/"decision-01.result.json",{"model_status":"invalid","review":{"status":"invalid"}})
    old_path=root/"legacy-plan.json";D.J._write_new(old_path,old)
    new_protocol=D.build_protocol(old_path,sources_lf={name:"0"*64 for name in D.NEW_SOURCES})
    path=root/"protocol-v2.json";D.J._write_new(path,new_protocol)
    return D.prepare_plan(old_path,path,source_anchor="b"*40,output=root/"v2-run")


def test_runner_training_frames_probe_replay_preserves_prefix(plan):
    prefix={name:D.J.file_sha(Path(value["path"])) for name,value in plan["legacy"]["prefix"].items()}
    sent=[]

    params_seen=[]
    def provider(messages,params):
        sent.append(messages);params_seen.append(params)
        assert messages[0]["content"]==D.A.SYSTEM
        assert params==D.CALL.params_for_messages(messages)
        return [response("read_evidence","evidence:training-01"),response("inspect_frames","evidence:clip-01"),
                response("run_probe","P_reward","vision:frame:f01")][len(sent)-1]

    result=D.run_case(plan,plan["cases"][0],provider)
    assert result["status"]=="probe_proposed_unexecuted" and len(sent)==3
    assert [row["image_count"] for row in result["decisions"]]==[0,0,2]
    assert "read_evidence" in params_seen[0]["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"]
    assert "read_evidence" not in params_seen[1]["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"]
    assert "vision:frame:f01" in params_seen[2]["response_format"]["json_schema"]["schema"]["properties"]["fact_ids"]["items"]["enum"]
    assert [row["params_sha256"] for row in result["decisions"]]==[D.J.digest(params) for params in params_seen]
    assert result["api"]["reserved"]==6 and result["approval_granted"] is False
    assert D.run_case(plan,plan["cases"][0],lambda *_:pytest.fail("completed replay provider"))==result
    assert prefix=={name:D.J.file_sha(Path(value["path"])) for name,value in plan["legacy"]["prefix"].items()}


def test_pending_is_preserved_and_not_resent(plan):
    def provider(*_):raise D.ProviderPending("CPU fixture interrupted before provider outcome")
    with pytest.raises(D.ProviderPending):D.run_case(plan,plan["cases"][0],provider)
    result=D.run_case(plan,plan["cases"][0],lambda *_:pytest.fail("pending resend"))
    assert result["status"]=="blocked_pending" and result["api"]["reserved"]==4


@pytest.mark.parametrize("field",["api_root","max_decisions","index","budget","cases","prefix"])
def test_plan_tampering_is_rejected_before_provider(plan,field):
    if field=="api_root":plan[field]=str(Path(plan["output"])/"new-api")
    elif field=="max_decisions":plan[field]=5
    elif field=="index":plan[field]["sha256"]="0"*64
    elif field=="budget":plan[field]["combined_allocation_cap"]=144
    elif field=="cases":plan[field][0]["initial_facts"]="all_training"
    else:
        p=Path(plan["legacy"]["prefix"]["responses/0001.json"]["path"]);p.write_bytes(p.read_bytes()+b" ")
    with pytest.raises(ValueError):D.run_case(plan,plan["cases"][0],lambda *_:pytest.fail("tampered provider"))


def test_same_checkpoint_binding_is_rechecked_before_provider(plan):
    path=Path(plan["index"]["path"])
    index=D.J._read(path)
    index["spec"]["entries"][0]["binding"]["checkpoint_sha256"]="1"*64
    path.write_bytes(D.J._bytes(index))
    with pytest.raises(ValueError,match="SHA"):
        D.run_case(plan,plan["cases"][0],lambda *_:pytest.fail("cross-checkpoint provider"))


def test_current_source_freeze_includes_new_implementation_tests_and_reader(plan):
    sources=D.source_hashes(Path(plan["legacy"]["plan"]["path"]))
    assert set(D.NEW_SOURCES) | set(D.NEW_TESTS) <= set(sources)
    assert all(value==D.V1._lf_sha((D.ROOT/name).read_bytes()) for name,value in sources.items())


def test_same_public_messages_from_another_checkpoint_cannot_replace_completed_capsule(plan):
    row=plan["cases"][0]
    D.run_case(plan,row,lambda *_:response("hold",None))
    root=Path(plan["output"])
    ref_path=root/row["id"]/"decision-01.reference.json"
    _,capsule=D.CALL.prepared(D.J._read(ref_path))
    inputs=capsule["context"]["snapshot"]["inputs"]
    packet=json.loads(json.dumps(inputs["packet"]))
    packet["input_bindings"]["case"]["checkpoint_sha256"]="1"*64
    snapshot=D.A.C.V.build_snapshot(packet,inputs["run_id"],inputs["policy"],inputs["protocol_sha256"],root/"other-history.jsonl")
    old=capsule["context"]
    context=D.A.prepare(snapshot,**{key:old[key] for key in ["goal","evidence","revealed","extra_facts","decisions_left"]})
    assert D.A.messages(context)[1]["content"]==D.A.messages(old)[1]["content"]
    other=D.CALL.freeze(context,root/"wrong-cp",source_anchor=plan["source_anchor"],
                        protocol_sha256=plan["protocol"]["sha256"],messages=capsule["messages"])
    ref_path.write_bytes(D.J._bytes(other))
    with pytest.raises(ValueError,match="checkpoint"):
        D.run_case(plan,row,lambda *_:pytest.fail("wrong CP cached result"))


def source_fixture(repo,files):
    """SHA가 맞는 loose commit/tree/blob를 개인 CPU fixture에 만든다. Git 명령은 실행하지 않는다."""
    directory=repo/".git"; (directory/"objects").mkdir(parents=True)

    def object_(kind,body):
        raw=f"{kind} {len(body)}".encode()+b"\0"+body
        oid=hashlib.sha1(raw).hexdigest();path=directory/"objects"/oid[:2]/oid[2:]
        path.parent.mkdir(exist_ok=True);path.write_bytes(zlib.compress(raw));return oid

    tree={}
    for name,raw in files.items():
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        cursor=tree
        for part in Path(name).parts[:-1]:cursor=cursor.setdefault(part,{})
        cursor[Path(name).name]=object_("blob",raw)

    def tree_(value):
        rows=[]
        for name,child in sorted(value.items()):
            oid=tree_(child) if isinstance(child,dict) else child
            rows.append((b"40000 " if isinstance(child,dict) else b"100644 ")+name.encode()+b"\0"+bytes.fromhex(oid))
        return object_("tree",b"".join(rows))

    tree_id=tree_(tree)
    def commit(parent=""):
        return object_("commit",f"tree {tree_id}\n{('parent '+parent+chr(10)) if parent else ''}author CPU <cpu@example.invalid> 1 +0000\ncommitter CPU <cpu@example.invalid> 1 +0000\n\nfixture\n".encode())
    anchor=commit();head=commit(anchor);(directory/"HEAD").write_text(head,encoding="ascii")
    return anchor,head


def test_readonly_source_ancestor_blob_and_traversal_cpu_objects(tmp_path):
    repo=tmp_path/"objects-fixture";files={"evals/fixture.py":b"value = 1\n"}
    anchor,_=source_fixture(repo,files)
    protocol={"sources_lf":{"evals/fixture.py":D.V1._lf_sha(files['evals/fixture.py'])}}
    D.verify_sources(protocol,anchor,repo=repo)
    with pytest.raises(ValueError,match="조상"):
        D.verify_sources(protocol,"0"*40,repo=repo)
    protocol["sources_lf"]={"../outside.py":"0"*64}
    with pytest.raises(ValueError,match="탈출"):D.verify_sources(protocol,anchor,repo=repo)
    protocol["sources_lf"]={"evals/fixture.py":"0"*64}
    with pytest.raises(ValueError,match="blob"):D.verify_sources(protocol,anchor,repo=repo)


def provider_fixture_payload(root):
    context=D.A.prepare(F.snapshot(root,"case","model_adaptive"),goal="CPU schema 전송 fixture")
    messages=D.A.messages(context);params=D.CALL.params_for(context)
    return {"messages":messages,"params":params,"messages_sha256":D.J.digest(messages),"params_sha256":D.J.digest(params)}


def test_provider_once_and_cpu_wsl_transport_preserve_registered_schema_and_hash(tmp_path):
    payload=provider_fixture_payload(tmp_path)
    received=[];transport=[]

    def provider(messages,params):
        received.append((messages,params))
        return response("hold",None)

    def process(command,**kwargs):
        transport.append(command)
        incoming=json.loads(kwargs["input"])
        envelope=D.provider_once(incoming,complete=provider)
        return SimpleNamespace(returncode=0,stdout=json.dumps(envelope))

    answer=D.wsl_complete(payload["messages"],payload["params"],run_process=process)
    assert len(received)==1 and received[0]==(payload["messages"],payload["params"])
    assert "response_format" in received[0][1] and received[0][1]!=D.CALL.O.REQUEST_PARAMS
    assert transport[0][-1].endswith("run_agentic_development_v2.py")
    assert "provider-once" in transport[0][-3]
    assert answer["content"]==response("hold",None)["content"]


def test_provider_hash_mismatch_or_tampered_schema_fails_before_provider(tmp_path):
    payload=provider_fixture_payload(tmp_path)
    payload["params_sha256"]="0"*64
    with pytest.raises(ValueError,match="hash/schema"):
        D.provider_once(payload,complete=lambda *_:pytest.fail("invalid hash provider"))
    payload["params_sha256"]=D.J.digest(payload["params"])
    payload["params"]["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"].append("run_shell")
    payload["params_sha256"]=D.J.digest(payload["params"])
    with pytest.raises(ValueError,match="hash/schema"):
        D.provider_once(payload,complete=lambda *_:pytest.fail("unregistered schema provider"))


def test_wsl_returned_hash_mismatch_preserves_pending_request(tmp_path):
    context=D.A.prepare(F.snapshot(tmp_path,"case","model_adaptive"),goal="CPU schema 전송 fixture")
    reference=D.CALL.freeze(context,tmp_path/"call",source_anchor="a"*40,protocol_sha256="b"*64)

    def process(*_args,**kwargs):
        incoming=json.loads(kwargs["input"])
        envelope=D.provider_once(incoming,complete=lambda *_:response("hold",None))
        envelope["params_sha256"]="0"*64
        return SimpleNamespace(returncode=0,stdout=json.dumps(envelope))

    with pytest.raises(D.ProviderPending,match="hash"):
        D.CALL.advance(reference,tmp_path/"api",lambda m,p:D.wsl_complete(m,p,run_process=process))
    answer=D.CALL.advance(reference,tmp_path/"api",lambda *_:pytest.fail("pending resend"))
    assert answer["journal"]["status"]=="blocked_pending" and answer["journal"]["api"]["pending"]==1
