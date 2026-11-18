"""Offline episode quality dashboard; no external scripts or services required."""

import json
from pathlib import Path
from typing import Any


CHECKS = {
    "structure": ("数据完整性", {"read_error", "capture_rejected", "missing_configuration",
        "configuration_hash_mismatch", "empty_stream", "invalid_message", "missing_camera_info",
        "image_shape_mismatch", "calibration_changed", "invalid_state_shape", "nonfinite_action"}),
    "timestamps": ("时间戳顺序", {"non_increasing_timestamp", "insufficient_common_interval"}),
    "alignment": ("多模态时间对齐", {"alignment_gap", "primary_frame_drop", "primary_period_anomaly"}),
    "action_freshness": ("指令时效", {"stale_command"}),
    "action_jumps": ("动作跳变", {"action_jump"}),
    "action_bounds": ("URDF 关节限位", {"urdf_limit", "missing_urdf"}),
    "image_quality": ("图像清晰度与亮度", {"image_quality"}),
    "image_reuse": ("图像复用", {"reused_source_image", "primary_filled_frame"}),
}


def dashboard_data(report: dict[str, Any]) -> dict[str, Any]:
    """Only explicitly completed checks may be shown as passed."""
    data = dict(report)
    data["episodes"] = []
    for index, source in enumerate(report.get("episodes", [])):
        ep = dict(source)
        ep["input_index"] = index
        ep["name"] = source.get("source_episode") or Path(source.get("source", "unknown")).stem
        completed = set(source.get("completed_checks", []))
        rows = []
        for key, (label, kinds) in CHECKS.items():
            issues = [i for i in source.get("issues", []) if i.get("kind") in kinds]
            status = ("failed" if any(i.get("severity") == "error" for i in issues)
                      else "warning" if issues else "passed" if key in completed else "not_run")
            rows.append({"key": key, "label": label, "status": status, "issues": issues})
        known = set().union(*(kinds for _, kinds in CHECKS.values()))
        other = [i for i in source.get("issues", []) if i.get("kind") not in known]
        if other:
            rows.append({"key": "other", "label": "其他问题", "status": "failed"
                         if any(i.get("severity") == "error" for i in other) else "warning", "issues": other})
        ep["checks"] = rows
        data["episodes"].append(ep)
    return data


def episode_status_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact QC ledger. Frame ranges refer only to the aligned output grid."""
    rows = []
    for input_index, episode in enumerate(report.get("episodes", [])):
        issues = []
        for original in episode.get("issues", []):
            kind = original.get("kind")
            field = original.get("field")
            check = next((key for key, (_, kinds) in CHECKS.items() if kind in kinds), "other")
            # 'frames' may also be an integer count, not a list of bad frames.
            frames = original.get("frames")
            aligned = isinstance(frames, list) and bool(frames)
            indices = sorted(set(frames)) if aligned else []
            intervals = []
            for frame in indices:
                if intervals and frame == intervals[-1][1]:
                    intervals[-1][1] = frame + 1
                else:
                    intervals.append([frame, frame + 1])
            details = {k: v for k, v in original.items()
                       if k not in {"kind", "severity", "field"} and not (aligned and k == "frames")}
            camera = field if (check in {"image_quality", "image_reuse", "alignment"}
                               and field in episode.get("cameras", {})) else None
            for start, end in intervals or [[None, None]]:
                issues.append({"check": check, "kind": kind,
                               "severity": original.get("severity"), "field": field,
                               "camera": camera, "start_frame": start, "end_frame": end,
                               "details": details})
        name = episode.get("source_episode") or Path(episode.get("source", "unknown")).stem
        rows.append({"schema_version": "1.0", "dataset_id": report.get("repo_id"),
                     "episode_id": name, "source_episode": name,
                     "source": episode.get("source"), "input_index": input_index,
                     "episode_index": (episode.get("output_episode_index")
                                       if report.get("status") == "converted" else None),
                     "status": episode.get("status"), "total_frames": episode.get("frames", 0),
                     "frame_interval": "[start_frame, end_frame)",
                     "frame_reference": "aligned_grid", "fps": report.get("config", {}).get("fps"),
                     "grid_start_ns": episode.get("grid_start_ns"),
                     "grid_mode": episode.get("grid_mode"),
                     "timeline": episode.get("timeline"),
                     "devices_id": episode.get("devices_id"), "calib_id": episode.get("calib_id"),
                     "operator_id": episode.get("operator_id"), "session": episode.get("session"),
                     "capture_state": episode.get("capture_state"),
                     "task_result": episode.get("task_result"),
                     "completed_checks": episode.get("completed_checks", []), "issues": issues})
    return rows


def write_episode_status(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace the previous run's ledger, never append duplicate episode rows.
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                      for row in episode_status_rows(report))
    path.write_text(payload, encoding="utf-8")


def write_quality_report(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(dashboard_data(report), ensure_ascii=False, allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HTML.replace("__REPORT_DATA__", payload), encoding="utf-8")


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>数据质量看板</title><style>
:root{font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#243247;background:#f4f7fb}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:30px}
h1{font-size:27px;margin:0;color:#14273e}h2{font-size:18px;margin:0 0 14px}h3{font-size:15px}
.muted{color:#68788e}.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:22px 0}
.card,.panel{background:#fff;border:1px solid #dfe6ef;border-radius:10px;padding:18px}.card strong{display:block;font-size:27px;color:#163653}
.panel{margin-bottom:18px}.layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:18px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;border-bottom:1px solid #e7edf4;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{background:#f7f9fc;color:#52647b}tbody tr:last-child td{border-bottom:0}
.badge{display:inline-block;border-radius:20px;padding:2px 9px;font-size:12px;font-weight:600;white-space:nowrap}
.passed{background:#e4f6ed;color:#117044}.warning{background:#fff2d5;color:#8b5b00}.failed{background:#ffebeb;color:#b02b2b}.not_run{background:#edf0f4;color:#667386}
input,select,button{font:inherit;border:1px solid #ccd7e5;border-radius:6px;padding:8px;background:white;color:inherit}
input{width:100%;margin-bottom:9px}select{width:100%}button{cursor:pointer}.episode{display:block;width:100%;text-align:left;margin-top:8px;padding:12px;overflow-wrap:anywhere}
.episode.selected{border-color:#367ac4;background:#eef5ff}.episode small{display:block;color:#6b7c90}
#episode-list{max-height:680px;overflow:auto}.toolbar{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-bottom:12px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f5f8fc;padding:12px;border-radius:6px}details{margin:8px 0}summary{cursor:pointer;color:#31577c}
.scroll{overflow-x:auto}.notice{padding:12px 16px;border-left:4px solid #a9bdd4;background:#edf3fa;margin:15px 0}.source{overflow-wrap:anywhere;font-size:12px}
.check-summary button{border:0;background:none;padding:0;color:#2268ad;text-decoration:underline}
@media(max-width:950px){.cards{grid-template-columns:repeat(3,1fr)}.layout{grid-template-columns:1fr}#episode-list{max-height:280px}}
@media(max-width:550px){main{padding:14px}.cards{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main><h1>数据质量看板</h1><div id="subtitle" class="muted"></div>
<div id="cards" class="cards"></div><div id="global-notice"></div>
<section class="panel"><h2>各检查项问题集数</h2><div class="scroll" id="summary"></div>
<p class="muted">错误会拒绝整集，警告集仍可保留。通过率指没有被拒绝的集占比；未执行的检查不计为通过。点击计数可筛选 episode。</p></section>
<div class="layout"><aside class="panel"><h2>逐 Episode 查看</h2>
<input id="search" type="search" placeholder="搜索源 episode 名称或输入序号" aria-label="搜索 episode">
<select id="filter" aria-label="筛选状态"><option value="all">全部 episode</option><option value="failed">仅拒绝</option><option value="warning">仅警告</option><option value="passed">仅通过</option><option value="issues">所有问题集</option></select>
<div id="active-check" class="muted"></div><div id="episode-list"></div></aside><section class="panel" id="detail"></section></div>
<section class="panel"><h2>输出验收</h2><div id="verification"></div></section>
<details class="panel"><summary>转换配置与阈值</summary><pre id="config"></pre></details>
<noscript>请启用浏览器 JavaScript 查看逐集报告。本文件可离线打开，不访问外部服务。</noscript>
</main><script id="report-data" type="application/json">__REPORT_DATA__</script><script>
const data=JSON.parse(document.getElementById('report-data').textContent), eps=data.episodes||[];
const labels={passed:'通过',warning:'警告',failed:'拒绝 / 错误',not_run:'未执行',converted:'转换完成',audited:'预检完成',conversion_failed:'转换失败'};
const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge=s=>`<span class="badge ${['passed','warning','failed','not_run'].includes(s)?s:'not_run'}">${esc(labels[s]||s)}</span>`;
const table=(headers,rows)=>`<table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.length?rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join(''):`<tr><td colspan="${headers.length}" class="muted">暂无记录</td></tr>`}</tbody></table>`;
const fmt=v=>typeof v==='number'?Number(v.toFixed(3)):v??'—';
const accepted=eps.filter(e=>e.status!=='failed').length, failed=eps.filter(e=>e.status==='failed').length, warnings=eps.filter(e=>e.status==='warning').length;
const frames=eps.filter(e=>e.status!=='failed').reduce((n,e)=>n+(e.frames||0),0),fps=data.config?.fps||30;
document.getElementById('subtitle').textContent=`数据集：${data.repo_id||data.profile?.name||'未指定'} · ${labels[data.status]||data.status} · ${fps} FPS`;
document.getElementById('cards').innerHTML=[['总 episodes',eps.length],['问题 episodes',eps.filter(e=>(e.issues||[]).length).length],['拒绝 episodes',failed],['警告 episodes',warnings],['质检通过率',eps.length?(100*accepted/eps.length).toFixed(1)+'%':'—'],[data.status==='converted'?'输出帧数':'候选帧数',frames]].map(([k,v])=>`<div class="card"><div class="muted">${esc(k)}</div><strong>${esc(v)}</strong></div>`).join('');
document.getElementById('global-notice').innerHTML=data.error?`<div class="notice">转换错误：${esc(data.error)}${data.staging_path?'<br>临时目录：'+esc(data.staging_path):''}</div>`:'';
let selected=eps.length?0:null,checkFilter=null,statusFilter=null;
const keys=[...new Set(eps.flatMap(e=>e.checks.map(c=>c.key)))];
document.getElementById('summary').innerHTML=table(['检查项','错误集数','警告集数','通过集数','未执行集数'],keys.map(key=>{
 const checks=eps.map(e=>e.checks.find(c=>c.key===key)).filter(Boolean),label=checks[0].label;
 return [esc(label),...['failed','warning','passed','not_run'].map(s=>`<button data-check="${esc(key)}" data-state="${s}">${checks.filter(c=>c.status===s).length}</button>`)];
}));
document.getElementById('summary').className='check-summary';
document.getElementById('summary').addEventListener('click',e=>{const b=e.target.closest('button[data-check]');if(!b)return;checkFilter=b.dataset.check;statusFilter=b.dataset.state;document.getElementById('filter').value='all';renderList()});
function renderList(){
 const q=document.getElementById('search').value.toLowerCase(),filter=document.getElementById('filter').value;
 const visible=eps.filter(e=>(`${e.input_index} ${e.name}`).toLowerCase().includes(q)&&(filter==='all'||(filter==='issues'?(e.issues||[]).length:e.status===filter))&&(!checkFilter||e.checks.some(c=>c.key===checkFilter&&c.status===statusFilter)));
 if(!visible.some(e=>e.input_index===selected))selected=visible.length?visible[0].input_index:null;
 document.getElementById('active-check').innerHTML=checkFilter?`按检查项筛选 · ${esc(labels[statusFilter])} <button id="clear-check">清除</button>`:`显示 ${visible.length} / ${eps.length} 集`;
 document.getElementById('clear-check')?.addEventListener('click',()=>{checkFilter=null;renderList()});
 document.getElementById('episode-list').innerHTML=visible.map(e=>`<button class="episode ${e.input_index===selected?'selected':''}" data-index="${e.input_index}">输入 #${e.input_index} ${badge(e.status)}<small>${esc(e.name)}</small><small>${e.frames||0} 帧 · 输出 ${e.output_episode_index??'未写入'} · ${(e.issues||[]).length} 项记录</small></button>`).join('')||'<p class="muted">没有符合条件的 episode。</p>';
 renderDetail();
}
function issueDetails(i){const details=Object.fromEntries(Object.entries(i).filter(([k])=>!['severity','kind','field','frames'].includes(k)));
 return `<div><b>${esc(i.kind)}</b> · ${esc(i.field)}${i.frames?` · 涉及 ${i.frames.length} 帧`:''}</div><details><summary>查看原因、阈值与帧索引</summary><pre>${esc(JSON.stringify({...details,...(i.frames?{frames:i.frames}:{})},null,2))}</pre></details>`;}
function renderDetail(){const e=eps.find(e=>e.input_index===selected);if(!e){document.getElementById('detail').innerHTML='<p class="muted">请选择一个 episode。</p>';return}
 document.getElementById('detail').innerHTML=`<div class="toolbar"><h2>输入 Episode #${e.input_index}</h2><div><button id="prev">上一集</button> <button id="next">下一集</button></div></div>${badge(e.status)} <b>${esc(e.name)}</b><p class="source muted">${esc(e.source)}</p><p>输出 episode：${esc(e.output_episode_index??'未写入')}　帧数：${e.frames||0}　时长：${fmt((e.frames||0)/fps)} 秒</p>`+
 table(['检查项','结果','详情'],e.checks.map(c=>[esc(c.label),badge(c.status),c.issues.length?c.issues.map(issueDetails).join(''):c.status==='passed'?'检查已执行，未触发对应规则。':'未执行或不适用；可能未开启，或该集在早期检查中被拒绝。']))+
 (e.timeline?.mode==='primary_camera'?`<h3>时间轴</h3><p>以主相机 <b>${esc(e.timeline.primary_camera)}</b> 帧为基准：源帧 ${esc(e.timeline.primary_source_frames)}，补帧 ${esc(e.timeline.filled_frames)}，最长连续掉帧 ${esc(e.timeline.max_primary_drop_frames)}，实测 ${esc(fmt(e.timeline.primary_rate_hz))} Hz（漂移 ${esc(fmt(e.timeline.drift_ppm))} ppm）</p>`:'')+
 '<h3>相机对齐与质量</h3>'+table(['相机','源帧','输出帧','复用','最大误差 ms','质量警告帧'],Object.entries(e.cameras||{}).map(([n,c])=>[esc(n),c.source_frames,c.output_frames,c.reused_frames,fmt(c.max_alignment_gap_ms),c.quality_warning_frames].map(esc)))+
 '<details><summary>各流频率和时间间隔</summary>'+table(['数据流','样本数','中位频率 Hz','最大源间隔 ms'],Object.entries(e.timing||{}).map(([n,t])=>[n,t.count,fmt(t.median_hz),fmt(t.max_gap_ms)].map(esc)))+'</details>';
 document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);
}
function move(offset){const buttons=[...document.querySelectorAll('#episode-list button')],i=buttons.findIndex(b=>Number(b.dataset.index)===selected),b=buttons[i+offset];if(b){selected=Number(b.dataset.index);renderList();document.querySelector('.episode.selected')?.scrollIntoView({block:'nearest'})}}
document.getElementById('episode-list').onclick=e=>{const b=e.target.closest('button[data-index]');if(b){selected=Number(b.dataset.index);renderList()}};
document.getElementById('search').oninput=renderList;document.getElementById('filter').onchange=()=>{checkFilter=null;renderList()};
document.getElementById('verification').innerHTML=table(['检查项','结果'],Object.entries(data.verification||{'正式写出验收':'未执行或未完成'}).map(([k,v])=>[esc(({schema_counts_video_frames:'数据结构、行数及视频帧数',official_loader:'官方加载器首尾帧检查'})[k]||k),typeof v==='boolean'?badge(v?'passed':'not_run'):esc(v)]));
document.getElementById('config').textContent=JSON.stringify({quality:data.config,profile:data.profile},null,2);
renderList();
</script></body></html>'''
