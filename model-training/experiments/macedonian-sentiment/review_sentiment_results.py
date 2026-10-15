"""Build an offline error-review page from completed test predictions, without inference."""

import argparse
import hashlib
import json
from pathlib import Path

LABELS = ["negative", "neutral", "positive"]

if not __debug__:
    raise RuntimeError(
        "Run without -O or PYTHONOPTIMIZE; integrity checks must stay enabled."
    )


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_review(results, output):
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["split"] == "test"
    aliases = list(summary["models"])
    assert len(aliases) == 6
    predictions = {}
    analysis = {
        "dataset_version": summary["dataset_version"],
        "summary_sha256": checksum(summary_path),
        "scope": "Post-test error analysis; no model fitting, selection, or relabelling.",
        "models": {},
    }
    reference = None
    for alias, report in summary["models"].items():
        path = results / alias / "predictions.jsonl"
        assert checksum(path) == report["predictions_sha256"]
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == summary["examples"]
        identities = [(row["id"], row["text_mk"], row["gold"]) for row in rows]
        if reference is None:
            reference = identities
        assert identities == reference
        confusion = [[0] * 3 for _ in LABELS]
        for row in rows:
            confusion[LABELS.index(row["gold"])][LABELS.index(row["predicted"])] += 1
        assert confusion == report["raw"]["confusion_matrix"]
        errors = [row for row in rows if row["gold"] != row["predicted"]]
        analysis["models"][alias] = {
            "errors": len(errors),
            "class_recall": {
                label: confusion[i][i] / sum(confusion[i])
                for i, label in enumerate(LABELS)
            },
            "wrong_at_least_95_percent_raw": sum(
                max(row["probabilities_raw"]) >= 0.95 for row in errors
            ),
        }
        if "calibrated" in report:
            analysis["models"][alias]["wrong_at_least_95_percent_calibrated"] = sum(
                max(row["probabilities_calibrated"]) >= 0.95 for row in errors
            )
        predictions[alias] = rows
    first, second = predictions["multilingual-full"], predictions["multilingual-frozen"]
    analysis["multilingual_full_vs_frozen"] = {
        "full_correct_frozen_wrong": sum(
            a["predicted"] == a["gold"] and b["predicted"] != b["gold"]
            for a, b in zip(first, second)
        ),
        "frozen_correct_full_wrong": sum(
            b["predicted"] == b["gold"] and a["predicted"] != a["gold"]
            for a, b in zip(first, second)
        ),
    }
    data = {
        "labels": LABELS,
        "aliases": aliases,
        "summary": summary,
        "analysis": analysis,
        "rows": [],
    }
    for index, (row_id, text, gold) in enumerate(reference):
        outcomes = []
        for alias in aliases:
            row = predictions[alias][index]
            outcomes.append(
                {
                    "prediction": row["predicted"],
                    "raw": row["probabilities_raw"],
                    "calibrated": row.get("probabilities_calibrated"),
                }
            )
        data["rows"].append(
            {"id": row_id, "text": text, "gold": gold, "outcomes": outcomes}
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    # Escape '<' so tweet text cannot terminate the JSON script element.
    embedded = json.dumps(
        data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("<", "\\u003c")
    (output / "review.html").write_text(
        TEMPLATE.replace("__DATA__", embedded), encoding="utf-8"
    )
    print("Verified all six prediction files and confusion matrices.")
    print("Review:", (output / "review.html").resolve())
    print(
        "Original result files unchanged. Human notes remain separate from dataset labels."
    )


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laya Macedonian sentiment — error review</title>
<style>
:root{color-scheme:light dark;--bg:#f5f6f8;--fg:#19232d;--panel:#fff;--muted:#586575;--line:#d5dce4;--accent:#245bb5;--bad:#982727;--good:#17664f}
@media(prefers-color-scheme:dark){:root{--bg:#111720;--fg:#e3eaf2;--panel:#1a2532;--muted:#b0becc;--line:#3c4b5d;--accent:#8fb8ff;--bad:#ffaaaa;--good:#9fd9c7}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:28px 20px 60px}h1{font-size:29px;line-height:1.25;margin:0 0 12px}h2{font-size:20px;margin:0 0 12px}p{margin:10px 0}.muted{color:var(--muted)}.intro{max-width:880px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;margin-top:20px}.controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}label{display:block;font-weight:600}select,input,button,textarea{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px;max-width:100%}select,input,textarea{width:100%}button{cursor:pointer}button:disabled{cursor:default;opacity:.5}.check{display:flex;align-items:center;gap:8px;font-weight:400}.check input{width:auto}.stats,.models{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.stats{margin-top:18px}.stat strong{font-size:23px;display:block}.stat p{margin:0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line)}th{font-size:14px}td button{width:100%;padding:8px;border:0;font-variant-numeric:tabular-nums}.diagonal{background:color-mix(in srgb,var(--good) 14%,var(--panel))}.offdiag{background:color-mix(in srgb,var(--bad) 9%,var(--panel))}.tweet{margin:18px 0 8px;padding:20px;background:var(--panel);border:1px solid var(--line);border-radius:12px}.tweet-text{font-size:19px;overflow-wrap:anywhere;margin-top:8px}.tweet-meta{font-size:13px;color:var(--muted);overflow-wrap:anywhere}.correct{color:var(--good)}.incorrect{color:var(--bad)}.model-name{font-weight:600;margin-bottom:6px}.probability{display:grid;grid-template-columns:78px 1fr 58px;align-items:center;gap:8px;font-size:14px;margin:4px 0}.track{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.fill{height:100%;background:var(--accent)}.notes{display:grid;grid-template-columns:minmax(180px,1fr) 2fr;gap:12px;margin-top:16px}.notes label{font-size:14px}.footer{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:18px}.footer span{margin-right:auto}.warning{border-left:3px solid var(--accent);padding-left:12px}.note-status{min-height:22px;font-size:14px}.badge{font-size:14px}button:focus-visible,select:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid var(--accent);outline-offset:2px}@media(max-width:640px){main{padding:20px 12px}.controls,.stats,.models,.notes{grid-template-columns:1fr}.tweet{padding:14px}.tweet-text{font-size:17px}h1{font-size:25px}}
</style></head><body><main>
<h1>Laya: inspect the Macedonian sentiment errors</h1>
<p class="intro">These are the 1,139 test tweets from your completed experiment. Compare two models and read individual mistakes. <strong>Dataset labels are reference labels, not guaranteed truth.</strong> Review notes do not change labels or model results.</p>
<p class="muted intro warning">This is post-test diagnosis. The initial view selects wrong predictions with at least 95% confidence after calibration; it is a deliberately difficult subset, not a representative accuracy estimate. Changes motivated by this review belong to a new experiment.</p>
<section class="panel"><h2>Choose what to inspect</h2><div class="controls">
<label>Primary model<select id="primary"></select></label><label>Comparison model<select id="comparison"></select></label>
<label>Cases<select id="cases"><option value="wrong">Primary model is wrong</option><option value="rescued">Primary wrong, comparison correct</option><option value="both">Both models are wrong</option><option value="all">All tweets</option></select></label>
<label>Reference sentiment<select id="gold"><option value="all">All classes</option><option>negative</option><option>neutral</option><option>positive</option></select></label>
<label>Minimum primary confidence (%)<input id="minimum" type="number" min="0" max="100" step="1" value="95"></label>
<label>Primary predicted sentiment<select id="predicted"><option value="all">All predictions</option><option>negative</option><option>neutral</option><option>positive</option></select></label>
<label class="check"><input id="calibrated" type="checkbox" checked>Use calibrated probabilities when available; bases always use raw probabilities</label>
<label class="check"><input id="unreviewed" type="checkbox">Show only tweets without a review judgment</label>
</div><div class="stats" id="stats"></div></section>
<section class="panel"><h2 id="matrix-title">Primary model: confusion matrix</h2><p class="muted">Rows are reference labels; columns are model predictions. Click a cell to inspect that kind of outcome. Counts cover the full test set, regardless of the filters above.</p><div class="scroll" id="matrix"></div><button id="reset" type="button">Clear class filters and confidence threshold</button></section>
<div class="footer"><span id="count" aria-live="polite"></span><button id="export" type="button">Export review notes as CSV</button></div>
<p class="note-status" id="note-status" aria-live="polite">Notes are saved in this browser when local storage is available. Export CSV to keep a separate copy.</p>
<div id="tweets"></div><div class="footer"><button id="previous" type="button">Previous 20</button><span id="page" aria-live="polite"></span><button id="next" type="button">Next 20</button></div>
<p class="muted">Source: fixed test reports. Scalar temperature scaling changes probabilities, not sentiment predictions. Calibration quality here is measured on test, not on its fitting cohort. This page runs offline and sends no requests.</p>
</main><script type="application/json" id="data">__DATA__</script><script>
(()=>{'use strict';
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const storageKey='laya-sentiment-review:'+data.summary.dataset_version;let notes={},page=0,visible=[];
try{notes=JSON.parse(localStorage.getItem(storageKey)||'{}');if(notes===null||Array.isArray(notes)||typeof notes!=='object')throw new Error('Invalid saved notes');}catch(error){notes={};$('note-status').textContent='Saved notes could not be loaded. Export CSV before closing this page.';}
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function outcome(row,alias){return row.outcomes[data.aliases.indexOf(alias)];}
function probs(result){return $('calibrated').checked&&result.calibrated?result.calibrated:result.raw;}
function confidence(result){return Math.max(...probs(result));}
function save(){try{localStorage.setItem(storageKey,JSON.stringify(notes));$('note-status').textContent='Notes saved in this browser. Export CSV to keep a separate copy.';}catch(error){$('note-status').textContent='Notes are in memory only. Export CSV before closing the page.';}}
data.aliases.forEach(alias=>{for(const id of ['primary','comparison']){$(id).append(el('option',alias));}});
$('primary').value='multilingual-full';$('comparison').value='multilingual-frozen';
function renderMatrix(){const alias=$('primary').value,m=data.summary.models[alias].raw.confusion_matrix;const table=el('table'),head=el('tr');head.append(el('th','Reference ↓ / prediction →'));data.labels.forEach(label=>head.append(el('th',label)));const thead=el('thead');thead.append(head);table.append(thead);const body=el('tbody');m.forEach((row,i)=>{const tr=el('tr');tr.append(el('th',data.labels[i]));row.forEach((count,j)=>{const cell=el('td'),button=el('button',String(count),i===j?'diagonal':'offdiag');button.type='button';button.setAttribute('aria-label',`${data.labels[i]} reference, ${data.labels[j]} prediction: ${count} tweets`);button.addEventListener('click',()=>{$('gold').value=data.labels[i];$('predicted').value=data.labels[j];$('minimum').value='0';$('cases').value='all';page=0;render();});cell.append(button);tr.append(cell);});body.append(tr);});table.append(body);$('matrix').replaceChildren(table);$('matrix-title').textContent=alias+': confusion matrix';}
function modelPanel(row,alias){const result=outcome(row,alias),p=probs(result),correct=result.prediction===row.gold;const panel=el('div');panel.append(el('div',alias,'model-name'));panel.append(el('p',`${result.prediction} · ${(Math.max(...p)*100).toFixed(1)}% confidence · ${correct?'correct':'wrong'}`,correct?'correct':'incorrect'));panel.append(el('div',$('calibrated').checked&&result.calibrated?'After task calibration':'Raw probabilities','muted badge'));data.labels.forEach((label,index)=>{const line=el('div',undefined,'probability'),track=el('div',undefined,'track'),fill=el('div',undefined,'fill');fill.style.width=(p[index]*100)+'%';track.append(fill);line.append(el('span',label),track,el('span',(p[index]*100).toFixed(1)+'%'));panel.append(line);});return panel;}
function reviewPanel(row){const grid=el('div',undefined,'notes'),ratingLabel=el('label','Your judgment of the reference label'),select=el('select'),noteLabel=el('label','Why? (optional)'),textarea=el('textarea');textarea.rows=2;textarea.placeholder='Sarcasm? Dialect? Label ambiguity? Missing context?';for(const [value,text] of [['','Not reviewed'],['plausible','Reference label seems plausible'],['unclear','Unclear / needs context'],['questionable','Reference label is questionable']]){const option=el('option',text);option.value=value;select.append(option);}select.value=notes[row.id]?.judgment||'';textarea.value=notes[row.id]?.note||'';function record(){notes[row.id]={judgment:select.value,note:textarea.value};save();}select.addEventListener('change',record);textarea.addEventListener('input',record);ratingLabel.append(select);noteLabel.append(textarea);grid.append(ratingLabel,noteLabel);return grid;}
function render(){const primary=$('primary').value,comparison=$('comparison').value,minimum=Math.min(100,Math.max(0,Number($('minimum').value)||0))/100;visible=data.rows.filter(row=>{const a=outcome(row,primary),b=outcome(row,comparison),wrong=a.prediction!==row.gold,mode=$('cases').value;return (mode==='all'||mode==='wrong'&&wrong||mode==='rescued'&&wrong&&b.prediction===row.gold||mode==='both'&&wrong&&b.prediction!==row.gold)&&($('gold').value==='all'||row.gold===$('gold').value)&&($('predicted').value==='all'||a.prediction===$('predicted').value)&&confidence(a)>=minimum&&(!$('unreviewed').checked||!notes[row.id]?.judgment);});const pages=Math.max(1,Math.ceil(visible.length/20));page=Math.min(page,pages-1);$('count').textContent=`${visible.length} matching tweets / ${data.rows.length} test tweets`;$('page').textContent=`Page ${page+1} of ${pages}`;$('previous').disabled=page===0;$('next').disabled=page+1>=pages;const stats=el('div',undefined,'stat'),stats2=el('div',undefined,'stat');const report=data.summary.models[primary],metrics=$('calibrated').checked&&report.calibrated?report.calibrated:report.raw;stats.append(el('strong',(metrics.accuracy*100).toFixed(1)+'%'),el('p','Primary accuracy on the entire test set'));stats2.append(el('strong',(metrics.mean_confidence*100).toFixed(1)+'%'),el('p','Primary mean confidence on the entire test set'));$('stats').replaceChildren(stats,stats2);renderMatrix();const cards=visible.slice(page*20,(page+1)*20).map(row=>{const card=el('article',undefined,'tweet');card.append(el('div',row.id,'tweet-meta'),el('p','Reference label: '+row.gold),el('p',row.text,'tweet-text'));const models=el('div',undefined,'models');models.append(modelPanel(row,primary),modelPanel(row,comparison));card.append(models,reviewPanel(row));return card;});if(!cards.length)cards.push(el('p','No tweets match these filters. Lower the confidence threshold or change the case filter.'));$('tweets').replaceChildren(...cards);}
for(const id of ['primary','comparison','cases','gold','predicted','minimum','calibrated','unreviewed'])$(id).addEventListener('change',()=>{page=0;render();});
$('previous').addEventListener('click',()=>{page--;render();});$('next').addEventListener('click',()=>{page++;render();});$('reset').addEventListener('click',()=>{$('gold').value='all';$('predicted').value='all';$('minimum').value='0';$('cases').value='all';page=0;render();});
$('export').addEventListener('click',()=>{const quote=value=>{let text=String(value??'');if(/^[=+@-]/.test(text))text="'"+text;return '"'+text.replaceAll('"','""')+'"';};const rows=[['id','reference_label','review_judgment','review_note','dataset_version']];for(const row of data.rows){const note=notes[row.id];if(note?.judgment||note?.note)rows.push([row.id,row.gold,note.judgment,note.note,data.summary.dataset_version]);}const blob=new Blob(['\ufeff'+rows.map(row=>row.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),link=el('a');link.href=url;link.download='sentiment-error-review.csv';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
render();})();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_review(args.results, args.output)


if __name__ == "__main__":
    main()
