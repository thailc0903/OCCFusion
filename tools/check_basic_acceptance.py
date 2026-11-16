#!/usr/bin/env python3
"""Integrated saved-artifact acceptance, reusing camera QA; no model/worker."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import check_web_cameras as cameras


EXTRA=r"""
  // Copy an already evaluated fixture, never execute an evaluator.
  fs.cpSync(path.join(root,'work_dirs/eval_mini_val_scene0103_f1'),path.join(process.env.VIEWER_RUNS_ROOT,'eval_saved_occ'),{recursive:true});
  await page.goto(base+'/?run=occupancy');
  await expect(page.getByTestId('evaluation-panel')).toHaveAttribute('data-status','matched');
  const occ=await(await page.request.get(base+'/api/runs/occupancy')).json();
  const percent=new Intl.NumberFormat('vi-VN',{style:'percent',maximumFractionDigits:4});
  assert.equal(occ.evaluation.status,'matched');assert.equal(occ.sample_token,sample);
  await expect(page.getByTestId('iou-value')).toHaveText(percent.format(occ.evaluation.matched.metrics.occupancy_iou));
  await expect(page.getByTestId('miou-value')).toHaveText(percent.format(occ.evaluation.matched.metrics.semantic_miou));
  await expect(page.locator('canvas')).toHaveAttribute('data-webgl-ready','true');
  await expect.poll(()=>page.getByTestId('visible-count').innerText()).not.toBe('0');
  await expect.poll(()=>page.locator('.bev-image img').evaluate(e=>e.complete&&e.naturalWidth>0)).toBe(true);
  await page.locator('.visual-grid').screenshot({path:path.join(out,'acceptance-occupancy-metric.png')});
  fs.writeFileSync(path.join(out,'occupancy-evidence.json'),JSON.stringify({sample_token:sample,status:occ.evaluation.status,
   metrics:occ.evaluation.matched.metrics,confidence:occ.confidence.note,source:'copied saved prediction/evaluation; no evaluator invoked'},null,2));
  await page.goto(base+'/detections?run=detection');
  await expect(page.getByTestId('detection-token')).toHaveText(sample);
"""

METRICS=r"""
  const benchmark=await(await page.request.get(base+'/api/benchmarks/benchmark_detection_mini_val')).json();
  const metricPercent=new Intl.NumberFormat('vi-VN',{style:'percent',maximumFractionDigits:4});
  assert.equal(benchmark.successful_frame_count,81);assert.equal(benchmark.expected_frame_count,81);
  await expect(page.getByTestId('detection-map')).toHaveText(metricPercent.format(benchmark.metrics.mAP));
  await expect(page.getByTestId('detection-nds')).toHaveText(metricPercent.format(benchmark.metrics.NDS));
  fs.writeFileSync(path.join(out,'detection-metric-evidence.json'),JSON.stringify({status:benchmark.status,
   coverage:[benchmark.successful_frame_count,benchmark.expected_frame_count],metrics:benchmark.metrics,
   latency:benchmark.latency,loader_audit:benchmark.loader_audit},null,2));
"""


def main():
    sign="  await sign(page,'qa_engineer',process.env.CAMERA_PASSWORD);"
    assert cameras.BROWSER.count(sign)==1
    cameras.BROWSER=cameras.BROWSER.replace(sign,sign+EXTRA)
    marker="  await expect(page.getByTestId('detection-map')).toBeVisible();"
    assert cameras.BROWSER.count(marker)==1
    cameras.BROWSER=cameras.BROWSER.replace(marker,marker+METRICS)
    output=ROOT/'work_dirs/basic_acceptance'
    cameras.main(evidence_root=output)
    result=json.loads((output/'browser_evidence.json').read_text());qa=Path(result['qa_directory'])
    result.update(occupancy=json.loads((qa/'occupancy-evidence.json').read_text()),
        detection_benchmark=json.loads((qa/'detection-metric-evidence.json').read_text()),
        acceptance_scope='Engineer saved occupancy/metrics -> detection/camera -> submitted -> Reviewer camera approval -> exact-hash export -> restart; benchmark metrics and frame switch')
    (output/'integrated_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('BASIC_ACCEPTANCE_BROWSER_PASSED')


if __name__=='__main__':main()
