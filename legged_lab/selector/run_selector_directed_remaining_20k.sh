#!/usr/bin/env bash
# Keep the verified LP bank; collect fresh VP and dual banks, 20k each.
# Run with bash; strict shell options apply inside this script, not the terminal.
set -euo pipefail

selector_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$selector_script_dir/../.."

export PYTHONUNBUFFERED=1
export SELECTOR_COLLECTION_PHASE=pilot
case "$SELECTOR_COLLECTION_PHASE" in
    pilot) selector_default_target=20000; selector_default_span=120 ;;
    formal) selector_default_target=100000; selector_default_span=120 ;;
    *) printf 'SELECTOR_COLLECTION_PHASE 必须为 pilot 或 formal。\n' >&2; exit 2 ;;
esac
export SELECTOR_BASE="${SELECTOR_BASE:-$PWD/outputs/g1_renet_selector_v4_1s_normal_20261012_170014/dataset}"
export SELECTOR_TERRAIN="${SELECTOR_TERRAIN:-$(dirname -- "$SELECTOR_BASE")/terrain_distribution.json}"
export SELECTOR_CHECKPOINT="${SELECTOR_CHECKPOINT:-/data/Tienkung_lab/logs/g1_renet/2026-09-10_19-25-08/model_60000.pt}"
export SELECTOR_REFERENCE="${SELECTOR_REFERENCE:-$PWD/Exported_policy/policy.pt}"
export SELECTOR_OUTCOME_RUN="${SELECTOR_OUTCOME_RUN:-$PWD/outputs/g1_renet_selector_directed_${SELECTOR_COLLECTION_PHASE}_$(date +%Y%m%d_%H%M%S)}"
export SELECTOR_TARGET_PER_OUTCOME="${SELECTOR_TARGET_PER_OUTCOME:-$selector_default_target}"
export SELECTOR_SEED_BASE="${SELECTOR_SEED_BASE:-$(date +%s)}"
export SELECTOR_MIN_SPAN_SECONDS="${SELECTOR_MIN_SPAN_SECONDS:-$selector_default_span}"
export SELECTOR_MAX_CANDIDATE_BATCHES="${SELECTOR_MAX_CANDIDATE_BATCHES:-20000}"
export SELECTOR_MAX_NO_PROGRESS_BATCHES="${SELECTOR_MAX_NO_PROGRESS_BATCHES:-500}"
export SELECTOR_SHARD_SIZE="${SELECTOR_SHARD_SIZE:-1024}"
export SELECTOR_COMPLETED_LP_DATASET="${SELECTOR_COMPLETED_LP_DATASET:-$PWD/outputs/g1_renet_selector_directed_20k_20261014_104733/lp_failure/dataset}"
selector_python="${SELECTOR_PYTHON:-python}"
selector_gpu="${SELECTOR_GPU:-1}"

if (( $# != 0 )); then
    printf 'Usage: bash legged_lab/selector/run_selector_directed_remaining_20k.sh\n' >&2
    exit 2
fi

# Refuse incomplete bases, reused output roots and invalid quotas before Kit starts.
"$selector_python" - <<'PY'
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os

def require(condition, message):
    if not condition:
        raise SystemExit(message)

paths = {key: Path(os.environ[key]).expanduser().resolve() for key in (
    'SELECTOR_BASE', 'SELECTOR_TERRAIN', 'SELECTOR_CHECKPOINT',
    'SELECTOR_REFERENCE', 'SELECTOR_OUTCOME_RUN')}
for key, path in paths.items():
    if key == 'SELECTOR_OUTCOME_RUN':
        require(not path.exists(), f'输出目录已存在，请设置新的 SELECTOR_OUTCOME_RUN：{path}')
    elif key != 'SELECTOR_BASE':
        require(path.is_file(), f'找不到文件 {key}：{path}')
manifest_path = paths['SELECTOR_BASE'] / 'manifest.json'
require(manifest_path.is_file(),
        f'主数据尚无 manifest.json，请等主采集完成后再运行：{manifest_path}')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
require(manifest.get('version') == 4 and manifest.get('label_order') == ['lp_1.0s', 'vp_1.0s'],
        '补采基准必须是 V4 两标签、1 秒数据。')
metadata = manifest.get('metadata', {})
require(metadata.get('collection_complete') is True,
        '主采集尚未成功完成，请等主采集完成后再运行。')
require(metadata.get('terrain_mode') == 'main' and not metadata.get('smoke_test'),
        '补采基准必须是正式 main 数据。')
require(manifest.get('counts', {}).get('train', 0) > 0, '主数据没有训练样本。')
terrain_hash = hashlib.sha256(paths['SELECTOR_TERRAIN'].read_bytes()).hexdigest()
require(terrain_hash == metadata.get('terrain_distribution_sha256'),
        '地形分布与主数据不匹配，请使用该主数据原来的 terrain_distribution.json。')
target = int(os.environ['SELECTOR_TARGET_PER_OUTCOME'])
seed_base = int(os.environ['SELECTOR_SEED_BASE'])
span = float(os.environ['SELECTOR_MIN_SPAN_SECONDS'])
candidate_limit = int(os.environ['SELECTOR_MAX_CANDIDATE_BATCHES'])
no_progress_limit = int(os.environ['SELECTOR_MAX_NO_PROGRESS_BATCHES'])
phase = os.environ['SELECTOR_COLLECTION_PHASE']
minimum, min_span = (10000, 30.) if phase == 'pilot' else (100000, 120.)
require(target >= minimum, f'{phase} 每类新增目标至少为 {minimum} 条。')
require(math.isfinite(span) and span >= min_span,
        f'{phase} SELECTOR_MIN_SPAN_SECONDS 至少为 {min_span:g} 秒。')
require(candidate_limit > 0 and span < candidate_limit * .1,
        '候选批次数上限不足以覆盖最小采样时间跨度。')
require(no_progress_limit > 0, '无进展上限必须为正整数。')
require(0 <= seed_base <= 2**32 - 2004, 'SELECTOR_SEED_BASE 超出允许范围。')
info = metadata.get('run_info', {})
base_fault_seed = int(info.get('fault_seed', int(info.get('seed', 42)) + 2000)) & 0xFFFFFFFF
require(base_fault_seed not in {seed_base + i + 2000 for i in (1, 2, 3)},
        '补采故障 seed 与主数据重复，请更换 SELECTOR_SEED_BASE。')
for module in ('torch', 'isaaclab', 'isaacsim'):
    require(importlib.util.find_spec(module) is not None,
            f'当前 Python 找不到 {module}，请切换到运行 g1_renet 的环境。')

def verified_completed_lp(base, target):
    lp = Path(os.environ['SELECTOR_COMPLETED_LP_DATASET']).expanduser().resolve()
    def check(condition, message):
        if not condition:
            raise SystemExit(f'已完成 LP 组验收失败：{message}；保留原目录，不启动重采。')
    inputs = (lp/'manifest.json', lp/'collection_analysis.json', lp.parent/'quota_verification.json')
    check(all(p.is_file() for p in inputs), f'完成记录缺失：{lp}')
    manifest, report, result = [json.loads(p.read_text(encoding='utf-8')) for p in inputs]
    expected = [0, target, 0, 0]
    check(manifest.get('version') == 4 and manifest.get('label_order') == ['lp_1.0s', 'vp_1.0s'], '数据合同不符')
    check(manifest.get('fixed_split') == 'train' and manifest.get('counts') == {'train': target, 'validation': 0, 'test': 0}, '配额或分区不符')
    check(manifest.get('metadata', {}).get('collection_complete') is True and report.get('collection_complete') is True, 'LP 组未完成')
    check(report.get('overall', {}).get('rows') == target and report.get('overall', {}).get('pair_counts') == [expected], '实际标签数量不符')
    directed = report.get('directed_acceptance', {})
    check(directed.get('verified') is True and directed.get('fault_source') == 'lp_only' and directed.get('available_rows') == target, '故障来源验收不符')
    safety = report.get('candidate_safety', {})
    check(safety.get('available_rows') == target and safety.get('invalid_rows') == 0 and safety.get('already_violated_rows') == 0, '起点安全验收不符')
    for field in ('future_metrics', 'candidate_time', 'feature_end_time', 'fault_parameters', 'event_identity', 'first_failure_flags', 'first_failure_cause_stable', 'source_metadata', 'vp_fault_metadata'):
        check(report.get('field_rows', {}).get(field) == target, f'缺少审计字段 {field}')
    base_hash = hashlib.sha256((Path(base)/'manifest.json').read_bytes()).hexdigest()
    check(manifest.get('metadata', {}).get('supplement_source', {}).get('base_manifest_sha256') == base_hash, '原始数据来源不同')
    check(result.get('verified') is True and result.get('source_verified') is True and result.get('bank') == 'lp_failure' and result.get('new_pairs_1s') == expected, '配额验收不符')
    fault_seed = manifest.get('metadata', {}).get('run_info', {}).get('fault_seed')
    check(fault_seed is not None and int(fault_seed) == int(result['fault_seed']), '故障 seed 记录不一致')
    for split in ('train', 'validation', 'test'):
        shards = sorted((lp/split).glob('shard_*.pt'))
        count = int(manifest.get('shards', {}).get(split, -1))
        check(count >= 0 and len(shards) == count, f'{split} 分片数量不符')
        check(all(p.name == f'shard_{i:05d}.pt' and p.stat().st_size > 0 for i,p in enumerate(shards)), f'{split} 分片缺失或为空')
    return {**result, 'dataset': str(lp), 'reused': True}

require(int(os.environ['SELECTOR_SHARD_SIZE']) > 0, 'SELECTOR_SHARD_SIZE 必须为正整数。')
completed_lp = verified_completed_lp(paths['SELECTOR_BASE'], target)
require(int(completed_lp['fault_seed']) not in {seed_base+2002, seed_base+2003}, '故障 seed 与已完成 LP 组重复，请更换 SELECTOR_SEED_BASE。')
print('复用 LP 数据：', completed_lp['dataset'])
old_vp = Path(completed_lp['dataset']).parent.parent/'vp_failure/dataset/train'
print('旧 VP 完整命名分片数量：', len(list(old_vp.glob('shard_*.pt'))), '；旧目录保留，本入口用新 seed 重采。')
print('主数据：', paths['SELECTOR_BASE'])
print('采集阶段：', phase)
print('复用已经验收的 LP 组；本次重采 VP 组和双路组，各', target, '条。')
print('结果配额顺序：[两路安全, 仅LP失败, 仅VP失败, 两路失败]')
PY

mkdir -p -- "$SELECTOR_OUTCOME_RUN/logs"
printf '补采输出：%s\n' "$SELECTOR_OUTCOME_RUN"

for selector_bank in vp_failure both_failure; do
    selector_quotas=(0 0 0 0)
    case "$selector_bank" in
        lp_failure)
            selector_profile=directed_lp
            selector_source=lp_only
            selector_pair_index=1
            selector_seed="$((SELECTOR_SEED_BASE + 1))"
            selector_title='LP失败、VP安全'
            ;;
        vp_failure)
            selector_profile=directed_vp
            selector_source=vp_only
            selector_pair_index=2
            selector_seed="$((SELECTOR_SEED_BASE + 2))"
            selector_title='VP失败、LP安全'
            ;;
        both_failure)
            selector_profile=directed_dual
            selector_source=dual
            selector_pair_index=3
            selector_seed="$((SELECTOR_SEED_BASE + 3))"
            selector_title='两路都失败'
            ;;
    esac
    selector_quotas[selector_pair_index]="$SELECTOR_TARGET_PER_OUTCOME"
    selector_dataset="$SELECTOR_OUTCOME_RUN/$selector_bank/dataset"
    printf '\n开始补采：%s；结果配额=[%s]\n' "$selector_title" "${selector_quotas[*]}"

    CUDA_VISIBLE_DEVICES="$selector_gpu" "$selector_python" -u legged_lab/scripts/collect_selector_data.py \
        --headless --device cuda:0 --task g1_renet --terrain-mode main \
        --checkpoint-path "$SELECTOR_CHECKPOINT" --reference-policy "$SELECTOR_REFERENCE" \
        --terrain-histogram "$SELECTOR_TERRAIN" --base-dataset "$SELECTOR_BASE" \
        --skip-dual-branch-export --num-envs 4096 --supplement-split train \
        --fault-profile "$selector_profile" --fault-source "$selector_source" \
        --fault-severity-probabilities 0.10 0.30 0.60 \
        --pair-quotas "${selector_quotas[@]}" --target-samples "$SELECTOR_TARGET_PER_OUTCOME" \
        --shard-size "$SELECTOR_SHARD_SIZE" \
        --seed "$selector_seed" --warmup-seconds 15 \
        --min-collection-span-seconds "$SELECTOR_MIN_SPAN_SECONDS" \
        --label-interval-frames 10 --randomize-label-phase --max-samples-per-labelled-batch 1024 \
        --max-samples-per-event 8 --min-event-spacing-s 0.2 \
        --max-candidate-batches "$SELECTOR_MAX_CANDIDATE_BATCHES" \
        --max-no-progress-batches "$SELECTOR_MAX_NO_PROGRESS_BATCHES" \
        --heartbeat-seconds 30 --output "$selector_dataset" \
        2>&1 | tee "$SELECTOR_OUTCOME_RUN/logs/collect_${selector_bank}.log"

    # A bank is accepted only after both the manifest and actual label report agree.
    "$selector_python" - "$selector_dataset" "$selector_bank" "$selector_pair_index" "$selector_seed" "$selector_source" <<'PY' \
        | tee "$SELECTOR_OUTCOME_RUN/logs/verify_${selector_bank}.log"
from pathlib import Path
import hashlib
import json
import math
import os
import sys

root = Path(sys.argv[1])
bank, pair_index, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
fault_source = sys.argv[5]
target = int(os.environ['SELECTOR_TARGET_PER_OUTCOME'])
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
report = json.loads((root / 'collection_analysis.json').read_text(encoding='utf-8'))
expected = [0, 0, 0, 0]
expected[pair_index] = target
def require(condition, message):
    if not condition:
        raise SystemExit(f'{bank} 验收失败：{message}')
require(manifest.get('version') == 4 and manifest.get('label_order') == ['lp_1.0s', 'vp_1.0s'],
        '格式或标签不匹配')
require(manifest['metadata'].get('collection_complete') is True, '采集未完成')
require(report.get('collection_complete') is True, '分析报告标记未完成')
require(manifest.get('fixed_split') == 'train', '补采不是训练集')
require(manifest['counts'] == {'train': target, 'validation': 0, 'test': 0}, '样本数或分区不匹配')
require(report['overall']['rows'] == target and report['overall']['pair_counts'] == [expected],
        f'实际失败结果不匹配，目标={expected}，实际={report["overall"]["pair_counts"]}')
source = manifest['metadata']['supplement_source']
base_hash = hashlib.sha256((Path(os.environ['SELECTOR_BASE']) / 'manifest.json').read_bytes()).hexdigest()
require(source['base_manifest_sha256'] == base_hash, '补采来源不匹配')
require(int(manifest['metadata']['run_info']['fault_seed']) == seed + 2000, '故障 seed 不匹配')
coverage = report['supplement_coverage']
require(coverage['accepted_1s'] == expected, '结果配额记录不匹配')
require(coverage['config']['pair_quotas'] == expected, '配置不是指定结果配额')
require(coverage['config'].get('fault_source') == fault_source, '故障来源过滤配置不匹配')
require(coverage['source_acceptance']['written_1s'] == expected, '来源验收后的标签数不匹配')
directed = report.get('directed_acceptance', {})
require(directed.get('verified') is True and directed.get('fault_source') == fault_source
        and directed.get('available_rows') == target, '实际保存样本未全部通过来源与可观测效应验收')
require(report['field_rows'].get('vp_fault_metadata') == target, '缺少新 VP 故障审计字段')
for field in ('future_metrics', 'candidate_time', 'feature_end_time', 'fault_parameters',
              'event_identity', 'first_failure_flags', 'first_failure_cause_stable'):
    require(report['field_rows'].get(field) == target, f'缺少验收字段：{field}')
require(report['candidate_safety']['available_rows'] == target
        and report['candidate_safety']['invalid_rows'] == 0
        and report['candidate_safety']['already_violated_rows'] == 0, '采样起点无效或已经失败')
require(coverage['independent_events'] >= math.ceil(target / 8), '故障事件数量不足')
require(report['events']['plans']['unique'] == coverage['independent_events'], '实际事件数量与覆盖报告不符')
require(report['events']['max_rows_per_plan'] <= 8, '同一父事件超过 8 条')
spacing = report['events']['min_spacing_s_within_plan']
require(spacing is None or spacing >= .2-1e-6, '同一父事件的实际样本间隔小于 0.2 秒')
require(coverage['temporal_coverage']['collection_elapsed_s'] + 1e-6 >= float(os.environ['SELECTOR_MIN_SPAN_SECONDS']),
        '采样时间跨度不足')
result = {'bank': bank, 'dataset': str(root.resolve()), 'new_pairs_1s': expected,
          'fault_seed': seed + 2000, 'events': coverage['independent_events'],
          'fault_source': fault_source, 'phase': os.environ['SELECTOR_COLLECTION_PHASE'],
          'source_verified': True, 'training_only': True, 'verified': True}
(root.parent / 'quota_verification.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('结果配额验收通过：', json.dumps(result, ensure_ascii=False))
PY
done

"$selector_python" - <<'PY'
from pathlib import Path
import json
import os

import hashlib

def verified_completed_lp(base, target):
    lp = Path(os.environ['SELECTOR_COMPLETED_LP_DATASET']).expanduser().resolve()
    def check(condition, message):
        if not condition:
            raise SystemExit(f'已完成 LP 组验收失败：{message}；保留原目录，不启动重采。')
    inputs = (lp/'manifest.json', lp/'collection_analysis.json', lp.parent/'quota_verification.json')
    check(all(p.is_file() for p in inputs), f'完成记录缺失：{lp}')
    manifest, report, result = [json.loads(p.read_text(encoding='utf-8')) for p in inputs]
    expected = [0, target, 0, 0]
    check(manifest.get('version') == 4 and manifest.get('label_order') == ['lp_1.0s', 'vp_1.0s'], '数据合同不符')
    check(manifest.get('fixed_split') == 'train' and manifest.get('counts') == {'train': target, 'validation': 0, 'test': 0}, '配额或分区不符')
    check(manifest.get('metadata', {}).get('collection_complete') is True and report.get('collection_complete') is True, 'LP 组未完成')
    check(report.get('overall', {}).get('rows') == target and report.get('overall', {}).get('pair_counts') == [expected], '实际标签数量不符')
    directed = report.get('directed_acceptance', {})
    check(directed.get('verified') is True and directed.get('fault_source') == 'lp_only' and directed.get('available_rows') == target, '故障来源验收不符')
    safety = report.get('candidate_safety', {})
    check(safety.get('available_rows') == target and safety.get('invalid_rows') == 0 and safety.get('already_violated_rows') == 0, '起点安全验收不符')
    for field in ('future_metrics', 'candidate_time', 'feature_end_time', 'fault_parameters', 'event_identity', 'first_failure_flags', 'first_failure_cause_stable', 'source_metadata', 'vp_fault_metadata'):
        check(report.get('field_rows', {}).get(field) == target, f'缺少审计字段 {field}')
    base_hash = hashlib.sha256((Path(base)/'manifest.json').read_bytes()).hexdigest()
    check(manifest.get('metadata', {}).get('supplement_source', {}).get('base_manifest_sha256') == base_hash, '原始数据来源不同')
    check(result.get('verified') is True and result.get('source_verified') is True and result.get('bank') == 'lp_failure' and result.get('new_pairs_1s') == expected, '配额验收不符')
    fault_seed = manifest.get('metadata', {}).get('run_info', {}).get('fault_seed')
    check(fault_seed is not None and int(fault_seed) == int(result['fault_seed']), '故障 seed 记录不一致')
    for split in ('train', 'validation', 'test'):
        shards = sorted((lp/split).glob('shard_*.pt'))
        count = int(manifest.get('shards', {}).get(split, -1))
        check(count >= 0 and len(shards) == count, f'{split} 分片数量不符')
        check(all(p.name == f'shard_{i:05d}.pt' and p.stat().st_size > 0 for i,p in enumerate(shards)), f'{split} 分片缺失或为空')
    return {**result, 'dataset': str(lp), 'reused': True}

run = Path(os.environ['SELECTOR_OUTCOME_RUN'])
target = int(os.environ['SELECTOR_TARGET_PER_OUTCOME'])
completed_lp = verified_completed_lp(os.environ['SELECTOR_BASE'], target)
new_results = [json.loads((run / bank / 'quota_verification.json').read_text(encoding='utf-8'))
               for bank in ('vp_failure', 'both_failure')]
results = [completed_lp, *new_results]
total = [sum(r['new_pairs_1s'][i] for r in new_results) for i in range(4)]
combined = [sum(r['new_pairs_1s'][i] for r in results) for i in range(4)]
summary = {'base_dataset': os.environ['SELECTOR_BASE'],
           'phase': os.environ['SELECTOR_COLLECTION_PHASE'],
           'pair_order': ['both_safe', 'lp_only_failure', 'vp_only_failure', 'both_failure'],
           'new_pairs_1s': total, 'new_training_rows': sum(total),
           'combined_pairs_1s': combined, 'combined_training_rows': sum(combined),
           'reused_training_rows': target,
           'supplement_datasets': [r['dataset'] for r in results], 'banks': results}
path = run / 'outcome_supplement_summary.json'
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('剩余两组补采完成，新增结果数量：', total)
print('含已完成 LP 组，三组总数量：', combined)
print('训练补采目录：', *summary['supplement_datasets'], sep='\n')
print('汇总：', path)
PY
