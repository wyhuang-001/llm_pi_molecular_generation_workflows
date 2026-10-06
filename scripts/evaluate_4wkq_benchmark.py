"""Offline evaluator: private labels are NEVER imported by the design workflow.

First freeze exact-assay labels/policy; then evaluate only a completed, sealed run.
Unknown backgrounds remain UNKNOWN, not inactive. No docking/LLM calls here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from rdkit import Chem

from molecular_agent.closed_pool import canonical, sha256
from molecular_agent.pose_retention import save_json
from molecular_agent.structure import ComplexContext

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT/'4WKQ/benchmark-private'


def load_activity_labels(activity_path: Path) -> dict:
    """Validated, structure-keyed same-assay label table."""
    raw = json.loads(activity_path.read_text())
    if raw.get('page_meta', {}).get('next'):
        raise ValueError('Incomplete activity pagination')
    labels = {}
    for row in raw['activities']:
        if (row.get('document_chembl_id') != 'CHEMBL1134804' or row.get('assay_chembl_id') != 'CHEMBL677210'
                or row.get('target_chembl_id') != 'CHEMBL203' or row.get('standard_type') != 'IC50'
                or row.get('standard_relation') != '=' or row.get('standard_units') != 'nM'
                or row.get('data_validity_comment')):
            raise ValueError('Activity source contains unreviewed endpoint/relation/validity flags')
        value = float(row['standard_value'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('IC50 must be finite, positive and exact')
        smiles = canonical(row['canonical_smiles'])
        if smiles in labels:
            raise ValueError('Duplicate structure/assay measurements need an explicit aggregation policy')
        labels[smiles] = {'activity_id': row['activity_id'], 'compound_id': row['molecule_chembl_id'],
                          'ic50_nM': value, 'pIC50': 9-math.log10(value),
                          'potential_duplicate_elsewhere': bool(row.get('potential_duplicate'))}
    return labels


def freeze_labels_multisite(activity_path: Path, task_path: Path, reachability_path: Path,
                            public_dir: Path, out: Path):
    """Freeze a structure-keyed label policy for the cut-point-free pool.

    There is no fragment->compound table here: a compound can be reached through more
    than one site and fragment, so the answer key is the **molecule**.
    """
    if out.exists():
        raise ValueError('Evaluator output must be new; do not revise labels/thresholds after a run')
    labels = load_activity_labels(activity_path)
    reference = Chem.MolToSmiles(ComplexContext(task_path).ligand, isomericSmiles=True)
    if reference not in labels:
        raise ValueError('Same-assay reference is missing')
    coverage = json.loads(reachability_path.read_text())
    covered = {compound: item for compound, item in coverage['compound_detail'].items()
               if item.get('reachable')}
    fold_threshold = 3.0
    reference_value = labels[reference]['ic50_nM']
    by_structure = {}
    for compound, item in covered.items():
        structure = canonical(item['structure'])
        measured = labels.get(structure)
        if measured is None:
            raise ValueError(f'Covered compound {compound} has no same-assay label')
        by_structure[structure] = {'compound_id': compound, 'tier': item['tier'],
                                   'experimental': measured,
                                   'known_high': bool(reference_value/measured['ic50_nM'] >= fold_threshold)}
    if len(by_structure) != len(coverage['product_index']):
        raise ValueError('Product index and covered-compound set disagree')
    policy = {'schema_version': 1, 'role': 'evaluator_only', 'mode': 'multisite',
              'library_sha256': sha256(public_dir/'fragments-multisite.json'),
              'site_table_sha256': sha256(public_dir/'sites-multisite.json'),
              'pool_manifest_sha256': sha256(public_dir/'pool-manifest-multisite.json'),
              'task_sha256': sha256(task_path),
              'reachability_sha256': sha256(reachability_path),
              'activity_source_sha256': sha256(activity_path),
              'assay': 'CHEMBL677210', 'standard_relation': '=', 'units': 'nM',
              'maximum_unique_candidates': coverage.get('maximum_unique_candidates')
                  or json.loads((public_dir/'pool-manifest-multisite.json').read_text())['maximum_unique_candidates'],
              'top_k': [1, 3, 5],
              'known_high_definition': {'minimum_fold_vs_same_assay_reference': fold_threshold},
              'hit_key': 'product_molecule_canonical_smiles',
              'random_repetitions': 10000, 'random_seed': 793410,
              'potential_duplicate_policy': 'retain flag; unique structure/assay records only, never count as independent replication',
              'provenance_limit': 'Exact ChEMBL activity snapshot; not an independent transcription/verification of the paper tables',
              'unknown_policy': 'products outside the measured literature set are unknown, not experimental negatives'}
    out.mkdir(parents=True)
    save_json(out/'policy.json', policy)
    save_json(out/'labels.json', {'reference': labels[reference], 'by_structure': by_structure})
    save_json(out/'seal.json', {'files': {name: sha256(out/name) for name in ('policy.json', 'labels.json')},
                                'evaluator_sha256': sha256(Path(__file__))})
    return {'status': 'frozen', 'mode': 'multisite', 'measured_compounds': len(by_structure),
            'known_high_compounds': sum(v['known_high'] for v in by_structure.values()),
            'activity_records_verified': len(labels)}


def score_product_attempts(attempts: list, by_structure: dict, maximum_unique_candidates: int):
    """Product-keyed scoring rows. The fragment id is deliberately never used."""
    if len(attempts) > maximum_unique_candidates:
        raise ValueError('Candidate budget exceeded')
    rows, seen = [], set()
    for number, item in enumerate(attempts, 1):
        product = canonical(item['product_smiles'])
        if product in seen:
            raise ValueError('Repeated candidate molecule in a completed run')
        seen.add(product)
        label = by_structure.get(product)
        rows.append({
            'proposal_number': number,
            'attempt': item.get('attempt'),
            'product_smiles': product,
            'site_id': item.get('site_id'),
            'change_type': item.get('change_type'),
            'fragment_id': item.get('fragment_id'),
            'in_measured_set': label is not None,
            'experimental': (label or {}).get('experimental'),
            'known_high': bool((label or {}).get('known_high')),
            'compound_id': (label or {}).get('compound_id'),
            'tier': (label or {}).get('tier'),
            'quality': item.get('quality'),
            'pose_passed': item.get('pose_passed'),
            'docking_status': item.get('docking_status'),
        })
    return rows


def freeze_labels(activity_path: Path, membership_path: Path, public_dir: Path, task_path: Path, out: Path):
    if out.exists():
        raise ValueError('Evaluator output must be new; do not revise labels/thresholds after a run')
    membership = json.loads(membership_path.read_text())
    if membership['library_sha256'] != sha256(public_dir/'fragments.json'):
        raise ValueError('Membership/public library mismatch')
    raw = json.loads(activity_path.read_text())
    if raw.get('page_meta', {}).get('next'):
        raise ValueError('Incomplete activity pagination')
    labels = {}
    for row in raw['activities']:
        if (row.get('document_chembl_id') != 'CHEMBL1134804' or row.get('assay_chembl_id') != 'CHEMBL677210'
                or row.get('target_chembl_id') != 'CHEMBL203' or row.get('standard_type') != 'IC50'
                or row.get('standard_relation') != '=' or row.get('standard_units') != 'nM'
                or row.get('data_validity_comment')):
            raise ValueError('Activity source contains unreviewed endpoint/relation/validity flags')
        value = float(row['standard_value'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('IC50 must be finite, positive and exact')
        smiles = canonical(row['canonical_smiles'])
        if smiles in labels:
            raise ValueError('Duplicate structure/assay measurements need an explicit aggregation policy')
        labels[smiles] = {'activity_id': row['activity_id'], 'compound_id': row['molecule_chembl_id'],
                          'ic50_nM': value, 'pIC50': 9-math.log10(value),
                          'potential_duplicate_elsewhere': bool(row.get('potential_duplicate'))}
    reference = Chem.MolToSmiles(ComplexContext(task_path).ligand, isomericSmiles=True)
    if reference not in labels:
        raise ValueError('Same-assay reference is missing')
    # Predeclared BEFORE any model run. This is a benchmark threshold, not a claim of statistical significance.
    fold_threshold = 3.0
    reference_value = labels[reference]['ic50_nM']
    members = {}
    for row in membership['records']:
        measured = labels.get(row['product_smiles'])
        if (row['group'] == 'literature') != (measured is not None):
            raise ValueError('Unexpected measured-background overlap or missing literature label')
        members[row['fragment_id']] = {**row, 'experimental': measured,
            'known_high': bool(measured and reference_value/measured['ic50_nM'] >= fold_threshold)}
    if len(members) != 78 or sum(v['experimental'] is not None for v in members.values()) != 26:
        raise ValueError('Unexpected frozen pool/label coverage')
    policy = {'schema_version': 1, 'role': 'evaluator_only', 'library_sha256': membership['library_sha256'],
              'pool_manifest_sha256': sha256(public_dir/'pool-manifest.json'), 'task_sha256': sha256(task_path),
              'membership_sha256': sha256(membership_path), 'activity_source_sha256': sha256(activity_path),
              'assay': 'CHEMBL677210', 'standard_relation': '=', 'units': 'nM',
              'maximum_unique_candidates': 20, 'top_k': [1, 3, 5],
              'known_high_definition': {'minimum_fold_vs_same_assay_reference': fold_threshold},
              'random_repetitions': 10000, 'random_seed': 793410,
              'potential_duplicate_policy': 'retain flag; unique structure/assay records only, never count as independent replication',
              'provenance_limit': 'Exact ChEMBL activity snapshot; not an independent transcription/verification of the paper tables',
              'unknown_policy': 'unmeasured backgrounds are unknown, not experimental negatives'}
    out.mkdir(parents=True)
    save_json(out/'policy.json', policy)
    save_json(out/'labels.json', {'reference': labels[reference], 'members': members})
    save_json(out/'seal.json', {'files': {name: sha256(out/name) for name in ('policy.json', 'labels.json')},
                              'evaluator_sha256': sha256(Path(__file__))})
    return {'status': 'frozen', 'candidate_count': len(members), 'measured_nonreference_count': 26,
            'unknown_count': 52, 'activity_records_verified': len(labels)}


def stage_metrics(rows):
    measured = [r for r in rows if r['experimental'] is not None]
    hits = [r for r in rows if r['known_high']]
    return {'n': len(rows), 'experimentally_measured_n': len(measured), 'unknown_n': len(rows)-len(measured),
            'experimental_coverage': len(measured)/len(rows) if rows else None,
            'known_high_count': len(hits), 'known_high_yield_all_proposals': len(hits)/len(rows) if rows else None,
            'hit_fraction_among_measured': len(hits)/len(measured) if measured else None,
            'best_measured_ic50_nM': min((r['experimental']['ic50_nM'] for r in measured), default=None)}


def random_reference(members, n, observed, repetitions, seed):
    if not 0 <= n <= len(members):
        raise ValueError('Invalid random sampling budget')
    keys = sorted(members)
    rng = random.Random(seed)
    values = [sum(members[i]['known_high'] for i in rng.sample(keys, n)) for _ in range(repetitions)]
    high = sum(row['known_high'] for row in members.values())
    return {'proposal_level_only': True, 'budget': n, 'repetitions': repetitions,
            'expected_known_high_count_exact': n*high/len(members),
            'known_high_count_quantiles_025_50_975': np.quantile(values, [.025, .5, .975]).tolist(),
            'probability_at_least_one_exact': 1-math.comb(len(keys)-high, n)/math.comb(len(keys), n)
                if n <= len(keys)-high else 1.0,
            'observed_known_high_count': observed,
            'monte_carlo_fraction_at_least_observed': sum(v >= observed for v in values)/repetitions,
            'limitation': 'Uniform proposals without replacement; not a simulated docking/pose filter or experimental significance test'}


def tied_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0]*len(values)
    pos = 0
    while pos < len(order):
        end = pos+1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        for i in order[pos:end]:
            ranks[i] = (pos+end-1)/2 + 1
        pos = end
    return ranks


def evaluate(run: Path, evaluator: Path):
    seal = json.loads((evaluator/'seal.json').read_text())
    for name, value in seal['files'].items():
        if sha256(evaluator/name) != value:
            raise ValueError('Frozen experimental labels/policy changed')
    if sha256(Path(__file__)) != seal['evaluator_sha256']:
        raise ValueError('Evaluator implementation changed after freezing; explicitly version a new evaluator')
    policy = json.loads((evaluator/'policy.json').read_text())
    members = json.loads((evaluator/'labels.json').read_text())['members']
    result_path = run/'result.json'
    if not result_path.exists():
        raise ValueError('Evaluate only a completed run with result.json, never feed live experimental feedback')
    run_manifest = json.loads((run/'closed-pool-run.json').read_text())
    for key in ('library_sha256', 'pool_manifest_sha256', 'task_sha256'):
        if run_manifest[key] != policy[key]:
            raise ValueError(f'Run/evaluator protocol mismatch: {key}')
    if sha256(run/'closed-pool-library.json') != policy['library_sha256']:
        raise ValueError('Run library snapshot changed')
    catalog = json.loads((run/'closed-pool-library.json').read_text())['fragments']
    ids = [row['fragment_id'] for row in catalog]
    requests = sorted(run.glob('closed-pool-request-*.json'))
    for path in requests:
        request = json.loads(path.read_text())
        encoded = json.dumps(request['payload'], ensure_ascii=False, sort_keys=True, allow_nan=False)
        if hashlib.sha256(encoded.encode()).hexdigest() != request['payload_sha256']:
            raise ValueError('Audited payload hash mismatch')
        visible = request['payload']['design_dossier']['fragment_catalog']
        if request['visible_fragment_ids'] != ids or [r['fragment_id'] for r in visible] != ids:
            raise ValueError('Incomplete/changed catalog visibility')
        if any(canonical(r['smiles']) != canonical(members[r['fragment_id']]['smiles']) for r in visible):
            raise ValueError('Wrong fragment chemistry in visible catalog')
    envelope = json.loads(result_path.read_text())
    result = envelope['result']
    attempts = result.get('attempts', [])
    if attempts and not requests:
        raise ValueError('Missing designer request/exposure audit')
    if len(attempts) > policy['maximum_unique_candidates']:
        raise ValueError('Candidate budget exceeded')
    if not attempts and result.get('status') == 'reference_calibration_failed':
        return {'status': 'not_evaluable', 'reason': 'Reference calibration failed before design'}
    history = {r['attempt']: r for r in result.get('docking_history', envelope.get('state', {}).get('docking_history', []))}
    rows, seen = [], set()
    for number, report in enumerate(attempts, 1):
        transformation = report['transformation']
        fragment_id = transformation['fragment_id']
        if fragment_id not in members or fragment_id in seen:
            raise ValueError('Out-of-pool or repeated candidate in completed run')
        seen.add(fragment_id)
        member = members[fragment_id]
        if (transformation.get('change_type') != 'replacement' or transformation.get('cut_bond') != [10, 9]
                or transformation.get('parent_attempt') is not None or transformation.get('generation', 1) != 1
                or canonical(transformation['fragment_smiles']) != canonical(member['smiles'])):
            raise ValueError('Attempt violates original-ligand/C6/catalog contract')
        validation = report.get('validation', {})
        built_smiles = validation.get('candidate', {}).get('canonical_smiles')
        if built_smiles and canonical(built_smiles) != member['product_smiles']:
            raise ValueError('Host constructed a different full molecular graph than the selected pool candidate')
        docking = report.get('docking', {})
        pose_passed = docking.get('status') == 'complete' and docking.get('pose_retention', {}).get('status') == 'passed'
        quality = history.get(report['attempt'], {}).get('quality')
        if quality is not None and (not pose_passed or not math.isfinite(float(quality))):
            raise ValueError('Invalid ranked quality for a failed/unavailable pose')
        delta = docking.get('comparison', {}).get('metrics', {}).get('minimizedAffinity', {}).get('delta_candidate_minus_reference', {}).get('mean')
        rows.append({'proposal_number': number, 'attempt': report['attempt'], 'fragment_id': fragment_id,
                     'experimental': member['experimental'], 'known_high': member['known_high'],
                     'constructed': bool(built_smiles), 'pre_docking_eligible': validation.get('status') == 'accepted',
                     'initial_clash_deferred': validation.get('initial_receptor_clash_check', {}).get('status') == 'deferred_to_docking',
                     'geometry_passed': validation.get('status') == 'accepted' and not validation.get('initial_receptor_clash_check'),
                     'pose_passed': pose_passed, 'quality': quality,
                     'docking_improvement': -float(delta) if pose_passed and delta is not None else None,
                     'docking_status': docking.get('status')})
    ranked = sorted([r for r in rows if r['quality'] is not None], key=lambda r: (-r['quality'], r['attempt']))
    best = result.get('best_attempt')
    if best is not None and (not ranked or ranked[0]['attempt'] != best):
        raise ValueError('Reported final best is inconsistent with frozen quality ranking')
    correlation_rows = [r for r in rows if r['experimental'] and r['docking_improvement'] is not None]
    spearman = None
    if len(correlation_rows) >= 3:
        x = tied_ranks([r['docking_improvement'] for r in correlation_rows])
        y = tied_ranks([r['experimental']['pIC50'] for r in correlation_rows])
        if np.std(x) > 0 and np.std(y) > 0:
            spearman = float(np.corrcoef(x, y)[0, 1])
    observed = sum(r['known_high'] for r in rows)
    return {'status': 'complete', 'run_result_sha256': sha256(result_path), 'task_sha256': policy['task_sha256'],
            'library_sha256': policy['library_sha256'], 'evaluation_policy_sha256': sha256(evaluator/'policy.json'),
            'design_requests': len(requests), 'unique_proposals': len(rows),
            'first_known_high_proposal': next((r['proposal_number'] for r in rows if r['known_high']), None),
            'proposals': stage_metrics(rows), 'constructed': stage_metrics([r for r in rows if r['constructed']]),
            'initial_clash_free': stage_metrics([r for r in rows if r['geometry_passed']]),
            'initial_clash_deferred': stage_metrics([r for r in rows if r['initial_clash_deferred']]),
            'pre_docking_eligible': stage_metrics([r for r in rows if r['pre_docking_eligible']]),
            'pose_passed': stage_metrics([r for r in rows if r['pose_passed']]),
            'final_top_k': {str(k): stage_metrics(ranked[:k]) for k in policy['top_k']},
            'score_experiment_correlation': {'n': len(correlation_rows), 'spearman': spearman,
                'scope': 'Observed, measured, pose-passing subset only; not an unbiased full-series correlation'},
            'random_same_actual_budget': random_reference(members, len(rows), observed, policy['random_repetitions'], policy['random_seed']),
            'random_fixed_maximum_budget': random_reference(members, policy['maximum_unique_candidates'], observed,
                                                             policy['random_repetitions'], policy['random_seed']),
            'rows': rows, 'warnings': [policy['unknown_policy'], policy['provenance_limit'],
                'Known structures are deliberately present: this is label-blind retrospective selection, not structure-held-out discovery.',
                'Public drug/scaffold knowledge may be present in model pretraining.',
                'Candidate construction and charge/stereochemical microstate limitations still apply; do not infer experimental activity for unmeasured isomers.']}


def enumerate_pool_products(task_path: Path, public_dir: Path):
    """Every molecule the closed pool can build, with its measured label if known.

    Used only as the null distribution for the random baseline.  It is the exact
    proposal space of the pool, not a docking or pose simulation.
    """
    from molecular_agent.editing import transformation_product_smiles
    from molecular_agent.fragment_library import FragmentLibrary
    from molecular_agent.tools import ToolRegistry

    context = ComplexContext(task_path)
    library = FragmentLibrary(public_dir / 'fragments-multisite.json')
    tools = ToolRegistry(context, library)
    parent = Chem.Mol(context.ligand)
    products = {}
    for site in tools.list_bond_sites(limit=200)['sites']:
        for change_type in site.get('allowed_change_types') or []:
            base = {'site_type': 'bond', 'change_type': change_type,
                    'cut_bond': list(site['cut_bond']),
                    'edit_atom_index': site['retained_atom_index']}
            combos = ([None] if change_type == 'deletion'
                      else [{'fragment_id': record['fragment_id'],
                             'fragment_smiles': record['smiles']} for record in library.records])
            for combo in combos:
                try:
                    products[canonical(transformation_product_smiles(parent, {**base, **(combo or {})}))] = True
                except Exception:
                    continue
    for site in tools.get_edit_site_candidates()['atom_sites']:
        for change_type in site.get('allowed_change_types') or []:
            if change_type != 'addition':
                continue
            for record in library.records:
                try:
                    products[canonical(transformation_product_smiles(parent, {
                        'site_type': 'atom', 'change_type': 'addition',
                        'edit_atom_index': site['target_id'],
                        'fragment_id': record['fragment_id'],
                        'fragment_smiles': record['smiles']}))] = True
                except Exception:
                    continue
    native = canonical(Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(context.ligand)), isomericSmiles=True))
    products.pop(native, None)
    return products


def evaluate_multisite(run: Path, evaluator: Path, task: Path, public_dir: Path) -> dict:
    seal = json.loads((evaluator / 'seal.json').read_text())
    for name, value in seal['files'].items():
        if sha256(evaluator / name) != value:
            raise ValueError('Frozen experimental labels/policy changed')
    if sha256(Path(__file__)) != seal['evaluator_sha256']:
        raise ValueError('Evaluator implementation changed after freezing; explicitly version a new evaluator')
    policy = json.loads((evaluator / 'policy.json').read_text())
    labels = json.loads((evaluator / 'labels.json').read_text())
    by_structure = labels['by_structure']

    result_path = run / 'result.json'
    if not result_path.exists():
        raise ValueError('Evaluate only a completed run with result.json')
    run_manifest = json.loads((run / 'closed-pool-run.json').read_text())
    for key in ('library_sha256', 'task_sha256'):
        if run_manifest[key] != policy[key]:
            raise ValueError(f'Run/evaluator protocol mismatch: {key}')
    if sha256(run / 'closed-pool-library.json') != policy['library_sha256']:
        raise ValueError('Run library snapshot changed')

    envelope = json.loads(result_path.read_text())
    result = envelope['result']
    reports = result.get('attempts', [])
    requests = sorted(run.glob('closed-pool-request-*.json'))
    if reports and not requests:
        raise ValueError('Missing designer request/exposure audit')
    catalog = json.loads((run / 'closed-pool-library.json').read_text())['fragments']
    catalog_ids = [row['fragment_id'] for row in catalog]
    for path in requests:
        request = json.loads(path.read_text())
        visible = request['payload']['design_dossier']['fragment_catalog']
        if [row['fragment_id'] for row in visible] != catalog_ids:
            raise ValueError('Incomplete/changed catalog visibility')

    history = {row['attempt']: row for row in result.get(
        'docking_history', envelope.get('state', {}).get('docking_history', []))}
    attempts = []
    for report in reports:
        transformation = report['transformation']
        validation = report.get('validation', {})
        docking = report.get('docking', {})
        built = (validation.get('candidate') or {}).get('canonical_smiles')
        if not built:
            continue
        try:
            rederived = canonical(Chem.MolToSmiles(
                Chem.MolFromSmiles(built), isomericSmiles=True))
        except Exception as error:
            raise ValueError(f'Unreadable built candidate molecule: {error}') from error
        pose_passed = (docking.get('status') == 'complete'
                       and docking.get('pose_retention', {}).get('status') == 'passed')
        attempts.append({
            'attempt': report.get('attempt'),
            'product_smiles': rederived,
            'site_id': transformation.get('bond_site_id') or transformation.get('edit_atom_index'),
            'change_type': transformation.get('change_type'),
            'fragment_id': transformation.get('fragment_id'),
            'quality': history.get(report.get('attempt'), {}).get('quality'),
            'pose_passed': pose_passed,
            'docking_status': docking.get('status'),
            'in_pool_fragment': (transformation.get('fragment_id') in catalog_ids
                                 if transformation.get('fragment_id') else True),
        })
    rows = score_product_attempts(attempts, by_structure, policy['maximum_unique_candidates'])
    ranked = sorted([row for row in rows if row['quality'] is not None],
                    key=lambda row: (-float(row['quality']), row['attempt']))
    observed = sum(row['known_high'] for row in rows)
    measured_products = {structure: True for structure in by_structure}
    pool_products = enumerate_pool_products(task, public_dir)
    null_space = {
        product: {'known_high': bool((by_structure.get(product) or {}).get('known_high'))}
        for product in sorted(set(pool_products) | set(measured_products))
    }
    return {
        'status': 'complete', 'mode': 'multisite',
        'run_result_sha256': sha256(result_path),
        'task_sha256': policy['task_sha256'],
        'library_sha256': policy['library_sha256'],
        'evaluation_policy_sha256': sha256(evaluator / 'policy.json'),
        'hit_key': policy['hit_key'],
        'design_requests': len(requests),
        'unique_proposals': len(rows),
        'pool_product_space': len(null_space),
        'first_known_high_proposal': next((row['proposal_number'] for row in rows if row['known_high']), None),
        'proposals': stage_metrics(rows),
        'pose_passed': stage_metrics([row for row in rows if row['pose_passed']]),
        'final_top_k': {str(k): stage_metrics(ranked[:k]) for k in policy['top_k']},
        'random_same_actual_budget': random_reference(
            null_space, len(rows), observed, policy['random_repetitions'], policy['random_seed']),
        'rows': rows,
        'warnings': [policy['unknown_policy'], policy['provenance_limit'],
                     'Known structures are deliberately present: this is label-blind retrospective '
                     'selection, not structure-held-out discovery.',
                     'The random baseline samples the pool product space uniformly; it does not '
                     'simulate docking, pose filtering or experimental significance.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--activities', type=Path, default=PRIVATE/'activity-source-v1/activities.json')
    freeze.add_argument('--membership', type=Path, default=PRIVATE/'pool-v1/membership.json')
    freeze.add_argument('--public-dir', type=Path, default=ROOT/'4WKQ/design')
    freeze.add_argument('--task', type=Path, default=ROOT/'4WKQ/task.benchmark.json')
    freeze.add_argument('--output-dir', type=Path, default=PRIVATE/'evaluator-v1')
    freeze_ms = sub.add_parser('freeze-multisite')
    freeze_ms.add_argument('--activities', type=Path, default=PRIVATE/'activity-source-v1/activities.json')
    freeze_ms.add_argument('--task', type=Path, default=ROOT/'4WKQ/task.benchmark-multisite.json')
    freeze_ms.add_argument('--reachability', type=Path, default=PRIVATE/'reachability-multisite.json')
    freeze_ms.add_argument('--public-dir', type=Path, default=ROOT/'4WKQ/design')
    freeze_ms.add_argument('--output-dir', type=Path, default=PRIVATE/'evaluator-multisite-v1')
    score = sub.add_parser('evaluate')
    score.add_argument('--run-dir', type=Path, required=True)
    score.add_argument('--evaluator-dir', type=Path, default=PRIVATE/'evaluator-v1')
    score.add_argument('--output', type=Path, required=True)
    score.add_argument('--multisite', action='store_true',
                       help='product-keyed scoring for the cut-point-free pool')
    score_ms = sub.add_parser('evaluate-multisite')
    score_ms.add_argument('--run-dir', type=Path, required=True)
    score_ms.add_argument('--evaluator-dir', type=Path, default=PRIVATE/'evaluator-multisite-v1')
    score_ms.add_argument('--task', type=Path, default=ROOT/'4WKQ/task.benchmark-multisite.json')
    score_ms.add_argument('--public-dir', type=Path, default=ROOT/'4WKQ/design')
    score_ms.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'freeze':
        report = freeze_labels(args.activities, args.membership, args.public_dir, args.task, args.output_dir)
        print(json.dumps(report, indent=2))  # Never print the answer key/high-activity IDs here.
    elif args.mode == 'freeze-multisite':
        report = freeze_labels_multisite(args.activities, args.task, args.reachability,
                                         args.public_dir, args.output_dir)
        print(json.dumps(report, indent=2))  # Never print the answer key/high-activity IDs here.
    elif args.mode == 'evaluate-multisite':
        output = args.output.resolve()
        if output.exists() or output.is_relative_to(args.run_dir.resolve()) or output.is_relative_to((ROOT/'4WKQ/design').resolve()):
            parser.error('Use a NEW evaluator-only output OUTSIDE the design run/public inputs')
        report = evaluate_multisite(args.run_dir.resolve(), args.evaluator_dir.resolve(),
                                    args.task.resolve(), args.public_dir.resolve())
        save_json(output, report)
        print('Evaluation saved outside designer inputs:', output)
    else:
        output = args.output.resolve()
        if output.exists() or output.is_relative_to(args.run_dir.resolve()) or output.is_relative_to((ROOT/'4WKQ/design').resolve()):
            parser.error('Use a NEW evaluator-only output OUTSIDE the design run/public inputs')
        report = evaluate(args.run_dir.resolve(), args.evaluator_dir.resolve())
        save_json(output, report)
        print('Evaluation saved outside designer inputs:', output)


if __name__ == '__main__':
    main()
