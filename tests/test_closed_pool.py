"""No real experimental activities or high-activity IDs are used by these tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem

from molecular_agent.closed_pool import assemble, canonical, scaffold, sha256
from molecular_agent.adapters import DockingAdapter
from molecular_agent.pose_retention import PoseRetention, heavy_steric_check
from scripts.audit_4wkq_benchmark import decompose
from molecular_agent.structure import ComplexContext
from molecular_agent.workflow import Workflow
from scripts.build_4wkq_pool import compatible, features, select_pool
from scripts.evaluate_4wkq_benchmark import evaluate, freeze_labels, random_reference, tied_ranks
from scripts.run_4wkq_baseline import PublicBaselineClient

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'4WKQ/task.benchmark.json'
PUBLIC = ROOT/'4WKQ/design'


def make_workflow(tmp_path, client=None):
    w = Workflow(TASK, client or object(), tmp_path)
    w._prepare_initial_context()
    return w


def decision(w, fragment_id=None):
    return {'action': 'READY', 'understanding': 'Fixed core and C6 site', 'edit_hypothesis': 'Test catalog selection',
            'operation': 'bond:replacement', 'bond_site_id': w.closed_pool.site['bond_site_id'],
            'fragment_id': fragment_id or w.closed_pool.catalog[0]['fragment_id']}


def test_frozen_pool_is_78_unique_reconstructable_public_chemistries(tmp_path):
    w = make_workflow(tmp_path)
    assert len(w.closed_pool.products) == len(set(w.closed_pool.products.values())) == 78
    for record in w.tools.fragment_library.records:
        assert set(record) == {'fragment_id', 'smiles', 'allowed_operations'}
    assert sha256(PUBLIC/'fragments.json') == w.closed_pool.manifest['library_sha256']
    assert all(Chem.MolFromSmiles(s) is not None for s in w.closed_pool.products.values())


def test_all_78_entries_visible_without_heuristic_panels_or_private_context(tmp_path):
    w = make_workflow(tmp_path)
    for _ in range(2):
        payload = w._direct_payload('choose')
        assert len(payload['design_dossier']['fragment_catalog']) == 78
        assert len(payload['design_dossier']['sites']) == 1
        assert payload['design_dossier']['sites'][0]['cut_bond'] == [10, 9]
        text = json.dumps(payload)
        for forbidden in ('CHEMBL1134804', 'CHEMBL677210', 'standard_value', 'activity_class',
                          'known_high', 'benchmark-private', 'matched_to_fragment', 'source_molecule_count', 'curated'):
            assert forbidden not in text
    # Even an unrelated persisted research memory must NOT be reused in this benchmark.
    (tmp_path/'external-research.json').write_text(json.dumps({'status': 'complete', 'structure_memory': {'secret': 'SENTINEL_LABEL'}}))
    restored = make_workflow(tmp_path)
    assert 'SENTINEL_LABEL' not in json.dumps(restored._direct_payload('resume'))


@pytest.mark.parametrize('change', [
    {'operation': 'atom:addition'}, {'bond_site_id': 'bond-site-001'},
    {'parent_attempt': 1}, {'parent_attempt': 0}, {'cut_bond': [10, 9]},
    {'edit_atom_index': 21}, {'fragment_id': 'F999'}, {'fragment_smiles': '[*:1]OCCCCCCCCCCCCCCCC'},
])
def test_out_of_contract_decisions_rejected_before_construction(tmp_path, change):
    w = make_workflow(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        w._transformation({**decision(w), **change})


def test_closed_pool_learning_summary_keeps_early_incumbent_without_bulk_or_tool_hints(tmp_path):
    w = make_workflow(tmp_path)
    first = ['F053', 'F027', 'F004', 'F046', 'F054', 'F003',
             'F052', 'F039', 'F006', 'F013', 'F056', 'F005']
    ids = first + [r['fragment_id'] for r in w.closed_pool.catalog
                   if r['fragment_id'] not in first][:8]
    by_id = {r['fragment_id']: r for r in w.closed_pool.catalog}
    for attempt, fragment_id in enumerate(ids, 1):
        w.state.docking_history.append({
            'attempt': attempt, 'status': 'complete',
            'transformation': {'fragment_id': fragment_id, 'fragment_smiles': by_id[fragment_id]['smiles']},
            'raw_quality_from_mean': 0.5 if attempt == 1 else 0.1,
            'delta_candidate_minus_reference': -0.5 if attempt == 1 else -0.1,
            'seed_win_fraction': 1.0, 'seed_stddev': 0.02,
            'pose_retention': {'status': 'passed'},
            'quality': 0.495 if attempt == 1 else 0.095, 'is_new_best': attempt == 1,
        })
        w.state.candidate_history.append({'attempt': attempt})
    w.state.convergence.update(best_attempt=1, best_quality=0.495)
    feedback = {'failure_class': 'docking_evaluation', 'recommended_next_queries': ['search_fragment_library'],
                'latest_docking': {'status': 'complete', 'pose_retention': {'status': 'passed'},
                                   'candidate_per_seed': {'17': {'status': 'eligible', 'pose_selection': {
                                       'rank': 2, 'reference_core_rmsd': 0.2, 'crystal_core_rmsd': 0.4,
                                       'audit_path': '/private/huge-pose.json'}}},
                                   'plip_comparison': {'17': {'status': 'complete', 'gained': ['contact'],
                                                               'report_path': '/private/report.xml'}}},
                'latest_trend_entry': {'attempt': 12, 'quality': 0.095, 'is_new_best': False},
                'convergence': {'global_search': {'huge_irrelevant_state': 'unused'}},
                'recommended_next_queries': ['search_fragment_library']}
    payload = w._direct_payload('continue', rejection=feedback)
    summary = payload['state']['learning_summary']
    assert len(payload['design_dossier']['fragment_catalog']) == 78
    assert [r['attempt'] for r in payload['state']['docking_history']] == list(range(1, 21))
    assert summary['incumbent']['attempt'] == 1
    assert summary['incumbent']['transformation']['fragment_id'] == 'F053'
    assert summary['no_incumbent_improvement_count'] == 19
    assert summary['nearest_structural_contrasts']
    assert len(summary['nearest_structural_contrasts']) <= 6
    assert 'candidate_history' not in payload['state']
    text = json.dumps(payload)
    assert 'recommended_next_queries' not in text
    assert 'search_fragment_library' not in text
    assert 'huge_irrelevant_state' not in text
    assert '/private/' not in text
    assert len(text) < 85000  # Twenty outcomes should not stack twenty raw PLIP reports.
    assert payload['rejection']['per_seed_pose']['17']['reference_core_rmsd'] == 0.2
    assert payload['rejection']['per_seed_interaction_changes']['17']['gained'] == ['contact']


def test_equivalent_smiles_normalize_to_frozen_id_and_conflicting_id_fails(tmp_path):
    w = make_workflow(tmp_path)
    first = w.closed_pool.catalog[0]
    alternate = Chem.MolToSmiles(Chem.MolFromSmiles(first['smiles']), canonical=False, doRandom=True)
    proposed = decision(w)
    proposed.pop('fragment_id')
    proposed['fragment_smiles'] = alternate
    assert w._transformation(proposed)['fragment_id'] == first['fragment_id']
    proposed['fragment_id'] = w.closed_pool.catalog[1]['fragment_id']
    with pytest.raises(ValueError, match='conflicts'):
        w._transformation(proposed)


def test_request_repair_and_stop_are_audited(tmp_path):
    class Client:
        calls = 0
        def complete_json(self, payload):
            self.calls += 1
            if self.calls == 1:
                return {'action': 'READY', 'understanding': 'x', 'edit_hypothesis': 'x',
                        'operation': 'bond:replacement', 'bond_site_id': payload['design_dossier']['sites'][0]['bond_site_id'],
                        'fragment_id': 'F999'}
            return {'action': 'STOP', 'stop_reason': 'unable_to_repair'}
    w = make_workflow(tmp_path, Client())
    assert w._direct_decision('choose')['action'] == 'STOP'
    assert len(list(tmp_path.glob('closed-pool-request-*.json'))) == 2
    assert len(json.loads((tmp_path/'closed-pool-request-002.json').read_text())['visible_fragment_ids']) == 78
    response = json.loads((tmp_path/'closed-pool-response-002.json').read_text())
    assert response['decision']['action'] == 'STOP'
    bad = copy.deepcopy(w._direct_payload('choose'))
    bad['design_dossier']['fragment_catalog'] = bad['design_dossier']['fragment_catalog'][:6]
    with pytest.raises(ValueError, match='truncated'):
        w.closed_pool.audit_request(tmp_path, bad)


@pytest.mark.parametrize('failure_point', ['during_docking', 'after_docking_before_next_response'])
def test_interrupted_closed_pool_run_keeps_audit_and_fresh_run_is_independent(tmp_path, failure_point):
    """Simulate process failure without GNINA/network; do not resume or alter the failed run."""
    class Client:
        def __init__(self, fail_after_docking):
            self.fail_after_docking = fail_after_docking

        def complete_json(self, payload):
            if payload['state']['docking_history']:
                if self.fail_after_docking:
                    # Abrupt termination bypasses the normal API-exception audit writer.
                    raise SystemExit('injected termination before next model response')
                return {'action': 'STOP', 'stop_reason': 'no_promising_edit'}
            return {'action': 'READY', 'understanding': 'test the C6 side chain',
                    'edit_hypothesis': 'test a listed fragment', 'operation': 'bond:replacement',
                    'bond_site_id': payload['design_dossier']['sites'][0]['bond_site_id'],
                    'fragment_id': 'F053'}

    class Docking:
        def __init__(self, fail):
            self.fail = fail

        def prepare_reference(self, *args):
            return {'status': 'complete', 'valid_seeds': [17, 29, 43]}

        def run_with_reference_baseline(self, **kwargs):
            if self.fail:
                raise RuntimeError('injected failure during docking')
            return {'status': 'complete', 'seed_count': 3, 'seeds': [17, 29, 43],
                    'paired_seed_fraction': 1.0, 'pose_retention': {'status': 'passed'},
                    'comparison': {'status': 'complete', 'metrics': {'minimizedAffinity': {
                        'direction': 'lower_is_better',
                        'delta_candidate_minus_reference': {'mean': -0.4, 'stddev': 0.05},
                        'candidate_better_seed_fraction': 1.0}}},
                    'pose_consensus': {'stable': True}, 'reference_baseline': {'status': 'complete'}}

    interrupted = tmp_path/'interrupted'
    workflow = Workflow(TASK, Client(fail_after_docking=True), interrupted)
    workflow.docking_adapter = Docking(failure_point == 'during_docking')
    with pytest.raises((RuntimeError, SystemExit), match='injected (failure|termination)'):
        workflow.run()
    requests = sorted(interrupted.glob('closed-pool-request-*.json'))
    responses = sorted(interrupted.glob('closed-pool-response-*.json'))
    assert len(requests) == (1 if failure_point == 'during_docking' else 2)
    assert len(responses) == 1
    assert all(len(json.loads(path.read_text())['visible_fragment_ids']) == 78 for path in requests)
    assert not (interrupted/'result.json').exists()
    assert (interrupted/'candidate-01.sdf').exists()
    if failure_point == 'after_docking_before_next_response':
        assert not (interrupted/'closed-pool-response-002.json').exists()
        assert json.loads((interrupted/'edit-attempt-01.json').read_text())['docking']['status'] == 'complete'
        assert len(json.loads((interrupted/'docking-history.json').read_text())['history']) == 1
    else:
        assert not (interrupted/'edit-attempt-01.json').exists()
        assert not (interrupted/'docking-history.json').exists()

    # A new run, not --resume, starts from request 001 and finishes independently.
    fresh = tmp_path/'fresh'
    new_workflow = Workflow(TASK, Client(fail_after_docking=False), fresh)
    new_workflow.docking_adapter = Docking(False)
    result = new_workflow.run()['result']
    assert result['status'] == 'candidate_accepted'
    assert len(result['attempts']) == 1
    assert (fresh/'result.json').exists()
    assert len(list(fresh.glob('closed-pool-request-*.json'))) == 2
    assert len(list(fresh.glob('closed-pool-response-*.json'))) == 2
    assert len(list(interrupted.glob('closed-pool-response-*.json'))) == 1
    assert not (interrupted/'result.json').exists()


def test_pool_snapshot_tampering_prevents_resume(tmp_path):
    make_workflow(tmp_path)
    (tmp_path/'closed-pool-library.json').write_text('{}')
    with pytest.raises(ValueError, match='snapshot'):
        make_workflow(tmp_path)


def test_generator_is_structure_order_independent_and_frozen_rules_match():
    # Read SMILES ONLY; no activity-dependent input can enter select_pool().
    context = ComplexContext(TASK)
    structures = [r['canonical_smiles'] for r in json.loads((ROOT/'research/test_system_selection_egfr/gefitinib_discovery_series_28.json').read_text())]
    source = [r['smiles'] for r in json.loads((ROOT/'molecular_agent/data/fragments_unified.json').read_text())['fragments']]
    a, products, membership, _ = select_pool(context.ligand, structures, source)
    b, _, _, _ = select_pool(context.ligand, list(reversed(structures)), list(reversed(source)))
    assert a == b == json.loads((PUBLIC/'fragments.json').read_text())['fragments']
    assert len(set(products)) == 78
    assert {group: sum(v['group'] == group for v in membership.values()) for group in
            ('literature', 'near_neighbor', 'property_matched_diverse')} == {
                'literature': 26, 'near_neighbor': 26, 'property_matched_diverse': 26}
    for smiles, row in membership.items():
        if 'matched_to_fragment' in row:
            assert compatible(features(smiles), features(row['matched_to_fragment']))


def synthetic_evaluator(tmp_path):
    """Deliberately invented labels: public IDs do not imply their real literature origin."""
    context = ComplexContext(TASK)
    core = scaffold(context.ligand, [10, 9])
    records = json.loads((PUBLIC/'fragments.json').read_text())['fragments']
    rows, activities = [], []
    for i, record in enumerate(records):
        product = Chem.MolToSmiles(assemble(core, record['smiles']), isomericSmiles=True)
        rows.append({**record, 'product_smiles': product, 'group': 'literature' if i < 26 else 'background'})
        if i < 26:
            activities.append({'canonical_smiles': product, 'standard_value': str((i+1)*10)})
    activities += [{'canonical_smiles': Chem.MolToSmiles(context.ligand), 'standard_value': '90'},
                   {'canonical_smiles': 'CC', 'standard_value': '300'}]
    for i, row in enumerate(activities):
        row.update(document_chembl_id='CHEMBL1134804', assay_chembl_id='CHEMBL677210', target_chembl_id='CHEMBL203',
                   standard_type='IC50', standard_relation='=', standard_units='nM', data_validity_comment=None,
                   activity_id=i, molecule_chembl_id=f'SYNTHETIC-{i}', potential_duplicate=False)
    raw = tmp_path/'synthetic-activities.json'
    raw.write_text(json.dumps({'activities': activities, 'page_meta': {'next': None}}))
    membership = tmp_path/'synthetic-membership.json'
    membership.write_text(json.dumps({'library_sha256': sha256(PUBLIC/'fragments.json'), 'records': rows}))
    out = tmp_path/'evaluator'
    freeze_labels(raw, membership, PUBLIC, TASK, out)
    return out, raw, membership


def test_evaluator_rejects_missing_relation_and_label_tampering(tmp_path):
    evaluator, raw, membership = synthetic_evaluator(tmp_path)
    data = json.loads(raw.read_text())
    data['activities'][0].pop('standard_relation')
    raw.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='endpoint/relation'):
        freeze_labels(raw, membership, PUBLIC, TASK, tmp_path/'bad-freeze')
    (evaluator/'labels.json').write_text('{}')
    with pytest.raises(ValueError, match='labels/policy changed'):
        evaluate(tmp_path/'nonexistent-run', evaluator)


def test_random_reference_and_ties_do_not_call_unknowns_inactive():
    members = {str(i): {'known_high': i < 3, 'experimental': None if i >= 5 else {}} for i in range(10)}
    result = random_reference(members, 4, 1, 1000, 17)
    assert result['expected_known_high_count_exact'] == 1.2
    assert result['proposal_level_only']
    assert tied_ranks([2, 1, 2]) == [2.5, 1.0, 2.5]


def test_public_baselines_are_distinct_without_any_label_input(tmp_path):
    w = make_workflow(tmp_path)
    payload = w._direct_payload('test')
    for strategy in ('random', 'score-guided'):
        client = PublicBaselineClient(strategy, 17)
        decisions = [client.complete_json(payload) for _ in range(20)]
        assert len({d['fragment_id'] for d in decisions}) == 20
        assert all(w.closed_pool.normalize(d) for d in decisions)


def test_score_guided_baseline_uses_only_observed_scores_for_exploitation(tmp_path):
    from rdkit import DataStructs
    w = make_workflow(tmp_path)
    payload = w._direct_payload('test')
    client = PublicBaselineClient('score-guided', 17)
    prior = [client.complete_json(payload) for _ in range(4)]
    incumbent = prior[0]['fragment_id']
    payload['state']['docking_history'] = [{'status': 'complete', 'raw_quality_from_mean': 1.0,
                                           'transformation': {'fragment_id': incumbent}}]
    available = [r['fragment_id'] for r in w.closed_pool.catalog if r['fragment_id'] not in client.used]
    expected = min(available, key=lambda i: (-DataStructs.TanimotoSimilarity(client.fingerprints[incumbent], client.fingerprints[i]), i))
    selected = client.complete_json(payload)
    assert selected['fragment_id'] == expected
    assert 'Nearest unused' in selected['edit_hypothesis']


def test_native_reconstruction_clashes_are_deferred_only_with_final_gate(tmp_path):
    w = make_workflow(tmp_path)
    transformation = {'operation': 'bond:replacement', 'cut_bond': [10, 9],
                      'fragment_smiles': decompose(w.context.ligand, w.context.ligand)[0]}
    result = w._construct_candidate(w.context.ligand, transformation)
    assert result.report['status'] == 'accepted'
    assert result.report['initial_receptor_clash_check']['status'] == 'deferred_to_docking'
    assert result.report['severe_clash_count'] > 0
    w.context.task['pose_retention'].pop('maximum_heavy_atom_overlap')
    with pytest.raises(RuntimeError, match='final-pose steric gate'):
        w._construct_candidate(w.context.ligand, transformation)


def test_final_steric_gate_blocks_clashing_noncore_sidechain_even_with_good_rmsd(tmp_path):
    context = ComplexContext(TASK)
    native = Chem.AddHs(context.ligand, addCoords=True)
    native.SetProp('minimizedAffinity', '-8')
    receptor = context.write_receptor_pdb(tmp_path/'receptor.pdb')
    assert heavy_steric_check(native, receptor, .55)['status'] == 'passed'
    bad = Chem.Mol(native)
    bad.SetProp('minimizedAffinity', '-99')
    bad.GetConformer().SetAtomPosition(0, tuple(context.protein_atoms[0].xyz))
    assert heavy_steric_check(bad, receptor, .55)['status'] == 'failed'
    source, poses = tmp_path/'source.sdf', tmp_path/'poses.sdf'
    with Chem.SDWriter(str(source)) as writer:
        writer.write(native)
    with Chem.SDWriter(str(poses)) as writer:
        writer.write(bad)
        writer.write(native)
    adapter = DockingAdapter({'seeds': [17, 29, 43], 'pose_retention': {
        'core_atom_indices': list(range(10,31)), 'minimum_passing_seeds': 2,
        'maximum_heavy_atom_overlap': .55}}, tmp_path)
    protocol = PoseRetention(adapter)
    result = protocol.evaluate(source, source, receptor, {'pose_path': str(poses)}, tmp_path/'gate', 17)
    assert result['poses'][0]['reason'] == 'final_pose_heavy_atom_clash'
    assert result['poses'][0]['crystal_core_rmsd'] < .001
    assert protocol.select(result)['rank'] == 2


def test_full_budget_workflow_and_offline_evaluator_with_mock_docking(tmp_path):
    evaluator, _, _ = synthetic_evaluator(tmp_path)
    run = tmp_path/'run'
    client = PublicBaselineClient('random', 17)
    workflow = Workflow(TASK, client, run)
    # Real graph construction/geometry and feedback, but no GNINA/PLIP/LLM/network in this test.
    workflow.docking_adapter = SimpleNamespace(
        prepare_reference=lambda *args: {'status': 'complete'},
        run_with_reference_baseline=lambda **kwargs: {'status': 'no_eligible_pose', 'seed_count': 3,
            'pose_retention': {'status': 'failed'}, 'comparison': {'metrics': {}}, 'paired_seed_fraction': 0,
            'candidate_per_seed': {}, 'reference_baseline': {}})
    result = workflow.run()['result']
    assert len(result['attempts']) == 20
    assert result['status'] == 'no_candidate_accepted'
    assert len({r['transformation']['fragment_id'] for r in result['attempts']}) == 20
    report = evaluate(run, evaluator)
    assert report['unique_proposals'] == 20
    assert report['proposals']['unknown_n'] > 0
    assert report['pose_passed']['n'] == 0
    assert report['final_top_k']['1']['n'] == 0
    assert report['random_same_actual_budget']['budget'] == 20
    assert not (run/'evaluation.json').exists()
