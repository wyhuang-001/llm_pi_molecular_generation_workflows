"""Offline public-chemistry-only construction audit; NOT precomputed designer feedback."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem
from molecular_agent.editing import write_sdf
from molecular_agent.pose_retention import save_json
from molecular_agent.workflow import Workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, default=Path('4WKQ/task.benchmark.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Use a new offline audit directory')
    if args.output_dir.resolve().is_relative_to((args.task.parent/'design').resolve()):
        parser.error('Do not place construction outcomes in public designer inputs')
    workflow = Workflow(args.task, object(), args.output_dir)
    rows = []
    for fragment in workflow.closed_pool.catalog:
        fragment_id = fragment['fragment_id']
        decision = {'operation': 'replace_fragment', 'fragment_id': fragment_id,
                    'bond_site_id': workflow.closed_pool.site['bond_site_id']}
        transformation = workflow._transformation(decision)
        try:
            result = workflow._construct_candidate(workflow.context.ligand, transformation)
            smiles = Chem.MolToSmiles(Chem.RemoveHs(result.molecule), isomericSmiles=True)
            same_graph = smiles == workflow.closed_pool.products[fragment_id]
            write_sdf(result, workflow.run_dir/f'{fragment_id}.sdf', name=fragment_id)
            rows.append({'fragment_id': fragment_id, 'constructed': True, 'exact_expected_graph': same_graph,
                         'pre_docking_status': result.report['status'],
                         'initial_clash_status': (result.report.get('initial_receptor_clash_check') or {}).get('status',
                                                 'passed' if result.report['status'] == 'accepted' else 'rejected'),
                         'severe_clash_count': result.report.get('severe_clash_count')})
        except (ValueError, RuntimeError) as exc:
            rows.append({'fragment_id': fragment_id, 'constructed': False, 'exact_expected_graph': False, 'error': str(exc)})
    summary = {'role': 'offline_audit_not_designer_context', 'candidate_count': len(rows),
               'constructed_count': sum(r['constructed'] for r in rows),
               'exact_expected_graph_count': sum(r['exact_expected_graph'] for r in rows),
               'pre_docking_status_counts': dict(Counter(r.get('pre_docking_status', 'construction_failed') for r in rows)),
               'initial_clash_status_counts': dict(Counter(r.get('initial_clash_status', 'construction_failed') for r in rows)),
               'protocol': 'same Host construction as benchmark, seed=17; no GNINA/PLIP/LLM; do not filter frozen pool by these outcomes',
               'per_fragment': rows}
    save_json(workflow.run_dir/'construction-audit.json', summary)
    print({key: value for key, value in summary.items() if key != 'per_fragment'})
    if summary['exact_expected_graph_count'] != len(rows):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
