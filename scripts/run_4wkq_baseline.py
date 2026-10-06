"""Public-input-only random or score-guided baseline, same Host/GNINA budget as LLM.

No experimental labels, literature membership, network research or paid LLM calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from molecular_agent.structure import ComplexContext
from molecular_agent.workflow import Workflow


class PublicBaselineClient:
    """Public-input-only baseline. Works for both pool modes.

    Keyed by **product molecule** so a compound reached through different sites is one
    candidate, which is what the cut-point-free pool requires.
    """

    def __init__(self, strategy='random', seed=17, ligand=None):
        if strategy not in {'random', 'score-guided'}:
            raise ValueError('Unsupported public baseline strategy')
        self.strategy, self.seed = strategy, seed
        self.rng = random.Random(seed)
        self.used = set()
        self.best = None
        self.fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.fingerprints = {}
        self.ligand = ligand
        self.used_products = set()

    def _product(self, decision):
        if self.ligand is None:
            return json.dumps({k: decision.get(k) for k in
                               ('site_type', 'change_type', 'bond_site_id', 'edit_atom_index', 'fragment_id')},
                              sort_keys=True)
        from molecular_agent.editing import transformation_product_smiles
        try:
            return transformation_product_smiles(self.ligand, decision)
        except Exception:
            return None

    def _multisite_decision(self, dossier, catalog):
        sites = dossier['sites']
        candidates = []
        for site in sites:
            for change_type in site.get('allowed_change_types') or []:
                base = {'action': 'READY',
                        'understanding': 'Public structure and cut-point-free pool; no experimental labels.',
                        'site_type': site.get('site_type', 'atom'),
                        'change_type': change_type}
                if site.get('site_type') == 'bond':
                    # The host resolves cut_bond from bond_site_id; the multisite pool
                    # rejects decisions that guess cut_bond or parent_attempt.
                    base['bond_site_id'] = site['bond_site_id']
                else:
                    base['edit_atom_index'] = site['target_id']
                is_bond = site.get('site_type') == 'bond'
                if is_bond and change_type == 'replacement':
                    # a bond replacement consumes a fragment
                    for record in catalog:
                        if 'replacement' in (record.get('allowed_change_types') or []):
                            candidates.append({**base, 'fragment_id': record['fragment_id'],
                                               'fragment_smiles': record['smiles']})
                elif is_bond and change_type == 'deletion':
                    candidates.append(base)
                elif not is_bond and change_type == 'addition':
                    # an atom/ring replacement is an element swap, which this baseline
                    # does not explore; only the fragment-consuming addition is used
                    for record in catalog:
                        if 'addition' in (record.get('allowed_change_types') or []):
                            candidates.append({**base, 'fragment_id': record['fragment_id'],
                                               'fragment_smiles': record['smiles']})
        available = []
        for decision in candidates:
            product = self._product(decision)
            if product is None or product in self.used_products:
                continue
            available.append((decision, product))
        if not available:
            return {'action': 'STOP', 'reason': 'Closed pool exhausted'}
        decision, product = self.rng.choice(available)
        self.used_products.add(product)
        decision['edit_hypothesis'] = (
            'Uniform unused pool proposal keyed by product molecule under the same finite budget.')
        return decision

    def complete_json(self, payload):
        dossier = payload['design_dossier']
        catalog = dossier['fragment_catalog']
        if (dossier.get('selection_contract') or {}).get('mode') == 'closed_pool_multisite':
            return self._multisite_decision(dossier, catalog)
        if not self.fingerprints:
            self.fingerprints = {r['fragment_id']: self.fp.GetFingerprint(Chem.MolFromSmiles(r['smiles'])) for r in catalog}
        for row in payload['state']['docking_history']:
            fragment_id = row.get('transformation', {}).get('fragment_id')
            score = row.get('raw_quality_from_mean')
            if row.get('status') == 'complete' and fragment_id in self.fingerprints and score is not None:
                if self.best is None or score > self.best[0]:
                    self.best = (score, fragment_id)
        available = [r['fragment_id'] for r in catalog if r['fragment_id'] not in self.used]
        if not available:
            return {'action': 'STOP', 'reason': 'Closed pool exhausted'}
        # Three bootstrap uniform proposals; thereafter every fourth proposal explores.
        exploit = self.strategy == 'score-guided' and self.best is not None and len(self.used) >= 3 and len(self.used) % 4 != 3
        if exploit:
            best_fp = self.fingerprints[self.best[1]]
            selected = min(available, key=lambda i: (-DataStructs.TanimotoSimilarity(best_fp, self.fingerprints[i]), i))
        else:
            selected = self.rng.choice(available)
        self.used.add(selected)
        return {'action': 'READY', 'understanding': 'Public structure and fixed C6 pool; no experimental labels.',
                'edit_hypothesis': 'Nearest unused side-chain neighbor of the best observed pose-passing mean score.' if exploit
                                   else 'Uniform unused catalog proposal under the same finite budget.',
                'operation': 'replace_fragment', 'bond_site_id': dossier['sites'][0]['bond_site_id'],
                'fragment_id': selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', choices=['random', 'score-guided'], default='random')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--task', type=Path, default=Path('4WKQ/task.benchmark.json'))
    parser.add_argument('--config', type=Path, default=Path('config.single_edit.json'))
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error('Use a new run directory; baseline resume is not supported')
    client = PublicBaselineClient(args.strategy, args.seed, ligand=ComplexContext(args.task).ligand)
    workflow = Workflow(args.task, client, args.run_dir, args.config)
    (workflow.run_dir/'baseline-strategy.json').write_text(json.dumps({
        'strategy': args.strategy, 'seed': args.seed, 'uses_experimental_labels': False,
        'bootstrap_uniform_proposals': 3, 'uniform_exploration_every_fourth': True,
        'exploitation_metric': 'pose-passing raw mean docking improvement', 'candidate_budget': workflow.closed_pool.budget}, indent=2))
    result = workflow.run()['result']
    print(json.dumps({'status': result['status'], 'stopping_reason': result.get('stopping_reason'),
                      'unique_attempts': len(result.get('attempts', [])),
                      'result_path': str(workflow.run_dir/'result.json')}, indent=2))


if __name__ == '__main__':
    main()
