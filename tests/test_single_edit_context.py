import base64
import json
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from molecular_agent.llm import ResponsesClient
from molecular_agent.research import MCPStdioClient, PlaywrightResearchAdapter, ExternalResearchError
from molecular_agent.research_memory import evidence_projection, validate_memory
from molecular_agent.plip_adapter import PLIPAdapter, compare_plip
from molecular_agent.workflow import Workflow

ROOT = Path(__file__).resolve().parents[1]


def test_summary_once_and_no_images_in_design_or_restart(tmp_path):
    class Client:
        calls = []
        def complete_json(self, payload):
            self.calls.append(payload)
            return {"observations": [{"statement": "RCSB entry 1H1Q", "source_ids": ["call-001"],
                     "interpretation": "reported_fact", "limitation": "not an affinity measurement"}],
                    "uncertainties": []}
    client = Client()
    w = Workflow(ROOT / 'input/task.single_edit.json', client, tmp_path)
    w.state.design_dossier = {"sites": []}
    w.state.external_research = {"status": "complete", "calls": [{"source_id": "call-001",
        "text": ["1H1Q"], "images": [{"type": "image", "data_base64": "UE5H"}]}]}
    w._prepare_initial_context()
    w._prepare_initial_context()
    assert len(client.calls) == 1
    assert ResponsesClient._image_records(client.calls[0])
    for instruction in ['first design', 'next design']:
        payload = w._direct_payload(instruction)
        assert not ResponsesClient._image_records(payload)
        assert payload['state']['available_parents'] == []
        assert 'molecular_weight' not in payload['design_dossier']['ligand']
        assert payload['external_research']['observations']
    restored = Workflow(ROOT / 'input/task.single_edit.json', client, tmp_path)
    restored.state.design_dossier = {'sites': []}
    restored._prepare_initial_context()
    assert len(client.calls) == 1
    with pytest.raises(RuntimeError, match='original ligand'):
        w._transformation({'operation': 'atom:addition', 'edit_atom_index': 10,
                           'fragment_smiles': '[*:1]F', 'parent_attempt': 1})


def test_graph_arrays_not_truncated():
    value = {'atoms': [{'atom_index': i} for i in range(100)], 'bonds': list(range(99))}
    assert Workflow._llm_safe_value(value) == value


def test_summary_unknown_sources_rejected():
    with pytest.raises(ValueError, match='unknown'):
        validate_memory({'observations': [{'statement': 'x', 'source_ids': ['invented']}]},
                        {'sources': [{'source_id': 'real'}]})


def test_screenshot_files_are_distinct(tmp_path):
    adapter = PlaywrightResearchAdapter()
    results = [adapter._normalize_mcp_result('screenshot', {'content': [{'type': 'image',
               'data': base64.b64encode(data).decode(), 'mimeType': 'image/png'}]}, tmp_path, i)
               for i, data in enumerate([b'one', b'two'], 1)]
    assert [Path(r['images'][0]['path']).read_bytes() for r in results] == [b'one', b'two']
    with pytest.raises(ExternalResearchError, match='not a structure'):
        adapter._save_text_artifact({'text': ['navigation text']}, 'fake.sdf', tmp_path, 1)


def test_mcp_buffer_notifications_errors_and_timeout(tmp_path):
    server = tmp_path / 'server.py'
    server.write_text('''import sys,json
for line in sys.stdin:
 m=json.loads(line)
 if 'id' not in m: continue
 if m['method']=='tools/call':
  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{'isError':True,'content':[]}}),flush=True)
 else:
  print(json.dumps({'jsonrpc':'2.0','method':'notifications/message'})+'\\n'+json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{}}),flush=True)
''')
    cfg = {'transport': 'mcp_stdio', 'required': True, 'mcp_command': [sys.executable, str(server)],
           'calls': [{'tool': 'bad'}]}
    with pytest.raises(ExternalResearchError, match='Required MCP'):
        PlaywrightResearchAdapter(cfg).collect({}, tmp_path / 'out')
    assert json.loads((tmp_path/'out/research-response.json').read_text())['status'] == 'failed'
    with pytest.raises(ExternalResearchError, match='timed out'):
        with MCPStdioClient([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=0.1):
            pass


def test_plip_selected_ligand_and_comparison(tmp_path):
    report = tmp_path/'report.xml'
    report.write_text('''<report><bindingsite><identifiers><hetid>LIG</hetid><chain>Z</chain><position>1</position></identifiers>
<interactions><hydrogen_bonds><hydrogen_bond><restype>ASP</restype><reschain>A</reschain><resnr>86</resnr><dist_d-a>2.9</dist_d-a></hydrogen_bond></hydrogen_bonds></interactions></bindingsite></report>''')
    result = dict(PLIPAdapter.parse_report(report), status='complete', report_path=str(report))
    assert result['counts'] == {'hydrogen_bond': 1}
    assert compare_plip(result, result)['gained'] == []
    assert compare_plip({'status': 'failed'}, result)['status'] == 'unavailable'
    report.write_text('<report/>')
    with pytest.raises(ValueError, match='selected ligand'):
        PLIPAdapter.parse_report(report)


def test_plip_complex_preserves_pose_and_reports_failure(tmp_path):
    from molecular_agent.structure import ComplexContext
    ctx = ComplexContext(ROOT / 'input/task.single_edit.json')
    receptor = ctx.write_receptor_pdb(tmp_path / 'receptor.pdb')
    pose = tmp_path / 'pose.sdf'
    with Chem.SDWriter(str(pose)) as writer:
        writer.write(ctx.ligand)
    complex_path = tmp_path / 'complex.pdb'
    provenance = PLIPAdapter.build_complex(receptor, pose, complex_path)
    rows = [r for r in complex_path.read_text().splitlines()
            if r.startswith('HETATM') and r[17:20] == 'LIG' and r[21:22] == 'Z']
    assert len(rows) == ctx.ligand.GetNumAtoms()
    assert len({r[6:11] for r in complex_path.read_text().splitlines()
                if r.startswith(('ATOM  ', 'HETATM'))}) == len(ctx.protein_atoms) + len(rows)
    for i, row in enumerate(rows):
        xyz = [float(row[a:a+8]) for a in (30, 38, 46)]
        assert xyz == pytest.approx(list(ctx.ligand.GetConformer().GetAtomPosition(i)), abs=0.0006)
    assert provenance['explicit_hydrogen_count'] == 0
    result = PLIPAdapter({'enabled': True, 'executable': str(tmp_path / 'missing-plip')}).run(
        receptor, pose, tmp_path / 'plip')
    assert result['status'] == 'failed'
    assert (tmp_path / 'plip/interaction-result.json').is_file()


def test_inline_key_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'wrong-fallback')
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'api_key': 'local-test-only', 'base_url': 'https://example.invalid', 'model': 'test'}))
    assert ResponsesClient(config).api_key == 'local-test-only'


def test_disabled_research_short_circuits_without_writing_a_request(tmp_path):
    """A disabled research pass must not build or persist the structure request."""
    import json as _json
    from pathlib import Path as _Path

    from molecular_agent.workflow import Workflow

    task = _Path("4WKQ/task.v2.json")
    if not task.is_file():
        import pytest

        pytest.skip("v2 task has not been built in this checkout")

    class StopClient:
        def complete_json(self, payload):
            return {"action": "STOP", "reason": "smoke"}

    run_dir = tmp_path / "run"
    workflow = Workflow(task, StopClient(), run_dir)
    workflow._prepare_initial_context()

    research = workflow.state.external_research or {}
    assert research.get("status") == "disabled"
    assert research.get("query_executed") is False
    assert not (run_dir / "external-research").exists()
    assert (run_dir / "external-research.json").stat().st_size < 1000

    summary = workflow._research_structure_summary()
    assert set(summary) >= {"ligand", "pocket", "reference_interactions", "omitted"}
    # The per-atom ligand graph and fragment catalog must not be forwarded.
    assert "molecule_graph" not in _json.dumps(summary)
    assert "fragment" not in _json.dumps(summary["ligand"]).lower()
    assert len(_json.dumps(summary)) < 40_000
