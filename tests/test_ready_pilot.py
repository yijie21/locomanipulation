import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
path=Path(__file__).resolve().parents[1]/'src/teleop_free/tools/run_ready_pilot.py'
spec=importlib.util.spec_from_file_location('run_ready_pilot',path)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

class ReuseDataGateTests(unittest.TestCase):
    def test_incomplete_conversion_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'old';old.mkdir()
            with patch.object(sys,'argv',['pilot','--out',str(root/'new'),'--reuse-data',str(old)]), patch.object(pilot.subprocess,'call') as execute:
                with self.assertRaises(SystemExit):pilot.main()
                execute.assert_not_called()
    def test_receipt_for_changed_spec_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'old';old.mkdir()
            (old/'spec.json').write_text('{}')
            (old/'convert.done.json').write_text(json.dumps(dict(fingerprint='wrong',artifacts=[])))
            with patch.object(sys,'argv',['pilot','--out',str(root/'new'),'--reuse-data',str(old)]), patch.object(pilot.subprocess,'call') as execute:
                with self.assertRaises(SystemExit):pilot.main()
                execute.assert_not_called()


class FixedBudgetTests(unittest.TestCase):
    def exercise(self, root, count=25, successes=True, fail_audit=False):
        out=root/'run';(out/'scene').mkdir(parents=True)
        (out/'eval_cases.json').write_text(json.dumps({'cases':[{}]*20}))
        cases=[{'case_id':f'fixed_{i:02d}'} for i in range(count)]
        bank=root/'cases.json';bank.write_text(json.dumps({'cases':cases,'schema_version':1}))
        commands=[];attempted=[];session_index=[0]
        def execute(cmd,**kwargs):
            commands.append(cmd)
            if any(x.endswith('/start_stack.sh') for x in cmd):session_index[0]+=1
            if '--manifest' in cmd:
                selected=json.loads(Path(cmd[cmd.index('--cases')+1]).read_text())['cases']
                self.assertLessEqual(len(selected),10)
                self.assertEqual(int(cmd[cmd.index('--rounds')+1]),len(selected))
                limit=int(cmd[cmd.index('--until-ok')+1])
                played=selected[:limit] if successes else selected
                attempted.extend(r['case_id'] for r in played)
                Path(cmd[cmd.index('--manifest')+1]).write_text(''.join(
                    json.dumps(dict(r,ok=successes,produced=True))+'\n' for r in played))
            if any(x.endswith('/audit_frame_sources.py') for x in cmd):
                if fail_audit:return 1
                Path(cmd[cmd.index('--out')+1]).write_text('{}')
            return 0
        argv=['pilot','--out',str(out),'--collection-cases',str(bank),
              '--eval-rounds','20','--stand-retries','0','--through','collect']
        error=None
        with patch.object(sys,'argv',argv), patch.object(pilot,'latest_session',side_effect=lambda:root/f'raw_{session_index[0]}'), \
             patch.object(pilot.subprocess,'call',side_effect=execute):
            try:pilot.main()
            except RuntimeError as exc:error=exc
        self.assertFalse(any('train_bottle.sh' in x or 'convert_to_hf_format.py' in x for c in commands for x in c))
        for c in commands:
            if '--manifest' in c:self.assertEqual(c[c.index('--stand-retries')+1],'0')
        return out,commands,attempted,error

    def test_collect_boundary_never_starts_conversion_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,commands,attempted,error=self.exercise(Path(tmp))
            self.assertIsNone(error)
            self.assertEqual(attempted,[f'fixed_{i:02d}' for i in range(24)])
            self.assertEqual(len([c for c in commands if '--manifest' in c]),3)
            inputs=json.loads((out/'collect/inputs.json').read_text())
            self.assertEqual(len(set(inputs['sessions'])),3)
            audits=[c for c in commands if any(x.endswith('/audit_frame_sources.py') for x in c)]
            self.assertEqual([c[c.index('--session')+1] for c in audits],inputs['sessions'])
            self.assertEqual(len(set(c[2] for c in audits)),3)
            self.assertTrue((out/'collect.done.json').exists())
            self.assertEqual(json.loads((out/'spec.json').read_text())['eval_rounds'],20)
            self.assertEqual(json.loads((out/'spec.json').read_text())['collection_chunk_rounds'],10)

    def test_fixed_bank_failure_never_resamples_or_exceeds_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,commands,attempted,error=self.exercise(Path(tmp),count=40,successes=False)
            self.assertIsNotNone(error)
            self.assertIn('only 0/24',str(error))
            self.assertEqual(attempted,[f'fixed_{i:02d}' for i in range(40)])
            self.assertEqual(len([c for c in commands if '--manifest' in c]),4)
            self.assertFalse((out/'collect.done.json').exists())
            self.assertEqual(len(list((out/'collect').glob('manifest_*.jsonl'))),4)

    def test_source_audit_failure_stops_before_next_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,commands,attempted,error=self.exercise(Path(tmp),fail_audit=True)
            self.assertIsNotNone(error)
            self.assertEqual(len(attempted),10)
            self.assertEqual(len([c for c in commands if '--manifest' in c]),1)
            self.assertFalse((out/'collect.done.json').exists())

if __name__=='__main__':unittest.main()
