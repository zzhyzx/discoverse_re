"""Exercise fixed-five scheduling branches from the actual simulation evaluator."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

TREE=ast.parse((Path(__file__).resolve().parents[1]/"policies/smolvla/eval_sim_ur5_low_speed.py").read_text())
BRANCH=next(n for n in ast.walk(TREE) if isinstance(n,ast.If)
            and ast.unparse(n.test)=="a.fixed_five and active is not None")
CODE=compile(ast.Module(body=[BRANCH],type_ignores=[]),"<actual-fixed-five>","exec")
class FixedFiveTests(unittest.TestCase):
    def decide(self,steps,n=5,aligned=True,busy=False,pending=None,newest=True,pending_close=False):
        ns=dict(a=SimpleNamespace(fixed_five=True,action_steps=n,fixed_timing=False),active=object(),
                newest=object() if newest else None,
                progress=SimpleNamespace(steps=steps),aligned=aligned,
                adapter=SimpleNamespace(busy=busy,pending_close=pending_close,decision=SimpleNamespace(candidate=pending)),
                reason_takeover="expiry_margin")
        exec(CODE,ns)
        return ns["reason_takeover"]
    def test_no_early_takeover_even_at_expiry(self):
        for steps in [0,1,2,4,4.99]:
            self.assertIsNone(self.decide(steps))
    def test_fixed_timing_replaces_without_alignment(self):
        ns=dict(a=SimpleNamespace(fixed_five=True,action_steps=50,fixed_timing=True),
            active=object(),newest=object(),progress=SimpleNamespace(steps=50),
            aligned=False,adapter=SimpleNamespace(busy=False,pending_close=False,decision=SimpleNamespace(candidate=None)))
        exec(CODE,ns)
        self.assertEqual(ns['reason_takeover'],'fixed_steps_complete')
        ns['progress'].steps=49
        exec(CODE,ns)
        self.assertIsNone(ns['reason_takeover'])
    def test_full_chunk_boundary(self):
        for steps in [0,10,30,49,49.99]:
            self.assertIsNone(self.decide(steps,n=50))
        self.assertEqual(self.decide(50,n=50),"fixed_steps_complete")
    def test_ten_step_boundary(self):
        for steps in [0,1,5,9,9.99]:
            self.assertIsNone(self.decide(steps,n=10))
        self.assertEqual(self.decide(10,n=10),"fixed_steps_complete")
    def test_endpoint_and_gripper_gates(self):
        self.assertEqual(self.decide(5),"fixed_steps_complete")
        for kwargs in [dict(aligned=False),dict(busy=True),dict(pending=True),dict(pending_close=True),dict(newest=False)]:
            self.assertIsNone(self.decide(5,**kwargs))
    def test_wall_clock_abort_disabled_only_in_fixed_mode(self):
        messages={"Inference exceeds configured deadline","Next prediction observation too old",
                  "No initial fresh prediction","Active prediction expired"}
        found=set()
        for n in ast.walk(TREE):
            if not isinstance(n,ast.If):continue
            for stmt in n.body:
                if not isinstance(stmt,ast.Raise) or not isinstance(stmt.exc,ast.Call):continue
                if not stmt.exc.args or not isinstance(stmt.exc.args[0],ast.Constant):continue
                message=stmt.exc.args[0].value
                if message in messages:
                    self.assertTrue(ast.unparse(n.test).startswith("not a.fixed_five and "))
                    found.add(message)
        self.assertEqual(found,messages)
    def test_progress_capped_at_five(self):
        branch=next(n for n in ast.walk(TREE) if isinstance(n,ast.If)
                    and ast.unparse(n.test)=="a.fixed_five"
                    and any(isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=="progress.elapsed" for x in n.body))
        code=compile(ast.Module(body=[branch],type_ignores=[]),"<actual-cap>","exec")
        progress=SimpleNamespace(elapsed=6/30)
        exec(code,dict(a=SimpleNamespace(fixed_five=True,action_steps=5),progress=progress))
        self.assertAlmostEqual(progress.elapsed,5/30)
if __name__=="__main__":unittest.main()
