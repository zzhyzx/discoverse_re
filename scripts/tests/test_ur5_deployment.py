"""Deployment regression tests with synthetic frames and fake hardware only."""
import importlib.util
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from ur5_camera_wire import CameraClient, FRAME_BYTES, read_packet, send_packet
from ur5_camera_wire import CameraTimingError
from ur5_windows_camera_bridge import BridgeServer, Frames, Handler
from ur5_deploy_core import GripperDecision, Limiter, Plan, interpolate, validate_actions, remaining_plan
spec = importlib.util.spec_from_file_location('ur5_deploy',ROOT/'policies/smolvla/deploy_real_ur5.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

class Log:
    def __init__(self): self.rows=[]
    def event(self,event,**values): self.rows.append(dict(event=event,**values))

class Receive:
    def __init__(self, *args):
        self.q=np.zeros(6); self.runtime=2; self.status=3; self.stamp=None
    def isConnected(self): return True
    def getTimestamp(self): return self.stamp if self.stamp is not None else time.monotonic()
    def getRobotMode(self): return 7
    def getSafetyMode(self): return 1
    def getActualQ(self): return self.q.tolist()
    def getActualQd(self): return [0.]*6
    def getRuntimeState(self): return self.runtime
    def getRobotStatus(self): return self.status
    def getSpeedScaling(self): return 1.
    def getTargetSpeedFraction(self): return 1.
    def disconnect(self): pass

class Control:
    def __init__(self,receive):
        self.receive=receive;self.sent=[];self.stopped=False;self.watchdog=True;self.allowed=True
    def initPeriod(self): return time.monotonic()
    def waitPeriod(self,tick): time.sleep(max(0,.008-(time.monotonic()-tick)))
    def isConnected(self): return True
    def isJointsWithinSafetyLimits(self,q): return self.allowed
    def kickWatchdog(self): return self.watchdog
    def servoJ(self,q,*args): self.sent.append(q);self.receive.q=np.array(q);return True
    def servoStop(self,a): self.stopped=True;return True
    def stopJ(self,a): self.stopped=True

class DeploymentTests(unittest.TestCase):
    def test_homing_cancel_never_creates_control(self):
        with patch('builtins.input',return_value='cancel'):
            with self.assertRaisesRegex(RuntimeError,'Homing cancelled'):
                deploy.return_to_start(Receive(),np.zeros(6),np.ones(6)*.02,
                    dict(POS=3,OBJ=3),self.cfg,Log())

    def test_homing_refuses_closed_tool(self):
        with self.assertRaisesRegex(RuntimeError,'open EMPTY'):
            deploy.return_to_start(Receive(),np.zeros(6),np.ones(6)*.02,
                dict(POS=100,OBJ=2),self.cfg,Log())

    def test_camera_rejects_one_late_packet_without_publishing_it(self):
        shared=deploy.Shared();log=Log();calls=[]
        class Client:
            def get(self,*args):
                calls.append(1)
                if len(calls)==1: raise CameraTimingError('late packet')
                shared.stop.set()
                return {'local_observed_monotonic':time.monotonic()}, {r:bytes(FRAME_BYTES) for r in ['front','wrist']}
            def close(self): pass
        log.frames=lambda *args: None
        deploy.camera_worker(Client(),shared,log,self.cfg)
        self.assertIsNone(shared.error)
        self.assertIsNotNone(shared.camera)
        self.assertEqual(log.rows[0]['event'],'camera_frame_rejected')

    def test_continuous_late_packets_abort_without_publication(self):
        shared=deploy.Shared();cfg=self.cfg.copy();cfg['max_camera_age_s']=.01
        class Client:
            def get(self,*args):
                time.sleep(.006)
                raise CameraTimingError('late packet')
            def close(self): pass
        deploy.camera_worker(Client(),shared,Log(),cfg)
        self.assertIn('late packet',shared.error)
        self.assertIsNone(shared.camera)

    def setUp(self):
        self.cfg=deploy.configuration(ROOT/'policies/smolvla/configs/ur5_real_deploy.json')

    def test_velocity_and_acceleration_limits_with_reversal(self):
        limiter=Limiter(np.zeros(6),.03,.05)
        previous=np.zeros(6)
        for i in range(1000):
            target=np.ones(6)*(.2 if i<500 else -.2)
            limiter.step(target,.008)
            self.assertLessEqual(np.abs(limiter.velocity).max(),.03+1e-12)
            self.assertLessEqual(np.abs(limiter.velocity-previous).max(),.05*.008+1e-12)
            previous=limiter.velocity.copy()

    def test_invalid_angles_are_not_wrapped(self):
        a=np.zeros((10,7));a[:,3]=2*np.pi
        with self.assertRaisesRegex(RuntimeError,'envelope'):
            validate_actions(a,np.zeros(6),-np.ones(6),np.ones(6),.2)

    def test_nonfinite_and_far_prediction(self):
        a=np.zeros((10,7));a[0,0]=np.nan
        with self.assertRaises(ValueError): validate_actions(a,np.zeros(6),-np.ones(6),np.ones(6),.2)
        a[0,0]=.3
        with self.assertRaisesRegex(RuntimeError,'too far'):
            validate_actions(a,np.zeros(6),-np.ones(6),np.ones(6),.2)

    def test_gripper_dwell_resets_on_transient(self):
        g=GripperDecision(False,.2)
        self.assertIsNone(g.update(.9,0))
        self.assertIsNone(g.update(.1,.1))
        self.assertIsNone(g.update(.9,.2))
        self.assertIsNone(g.update(.9,.3))
        self.assertTrue(g.update(.9,.41))
        self.assertIsNone(g.update(.5,.42))

    def test_interpolation_has_correct_action_time(self):
        a=np.ones((10,7))
        np.testing.assert_allclose(interpolate(a,np.zeros(6),0)[:6],0)
        np.testing.assert_allclose(interpolate(a,np.zeros(6),1/60)[:6],.5)
        np.testing.assert_allclose(interpolate(a,np.zeros(6),1/30)[:6],1)

    def test_frame_protocol_exact_rgb_and_auth(self):
        frames=Frames()
        def update(number):
            for role,value in [('front',17),('wrist',231)]:
                frames.put(role,bytes([value])*FRAME_BYTES,
                    dict(frame_number=number,host_receipt_monotonic=time.monotonic(),serial=role))
        update(1)
        server=BridgeServer(('127.0.0.1',0),Handler)
        server.frames,server.token=frames,'test-token'
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            client=CameraClient('127.0.0.1',dict(port=server.server_address[1],token='test-token'))
            metadata,pixels=client.get(.25,.08)
            self.assertEqual(pixels['front'],bytes([17])*FRAME_BYTES)
            self.assertEqual(pixels['wrist'],bytes([231])*FRAME_BYTES)
            timer=threading.Timer(.04,lambda:update(2));timer.start()
            metadata,_=client.get(.25,.08)
            self.assertEqual(metadata['cameras']['front']['frame_number'],2)
            timer.join();client.close()
            bad=CameraClient('127.0.0.1',dict(port=server.server_address[1],token='wrong'))
            with self.assertRaisesRegex(RuntimeError,'Authentication'): bad.get(.25,.08)
            bad.close()
        finally:
            server.shutdown();server.server_close();worker.join(1)

    def test_oversized_protocol_header(self):
        left,right=socket.socketpair()
        try:
            left.sendall(struct.pack('!I',20000))
            with self.assertRaises(ValueError): read_packet(right)
        finally: left.close();right.close()

    def servo_setup(self):
        cfg=self.cfg.copy();cfg['duration_seconds']=.04
        shared=deploy.Shared();now=time.monotonic()
        shared.camera=({'local_observed_monotonic':now},None)
        shared.gripper=(now,dict(POS=3))
        shared.plan=Plan(1,now,now,np.zeros((10,7)),np.zeros(7),{})
        receive=Receive();feedback=deploy.Feedback(receive,cfg)
        control=Control(receive)
        return cfg,shared,feedback,control

    def test_servo_tracks_and_stops_at_duration(self):
        cfg,s,f,c=self.servo_setup()
        deploy.run_servo(c,f,s,Log(),cfg,-np.ones(6),np.ones(6),False)
        self.assertTrue(c.sent);self.assertTrue(c.stopped)

    def test_stale_camera_stops_before_command(self):
        cfg,s,f,c=self.servo_setup();s.camera[0]['local_observed_monotonic']-=1
        with self.assertRaisesRegex(RuntimeError,'Camera stream stale'):
            deploy.run_servo(c,f,s,Log(),cfg,-np.ones(6),np.ones(6),False)
        self.assertFalse(c.sent);self.assertTrue(c.stopped)

    def test_watchdog_failure_stops_before_command(self):
        cfg,s,f,c=self.servo_setup();c.watchdog=False
        with self.assertRaisesRegex(RuntimeError,'watchdog'):
            deploy.run_servo(c,f,s,Log(),cfg,-np.ones(6),np.ones(6),False)
        self.assertFalse(c.sent);self.assertTrue(c.stopped)

    def test_paused_program_stops_before_command(self):
        cfg,s,f,c=self.servo_setup();f.receive.runtime=4
        with self.assertRaisesRegex(RuntimeError,'paused'):
            deploy.run_servo(c,f,s,Log(),cfg,-np.ones(6),np.ones(6),False)
        self.assertFalse(c.sent);self.assertTrue(c.stopped)

    def test_stale_feedback(self):
        receive=Receive();receive.stamp=1
        feedback=deploy.Feedback(receive,self.cfg);feedback.sample()
        feedback.last_fresh-=1
        with self.assertRaisesRegex(RuntimeError,'Stale RTDE'): feedback.sample()

    def test_expired_prediction_stops_before_command(self):
        cfg,s,f,c=self.servo_setup();s.plan.observed-=cfg['max_observation_age_s']+1
        with self.assertRaisesRegex(RuntimeError,'observation too old'):
            deploy.run_servo(c,f,s,Log(),cfg,-np.ones(6),np.ones(6),False)
        self.assertFalse(c.sent);self.assertTrue(c.stopped)

    def test_controller_endpoint_rejection_stops(self):
        cfg,s,f,c=self.servo_setup();c.allowed=False
        with self.assertRaisesRegex(RuntimeError,'rejected chunk endpoint'):
            deploy.run_servo(c,f,s,Log(),cfg,-np.ones(6),np.ones(6),False)
        self.assertFalse(c.sent);self.assertTrue(c.stopped)

    def test_stop_during_gripper_read_does_not_issue_queued_command(self):
        shared=deploy.Shared();shared.gripper_request=True
        class Grip:
            def status(self):
                shared.stop.set()
                return dict(ACT=1,STA=3,FLT=0,POS=3,PRE=3,OBJ=3,GTO=1)
            def set(self,**values): raise AssertionError('SET after stop requested')
            def close(self): pass
        deploy.gripper_worker(Grip(),shared,Log(),self.cfg,True)
        self.assertIsNone(shared.error)

    def test_no_chunk_repetition(self):
        cfg,s,f,c=self.servo_setup()
        cfg['duration_seconds']=.5;cfg['max_camera_age_s']=2;cfg['max_gripper_status_age_s']=2
        with self.assertRaisesRegex(RuntimeError,'No fresh next action'):
            deploy.run_servo(c,f,s,Log(),cfg,-np.ones(6),np.ones(6),False)
        self.assertTrue(c.stopped)

    def test_latency_does_not_skip_unexecuted_prefix(self):
        plan=Plan(1,0,.813,np.repeat(np.arange(50)[:,None],7,axis=1),np.zeros(7),{})
        trimmed,skip,fraction=remaining_plan(plan,.813,30,30)
        self.assertEqual(skip,0)
        self.assertEqual(len(trimmed.actions),30)
        self.assertEqual(trimmed.actions[0,0],0)
        self.assertEqual(fraction,0)

    def test_new_prediction_replaces_active_before_end(self):
        cfg,s,f,c=self.servo_setup();cfg['duration_seconds']=.06
        original=c.waitPeriod
        def publish(tick):
            original(tick)
            now=time.monotonic()
            with s.lock:
                s.plan=Plan(2,now,now,np.zeros((50,7)),np.zeros(7),{})
        c.waitPeriod=publish
        log=Log()
        deploy.run_servo(c,f,s,log,cfg,-np.ones(6),np.ones(6),False)
        self.assertEqual([r['sequence'] for r in log.rows if r['event']=='chunk_started'],[1,2])

    def test_slow_completed_plan_keeps_prefix_in_servo(self):
        cfg,s,f,c=self.servo_setup()
        s.plan.actions=np.zeros((50,7));s.plan.observed-=.813
        log=Log()
        deploy.run_servo(c,f,s,log,cfg,-np.ones(6),np.ones(6),False)
        row=next(r for r in log.rows if r['event']=='chunk_started')
        self.assertEqual(row['skipped_steps'],0)
        self.assertEqual(row['execution_steps'],30)
        self.assertTrue(c.stopped)

    def test_drift_since_observation_stops_before_command(self):
        cfg,s,f,c=self.servo_setup();s.plan.state[0]=np.radians(4)
        with self.assertRaisesRegex(RuntimeError,'moved too far'):
            deploy.run_servo(c,f,s,Log(),cfg,-np.ones(6),np.ones(6),False)
        self.assertFalse(c.sent);self.assertTrue(c.stopped)

    def test_twelve_degree_limit_and_diagnostics(self):
        a=np.zeros((30,7));a[20,2]=np.radians(11)
        validate_actions(a,np.zeros(6),-np.ones(6),np.ones(6),np.radians(12))
        a[20,2]=np.radians(13)
        with self.assertRaisesRegex(RuntimeError,'step=21, joint=3.*delta=13.000deg, limit=12.000deg'):
            validate_actions(a,np.zeros(6),-np.ones(6),np.ones(6),np.radians(12))

    def test_video_writer_finalizes_both_views(self):
        with tempfile.TemporaryDirectory() as folder:
            shared=deploy.Shared();recorder=deploy.Recorder(Path(folder),shared,True)
            for index in range(4):
                recorder.frames(dict(local_observed_monotonic=index/30),
                    {'front':np.full((480,640,3),17,np.uint8),'wrist':np.full((480,640,3),231,np.uint8)})
            recorder.close();shared.check()
            import av
            for role,value in [('front',17),('wrist',231)]:
                with av.open(str(Path(folder)/f'{role}.mp4')) as video:
                    decoded=[frame.to_ndarray(format='rgb24') for frame in video.decode(video=0)]
                    self.assertEqual(len(decoded),4)
                    self.assertTrue(all(np.all(image==value) for image in decoded))

    def test_default_main_does_not_create_control_or_send_gripper_set(self):
        self.check_main(False)

    def test_execute_main_with_fake_control_only(self):
        self.check_main(True)

    def check_main(self, execute):
        class FakeCamera:
            def __init__(self): self.number=0
            def get(self,*args):
                self.number+=1
                return (dict(local_observed_monotonic=time.monotonic(),
                    cameras={r:dict(frame_number=self.number) for r in ['front','wrist']}),
                    {r:bytes(FRAME_BYTES) for r in ['front','wrist']})
            def close(self): pass
        class FakeGripper:
            def __init__(self,*args): pass
            def status(self): return dict(ACT=1,STA=3,FLT=0,POS=3,PRE=3,OBJ=3,GTO=1)
            def set(self,**values): raise AssertionError('Unexpected gripper SET')
            def close(self): pass
        class IdleReceive(Receive):
            def __init__(self,*args):
                super().__init__();self.runtime=1;self.status=1
                self.q=np.radians(self_cfg['start_reference_deg'])
                captured['receive']=self
        captured={}
        self_cfg=self.cfg.copy()
        self_cfg['train_stats']=str(ROOT/self_cfg['train_stats'])
        def fake_predict(policy,state,images,task): return np.tile(state,(50,1))
        class FakePolicy:
            def reset(self): pass
        class FakeControl(Control):
            def __init__(self,*args):
                if not execute: raise AssertionError('Unexpected control interface')
                receive=captured['receive'];receive.runtime=2;receive.status=3
                super().__init__(receive);captured['control']=self
            def setWatchdog(self,hz): captured['watchdog_hz']=hz;return True
            def stopScript(self): pass
            def disconnect(self): pass
        cli=['deploy','--config',str(ROOT/'policies/smolvla/configs/ur5_real_deploy.json'),'--no-video','--duration','.12']
        if execute: cli.append('--execute')
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(deploy,'ROOT',Path(folder)), patch.object(deploy,'configuration',return_value=self_cfg), patch.object(deploy,'load_model',return_value=(FakePolicy(),-np.ones(6)*10,np.ones(6)*10)), \
                patch.object(deploy,'bridge_client',return_value=FakeCamera()), patch.object(deploy,'predict',side_effect=fake_predict), \
                patch.dict(sys.modules,{'rtde_receive':types.SimpleNamespace(RTDEReceiveInterface=IdleReceive),
                    'rtde_control':types.SimpleNamespace(RTDEControlInterface=FakeControl)}), \
                patch('robotiq_gripper_test.Gripper',FakeGripper), \
                patch('builtins.input',return_value='MOVE'), patch.object(sys,'argv',cli):
                self.assertEqual(deploy.main(),0)
            result=list(Path(folder).glob('data/real_ur5/smolvla_deploy/*/result.json'))
            self.assertTrue(json.loads(result[0].read_text())['completed'])
        if execute:
            self.assertTrue(captured['control'].sent)
            self.assertTrue(captured['control'].stopped)
            self.assertEqual(captured['watchdog_hz'],10)
        else: self.assertNotIn('control',captured)

    def test_gripper_contact_requires_stable_feedback(self):
        shared=deploy.Shared();shared.gripper_request=True;shared.gripper_busy=True
        class Grip:
            def __init__(self): self.commands=[];self.sent=False;self.reads=0
            def status(self):
                self.reads+=1
                return dict(ACT=1,STA=3,FLT=0,POS=100 if self.sent else 3,
                            PRE=255 if self.sent else 3,OBJ=2 if self.sent else 3,GTO=1)
            def set(self,**values): self.commands.append(values);self.sent=True
            def close(self): pass
        class ContactLog(Log):
            def event(self,event,**values):
                super().event(event,**values)
                if event=='gripper_terminal': shared.stop.set()
        grip=Grip();log=ContactLog()
        deploy.gripper_worker(grip,shared,log,self.cfg,True)
        shared.check()
        self.assertFalse(shared.gripper_busy)
        self.assertGreaterEqual(grip.reads,5)
        self.assertEqual(grip.commands[0]['POS'],255)
        self.assertEqual(grip.commands[-1],{'GTO':0})

    def test_empty_grasp_aborts_without_reopening(self):
        shared=deploy.Shared();shared.gripper_request=True;shared.gripper_busy=True
        class Grip:
            def __init__(self): self.commands=[];self.sent=False
            def status(self):
                return dict(ACT=1,STA=3,FLT=0,POS=255 if self.sent else 3,
                            PRE=255 if self.sent else 3,OBJ=3,GTO=1)
            def set(self,**values): self.commands.append(values);self.sent=True
            def close(self): pass
        grip=Grip();deploy.gripper_worker(grip,shared,Log(),self.cfg,True)
        self.assertIn('without contact',shared.error)
        self.assertFalse(any(cmd.get('POS')==0 for cmd in grip.commands))

if __name__=='__main__': unittest.main()
