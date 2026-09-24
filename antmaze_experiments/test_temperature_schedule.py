"""Check warmup exclusion, exact endpoints, and legacy schedule behavior."""
import importlib.util
import math
from pathlib import Path
import subprocess
from types import ModuleType
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('trg_temperature',ROOT/'analysis_tools/experiments/20260920_truncated_mll/optiq_dime/temperature.py')
temperature=importlib.util.module_from_spec(spec);spec.loader.exec_module(temperature)


class ScheduleTests(unittest.TestCase):
    def test_increasing_schedule_excludes_warmup_and_holds_endpoint(self):
        for decay in ('linear', 'log_linear'):
            s=temperature.parse_temperature_schedule(dict(temperature=1.,temperature_schedule=
                dict(enabled=True,final_temperature=3.,anneal_steps=1000000,decay=decay)),'td')
            for step in (0,8191,8192):
                self.assertEqual(temperature.scheduled_temperature(1.,s,step,8192),(1.,0.))
            middle=temperature.scheduled_temperature(1.,s,508192,8192)
            self.assertAlmostEqual(middle[0],2. if decay=='linear' else math.sqrt(3.))
            self.assertEqual(middle[1],.5)
            for step in (1008192,1008384,5008384):
                self.assertEqual(temperature.scheduled_temperature(1.,s,step,8192),(3.,1.))
            values=[temperature.scheduled_temperature(1.,s,x,8192)[0] for x in range(8192,1200000,256)]
            self.assertTrue(all(1.<=a<=b<=3. for a,b in zip(values,values[1:])))

    def test_existing_schedules_match_frozen_parent_exactly(self):
        path='analysis_tools/experiments/20260920_truncated_mll/optiq_dime/temperature.py'
        old=ModuleType('frozen_temperature')
        content=subprocess.check_output(['git','show','f5b3fcbfff3e61ecac4517facaa446393045c5a9:'+path],cwd=ROOT)
        exec(compile(content,path,'exec'),old.__dict__)
        for final in (1.,.25,.5):
            for decay in ('linear','log_linear'):
                cfg=dict(temperature=10.,temperature_schedule=dict(enabled=True,
                    final_temperature=final,anneal_steps=1000000,decay=decay))
                s=temperature.parse_temperature_schedule(cfg,'td')
                previous=old.parse_temperature_schedule(cfg,'td')
                for step in (0,8192,8448,100096,250112,508192,1008384,5008384):
                    self.assertEqual(temperature.scheduled_temperature(10.,s,step,8192),
                                     old.scheduled_temperature(10.,previous,step,8192))

    def test_invalid_temperature_and_non_td_backup_remain_rejected(self):
        for initial,final in ((0,3),(1,0),(-1,3),(1,float('inf')),(float('nan'),3)):
            cfg=dict(temperature=initial,temperature_schedule=dict(enabled=True,
                final_temperature=final,anneal_steps=1000000,decay='linear'))
            with self.assertRaises(ValueError):temperature.parse_temperature_schedule(cfg,'td')
        cfg=dict(temperature=1.,temperature_schedule=dict(enabled=True,
            final_temperature=3.,anneal_steps=1000000,decay='linear'))
        with self.assertRaises(ValueError):temperature.parse_temperature_schedule(cfg,'soft_td')

    def test_linear_after_warmup_and_final_hold(self):
        for final in (1.,.25,.5):
            schedule=temperature.parse_temperature_schedule(dict(temperature=10.,temperature_schedule=
                dict(enabled=True,final_temperature=final,anneal_steps=1000000,decay='linear')),'td')
            for step in (0,8191,8192):
                self.assertEqual(temperature.scheduled_temperature(10.,schedule,step,8192),(10.,0.))
            self.assertEqual(temperature.scheduled_temperature(10.,schedule,508192,8192),((10.+final)/2,.5))
            for step in (1008192,1008384,5008384):
                self.assertEqual(temperature.scheduled_temperature(10.,schedule,step,8192),(final,1.))

    def test_first_vector_update_and_no_reverse_after_end(self):
        s=temperature.TemperatureSchedule(.25,1000000,'linear')
        value,fraction=temperature.scheduled_temperature(10.,s,8448,8192)
        self.assertAlmostEqual(fraction,256/1000000)
        self.assertAlmostEqual(value,10-9.75*256/1000000)
        values=[temperature.scheduled_temperature(10.,s,x,8192)[0] for x in range(8192,1200000,256)]
        self.assertTrue(all(a>=b>=.25 for a,b in zip(values,values[1:])))

    def test_legacy_log_linear_default_is_unchanged(self):
        s=temperature.parse_temperature_schedule(dict(temperature=10.,temperature_schedule=
            dict(enabled=True,final_temperature=.25,anneal_steps=1000000)),'td')
        self.assertEqual(s.decay,'log_linear')
        self.assertAlmostEqual(temperature.scheduled_temperature(10.,s,508192,8192)[0],math.sqrt(2.5))
        self.assertIsNone(temperature.parse_temperature_schedule(dict(temperature=1.),'td'))

    def test_invalid_decay_rejected(self):
        with self.assertRaises(ValueError):
            temperature.parse_temperature_schedule(dict(temperature=10.,temperature_schedule=
                dict(enabled=True,final_temperature=1.,anneal_steps=1000000,decay='typo')),'td')


if __name__=='__main__':unittest.main()
