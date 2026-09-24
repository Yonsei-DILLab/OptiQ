"""Analytic geometry, return semantics, and compatibility tests (CPU only)."""
import unittest
import numpy as np
from .progress_reward import (Geodesic,geodesic,maze_geometry,distance,progress_reward,
                              success_bonus,specification,value_support,PROFILES,bonus_enabled,progress_scale,step_cost)
from .settings import reward_description

class ProgressRewardTests(unittest.TestCase):
    def test_100x_formula_keeps_bonus_unscaled(self):
        for profile in ('progress100_euclidean','progress100_geodesic'):
            for before,after,wanted in [([-7.,0.],[-7.05,0.],4.),
                                        ([-7.,0.],[-7.,0.],-1.),
                                        ([-7.05,0.],[-7.,0.],-6.)]:
                r,*_=progress_reward(before,after,'v1',profile,0.)
                self.assertAlmostEqual(float(r),wanted)
            for bonus in (10.,20.):
                r,*_=progress_reward([-7.4,0.],[-7.7,0.],'v1',profile,bonus)
                self.assertAlmostEqual(float(r),29.+bonus)
                off,*_=progress_reward([-7.4,0.],[-7.7,0.],'v1',profile+'_no_bonus',bonus)
                self.assertAlmostEqual(float(off),29.)
        old,*_=progress_reward([-7.4,0.],[-7.7,0.],'v1','progress_euclidean',10.)
        self.assertAlmostEqual(float(old),10.29)

    def test_bonus_only_changes_terminal_reward(self):
        for base in (p for p in PROFILES if not p.endswith('_no_bonus')):
            for task in ('v1','v2','v3','v4'):
                for goal in maze_geometry(task)[1]:
                    before=goal+[.6,0];after=goal+[.3,0]
                    bonus=success_bonus(after,task)
                    r_on,*_=progress_reward(before,after,task,base,bonus)
                    r_off,*_=progress_reward(before,after,task,base+'_no_bonus',bonus)
                    self.assertAlmostEqual(float(r_on-r_off),float(bonus))
                    self.assertFalse(specification(task,base+'_no_bonus')['success_bonus_enabled'])

    def test_empty_space_is_euclidean(self):
        g=Geodesic([],[[3,4]],margin=0)
        np.testing.assert_allclose(g.distances([[0,0],[3,4]]),[[5],[0]])

    def test_rectangle_analytic_detour_and_boundary(self):
        g=Geodesic([[-1,-1,1,1]],[[3,0]],margin=0)
        self.assertAlmostEqual(g.distances([-3,0])[0],2*np.sqrt(5)+2)
        self.assertAlmostEqual(g.distances([-1,-1])[0],2+np.sqrt(5))
        self.assertFalse(g.visible([-3,0],np.array([[3,0]]))[0])
        with self.assertRaises(ValueError):g.distances([0,0])

    def test_shared_walls_do_not_create_shortcut(self):
        g=Geodesic([[-1,-2,1,0],[-1,0,1,2]],[[3,0]])
        self.assertFalse(g.visible([-3,0],np.array([[3,0]]))[0])
        self.assertAlmostEqual(g.distances([-3,0])[0],2*np.sqrt(8)+2,places=4)

    def test_v1_two_equal_routes_and_continuity(self):
        g=geodesic('v1')
        self.assertAlmostEqual(g.distances([0,0])[0],4+2*np.sqrt(8),places=4)
        for x in [-5,-4,-3,0]:
            np.testing.assert_allclose(g.distances([x,3]),g.distances([x,-3]),atol=1e-8)
        a,b=np.array([0.,.001]),np.array([0.,-.001])
        self.assertLessEqual(abs(distance(a,'v1',PROFILES[1])-distance(b,'v1',PROFILES[1])),np.linalg.norm(a-b)+1e-8)

    def test_all_mazes_and_goal_bonuses(self):
        for task in ('v1','v2','v3','v4'):
            goals=maze_geometry(task)[1]
            for profile in PROFILES:
                d=distance([0.,0.],task,profile)
                self.assertTrue(np.isfinite(d))
                for goal in goals:
                    bonus=20 if tuple(goal)==(-8,8) else 10
                    self.assertEqual(success_bonus(goal,task),bonus)
                    self.assertAlmostEqual(float(distance(goal,task,profile)),0.)
                    r,_,_=progress_reward(goal,goal,task,profile,bonus)
                    self.assertAlmostEqual(float(r),(bonus if bonus_enabled(profile) else 0)-step_cost(profile))
                self.assertEqual(specification(task,profile)['physical_body_inflation_m'],0.)
                lo,hi=value_support(task,profile)
                self.assertLess(lo,-1);self.assertGreater(hi,20)
        np.testing.assert_allclose(geodesic('v4').distances([0,0]),[17.656856,17.656856],atol=1e-5)

    def test_stop_retreat_and_discount_not_inside_reward(self):
        for profile in PROFILES:
            a=np.array([-4.,3.]);b=np.array([-4.05,3.])
            stationary,_,_=progress_reward(a,a,'v1',profile,0)
            forward,_,_=progress_reward(a,b,'v1',profile,0)
            backward,_,_=progress_reward(b,a,'v1',profile,0)
            self.assertAlmostEqual(float(stationary),-step_cost(profile))
            self.assertAlmostEqual(float(forward+backward),-2*step_cost(profile))
            self.assertGreater(forward,stationary)
            # Terminal endpoint keeps the actual center distance; no artificial zero.
            end=np.array([-7.7,0.]);start=np.array([-7.4,0.])
            r,d0,d1=progress_reward(start,end,'v1',profile,10.)
            self.assertAlmostEqual(float(d1),.3)
            self.assertAlmostEqual(float(r),.3*progress_scale(profile)-step_cost(profile)+(10 if bonus_enabled(profile) else 0))

    def test_route_total_and_metadata_no_legacy_change(self):
        for profile in PROFILES:
            route=np.array([[0,0],[0,3],[-4,3],[-8,3],[-8,0]],float)
            r,_,_=progress_reward(route[:-1],route[1:],'v1',profile,np.array([0,0,0,10]))
            self.assertAlmostEqual(float(r.sum()),float(distance(route[0],'v1',profile))*progress_scale(profile)+(10 if bonus_enabled(profile) else 0)-4*step_cost(profile))
        self.assertEqual(reward_description('dense'),'negative Euclidean distance from next xy to nearest goal; no sparse bonus')
        self.assertIsNone(specification('v1','dense'))

if __name__=='__main__':unittest.main()
