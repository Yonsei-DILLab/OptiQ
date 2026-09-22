"""User-approved dense profile and original per-maze interaction budgets."""
CAMPAIGN = 'antmaze-upstream-dense-nativebudget-64env-s0-20260923'
BUDGETS = dict(v1=3000000, v2=3000000, v3=4000000, v4=5000000)
REWARD = 'negative Euclidean distance from next xy to nearest goal; no sparse bonus'
# The native DIPO [0,5] support cannot represent any negative dense return.
# Keep its upper endpoint, atom count, architecture and optimizer unchanged.
DIPO_DENSE_V_MIN = -6000.0
