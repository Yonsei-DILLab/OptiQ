"""Validate sample/metric pairing and native-baseline fallback without training."""
import json,tempfile
from pathlib import Path
from .visualization import select_result,select_history


def main():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);(p/'config.json').write_text(json.dumps(dict(method='optiq_trg')))
        d=p/'evaluations/step_0100000';d.mkdir(parents=True)
        raw=dict(step=100000,high_density_fraction=.5,mode_coverage=40,training={'Q_evaluations':123})
        (d/'metrics_mu_only.json').write_text(json.dumps(dict(high_density_fraction=.9,mode_coverage=35)))
        primary,path=select_result(p,raw)
        assert primary['high_density_fraction']==.9 and primary['mode_coverage']==35
        assert path.name=='samples_mu_only.npy' and primary['training']==raw['training']
        assert select_history(p,[raw])[0]==primary
        full,path=select_result(p,raw,'full_policy')
        assert full['high_density_fraction']==.5 and path.name=='samples.npy'
        assert raw['high_density_fraction']==.5 and 'visualization_mode' not in raw
        (d/'metrics_mu_only.json').unlink()
        try:select_result(p,raw)
        except FileNotFoundError:pass
        else:raise AssertionError('Never silently substitute full-policy metrics for missing mu-only')
        (p/'config.json').write_text(json.dumps(dict(method='dipo')))
        native,path=select_result(p,raw)
        assert native['visualization_mode']=='native_policy' and native['high_density_fraction']==.5
    print('PASS: mu-only sample/metric pairing, full-policy preservation, required-mu validation, native baseline fallback')


if __name__=='__main__':main()
