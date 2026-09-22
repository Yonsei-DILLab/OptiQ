"""Collect the approved 50/100 sweep with an explicit frozen-source assertion."""
import argparse
from pathlib import Path
from antmaze.noveld_strength import collect_report as report
from .campaign import NAME, COEFFICIENTS


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--source',required=True,help='Exact training commit from registration')
    parser.add_argument('--root',type=Path,default=Path('artifacts/antmaze_noveld_strength_high_100k'))
    parser.add_argument('--local-only',action='store_true')
    args=parser.parse_args()
    assert len(args.source)==40 and all(c in '0123456789abcdef' for c in args.source)
    report.NAME=NAME;report.COEFFICIENTS=COEFFICIENTS;report.TRAINING_SHA=args.source
    status=report.read(args.root/'status.json') if args.local_only else report.collect(args.root)
    if status['phase']=='completed':
        report.render(args.root,*report.audit(args.root))
        print('REPORT_READY',args.root/'report'/'REPORT_KO.md')
    elif (args.root/'failure.json').exists():
        raise RuntimeError(report.read(args.root/'failure.json'))


if __name__=='__main__':main()
