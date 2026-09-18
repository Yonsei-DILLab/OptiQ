import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import wandb
from optiq_dime.runtime import load_environment, provenance


@pytest.fixture(scope="session", autouse=True)
def validation_run(request):
    # Portable local verification must not require credentials or create runs.
    # Explicitly opt in to the historical online test reporting when needed.
    if os.environ.get("OPTIQ_TEST_WANDB") != "1":
        yield None
        return
    load_environment()
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "optiq_dime_no_anchor"),
        entity=os.environ.get("WANDB_ENTITY"), mode="online",
        job_type="validation", name="no-anchor-regression-tests",
        config={"runtime": provenance()}, save_code=False,
    )
    yield run
    run.summary["tests_finished"] = True
    run.finish(exit_code=int(request.session.testsfailed > 0))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and wandb.run is not None:
        wandb.run.log({"test_passed": int(report.passed), "test": item.nodeid})
        if report.failed:
            wandb.run.summary["tests_failed"] = True
