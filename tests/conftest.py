"""Release framework allocators between heavy CPU integration tests."""
import os
import subprocess
import sys


def pytest_pyfunc_call(pyfuncitem):
    heavy = ('original_architecture', 'medical_notebook_cells', 'fastai_',
             'real_unet', 'cli_training', 'saved_checkpoint')
    if os.environ.get('CV_ISOLATED_TEST') or not any(name in pyfuncitem.name for name in heavy):
        return None
    env = dict(os.environ, CV_ISOLATED_TEST='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', pyfuncitem.nodeid],
                            cwd=pyfuncitem.config.rootpath, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return True
