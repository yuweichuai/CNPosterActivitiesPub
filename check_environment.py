"""Check Python imports and the required Stata commands without displaying license details."""
import importlib
from importlib import metadata

for module, package in [
    ('pandas', 'pandas'), ('numpy', 'numpy'), ('pyarrow', 'pyarrow'),
    ('scipy', 'scipy'), ('matplotlib', 'matplotlib'), ('seaborn', 'seaborn'),
    ('statsmodels', 'statsmodels'), ('sklearn', 'scikit-learn'), ('causalml.match', 'causalml'),
    ('nbformat', 'nbformat'), ('nbclient', 'nbclient'), ('stata_setup', 'stata-setup'),
]:
    importlib.import_module(module)
    print(package, metadata.version(package))

from review_setup import prepare_workspace, initialize_stata
prepare_workspace()
stata = initialize_stata()
stata.run('display "Stata version: " c(stata_version)', echo=False)
for command in ('ppmlhdfe', 'reghdfe', 'ftools', 'sdid', 'honestdid', 'coefplot'):
    stata.run('which ' + command, echo=False)
