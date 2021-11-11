import pkg_resources
from subprocess import call

packages = [dist.project_name for dist 
    in pkg_resources.working_set]

for p in packages:
    if p.__contains__('-'):
        packages.pop(packages.index(p))
call("pip install --upgrade " + ' '.join(packages), shell=True)