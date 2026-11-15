"""Create an explicit private build context; no weights are copied into Git."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(assets,runtime,training,destination):
    assets=Path(assets).resolve();runtime=Path(runtime).resolve();training=Path(training).resolve()
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Preserve existing build context')
    if destination.is_relative_to(REPO):raise ValueError('Private context must be outside Git checkout')
    if any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in (assets,runtime,training)):
        raise ValueError('Context must not overlap asset/runtime/training sources')
    if not (training/'README.md').is_file():raise ValueError('Training handoff requires README.md')
    release=REPO/'llm/baseline_50_9_verified'
    for name,expected in json.loads((release/'CODE_MANIFEST.json').read_text())['files_sha256'].items():
        path=(release/name).resolve()
        if not path.is_relative_to(release) or sha(path)!=expected:
            raise ValueError('Release source differs from checksum: '+name)
    pin=json.loads((release/'assets.manifest.json').read_text())
    selected={name:info for name,info in pin['files'].items() if name.startswith('models/') or name in ('private_data/split.json','private_data/retrieval_index.json')}
    for name,info in selected.items():
        path=(assets/name).resolve()
        if not path.is_relative_to(assets) or path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:
            raise ValueError('Model asset does not match baseline pin')
    receipt=json.loads((runtime/'runtime_manifest.json').read_text())
    if (receipt['platform']!='linux-amd64-cuda12.8' or receipt['server']!='llama-server'
            or receipt['release']!='b11800' or 'llama-server' not in receipt['files']
            or receipt['archive_sha256']!='e5d489b995c2a0965ee746a246c934be9f4ffa00c78183b2d4f644f43ae5d7a7'):
        raise ValueError('Wrong runtime variant')
    for name,info in receipt['files'].items():
        path=(runtime/name).resolve()
        if not path.is_relative_to(runtime) or path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:
            raise ValueError('Changed runtime files')
    destination.mkdir(parents=True,exist_ok=False)
    for name in selected:
        path=destination/'private_assets'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(assets/name,path)
    for source,target in [(runtime,'linux_runtime'),(training,'training_code'),
                          (REPO/'ops/src','ops/src'),(release,'llm/baseline_50_9_verified')]:
        shutil.copytree(source,destination/target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store'))
    for name in ('Dockerfile','.dockerignore','requirements-linux.txt','predict.py','predict.sh','start_jupyter.sh','predict_notebook.ipynb','README.md'):
        shutil.copy2(Path(__file__).resolve().parent/name,destination/name)
    for name in ('predict.sh','start_jupyter.sh'):
        path=destination/name
        path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
        path.chmod(0o755)
    (destination/'BUILD_STATUS.json').write_text(json.dumps({'candidate':True,'gpu_validated':False,
        'training_reproduction_verified':False,'test_solution_submitted':False,
        'baseline':'50.9', 'placement':'serial', 'source_assets_manifest_sha256':sha(REPO/'llm/baseline_50_9_verified/assets.manifest.json')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','runtime','training','destination'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.assets,a.runtime,a.training,a.destination)
    print('PRIVATE_CANDIDATE_CONTEXT_PREPARED; GPU/training parity not certified')
