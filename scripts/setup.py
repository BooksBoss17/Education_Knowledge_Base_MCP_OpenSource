"""Public API-first installer. Model weights and GPU frameworks are opt-in."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api-config',type=Path,required=True,help='User-owned four-role loopback client configuration')
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--dependency-runtime',type=Path,help='Reuse an existing compatible environment without modifying it')
    args=p.parse_args()
    if sys.version_info[:2]!=(3,11):p.error('Use CPython 3.11')
    from verify_release import verify
    verify(ROOT)
    manifest=json.loads((ROOT/'TOOLS/bemarkdown/TOOL_MANIFEST.json').read_text('utf8'))
    wheel=ROOT/'TOOLS/bemarkdown'/manifest['wheel']['path']
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=manifest['wheel']['sha256']:raise ValueError('Wheel hash mismatch')
    runtime=ROOT/'.local/runtimes'/manifest['wheel']['sha256'][:12]
    python=runtime/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():venv.EnvBuilder(with_pip=True).create(runtime)
    if args.dependency_runtime:
        donor=args.dependency_runtime.resolve()
        if donor.is_dir():donor=donor/("Scripts/python.exe" if os.name=="nt" else "bin/python")
        # Exclude the donor's BeMarkdown core; this environment owns its wheel.
        site=subprocess.check_output([str(python),'-c','import site;print(site.getsitepackages()[0])'],text=True).strip()
        donor_sites=json.loads(subprocess.check_output([str(donor),'-I','-c','import sys,json;from pathlib import Path;print(json.dumps([p for p in sys.path if Path(p).is_dir() and (Path(p).name in {"site-packages","dependencies"})]))'],text=True))
        Path(site,'public_dependencies.pth').write_text('\n'.join(donor_sites)+'\n',encoding='utf8')
    else:
        subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'scripts/requirements-core.txt')],check=True)
    subprocess.run([str(python),'-m','pip','install','--no-deps','--ignore-installed',str(wheel)],check=True)
    subprocess.run([str(python),'-I','-c','import bemarkdown,lxml,fitz,PIL,numpy,psutil,mathtypejx;print(bemarkdown.__version__)'],check=True)
    config=args.api_config.resolve();subprocess.run([str(python),'-I','-c','from bemarkdown.pdf.local_model_api import read_config;import sys;read_config(sys.argv[1])',str(config)],check=True)
    local=ROOT/'.local';(local/'model-api.json').write_bytes(config.read_bytes())
    receipt={'status':'READY_FULL','python':python.relative_to(ROOT).as_posix(),'wheel_sha256':manifest['wheel']['sha256'],
        'runtime_route':'LOCAL_MODEL_API_V1','execution_mode':'EXTERNAL_API','models_installed':False}
    pending=local/'runtime-selection.pending';pending.write_text(json.dumps(receipt,indent=2)+'\n','utf8');os.replace(pending,local/'runtime-selection.json')
    client={'mcpServers':{'education-knowledge-base':{'command':str(python),'args':['-I','-X','utf8',str(ROOT/'TOOLS/education_mcp/launch.py'),
        '--mcp-root',str(ROOT),'--workspace',str(args.workspace.resolve())],'env':{name:os.environ[name] for name in {spec.get('token_env') for spec in json.loads(config.read_text('utf8'))['roles'].values()} if name}}}}
    (local/'mcp-client.json').write_text(json.dumps(client,indent=2)+'\n','utf8')
    print('Installed API client. Client registration: '+str(local/'mcp-client.json'))


if __name__=='__main__':main()
