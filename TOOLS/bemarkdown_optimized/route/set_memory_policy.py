"""Select adaptive RAM policy or restore the prior fixed6GiB policy."""
from pathlib import Path
import argparse,json,runpy,uuid

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['auto','legacy6'])
    args=parser.parse_args()
    tool=Path(__file__).resolve().parents[1]
    module=runpy.run_path(str(tool/'core/bemarkdown/host_memory_policy.py'))
    target=tool/'HOST_MEMORY_POLICY.json';tmp=target.with_name('ram-'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps({'schema':module['SCHEMA'],'mode':args.mode},indent=2)+'\n',encoding='utf-8')
    module['replace_with_retry'](tmp,target)
    print(json.dumps(module['configured_policy'](tool)))

if __name__=='__main__':main()
