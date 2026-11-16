from pathlib import Path
import json,runpy
tool=Path(__file__).resolve().parent
runpy.run_path(str(tool/"engine.py"))["engine_root"](tool)
from bemarkdown.pdf.local_model_api import read_config
from bemarkdown.model_session import SESSION_SCHEMA
read_config(tool.parents[1]/".local/model-api.json")
print(json.dumps({"status":"READY_EXTERNAL_API_CONFIGURED","remote_endpoints_probed":False,"optional_session_protocol":SESSION_SCHEMA}))
