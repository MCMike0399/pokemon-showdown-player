"""Execute the read-only default notebook and save its outputs locally."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
for key, sub in {'JUPYTER_CONFIG_DIR':'config','JUPYTER_DATA_DIR':'data',
                 'JUPYTER_RUNTIME_DIR':'runtime','IPYTHONDIR':'ipython',
                 'MPLCONFIGDIR':'matplotlib'}.items():
    path = ROOT/'data/jupyter'/sub
    path.mkdir(parents=True,exist_ok=True)
    os.environ[key] = str(path)
os.environ['PS_TORCH_THREADS'] = '4'
notebook = nbformat.read(ROOT/'notebooks/01_offline_ppo.ipynb',as_version=4)
NotebookClient(notebook,timeout=300,kernel_name='pokemon-offline',
               resources={'metadata':{'path':str(ROOT)}}).execute()
output = ROOT/'artifacts/offline-notebook-executed.ipynb'
nbformat.write(notebook,output)
print(output)
