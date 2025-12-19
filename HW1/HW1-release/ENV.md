# One way of setting of environment on PACE-ICE for HW1 - uv + Jupyter Notebook

## Step One
Go to https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard and "Interactive Apps" -> "Jupyter".

Setting the dropdown as follows:
- Python Environment: Anaconda3 2023.03
- Jupyter Interface: Jupyter Notebook
- Quality of Service: coc-ice
- Node Type: NVIDIA GPU A40 (Note that you can choose other GPU nodes as long as they have enough VRAM but some GPU types such as V100 might cause cascaded dependency conflict issues.)
- Nodes: 1 (change as needed)
- Cores Per Node: 1 (change as needed)
- GPUs Per Node: 1 (change as needed)
- Memory Per Core (GB): 64 (change as needed)
- Number of hours: 2 (change as needed, max number = 8)

Then click "Launch"

## Step Two
Download the HW1 folder to location you desire and create a terminal by "New" -> "Terminal". 

You should see `(base)` on the very left which means PACE has preloaded the chosen base conda environment (i.e. Anaconda3 2023.03 we chose above). 

To use uv instead to manage environment, you should first use `conda deactivate` to deactivate current conda container. 

Then use the following commands for initiating, activating uv environments, installing dependencies (including ipykernel which is required for the Jupyter Notebook) and configuring kernal names.

- `uv init`
- `source .venv/bin/activate`
- `uv sync`
- `uv pip install -r requirements.txt`
- `uv pip install ipykernel`
- `python -m ipykernel install --user --name="nlp-hw1-uv" --display-name="my_nlp_hw1_env"`

## Step Three
Hurray! You are done creating a uv kernal for your Jupyter Notebook. 

As the last step, open the .ipynb Notebook (CS4650_hw1_release_fa2025.ipynb) and change the kernal by "Kernel" -> "Change kernel" -> "the kernal name you set in the last command in step two"

Now you should be good to go! You might need to change some package versions to make all packages compatible. For more details on how to use uv for environment management, check out: https://github.com/astral-sh/uv