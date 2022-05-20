# UniLight

Source code of "Multi-Agent Reinforcement Learning for Traffic Signal Control 
through Universal Communication Method", which proposes a universal communication 
method UniComm between agents.

```
@article{DBLP:journals/corr/abs-2204-12190,
  author    = {Qize Jiang and
               Minhao Qin and
               Shengmin Shi and
               Weiwei Sun and
               Baihua Zheng},
  title     = {Multi-Agent Reinforcement Learning for Traffic Signal Control through
               Universal Communication Method},
  journal   = {CoRR},
  volume    = {abs/2204.12190},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2204.12190},
  doi       = {10.48550/arXiv.2204.12190},
  eprinttype = {arXiv},
  eprint    = {2204.12190},
}
```

# Usage

We use [CityFlow](https://github.com/zyr17/CityFlow) as traffic simulator. Note
it is a forked version of the original CityFlow, which added some functions to 
calculate some statistics in C++ instead of Python.

We suggest using Docker to run our code. Use the docker image: `zyr17/unlight`.
Alternatively, you can build an environment by your self. We run our codes on
Python 3.6.5, it should also work on higher version of Python, but the 
compatibility is not guaranteed. 

## Docker

```bash
git clone git@github.com:zyr17/UniLight.git
cd UniLight && tar zxf data.tgz
docker run -it --rm -v `pwd`:/code zyr17/unilight
```

## Build
```bash
git clone git@github.com:zyr17/CityFlow.git
cd CityFlow && pip install . --upgrade && cd ..
git clone git@github.com:zyr17/UniLight.git
cd UniLight && tar zxf data.tgz && pip install -r requirements.txt
```

## Run

For training:
```bash
cd /code && python main.py --config configs/main/UniLight.yml --cityflow-config configs/cityflow/SH1.yml
```
For testing:
```bash
cd /code && python main.py --config configs/main/UniLight.yml --cityflow-config configs/cityflow/SH1.yml --preload-model-file ${PATH_TO_MODEL_PT} --test-round 10
```

# Details

## Environment

The definition of environment is in `envs/CityFlowEnv.py`. It runs in another 
process managed by `envs/SubProcVecEnv.py`.

## Dataset

You can find the newly proposed datasets in `data/SH1` and `data/SH2`, as well
as three commonly used `JN` `HZ` and `NY` after extracting the `data.tgz` 
tarball.

## Agents

The agents and communication mainly defined in `rl/agents/NewModelAgent.py`,
which is wrapped by `rl/multiAgents/NewModelMultiAgent.py`, and its `nn.Module`
is defined in `rl/models/innerModels/NewModel.py`. 

## Arguments and Configs

We parse arguments by `utils/argument.py`. The config for CityFlow and main
process are two separate configs. All CityFlow configs are stored in 
`configs/cityflow`, and `configs/main` for main configs.

## Logs

When training starts, all logs will save in `logs/${TARGET}` folder, including
full logs, configs, CityFlow replay files, and saved models. The
target foler name is start with current time. There are two exceptions, 
output of TensorBoard is in `runs`, and output of wandb is in `wandb`. 