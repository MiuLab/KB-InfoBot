KB-InfoBot
==================================================

This repository contains all the code and data accompanying the paper [Towards End-to-End Reinforcement Learning of Dialogue Agents for Information Access](https://arxiv.org/abs/1609.00777).

Prerequisites
--------------------------------------------------
See [requirements.txt](./requirements.txt) for required packacges. Also download nltk data:
```sh
python -m nltk.downloader all
```
IMPORTANT: Download the data and pretrained models from [here](https://drive.google.com/file/d/0B7aCzQIaRTDUMDF1S3NVajlHTFk/view?usp=sharing), unpack the tar and place it at the root of the repository.

Code Organization
--------------------------------------------------
* All agents are in [deep_dialog/agents/](./deep_dialog/agents/) directory
* The user-simulator along with a template based and seq2seq NLG is in [deep_dialog/usersims/](./deep_dialog/usersims/) directory
* [deep_dialog/dialog_system/](./deep_dialog/dialog_system/) contains classes for dialog manager and the database

Interact with the pre-trained InfoBot!
--------------------------------------------------
```sh
$ python interact.py
```

This will launch the command line tool running the RL-SoftKB infobot trained on the "Medium-KB" split. Instructions on how to interact the system are displayed within the tool itself. You can also specify other agents to test:

```sh
$ python interact.py --help
usage: interact.py [-h] [--agent AGENT]

optional arguments:
  -h, --help     show this help message and exit
  --agent AGENT  Agent to run -- (rule-no / rl-no / rule-hard / rl-hard /
                 rule-soft / rl-soft / e2e-soft
```

Training
--------------------------------------------------
To train the RL agents, call `train.py` with the following options:
```sh
$ python train.py --help
usage: train.py [-h] [--agent AGENT_TYPE] [--db DB] [--model_name MODEL_NAME]
                [--N N] [--max_turn MAX_TURN] [--nlg_temp NLG_TEMP]
                [--max_first_turn MAX_FIRST_TURN] [--err_prob ERR_PROB]
                [--dontknow_prob DONTKNOW_PROB] [--sub_prob SUB_PROB]
                [--reload RELOAD]

optional arguments:
  -h, --help            show this help message and exit
  --agent AGENT_TYPE    agent to use (rl-no / rl-hard / rl-soft / e2e-soft)
  --db DB               imdb-(S/M/L/XL) -- This is the KB split to use, e.g.
                        imdb-M
  --model_name MODEL_NAME
                        model name to save
  --N N                 Number of simulations
  --max_turn MAX_TURN   maximum length of each dialog (default=20, 0=no
                        maximum length)
  --nlg_temp NLG_TEMP   Natural Language Generator softmax temperature (to
                        control noise)
  --max_first_turn MAX_FIRST_TURN
                        Maximum number of slots informed by user in first turn
  --err_prob ERR_PROB   the probability of the user simulator corrupting a
                        slot value
  --dontknow_prob DONTKNOW_PROB
                        the probability that user simulator does not know a
                        slot value
  --sub_prob SUB_PROB   the probability that user simulator substitutes a slot
                        value
  --reload RELOAD       Reload previously saved model (0-no, 1-yes)
```
Example:
```sh
python train.py --agent e2e-soft --db imdb-M --model_name e2e_soft_example.m
```

Testing
----------------------------------------------------
To evaluate both RL and Rule agents, call `sim.py` with the following options:
```sh
$ python sim.py --help
usage: sim.py [-h] [--agent AGENT_TYPE] [--N N] [--db DB]
              [--max_turn MAX_TURN] [--err_prob ERR_PROB]
              [--dontknow_prob DONTKNOW_PROB] [--sub_prob SUB_PROB]
              [--nlg_temp NLG_TEMP] [--max_first_turn MAX_FIRST_TURN]
              [--model_name MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --agent AGENT_TYPE    agent to use (rule-no / rl-no / rule-hard / rl-hard /
                        rule-soft / rl-soft / e2e-soft)
  --N N                 Number of simulations
  --db DB               imdb-(S/M/L/XL) -- This is the KB split to use, e.g.
                        imdb-M
  --max_turn MAX_TURN   maximum length of each dialog (default=20, 0=no
                        maximum length)
  --err_prob ERR_PROB   the probability of the user simulator corrupting a
                        slot value
  --dontknow_prob DONTKNOW_PROB
                        the probability that user simulator does not know a
                        slot value
  --sub_prob SUB_PROB   the probability that user simulator substitutes a slot
                        value
  --nlg_temp NLG_TEMP   Natural Language Generator softmax temperature (to
                        control noise)
  --max_first_turn MAX_FIRST_TURN
                        Maximum number of slots informed by user in first turn
  --model_name MODEL_NAME
                        model name to evaluate (This should be the same as
                        what you gave for training). Pass "pretrained" to use
                        pretrained models.
```
Run without the `--model_name` argument to test on pre-trained models. Example:
```sh
python sim.py --agent rl-soft --db imdb-M
```

Hyperparameters
-------------------------------------------------
The default hyperparameters for each KB split are in `settings/config_<db_name>.py`. These include:
1. RL agent options-
  * `nhid`: Number of hidden units
  * `batch`: Batch size
  * `ment`: Entropy regularization parameter
  * `lr`: Learning rate for initial supervised learning of policy. RL learning rate is fixed to 0.005.
  * `featN`: Only for end-to-end RL agent, *n* for n-gram feature extraction
  * `pol_start`: Number of supervised learning updates before switching to RL
  * `input`: Input type to the policy network - full/entropy
  * `sl`: Only for end-to-end RL agent, Type of supervised learning (bel-only belief tracker, pol-only policy, e2e (default)-both)
  * `rl`: Only for end-to-end RL agent, Type of reinforcement learning (bel-only belief tracker, pol-only policy, e2e (default)-both)
2. Rule agent options-
  * `tr`: Threshold for databse entropy to inform
  * `ts`: Threshold for slot entropy to request
  * `max_req`: Maximum requests allowed per slot
  * `frac`: Ratio to initial slot entropy, below which if the slot entropy falls it is not requested anymore
  * `upd`: Update count for bayesian belief tracking

## Note
Make sure to add `THEANO_FLAGS=device=cpu,floatX=float32` before any command if you are running on a CPU.

## Contributors
If you use this code please cite the following:

Dhingra, B., Li, L., Li, X., Gao, J., Chen, Y. N., Ahmed, F., & Deng, L. (2017). Towards End-to-end reinforcement learning of dialogue agents for information access. ACL.
```
@inproceedings{dhingra2017towards,
  title={Towards End-to-end reinforcement learning of dialogue agents for information access},
  author={Dhingra, Bhuwan and Li, Lihong and Li, Xiujun and Gao, Jianfeng and Chen, Yun-Nung and Ahmed, Faisal and Deng, Li},
  booktitle={Proceddings of ACL},
  year={2017}
}
```

Report bugs and missing info to bdhingraATandrewDOTcmuDOTedu (replace AT, DOT appropriately).
