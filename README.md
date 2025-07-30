# Detection of Obfuscation Malware: A Federated Transfer Learning-based Approach with Hybrid Neural Networks

**Manuscript ID:** IEEE LATAM Submission ID: 9689 
**Authors and Affiliation :**  
- Carlos J. Reis        - Department of Computing, São Paulo State University, Bauru-SP, Brazil
- Carlos A. C. Tojeiro  - Department of Computing, São Paulo State University, Bauru-SP, Brazil
- Thiago José Lucas     - Department. of Information Security São Paulo State College of Technology (Fatec Ourinhos), Ourinhos – SP, Brazil
- Kelton A. P. da Costa - Department of Computing, São Paulo State University, Bauru-SP, Brazil


---

## 📁 Included Scripts and Datasets

This repository contains all scripts required to reproduce the simulation and numerical results presented in the article.


- fl-tabular-MalMem2022 - repository with Scripts to reproduce test with dataset MalMem2022 descentralized and Flower framework. 
This will create a new directory called `fl-tabular` containing the following files:
```shell
fl-tabular-MalMem2022
├── data
│   ├── Obfuscated-MalMem2022.csv # Dataset of project
├── fltabular
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
```
### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fltabular` package.
```shell
# From a new python environment, run:
pip install -e .
```
## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```
You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=10
```

- fl-tabular-Malware    - repository with Scripts to reproduce test with dataset Malware dataset descentralized and Flower framework.
```shell
fl-tabular-Malware
├── data
│   ├── Malware_dataset.csv # Dataset of project
├── fltabular
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
```


### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fltabular` package.
```shell
# From a new python environment, run:
pip install -e .
```
## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```
You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=10
```
- pytorch-centralized-federated - repository with Scripts to reproduce test with datasets MalMem2022 e Malware dataset centralized and Flower framework.

```shell
pytorch-centralized-federated
├── data
│   ├── Malware_dataset.csv # Dataset of project
│   ├── Obfuscated-MalMem2022.csv # Dataset of project
├── CIC-MalMem2022.py   # Defines your CIC-MalMem2022.py model, training and data loading
├── Malware.py   # Defines your CIC-Malware.py model, training and data loading
```
### Run with the Simulation Engine
```bash
flwr run .
```
- TFF_CIC_MalMem2022 - repository with Scripts to reproduce test with dataset CIC_MalMem2022 centralized and descentralized with TFF framework.

```shell
TFF_CIC_MalMem2022
├── Obfuscated-MalMem2022.csv # Dataset of project
├── tff_cic_malware.py   # Defines your tff_cic_malware.py model, training and data loading
```
### Run with the Simulation Engine
```bash
flwr run .
```

- TFF_Malware - repository with Scripts to reproduce test with dataset Malware dataset centralized and descentralized with TFF framework.

```shell
TFF_Malware
├── Malware_dataset.csv # Dataset of project
├── tff_malware_dataset.py   # Defines your tff_malware_dataset.py model, training and data loading
```
### Run with the Simulation Engine
```bash
flwr run .

