# FedRAG: A Recall-Driven Byzantine Defense for Federated Retrieval Models

**FedRAG** is a privacy-preserving federated embedding architecture designed for dense Information Retrieval (IR) systems. It introduces a task-grounded evaluation mechanism to detect and mitigate Byzantine attacks in federated representation learning.

Unlike conventional proximity-based aggregation defenses such as Krum and coordinate-wise median, FedRAG evaluates intercepted client updates against a held-out validation query set using **Recall@10**. Updates that degrade retrieval performance trigger an exponential Soft Trust Decay mechanism, helping isolate malicious clients executing sign-flipping, Gaussian noise, or stealthy semantic-space corruption attacks.

## Table of Contents

- [Overview](#overview)
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Empirical Results](#empirical-results)
- [Technologies](#technologies)
- [Citation](#citation)

## Overview

Federated learning enables multiple clients to collaboratively train a shared model without centralizing their private datasets. However, malicious participants can manipulate model updates and compromise the quality of the resulting global model.

FedRAG addresses this challenge through retrieval-aware validation and trust-weighted aggregation.

Key features include:

- **Retrieval-aware Byzantine detection:** Evaluates candidate client updates using Recall@10 instead of relying exclusively on geometric similarity.
- **Soft Trust Decay:** Reduces the persistent trust assigned to clients whose updates degrade retrieval performance.
- **Privacy-preserving training:** Uses local training and bounded L2 gradient clipping to limit the exposure of private information.
- **Trust-weighted aggregation:** Aggregates model parameters using normalized trust weights from active clients.
- **Non-IID data support:** Evaluates federated retrieval across heterogeneous datasets representing different domains.
- **Retrieval-focused inference:** Uses an in-memory Qdrant vector index to retrieve document pointers and cosine relevance scores without requiring a generative LLM.

## Architecture Overview

The FedRAG pipeline consists of six stages.

1. **Local Contrastive Fine-Tuning**

   Honest clients fine-tune a local Sentence Transformer (`all-MiniLM-L6-v2`) on their private corpora using Multiple Negatives Ranking Loss (MNRL).

2. **Differential Privacy Shield**

   Client updates undergo bounded L2 norm clipping to constrain update magnitudes and help limit information leakage.

3. **Server Intercept**

   The central server intercepts candidate client updates before they are incorporated into the global model.

4. **Evaluation Gate: Recall@10 Validation**

   Each candidate update is applied to an isolated temporary global model. The resulting model is evaluated against a held-out query validation set to measure its retrieval performance.

5. **Soft Trust Decay**

   Updates that degrade retrieval performance trigger an exponential reduction in the corresponding client's persistent trust score. The decay constant is set to \(\gamma = 0.5\).

6. **Safe Aggregation**

   The server aggregates model parameters using normalized trust weights, dynamically excluding blocked clients from the active aggregation process.

### High-Level Workflow

```text
Private Client Datasets
          |
          v
Local Contrastive Training
          |
          v
L2 Gradient Clipping
          |
          v
Client Model Updates
          |
          v
Server Intercept
          |
          v
Recall@10 Evaluation Gate
          |
          v
Soft Trust Decay
          |
          v
Trust-Weighted Aggregation
          |
          v
Updated Global Retrieval Model
```

## Repository Structure

```text
Fedrag/
├── client/
│   ├── client.py          # Flower client initialization and communication
│   ├── local_train.py     # MNRL contrastive fine-tuning logic
│   └── data_loader.py     # BEIR dataset ingestion and partitioning
├── server/
│   ├── server.py          # Flower server orchestration
│   ├── strategy.py        # Trust-weighted aggregation strategy
│   └── validator.py       # Recall@10 evaluation and Soft Trust Decay
├── utils/
│   ├── metrics.py         # Recall@10, MRR@10, and NDCG@10
│   └── attacks.py         # Byzantine attack simulations
├── requirements.txt       # Python dependencies
└── README.md
```

## Prerequisites

Before running FedRAG, ensure that your environment has the following:

- Python 3.9 or later
- `pip`
- Git
- An NVIDIA GPU with a compatible CUDA-enabled PyTorch installation for efficient GPU-accelerated contrastive fine-tuning

CPU execution may be possible, but training performance will depend on dataset size and available hardware.

## Installation

Clone the repository and install its dependencies:

```bash
git clone https://github.com/bishnu2005/Fedrag.git
cd Fedrag
pip install -r requirements.txt
```

For GPU acceleration, ensure that your installed PyTorch version is compatible with your NVIDIA driver and CUDA environment.

## Datasets

FedRAG uses three datasets from the [BEIR benchmark](https://github.com/beir-cellar/beir) to evaluate retrieval performance across heterogeneous, non-IID client environments.

| Dataset | Domain | Experimental Role |
|---|---|---|
| TREC-COVID | Biomedical research and virology | Honest client |
| NFCorpus | Medical and nutritional information retrieval | Honest client |
| SciFact | Scientific claim verification | Byzantine client in the targeted poisoning experiment |

Each client operates on its assigned local dataset, simulating a federated environment in which participants possess different data distributions and retrieval requirements.

## Usage

The following commands illustrate the intended execution workflow. Run the server and each client in separate terminal sessions.

### 1. Start the FedRAG Server

Initialize the central Flower server with the custom aggregation strategy:

```bash
python server/server.py --rounds 3 --threshold 0.25
```

- `--rounds 3`: Configures three federated training rounds.
- `--threshold 0.25`: Sets the configured validation or trust threshold, depending on the server implementation.

### 2. Launch Honest Clients

Start the honest clients using their respective local datasets:

```bash
python client/client.py --dataset trec-covid --cid 1
```

In another terminal:

```bash
python client/client.py --dataset nfcorpus --cid 2
```

### 3. Launch a Byzantine Client

Simulate a malicious client using a targeted sign-flipping attack:

```bash
python client/client.py --dataset scifact --cid 3 --attack sign_flip --eta 5.0
```

The attack parameter \(\eta = 5.0\) controls the sign-flipping attack magnitude according to the implementation.

A Gaussian noise attack can also be configured with \(\sigma = 0.5\), provided the client implementation supports the corresponding attack arguments.

### Implementation Note

The commands above assume that the server and client scripts expose the specified command-line arguments. Ensure that the argument names, client identifiers, dataset identifiers, and network configuration match the actual implementation in the repository.

## Empirical Results

The reported experiments evaluate FedRAG against conventional federated aggregation baselines under targeted poisoning attacks.

| Method | Recall@10 | False Positive Rate |
|---|---:|---:|
| FedAvg | 0.426* | Not specified |
| FedMedian | 0.426* | Not specified |
| **FedRAG** | **0.948** | **0.0%** |

\*The reported Recall@10 value of 0.426 represents the degraded baseline performance described in the experimental results; individual baseline measurements should be reported separately if they differ.

The reported results indicate that FedRAG maintains stronger retrieval utility under poisoning attacks while preserving the utility of honest clients across heterogeneous data distributions.

### Key Findings

- **Higher retrieval utility:** FedRAG achieves a reported Recall@10 of 0.948.
- **Improved attack resilience:** Recall-driven validation identifies updates that negatively affect retrieval quality.
- **Honest-client preservation:** The reported false positive rate is 0.0%.
- **Support for heterogeneous data:** Validation is grounded in the retrieval task rather than relying solely on similarities between client updates.

These figures reflect the reported experimental configuration. Reproducibility depends on the dataset splits, validation queries, attack parameters, random seeds, and baseline configurations.

## Technologies

FedRAG is built around the following technologies:

- **Python:** Core implementation language
- **Flower (`flwr`):** Federated learning orchestration
- **PyTorch (`torch`):** Model training and tensor operations
- **Sentence Transformers:** Dense embedding generation and contrastive fine-tuning
- **BEIR:** Information retrieval benchmarks and datasets
- **Qdrant Client:** Vector indexing and similarity search
- **NumPy and SciPy:** Numerical computation and supporting evaluation utilities

Install the complete dependency set through `requirements.txt`.

## Citation

If you use FedRAG in your research, please cite the accompanying paper:

```bibtex
@inproceedings{mangaraj2026fedrag,
  title     = {Degrade to Decay: A Recall-Driven Byzantine Defense for Federated Retrieval Models},
  author    = {Mangaraj, Bishnu Prasad and Pandey, Tanishq and S., Gayathri},
  booktitle = {IEEE},
  year      = {2026}
}
```

Please replace the abbreviated venue information with the full conference name and publication details when they become available.

## License

No license information has been specified in this repository. Add a `LICENSE` file before distributing the code under an open-source license.

---

**FedRAG: Degrade to Decay. Defend the retrieval task, not just the geometry of model updates.**
