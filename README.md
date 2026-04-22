# MLB-s26-Project13

Cross-species protein function prediction pipeline for human and yeast.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py --config configs/default.yaml
python scripts/prepare_annotations.py --config configs/default.yaml
python scripts/generate_embeddings.py --config configs/default.yaml --model esm2_650m
python scripts/run_similarity_search.py --config configs/default.yaml --model esm2_650m --top-k 10
python scripts/run_evaluation.py --config configs/default.yaml --model esm2_650m --top-k 10
```
