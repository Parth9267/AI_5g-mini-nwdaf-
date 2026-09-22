# Dataset used by Mini NWDAF

These are the two original, unmodified CSV files used to train and demonstrate
the project. They are included here so a fresh clone can train the model without
downloading the data separately.

| File | Data rows | Contents |
|---|---:|---|
| [df_location.csv](df_location.csv) | 652 | Time, serving cell ID, subscriber ID and tracking area |
| [df_reg.csv](df_reg.csv) | 915 | Registration timestamps, subscriber IDs, states and durations |

## Original source and attribution

The data was published by the authors of **Enhanced Open-Source NWDAF for
Event-Driven Analytics in 5G Networks**: Henok Daniel, Omar Alhussein, Jie Liang,
Cheng Li, and Ernesto Damiani.

- [Research paper](https://arxiv.org/abs/2601.01838)
- [Authors' repository](https://github.com/HenokDanielbfg/5g-testbed-conference)
- [Original dataset folder at the pinned commit](https://github.com/HenokDanielbfg/5g-testbed-conference/tree/d6fae0dbd7cf70343201f9351fe60498b628bb86/core%20dataset/03%20Feb%202025%20-%2018%20Feb%202025)
- [Exact file links and SHA-256 checksums](../data/manifest.json)
- [Data interpretation and preprocessing decisions](../data/README.md)

Source commit: `d6fae0dbd7cf70343201f9351fe60498b628bb86`.
Source folder: `core dataset/03 Feb 2025 - 18 Feb 2025`.

The records come from the authors' simulated Free5GC/UERANSIM testbed. They were
not collected by this project's student. Ownership remains with the original
authors; this project does not grant a new license over their data. The upstream
root did not expose an explicit dataset license when inspected.

To restore these files from the pinned original source, run `python download_data.py`.
Normal setup only needs `python train.py` followed by `python -m streamlit run app.py`.
