# TabPFN-DIN_DIP
TabPFN-Based Retrieval Model for Water Quality Parameters (DIN/DIP)

## Project Introduction

This repository implements the regression retrieval functionality for Dissolved Inorganic Nitrogen (DIN) and Dissolved Inorganic Phosphorus (DIP) based on the TabPFN model. The code corresponds to the paper "An Interpretable Transformer-Based Framework for Monitoring Dissolved Inorganic Nitrogen and Phosphorus in Jiangsu–Zhejiang–Shanghai Offshore", and includes two core types of scripts for model training and data production, which can be directly used to reproduce the relevant experimental results in the paper.

Core Dependency: TabPFN (Official Repository: https://github.com/PriorLabs/TabPFN)

## File Description

- DIN_train.py: Implements regression training for DIN based on the TabPFN model.

- DIP_train.py: Implements regression training for DIP based on the TabPFN model.

- Production_DIN.py: Loads the pre-trained DIN model weights, performs DIN parameter prediction on input data, and outputs formatted retrieval results.

- Production_DIP.py: Loads the pre-trained DIP model weights, performs DIP parameter prediction on input data, and outputs formatted retrieval results.

## Citation
```bibtex
@article{hollmann2025tabpfn,
  title={Accurate predictions on small data with a tabular foundation model},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
          Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
          Schirrmeister, Robin Tibor and Hutter, Frank},
  journal={Nature},
  year={2025},
  month={01},
  day={09},
  doi={10.1038/s41586-024-08328-6},
  publisher={Springer Nature}
}
```
## License

The code of this project is open-sourced under the Apache 2.0 License, and details can be found in the LICENSE file in the repository.

Usage Notes:

- The license terms of this project follow the requirements of the original TabPFN project. Please carefully read the official TabPFN license before use (https://github.com/PriorLabs/TabPFN/blob/main/LICENSE).

- It is only for academic research and non-commercial use. Commercial use must comply with the license terms of the original TabPFN project and this repository.

Acknowledgements

We would like to thank the TabPFN development team for providing the basic model support, as well as the technical contributions from the relevant open-source communities. Official TabPFN repository link: https://github.com/PriorLabs/TabPFN.
