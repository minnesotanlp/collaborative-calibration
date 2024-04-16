# Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation

Code for the paper "Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation" ([R2-FM Workshop at ICLR 2024](https://openreview.net/group?id=ICLR.cc/2024/Workshop/R2-FM))

## Getting Started

```
# We suggest installing within a virtual environment
cd ~/collaborative-calibration
pip3 install -r requirements.txt
```

Input datasets are preprocessed in `collaborative-calibration/data/benchmarks/preprocessed`.

API keys (Openai, Cohere, etc.) can be kept in `~/collaborative-calibration/.env`

To run the multi-agent deliberation simulation:

```
python3 collaborative-calibration/deliberation_simulator.py \
-logging_level <logging_level> \
-input_dataset <input_dataset_name> \
-test_sample_size <test_size> \
-n_thread <number_of_threads> \
-group_size <group_size> \
--model_ensemble \
--agent_ensemble \
-logfile_path <logfile_path>
```
e.g.
```
python3 collaborative-calibration/deliberation_simulator.py \
-logging_level debug \
-input_dataset dateUnd \
-test_sample_size 300 \
-n_thread 20 \
-group_size 6 \
--model_ensemble \
--agent_ensemble \
-logfile_path "data/logs-final-dateUnd-300.log"
```

Scripts for result evaluation and analysis: `collaborative-calibration/collaborative-calibration/post_analysis.ipynb`

## Citation

```
@misc{yang2024confidence,
      title={Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation}, 
      author={Ruixin Yang and Dheeraj Rajagopal and Shirley Anugrah Hayati and Bin Hu and Dongyeop Kang},
      year={2024},
      eprint={2404.09127},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```