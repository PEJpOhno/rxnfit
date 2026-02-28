**先行文献**  
| 内容 | 出典 | リンク |
|------|------|--------|
| Neural ODE を化学反応速度論に適用し、stiff な化学反応 ODE を高速に解く ChemNODE を提案。PyTorch ベースで反応速度式の数値解と学習を統合。 | ChemNODE: A Neural Ordinary Differential Equations Approach for Chemical Kinetics Solvers（arXiv） | https://arxiv.org/abs/2101.04749 |
| 反応速度式を損失関数に組み込み、濃度データから速度定数を推定する PINN（Physics-Informed Neural Network）実装。PyTorch を用いて N₂O₅ 分解反応の速度定数を学習。 | Physics-Informed Neural Network for Chemical Kinetics（GitHub） | https://github.com/shehanp-dev/Physics-Informed-Neural-Network-for-modelling-reaction-kinetics |
| 化学反応速度論のニューラルネット近似を効率化するアーキテクチャを提案。PyTorch を用いて反応速度式の近似モデルを構築。 | Efficient neural network models of chemical kinetics（RSC Publishing） | https://pubs.rsc.org/en/content/articlelanding/2021/dd/d1dd00000a |  


**実装例（GitHubなど）**  
| 内容 | 出典 | リンク |
|------|------|--------|
| PyTorch + Neural ODE（torchdiffeq）を使って、化学反応速度式を数値的に解く実装。反応速度定数の推定や ODE モデルの学習に利用可能。 | torchdiffeq（Neural ODE 公式実装） | https://github.com/rtqichen/torchdiffeq |
| PyTorch ベースの動力学系ライブラリ。Neural ODE / Hamiltonian ODE / SDE などを含み、反応速度式の学習にも応用されている。 | torchdyn | https://github.com/DiffEqML/torchdyn |
| 反応速度式を物理インフォームド NN（PINN）として学習し、濃度データから速度定数を推定する PyTorch 実装。N₂O₅ 分解反応などの例を含む。 | Physics-Informed Neural Network for Chemical Kinetics | https://github.com/shehanp-dev/Physics-Informed-Neural-Network-for-modelling-reaction-kinetics |
| PyTorch を用いた化学反応ネットワーク（CRN）の学習実装。反応速度式の構造推定やパラメータ推定に利用可能。 | CRNN（Chemical Reaction Neural Network） | https://github.com/zhanglabtools/CRNN |
| PyTorch による ODE ソルバーの高速実装。大規模反応系の数値積分に向く。 | torchode | https://github.com/martenlienen/torchode |