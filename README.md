# *POLICEd-RL:* Learning to Provably Satisfy High Relative Degree Constraints for Black-Box Systems
Official Code Repository for the high relative degree POLICEd-RL Paper: https://arxiv.org/abs/2407.20456
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Repository containing code to implement [high relative degree POLICEd RL](https://arxiv.org/pdf/2407.20456.pdf) presented at [CDC 2024](https://cdc2024.ieeecss.org/).
The objective of POLICEd RL is to *guarantee* the satisfaction of an affine *hard constraint* of *high relative degree*
when learning a policy in closed-loop with a black-box deterministic environment.
The algorithm enforces a repulsive buffer in front of the constraint preventing trajectories to approach and violate this constraint.
To analytically verify constraint satisfaction, the policy is made affine in that repulsive buffer using the [POLICE](https://arxiv.org/pdf/2211.01340.pdf) algorithm.

POLICEd RL guarantees that this space shuttle will never land with a vertical velocity higher than 6ft/s thanks to the green repulsive buffer.

![POLICEd RL learns to reach the target while avoiding the constraint](media/shuttle_gif.gif)


We provide the code for our implementation of POLICEd RL on several systems:
- the Gymnasium Inverted Pendulum
- a space shuttle landing






## Organization
- [POLICEdRL](POLICEdRL) contains the project source code,
- [docs](docs) contains the code for our website.



## Credit
The following repositories have been instrumental from both an algorithm and
software architecture perspective in the development of this project:
- [RandallBalestriero/POLICE](https://github.com/RandallBalestriero/POLICE)
- [sfujim/TD3](https://github.com/sfujim/TD3)
- [labicon/POLICEd RL](https://github.com/labicon/POLICEd-RL/tree/main)
