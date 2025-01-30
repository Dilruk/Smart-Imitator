# Smart Imitator (SI)
*This is the official codebase for the paper titled "Smart Imitator: Learning from Imperfect Clinical Decision."*

## **Overview**
Smart Imitator (SI) is a two-phase reinforcement learning (RL) system designed to refine clinical decision-making by learning from **imperfect observational data**. It integrates **Adversarial Cooperative Imitation Learning (ACIL)** and **Inverse Reinforcement Learning (IRL)** to develop more effective treatment policies.

## **Features**
✅ **Structured Learning**: ACIL + IRL for refining decision policies  
✅ **Improved Treatment Outcomes**: Significant reduction in sepsis mortality and diabetes HbA1c-High rates  
✅ **Validated on Real-World Clinical Data**  

## **Project Structure and Description**
```
Smart-Imitator/
│── A_DataProcessing/                 # Data processing scripts
│   ├── a1_data_processing.py
│
│── B_ACIL/                           # Adversarial Cooperative Imitation Learning (ACIL) scripts
│   ├── b_learn_optimal_demo_policy.py
│   ├── result/                        # Stores results from ACIL phase
│   ├── SI_Trajectories/               # Stores generated trajectories
│
│── C_BC/                              # Behavior Cloning (BC) module
│   ├── bc_Utils.py
│   ├── c1_behaviourclone_nonoptimal_policy.py
│   ├── c2_behaviourclone_optimal_policy.py
│   ├── c3_blendedlearn_suboptimal_policy.py
│   ├── BC_trajectories/                # Stores behavior cloning trajectories
│
│── D_Reward_Policy_Learn/             # Reward function and policy learning
│   ├── d1_learn_reward_function.py
│   ├── d2_learn_RL_policy.py
│   ├── RewardModel.py
│
│── Data/                              # Placeholder for datasets
│
│── Utils/                             # Utility scripts
│   ├── evaluation_graphs.py
│   ├── settings.py
│   ├── utils_main.py
│   ├── utils_trajectories.py
│
│── README.md                          # Project documentation
│── requirements.txt                    # Dependencies
```


## **Running the Experiments**

### **1. Data Processing**
Process and prepare clinical datasets for model training:
```bash
python A_DataProcessing/a1_data_processing.py
```

### **2. Learning Optimal Demonstrator Policy using ACIL**
```
python B_ACIL/b_learn_optimal_demo_policy.py
```

### **3. Training Behavior Cloning Models**
#### Nonoptimal policy:
```
python C_BC/c1_behaviourclone_nonoptimal_policy.py
```

#### Optimal policy:
```
python C_BC/c2_behaviourclone_optimal_policy.py
```

#### Suboptimal blended policy:
```
python C_BC/c3_blendedlearn_suboptimal_policy.py
```

### **4. Learning Reward Function and RL Policy**
#### Learning reward function:
```
python D_Reward_Policy_Learn/d1_learn_reward_function.py

```
#### Training the RL policy using the learned reward function:
```
python D_Reward_Policy_Learn/d2_learn_RL_policy.py
```

### **Citation**
If you use this work in your research, please cite:
```
@article{10.1093/jamia/ocae320,
    author = {Perera, Dilruk and Liu, Siqi and See, Kay Choong and Feng, Mengling},
    title = {Smart Imitator: Learning from Imperfect Clinical Decisions},
    journal = {Journal of the American Medical Informatics Association},
    pages = {ocae320},
    year = {2025},
    month = {01},
    abstract = {This study introduces Smart Imitator (SI), a 2-phase reinforcement learning (RL) solution enhancing personalized treatment policies in healthcare, addressing challenges from imperfect clinician data and complex environments.Smart Imitator’s first phase uses adversarial cooperative imitation learning with a novel sample selection schema to categorize clinician policies from optimal to nonoptimal. The second phase creates a parameterized reward function to guide the learning of superior treatment policies through RL. Smart Imitator’s effectiveness was validated on 2 datasets: a sepsis dataset with 19 711 patient trajectories and a diabetes dataset with 7234 trajectories.Extensive quantitative and qualitative experiments showed that SI significantly outperformed state-of-the-art baselines in both datasets. For sepsis, SI reduced estimated mortality rates by 19.6\% compared to the best baseline. For diabetes, SI reduced HbA1c-High rates by 12.2\%. The learned policies aligned closely with successful clinical decisions and deviated strategically when necessary. These deviations aligned with recent clinical findings, suggesting improved outcomes.Smart Imitator advances RL applications by addressing challenges such as imperfect data and environmental complexities, demonstrating effectiveness within the tested conditions of sepsis and diabetes. Further validation across diverse conditions and exploration of additional RL algorithms are needed to enhance precision and generalizability.This study shows potential in advancing personalized healthcare learning from clinician behaviors to improve treatment outcomes. Its methodology offers a robust approach for adaptive, personalized strategies in various complex and uncertain environments.},
    issn = {1527-974X},
    doi = {10.1093/jamia/ocae320},
    url = {https://doi.org/10.1093/jamia/ocae320},
    eprint = {https://academic.oup.com/jamia/advance-article-pdf/doi/10.1093/jamia/ocae320/61411081/ocae320.pdf},
}
```

### **License**
This project is licensed under the Creative Commons Attribution License **(CC-BY 4.0)**.

