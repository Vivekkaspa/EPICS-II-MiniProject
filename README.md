# The PathFinder in the Pandora - An HEX-RL Agent

This code accompanies the paper [Inherently Explainable Reinforcement Learning in Natural Language](https://arxiv.org/abs/2112.08907).

## Diving into our Project (Abstract)

 The focus of this project is to develop a Reinforcement Learning (RL) agent that is inherently capable of providing clear explanations for its decisions. This agent, named ***Hierarchically Explainable Reinforcement Learning (HEX-RL)***, can offer immediate local explanations by ***thinking out loud*** as it interactswith an environment and can also conduct post-hoc analysis of entire trajectories to provide more extended explanations over time. The HEX-RL agent works in Interactive Fiction game environment where agents can interact with the world using natural language. These environments comprise of puzzles with complex dependencies where agents have to execute a sequence of actions to achieve rewards. Explainability of agent is achieved through knowledge graph representation with a Hierarchical Graph Attention mechanism that focuses on only the specific facts in its internal graph representation that influenced its action choices the most.

 ## Problem Description

-  **Input:** The input of this HEX-RL agent is a combination of Zork1 game state descriptions, actions that the agent performed previously and a knowledge graph (KG) representing entities and their relationships.
-  **Model Architecture & Representation:** The model’s architecture features knowledge graph state representation Hierarchical Graph Attention mechanism to focus on most influential elements guiding the agent’s action choices.
-   **Output:** The output consists of action selections, immediate local explanations and temporally extended explanations summarizing entire trajectories.
-   **HEX-RL Training:** Training involves reinforcement learning to maximize task-related rewards, while ensuring explainability through the attention mechanism.
-   **Evaluation & Ablation Study:** Testing assesses the agent’s performance across different game scenarios, with human evaluations used to gauge the understandability of explanations.

## Zork1 Game Description

Zork1 is text-based adventure game where the main goal is to collect all 19 treasures(rewards) found across game world. There are three types of commands for zork1 game which include directional commands like north, verb-object commands that consists of both verb and noun or noun phrase and third type is verb-object-prep-object commands that consists of verb and noun phrase followed by preposition and second noun phrase. The game then responds according to player’s commands describing what happens next. Zork1 was commercial success and it popularized the genre of Interactive Fiction.

![Screenshot (2422)](https://github.com/Vivekkaspa/EPICS-II-MiniProject/assets/110654004/78519e56-4eb8-4b81-83ee-031b6426b7fc)




## Knowledge Graph(KG) Extraction & KG State Representation

![Screenshot (2421)](https://github.com/Vivekkaspa/EPICS-II-MiniProject/assets/110654004/156ea783-1c7d-41f2-9e57-49a3ffef153b)


## Datasets for Question Answering Task in KG Prediction

- ***Stanford Question Answering Dataset (SQuAD) Dataset***  for Pretraining ALBERT-QA model
-  ***JerichoQA Dataset*** for Fine-tuning ALBERT-QA model


## Model Architecture of HEX-RL

![Screenshot (2417)](https://github.com/Vivekkaspa/EPICS-II-MiniProject/assets/110654004/f96b338a-361a-47f2-8b73-703bc47df293) 

## Temporally Extended Explanations Execution Pipeline
![Screenshot (2420) (2)](https://github.com/Vivekkaspa/EPICS-II-MiniProject/assets/110654004/fc2cb92e-6216-4e64-9f57-72a7fc064204)




## Tools and Technologies

### Programming Language 

- Python

### Important Libraries & Frameworks

-  Flask & Gunicorn
-  Stanford CoreNLP
-  NetworkX
-  Jericho Framework
-   SpaCy
-  Pattern.en.
-   Frotz
-   FuzzyWuzz
-    NumPy
-  Redis
-   PyTorch
-    Hugging face Transformers (Version2.5.1)
-    GoExplore Framework

###  Technologies & Key Domain-specific tasks

-  Deep Learning
-   Natural Language Processing(NLP)
    -  Entity Extraction
    -  Relation Extraction
    -   Sub-word Tokenization
    - Question Answering
-  Parallel Computing using T4 GPU
-   Reinforcement Learning



## QuickStart with Code

```ruby
cd qbert/extraction && gunicorn --workers 4 --bind 0.0.0.0:5000 wsgi:app
redis-server
```

* Open another terminal:
```ruby
cd qbert && python train.py --training_type base --reward_type game_only  --subKG_type QBert
```
```ruby
nohup python train.py --training_type chained --reward_type game_and_IM  --subKG_type QBert --batch_size 2 --seed 0 --preload_weights Q-BERT/qbert/logs/qbert.pt --eval_mode --graph_dropout 0 --mask_dropout 0 --dropout_ratio 0
```

#### :eye_speech_bubble: Features
* `--subKG_type`: What kind of subgraph you want to use. There are 3 choices, 'Full', 'SHA', 'QBert'.
    * 'Full': 4 subgraphs are all full graph_state.
    * 'QBert':
        1. __ 'is' __ (Attr of objects)
        2. 'you' 'have' __
        3. __ 'in' __
        4. others (direction)
    * 'SHA':
        1. room connectivity (history included)
        2. what's in current room
        3. your inventory
        4. remove you related nodes (history included)

* `--eval_mode`: Whether turning off the training and evaluation the pre-trained model
    * bool. True or False
    * use `--preload_weights` at the same time.
    
* `--random_action`: Whether to use random valid actions instead of QBERT actions.
    * bool. True or False


#### Debug Tricks
1. graph_dropout to .5 and mask_dropout to .5 in `train.py`.
2. The score should reach 5 in 10,000 steps.

## Contributions made to the existing Q-BERT 

- We added a code snippet called ***_load_bindings module*** from Jericho Framework to properly load the bindings of the .z5 extension of the Zork1 game within the environment setup module.
- We added a folder called ***Stanford CoreNLP- 2018*** Version which includes all the java files to setup the Stanford CoreNLP server. This is used in the source code especially for information extraction pin knowledge graph building and construction. This fixed a lot of path issues and environment initialization issues.
- We also included a repository clone of ***Z-machine games*** which includes ***Jericho framework game suite*** in which we find ***zork1.z5*** file. This is also something which existing source code does not specifies properly.

-------

## Team Members & Contributors to the Project

<table>
<tr>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/Vivekkaspa>
            <img src=https://avatars.githubusercontent.com/u/110654004?s=96&v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Kaspa Vivek/>
            <br />
            <sub style="font-size:14px"><b>Kaspa Vivek</b></sub>
        </a>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/yashwanth2212>
            <img src=https://avatars.githubusercontent.com/u/139617878?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Arikathota Yashwanth/>
            <br />
            <sub style="font-size:14px"><b>Arikathota Yashwanth</b></sub>
        </a>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/Mukesh-Eppili19>
            <img src=https://avatars.githubusercontent.com/u/132339498?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Eppili Mukesh/>
            <br />
            <sub style="font-size:14px"><b>Eppili Mukesh</b></sub>
        </a>
    </td>
</tr>
</table>





