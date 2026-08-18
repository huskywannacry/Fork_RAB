# Ring-A-Bell


Code for reproducing the results in the paper "Ring-A-Bell! How Reliable are Concept Removal Methods For Diffusion Models?" [Link](<https://arxiv.org/abs/2310.10012>).

This checkout is retargeted from nudity / violence to the **Obama / celebrity** concept. The original notebooks are unchanged in method; they now extract an Obama CLIP concept vector and search for 100 inverse prompts.

## Framework of Ring-A-Bell
![image](https://github.com/chiayi-hsu/Ring-A-Bell/blob/main/model_architecture.png)

### 📌 Concept Extraction
The code of **Concept Extraction** is in ```Get_Concept_Vector.ipynb``` (Obama section first).
Paired prompts: ```data/Prompts_For_ConceptVector/Obama_30.csv```.
### 📌 Prompt Discovery
The code of **Prompt Discovery** is in ```InversePrompt.ipynb``` (Obama section first).
Seed prompts: ```data/Obama_seeds.csv``` (100 rows).

Or run the single script (recommended on Kaggle T4 x2):

```bash
pip install -r requirements.txt
python train_obama.py --n-prompts 100 --output Obama_invprompts.csv
```

On Kaggle: set the accelerator to **GPU T4 x2**, turn **Internet** on, open `kaggle_obama.ipynb`, and Run All. The notebook clones `https://github.com/huskywannacry/Fork_RAB.git`, uses both T4s (50 seeds each), writes incrementally, and merges to `/kaggle/working/Obama_invprompts.csv`.

Defaults match the notebooks: population 200, 3000 generations, length 16, η = 3, with early-stop patience 250.

<br>
<br>
You can get the problematic prompts after you conduct ```InversePrompt.ipynb``` or ```train_obama.py```. Then you can use problematic prompts to evaluate the effectiveness of the concept removal methods.
<br>
<br>

## Citation
🔔🔔 If you need InvPrompts for nudity, please visit ``` https://huggingface.co/datasets/Chia15/RingABell-Nudity``` and send the request.
<br><br>
Please feel free to email us at ```chiayihsu8315@gmail.com```. If this work is useful in your own research, please consider citing our work!
```
@inproceedings{
ringabell,
title={Ring-A-Bell! How Reliable are Concept Removal Methods For Diffusion Models?},
author={Yu-Lin Tsai*, Chia-Yi Hsu*, Chulin Xie, Chih-Hsun Lin, Jia-You Chen, Bo Li, Pin-Yu Chen, Chia-Mu Yu, Chun-Ying Huang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=lm7MRcsFiS}
}
```
