# Comparison partially finetuned vs. LoRA fine-tuned Encoder Models

## Dataset

<details>
  <summary>Sample Preview</summary>

We use the [**Stanford Natural Language Inference (SNLI) Corpus**](https://nlp.stanford.edu/projects/snli/#:~:text=The%20Stanford%20Natural%20Language%20Inference%20%28SNLI%29%20corpus%20%28version,classification%20with%20the%20labels%20entailment%2C%20contradiction%2C%20and%20neutral.), a benchmark dataset for natural language understanding.  
It consists of sentence pairs labeled as **entailment**, **contradiction**, or **neutral**, making it ideal for evaluating classification models on semantic inference.  

This dataset is well-suited for testing various **fine-tuning methods** in multi-class text classification.  
Its large size and balanced label distribution allow for robust performance comparisons across different model architectures and training strategies.

| Text                                                                 | Judgments     | Hypothesis                                               |
|----------------------------------------------------------------------|---------------|----------------------------------------------------------|
| A man inspects the uniform of a figure in some East Asian country.  | contradiction | The man is sleeping                                      |
| An older and younger man smiling.                                   | neutral       | Two men are smiling and laughing at the cats playing on the floor. |
| A black race car starts up in front of a crowd of people.           | contradiction | A man is driving down a lonely road.                     |
| A soccer game with multiple males playing.                          | entailment    | Some men are playing a sport.                            |
| A smiling costumed woman is holding an umbrella.                    | neutral       | A happy woman in a fairy costume holds an umbrella.      |

</details>

<details>
  <summary>EDA</summary>

- **Label Distribution**: The dataset contains a nearly equal number of samples for each class, ensuring balanced training and evaluation.  
- **Text Length Histogram**: Most texts range between 50–150 characters, with a peak around 100, supporting efficient tokenization and batching.  
- **Box Plot of Text Length**: The median text length is ~100 characters, with a compact interquartile range and some long-text outliers above 500 characters, guiding preprocessing decisions like truncation or padding.  

<img width="715" height="468" alt="image" src="https://github.com/user-attachments/assets/d5d7c49b-3e3f-465e-9ecc-c334eeb1d48c" />

<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/be1d2287-509d-4772-89b3-22925a06aa64" />

</details>


## Approach
This project investigates **parameter-efficient fine-tuning strategies** for transformer-based text classification, focusing on **Low-Rank Adaptation (LoRA)** and **partial fine-tuning**.  
The aim is to evaluate how these lightweight approaches can efficiently adapt pretrained encoder models, such as **BERT and DistilBERT**, to a specific classification task while minimizing computational and memory requirements.

The underlying assumption is that large pretrained models already capture strong general linguistic knowledge, and effective task-specific adaptation can be achieved by updating only a small subset of parameters.

**Fine-Tuning Strategies**

- **LoRA** introduces small trainable matrices within the model’s attention layers, allowing adaptation without modifying the core pretrained weights.
- **Partial fine-tuning** selectively unfreezes only certain layers, such as the final encoder block or classification head, enabling limited but targeted learning.

Both methods are designed to reduce training time and resource consumption while maintaining high performance.  
The hypothesis is that LoRA and partial fine-tuning will deliver similar classification accuracy and generalization to full fine-tuning, making them effective and scalable alternatives for fine-tuning large models in constrained computational environments.

## Results
<details>
  <summary>BERT - partially finetuned</summary>
  
  **Parameters:**
  
  We fine-tuned the model by unfreezing the last two layers of BERT, allowing them to update during training.
  
  <img width="1000" height="600" alt="BERT_par_Parameters" src="https://github.com/user-attachments/assets/97ce4138-afbb-4a89-83d3-c69b73edc344" />

  **Training:**
  
  Training was stable across three epochs, as shown by steadily decreasing train and validation loss curves. 
  Key hyperparameters included a batch size of 18, learning rate of 3e-5, dropout rate of 0.5, and weight decay of 0.1.
  
  <img width="1000" height="600" alt="BERT_par_trainValLoss" src="https://github.com/user-attachments/assets/fb845482-bdc5-4d20-814b-a41e4d4583b7" />

  **Classification Results:**

  The classification model achieved an overall accuracy of 84.98% with a test loss of 0.4151, demonstrating strong performance across all three classes. 
  Precision and recall scores were consistently high, especially for the contradiction class (F1-score: 0.88), while entailment and neutral also showed solid metrics (F1-scores: 0.86 and 0.81 respectively). 
  Despite this, around 15% of the test set—roughly 16,500 samples—were misclassified, indicating room for further optimization.
  
  <img width="407" height="442" alt="BERT_par_Results" src="https://github.com/user-attachments/assets/23324776-664b-49e7-92c1-a1b285f9ad03" />
  <img width="1000" height="800" alt="BERT_par_ConfusionMatrix" src="https://github.com/user-attachments/assets/c19d63d7-071e-4edf-9147-98e4f80e195d" />

  

</details>

<details>
  <summary>BERT - LoRA</summary>

  **Parameters:**
  
  We fine-tuned the model by applying LoRA which is a PEFT method, known for being computational efficient since it uses a lot less trainable parameters.

  <img width="1000" height="600" alt="media_images_trainable_parameters_0_de377a5c3b6f8a693a54" src="https://github.com/user-attachments/assets/fe7f76a9-26b7-438a-b200-a4800d83b37d" />

  **Training:**
  
  Training was stable across three epochs, as shown by steadily decreasing train and validation loss curves. 
  Key hyperparameters included a batch size of 6, learning rate of 1e-4, dropout rate of 0.5, and weight decay of 0.1.
  
  <img width="1000" height="600" alt="media_images_loss_curves_3_09434284645fae728a8c" src="https://github.com/user-attachments/assets/c64e7f47-fa3f-47d9-b0b9-c6c1bab5d948" />


  **Classification Results:**

  The classification model achieved an overall accuracy of 85.20% with a test loss of 0.4338, demonstrating strong performance across all three classes. 
  Precision and recall scores were consistently high, especially for the contradiction class (F1-score: 0.88), while entailment and neutral also showed solid metrics (F1-scores: 0.87 and 0.81 respectively). 
  Despite this, around 14.8% of the test set—roughly 16,262 samples—were misclassified, indicating room for further optimization.
  
  <img width="402" height="437" alt="Screenshot 2025-11-25 192448" src="https://github.com/user-attachments/assets/9d541f28-51bb-4b16-a231-61e718eeb764" />
  <img width="1000" height="800" alt="media_images_confusion_matrix_8_1fd7908f3e63ed123900" src="https://github.com/user-attachments/assets/c934b887-50cf-4430-aeef-9134abaf7520" />


</details>

<details>
  <summary>Destilled BERT - partially finetuned</summary>

  **Parameters:**
  
  We fine-tuned the model by unfreezing the last two layers of BERT, allowing them to update during training.

  <img width="1000" height="600" alt="media_images_trainable_parameters_0_bbc74d19606837d90034 (1)" src="https://github.com/user-attachments/assets/20ccb2ae-8228-4ef4-9593-c3c84c9dfb8a" />

  **Training:**
  
  Training was stable across three epochs, as shown by steadily decreasing train and validation loss curves. 
  Key hyperparameters included a batch size of 13, learning rate of 5e-5, dropout rate of 0.5, and weight decay of 0.1.
  
  <img width="1000" height="600" alt="media_images_loss_curves_3_58e15d3b8603182ba5dd" src="https://github.com/user-attachments/assets/617fe9b3-8aa8-4f33-8229-e5677c21a6b7" />

  **Classification Results:**

  The classification model achieved an overall accuracy of 81.12% with a test loss of 0.4828, showing solid performance across all three classes. 
  Precision and recall scores were strong for entailment (F1-score: 0.84) and contradiction (F1-score: 0.83), while neutral performed slightly lower (F1-score: 0.77). 
  A total of 20,742 samples were misclassified, accounting for 18.9% of the test set, indicating potential for further refinement.
  
  <img width="407" height="441" alt="Screenshot 2025-11-25 194332" src="https://github.com/user-attachments/assets/82982424-0bd3-4f34-947b-8143490ba7f5" />
  <img width="1000" height="800" alt="media_images_confusion_matrix_8_407c43f9d599f3ea75db" src="https://github.com/user-attachments/assets/ef5e3958-372b-4277-a7f0-2b4c44206f30" />

</details>

<details>
  <summary>Destilled BERT - LoRA</summary>

  **Parameters:**
  
  We fine-tuned the model by applying LoRA which is a PEFT method, known for being computational efficient since it uses a lot less trainable parameters.

  <img width="1000" height="600" alt="media_images_trainable_parameters_0_9beecd02a95806b63a11" src="https://github.com/user-attachments/assets/98be8672-d310-4407-b754-89fe3b2fe373" />

  **Training:**
  
  Training was stable across three epochs, as shown by steadily decreasing train and validation loss curves. 
  Key hyperparameters included a batch size of 8, learning rate of 1e-4, dropout rate of 0.5, and weight decay of 0.1.
  
  <img width="1000" height="600" alt="media_images_loss_curves_3_ae8c92e0575b2a53b6c3" src="https://github.com/user-attachments/assets/35459e9b-654a-4333-833d-7b5712df7f28" />

  **Classification Results:**

  The classification model reached an overall accuracy of 83.48% with a test loss of 0.4479, indicating strong and consistent performance. 
  F1-scores were high for entailment and contradiction (both 0.86), while neutral maintained a respectable score of 0.79. 
  A total of 18,151 samples were misclassified, representing 16.5% of the test set, suggesting solid generalization with room for further tuning.
  
  <img width="402" height="438" alt="Screenshot 2025-11-25 194955" src="https://github.com/user-attachments/assets/cafd8b54-bf10-48ff-b5c4-cd873b40ce63" />
  <img width="1000" height="800" alt="media_images_confusion_matrix_8_98e1452325ef83bcaf5a" src="https://github.com/user-attachments/assets/4ea4eeb5-cbf4-4e86-a81b-8e398b88cce8" />
</details>

# Analysis
<details>
  <summary>Misclassifications</summary>
  
**Label Distributions**
  - Across all models, the **neutral class** is the dominant source of misclassifications.
  - In the **true label distribution**, neutral accounts for roughly half of all mistakes.
  - In the **predicted label distribution**, all models also over-predict neutral.
  - This confirms that neutral is the most ambiguous and error-prone category.

**Model Comparison**

- **DistilBERT LoRA** shows the highest share of neutral misclassifications and stronger bias toward predicting neutral overall.
- **BERT models** (LoRA and Partial) distribute errors more evenly and show slightly better discrimination between entailment and contradiction.
  
  <p align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/36222ad1-6ae1-4c5f-af44-fa29f5df8c90" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/0a87a9b4-fbd0-4b16-8d42-b24e13b15013" />
</p>

**Probability Differences**
- Boxplots of predicted minus true probabilities show that all models exhibit large probability gaps in their misclassifications.
- Neutral again has the **widest spread**, indicating inconsistent confidence levels.
- DistilBERT LoRA tends to produce the **largest overconfidence gaps**, while BERT Partial is more stable.

  <p align="center">
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/c2d239d5-9490-435e-b5ed-1b62658dc6d8" />
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/ae4ceced-7256-4a01-8265-29c2cccecd66" />
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/0a89e74d-de9a-4e23-99f8-3e05deffaa54" />
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/deca6990-2951-4712-bf49-9650215836c3" />
</p>

**Top Misclassified Words**

Check out this [visualization](./topWordsAcrossMethods.html)

- Distil Partial dominates overall frequency: The distil_partial model consistently shows the highest counts across nearly all keywords, suggesting it identifies more instances or has broader keyword detection compared to other configurations.
- Core person-related keywords are most common: "Man" and "woman" are by far the top keywords across all models, followed by "people," indicating the models are primarily detecting human subjects in images.
Descriptor words follow closely — Action and appearance keywords like "wearing," "young," "shirt," and color terms ("white," "black," "blue") are frequently detected, suggesting the models capture both entity and attribute information.
- BERT variants underperform Distil variants: Both Distil models (LoRA and Partial) consistently report higher keyword frequencies than their BERT counterparts, with Distil Partial being the highest overall.
- Partial fine-tuning outperforms LoRA: Comparing within architecture families, the "partial" variants have higher counts than "LoRA" variants, suggesting full parameter fine-tuning captures more keywords than the parameter-efficient LoRA approach.

**Overall Interpretation**
- All models share similar weaknesses:
  - They confuse neutral examples most frequently.
  - Their probability outputs reveal overconfidence even in wrong predictions.
- LoRA and partial fine-tuning yield **comparable misclassification behavior**, though BERT variants perform slightly more consistently than DistilBERT ones.
- Fine-tuning method affects the **extent** of misclassification but not the **type** — the neutral class remains the primary challenge across all setups.
</details>

<details>
  <summary>Compute</summary>

</details>

## Final Evaluation




