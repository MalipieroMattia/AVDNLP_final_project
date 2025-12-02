# Comparison of partially finetuned vs. LoRA fine-tuned BERT and DistilBERT Models

Mattia Malipiero 

Johannes Stärk 

## Central Problem & Domain
This project investigates how Low-Rank Adaptation (LoRA) compares to partial fine-tuning for text classification tasks using transformer-based models.
We focus on three guiding questions:

- **Classification Performance:** <br>
How does LoRA compare to partial fine-tuning in overall classification accuracy and F1-score?
- **Resource Efficiency:** <br>
How much GPU memory, time, and storage does LoRA save compared to partial fine-tuning?
- **Error Behavior:** <br>
Do misclassification patterns differ across models and fine-tuning methods?

Our hypotheses are that LoRA will achieve similar classification accuracy while requiring fewer trained parameters and less computational effort, and that different fine-tuning methods may lead to distinct misclassification patterns across models.

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

- **Label Distribution**: The dataset contains a nearly equal number of samples for each class, ensuring balanced training and evaluation  
- **Text Length Histogram**: Most texts range between 50–150 characters, with a peak around 100, supporting efficient tokenization and batching 
- **Box Plot of Text Length**: The median text length is ~100 characters, with a compact interquartile range and some long-text outliers above 500 characters, guiding preprocessing decisions like truncation or padding 

<img width="715" height="468" alt="image" src="https://github.com/user-attachments/assets/d5d7c49b-3e3f-465e-9ecc-c334eeb1d48c" />

<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/be1d2287-509d-4772-89b3-22925a06aa64" />

</details>


## Approach
This project investigates **parameter-efficient fine-tuning strategies** for transformer-based text classification, focusing on **Low-Rank Adaptation (LoRA)** and **partial fine-tuning**.  
The aim is to evaluate how these lightweight approaches can efficiently adapt pretrained encoder models, such as **BERT and DistilBERT**, to a specific classification task while minimizing computational and memory requirements.

The underlying assumption is that large pretrained models already capture strong general linguistic knowledge, and effective task-specific adaptation can be achieved by updating only a small subset of parameters.

### Fine-Tuning Strategies

- **LoRA** introduces small trainable matrices within the model’s attention layers, allowing adaptation without modifying the core pretrained weights
- **Partial fine-tuning** selectively unfreezes only certain layers, such as the final encoder block or classification head, enabling limited but targeted learning

Both methods are designed to reduce training time and resource consumption while maintaining high performance.  
The hypothesis is that LoRA and partial fine-tuning will deliver similar classification accuracy and generalization to full fine-tuning, making them effective and scalable alternatives for fine-tuning large models in constrained computational environments.

## Results
<details>
  <summary>BERT - partially finetuned</summary>
  
  ### Parameters
  
  We fine-tuned the model by unfreezing the last two layers of BERT, allowing them to update during training.
  
  <img width="1000" height="600" alt="BERT_par_Parameters" src="https://github.com/user-attachments/assets/97ce4138-afbb-4a89-83d3-c69b73edc344" />

  **Training:**
  
  Training was stable across three epochs, as shown by steadily decreasing train and validation loss curves. 
  Key hyperparameters included a batch size of 18, learning rate of 3e-5, dropout rate of 0.5, and weight decay of 0.1.
  
  <img width="1000" height="600" alt="BERT_par_trainValLoss" src="https://github.com/user-attachments/assets/fb845482-bdc5-4d20-814b-a41e4d4583b7" />

  ### Classification Results

  The classification model achieved an overall accuracy of 84.98% with a test loss of 0.4151, demonstrating strong performance across all three classes. 
  Precision and recall scores were consistently high, especially for the contradiction class (F1-score: 0.88), while entailment and neutral also showed solid metrics (F1-scores: 0.86 and 0.81 respectively). 
  Despite this, around 15% of the test set—roughly 16,500 samples—were misclassified, indicating room for further optimization.
  
  <img width="407" height="442" alt="BERT_par_Results" src="https://github.com/user-attachments/assets/23324776-664b-49e7-92c1-a1b285f9ad03" />
  <img width="1000" height="800" alt="BERT_par_ConfusionMatrix" src="https://github.com/user-attachments/assets/c19d63d7-071e-4edf-9147-98e4f80e195d" />

  

</details>

<details>
  <summary>BERT - LoRA</summary>

  ### Parameters
  
  We fine-tuned the model by applying LoRA which is a PEFT method, known for being computational efficient since it uses a lot less trainable parameters.

  <img width="1000" height="600" alt="media_images_trainable_parameters_0_de377a5c3b6f8a693a54" src="https://github.com/user-attachments/assets/fe7f76a9-26b7-438a-b200-a4800d83b37d" />

  ### Training
  
  Training was stable across three epochs, as shown by steadily decreasing train and validation loss curves. 
  Key hyperparameters included a batch size of 6, learning rate of 1e-4, dropout rate of 0.5, and weight decay of 0.1.
  
  <img width="1000" height="600" alt="media_images_loss_curves_3_09434284645fae728a8c" src="https://github.com/user-attachments/assets/c64e7f47-fa3f-47d9-b0b9-c6c1bab5d948" />


  ### Classification Results

  The classification model achieved an overall accuracy of 85.20% with a test loss of 0.4338, demonstrating strong performance across all three classes. 
  Precision and recall scores were consistently high, especially for the contradiction class (F1-score: 0.88), while entailment and neutral also showed solid metrics (F1-scores: 0.87 and 0.81 respectively). 
  Despite this, around 14.8% of the test set—roughly 16,262 samples—were misclassified, indicating room for further optimization.
  
  <img width="402" height="437" alt="Screenshot 2025-11-25 192448" src="https://github.com/user-attachments/assets/9d541f28-51bb-4b16-a231-61e718eeb764" />
  <img width="1000" height="800" alt="media_images_confusion_matrix_8_1fd7908f3e63ed123900" src="https://github.com/user-attachments/assets/c934b887-50cf-4430-aeef-9134abaf7520" />


</details>

<details>
  <summary>Destilled BERT - partially finetuned</summary>

  ### Parameters
  
  We fine-tuned the model by unfreezing the last two layers of BERT, allowing them to update during training.

  <img width="1000" height="600" alt="media_images_trainable_parameters_0_bbc74d19606837d90034 (1)" src="https://github.com/user-attachments/assets/20ccb2ae-8228-4ef4-9593-c3c84c9dfb8a" />

  ### Training
  
  Training was stable across three epochs, as shown by steadily decreasing train and validation loss curves. 
  Key hyperparameters included a batch size of 13, learning rate of 5e-5, dropout rate of 0.5, and weight decay of 0.1.
  
  <img width="1000" height="600" alt="media_images_loss_curves_3_58e15d3b8603182ba5dd" src="https://github.com/user-attachments/assets/617fe9b3-8aa8-4f33-8229-e5677c21a6b7" />

  ### Classification Results

  The classification model achieved an overall accuracy of 81.12% with a test loss of 0.4828, showing solid performance across all three classes. 
  Precision and recall scores were strong for entailment (F1-score: 0.84) and contradiction (F1-score: 0.83), while neutral performed slightly lower (F1-score: 0.77). 
  A total of 20,742 samples were misclassified, accounting for 18.9% of the test set, indicating potential for further refinement.
  
  <img width="407" height="441" alt="Screenshot 2025-11-25 194332" src="https://github.com/user-attachments/assets/82982424-0bd3-4f34-947b-8143490ba7f5" />
  <img width="1000" height="800" alt="media_images_confusion_matrix_8_407c43f9d599f3ea75db" src="https://github.com/user-attachments/assets/ef5e3958-372b-4277-a7f0-2b4c44206f30" />

</details>

<details>
  <summary>Destilled BERT - LoRA</summary>

  ### Parameters
  
  We fine-tuned the model by applying LoRA which is a PEFT method, known for being computational efficient since it uses a lot less trainable parameters.

  <img width="1000" height="600" alt="media_images_trainable_parameters_0_9beecd02a95806b63a11" src="https://github.com/user-attachments/assets/98be8672-d310-4407-b754-89fe3b2fe373" />

  ### Training
  
  Training was stable across three epochs, as shown by steadily decreasing train and validation loss curves. 
  Key hyperparameters included a batch size of 8, learning rate of 1e-4, dropout rate of 0.5, and weight decay of 0.1.
  
  <img width="1000" height="600" alt="media_images_loss_curves_3_ae8c92e0575b2a53b6c3" src="https://github.com/user-attachments/assets/35459e9b-654a-4333-833d-7b5712df7f28" />

  ### Classification Results

  The classification model reached an overall accuracy of 83.48% with a test loss of 0.4479, indicating strong and consistent performance. 
  F1-scores were high for entailment and contradiction (both 0.86), while neutral maintained a respectable score of 0.79. 
  A total of 18,151 samples were misclassified, representing 16.5% of the test set, suggesting solid generalization with room for further tuning.
  
  <img width="402" height="438" alt="Screenshot 2025-11-25 194955" src="https://github.com/user-attachments/assets/cafd8b54-bf10-48ff-b5c4-cd873b40ce63" />
  <img width="1000" height="800" alt="media_images_confusion_matrix_8_98e1452325ef83bcaf5a" src="https://github.com/user-attachments/assets/4ea4eeb5-cbf4-4e86-a81b-8e398b88cce8" />
</details>

# Analysis
<details>
  <summary>Misclassifications</summary>
  
### Top Misclassified Words

  <p align="center">
  <img width="1200" height="700" alt="topWordsMisclassifications" src="https://github.com/user-attachments/assets/58915b74-dda7-4a9d-92d1-50c3e4219513" />
</p>

- Same top words across models: All four methods misclassify the same core set of words — man, woman, people, wearing, and shirt/young
- Stable ranking: The relative order of misclassifications is nearly identical (e.g., man always #1, woman always #2)
- Human-related bias: The most misclassified words are generic, high-frequency human-appearance terms, suggesting shared difficulty rather than model-specific issues
- Magnitude varies, pattern doesn’t: Raw counts differ (highest in distil_partial), but the relative scale is consistent across methods

  
### Label Distributions
  - Across all models, the **neutral class** is the dominant source of misclassifications.
  - In the **true label distribution**, neutral accounts for roughly half of all mistakes.
  - In the **predicted label distribution**, all models also over-predict neutral.
  - This confirms that neutral is the most ambiguous and error-prone category.

### Model Comparison

- **DistilBERT LoRA** shows the highest share of neutral misclassifications and stronger bias toward predicting neutral overall
- **BERT models** (LoRA and Partial) distribute errors more evenly and show slightly better discrimination between entailment and contradiction
  
  <p align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/36222ad1-6ae1-4c5f-af44-fa29f5df8c90" />
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/0a87a9b4-fbd0-4b16-8d42-b24e13b15013" />
</p>

### Probability Differences
- Boxplots of predicted minus true probabilities show that all models exhibit large probability gaps in their misclassifications
- Neutral again has the **widest spread**, indicating inconsistent confidence levels
- DistilBERT LoRA tends to produce the **largest overconfidence gaps**, while BERT Partial is more stable

  <p align="center">
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/c2d239d5-9490-435e-b5ed-1b62658dc6d8" />
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/ae4ceced-7256-4a01-8265-29c2cccecd66" />
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/0a89e74d-de9a-4e23-99f8-3e05deffaa54" />
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/deca6990-2951-4712-bf49-9650215836c3" />
</p>

### Overall Interpretation

- All models share similar weaknesses:
  - They confuse neutral examples most frequently
  - Their probability outputs reveal overconfidence even in wrong predictions
- LoRA and partial fine-tuning yield **comparable misclassification behavior**, though BERT variants perform slightly more consistently than DistilBERT ones
- Fine-tuning method affects the **extent** of misclassification but not the **type** — the neutral class remains the primary challenge across all setups
</details>

<details>
  <summary>Compute</summary>

### Resource Usage Analysis

LoRA is presented in the literature as a parameter-efficient fine-tuning method that significantly reduces memory and computational requirements by training only low-rank adapter matrices while keeping the pretrained weights frozen. These claims originate primarily from experiments on very large models such as GPT-3 (175B parameters). We hypothesized that LoRA would demonstrate similar efficiency gains over partial fine-tuning in our experimental setting.

To evaluate this, we tracked peak GPU memory allocation, total training time, and throughput (samples per second) across all runs, using a fixed batch size of 8 for fair comparison.

| Model | Method | Peak Memory (GB) | Training Time (s) | Throughput (samples/s) |
|-------|--------|------------------|-------------------|------------------------|
| BERT (110M) | Partial FT | 0.89 | 2,566 | 450 |
| BERT (110M) | LoRA | 1.57 | 4,270 | 270 |
| DistilBERT (66M) | Partial FT | 0.64 | 1,347 | 856 |
| DistilBERT (66M) | LoRA | 0.87 | 2,238 | 516 |
| DeBERTa-XXL (1.5B)* | Partial FT | 10.80 | 14,454 | 27 |
| DeBERTa-XXL (1.5B)* | LoRA | 19.66 | 21,695 | 18 |

*DeBERTa-XXL was trained for only 1 epoch due to computational constraints thus the performance metrics are not comparable.

Contrary to expectations, LoRA consumed more memory and was slower than partial fine-tuning across all model sizes tested. On BERT, LoRA used 1.77× more memory and was 1.66× slower. Notably, this pattern persisted even at 1.5B parameters: DeBERTa-XXL with LoRA used 1.82× more memory and was 1.50× slower than partial fine-tuning.

### Interpreting the Memory Overhead

GPU memory during training consists of model weights, optimizer states, activations (intermediate outputs stored for backpropagation), and gradients. While LoRA trains far fewer parameters and thus requires smaller optimizer states, this advantage appears to be offset by other factors at BERT-scale models.

A likely explanation involves activation storage. In partial fine-tuning, early layers are frozen with `requires_grad=False`, which may allow PyTorch to discard their activations after the forward pass since they are not needed for gradient computation. In contrast, LoRA inserts trainable adapters throughout the model—even though the base weights are frozen, the computational graph must still flow through these layers to reach the adapters, potentially requiring activation retention across more layers.

This interpretation is supported by Zhang et al. (2023), who observe that LoRA "still requires expensive activation memory to update low-rank weights" and propose LoRA-FA to address this limitation ([arXiv:2308.03303](https://arxiv.org/abs/2308.03303)). However, the exact memory dynamics depend on implementation details and may vary across frameworks and configurations.

### Training Time Overhead

The observed slowdown with LoRA (1.66× on BERT) likely stems from the additional computations introduced by the adapter architecture. Each LoRA-adapted layer performs extra matrix operations for the low-rank decomposition. While each operation is small, they accumulate across all adapted layers and training steps.

### Scaling Considerations

LoRA's efficiency benefits are typically demonstrated on very large models (7B+ parameters), where optimizer state savings become substantial. At BERT scale (110M parameters), the optimizer state difference between methods is relatively small, and may be outweighed by activation-related overhead.

To explore whether LoRA becomes more efficient at larger scales, we ran experiments on DeBERTa-v2-XXLarge (1.5B parameters). Even at this scale, LoRA used 19.7 GB compared to 10.8 GB for partial fine-tuning. Training was limited to one epoch due to cost (each epoch took 6+ hours on an NVIDIA H100), so accuracy results are not comparable. However, the resource usage pattern suggests that the efficiency crossover point for LoRA may require models larger than 1.5B parameters—potentially in the 7B+ range where the original LoRA experiments were conducted.

</details>

## Final Evaluation

This study compared LoRA and partial fine-tuning on BERT and DistilBERT for natural language inference. All configurations achieved 81–85% accuracy, with no meaningful performance difference between fine-tuning methods. Error analysis showed consistent patterns across models—the neutral class dominated misclassifications, and errors correlated with dataset ambiguity rather than method choice.

Contrary to expectations, LoRA used 1.77× more memory and was 1.66× slower than partial fine-tuning on BERT. This pattern held even at 1.5B parameters (DeBERTa-XXL), where LoRA still used 1.82× more memory. A likely explanation is that LoRA's activation memory overhead outweighs its optimizer state savings at these model scales. For encoder models up to at least 1.5B parameters, partial fine-tuning appears to offer a better resource-accuracy tradeoff.

The original LoRA paper (Hu et al., 2021) demonstrated memory savings on GPT-3 175B, but our results suggest these benefits may not transfer to smaller encoder models. This aligns with observations from Zhang et al. (2023), who note that LoRA "still requires expensive activation memory" and propose LoRA-FA to address this overhead.

The main learning to take from this work is that having fewer trainable parameters doesn't automatically mean lower memory or faster training. It was also noticed that all our models made similar types of errors regardless of the fine-tuning method, which suggests that for a task like NLI the choice between LoRA and partial fine-tuning probably matters less than expected.

## References

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*. https://arxiv.org/abs/2106.09685

Zhang, L., Zhang, L., Shi, S., Chu, X., & Li, B. (2023). LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning. *arXiv preprint arXiv:2308.03303*. https://arxiv.org/abs/2308.03303
