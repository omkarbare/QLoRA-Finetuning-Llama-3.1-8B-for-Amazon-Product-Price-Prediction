# QLoRA Finetuning Llama 3.1-8B for Amazon Product Price Prediction

This project demonstrates advanced regression modeling for Amazon product price prediction by applying QLoRA-based parameter-efficient finetuning to the Llama 3.1-8B language model. Through rigorous experimentation and evaluation, the approach significantly improves predictive accuracy over the base model.

---

## Project Overview

- **Objective:** Predict product prices on Amazon using state-of-the-art large language models.
- **Models Used:** Base and finetuned Llama 3.1-8B with QLoRA.
- **Pipeline:** Data preprocessing, model training (QLoRA), and systematic evaluation—all implemented in Python notebooks.

---

## Dataset Description

The project utilizes a curated Amazon Product Price Prediction dataset tailored for training regression models on real-world e-commerce data. This dataset is derived from a subset of the McAuley-Lab/Amazon-Reviews-2023 dataset and is specifically engineered for price estimation tasks using product titles and descriptions. dataset link: [Amazon Product Price Prediction Dataset on Hugging Face by Ed Donner](https://huggingface.co/datasets/ed-donner/pricer-data/viewer/default/train?views%5B%5D=train)

---

## Notebooks

- `QLoRA_Finetuning_ProductPriceRegression.ipynb` – Details the QLoRA finetuning strategy.
- `evaluation_base_model_ProductPriceRegression.ipynb` – Contains baseline evaluation results.
- `evaluation_finetuned_model_ProductPriceRegression.ipynb` – Presents results after finetuning.

---

## Metrics

The model evaluation uses three intuitive metrics relevant for regression problems:

- **Predict Error ($):**  
  Represents the average absolute difference between the model's predicted prices and actual product prices, measured in dollars. This allows employers to understand how much, on average, the model deviates from the actual value for each prediction—a lower value indicates more accurate, reliable pricing for Amazon products.

- **Root Mean Squared Log Error (RMSLE):**  
  RMSLE penalizes larger differences between actual and predicted prices, especially when both values are logarithmically transformed. It is less sensitive to large absolute differences at higher price values and is useful for tasks where relative error matters, such as price prediction. Lower RMSLE indicates better model calibration and more trustworthy results for both low and high-value products.

- **Hits (%):**  
  The hit rate measures the percentage of predictions that fall within a predefined acceptable error threshold from the ground truth. This is an intuitive measure of how often the model is “close enough” for practical business use—higher percentages mean the model is more useful in real applications.

### Result Comparison

| Metric              | Base Model           | Finetuned Model     |
|---------------------|---------------------|---------------------|
| Predict Error ($)   | 395.72               | 52.09               |
| RMSLE               | 1.49                 | 0.39                |
| Hits (%)            | 28.0                 | 68.4                |


### Highlights of result after finetuning

- **Predict error reduced** by over 7x, from $395.72 to $52.09, after finetuning.
- **RMSLE** (Root Mean Squared Log Error) dropped from 1.49 to 0.39, indicating vastly improved model calibration.
- **Hit rate** (proportion of predictions within acceptable error) increased from 28% to 68.4%, reflecting the practical benefit of model improvements.
- Visual analysis shows scatter plots tightening along the diagonal and a major decrease in large outlier predictions after finetuning.


---

## Skills Demonstrated

- Advanced LLM and parameter-efficient finetuning with QLoRA.
- Model evaluation with custom metrics and comprehensive visualization.
- Best practices in code organization, experiment tracking, and reporting.

