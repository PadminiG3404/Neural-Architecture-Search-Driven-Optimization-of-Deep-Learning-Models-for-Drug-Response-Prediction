# Neural Architecture Search-Driven Optimization of Deep Learning Models for Drug Response Prediction

## 📖 Overview

<p align="center">
    <img width="600" src="https://github.com/user-attachments/assets/67360516-7b6a-443d-890a-01075ded9a22" alt="Sample Data">
</p>

This repository presents the implementation of **Neural Architecture Search (NAS)** techniques to optimize deep learning models for **drug response prediction**. Accurate prediction of drug responses is critical for **personalized medicine**, minimizing adverse effects, and enhancing therapeutic outcomes. Traditional deep learning models rely on manually designed architectures, which often struggle to capture the intricate relationships between genomic features and drug interactions.

To address this challenge, the project explores **three NAS approaches—Random Search, Q-Learning, and Bayesian Optimization**—to automatically identify optimal model architectures. Experimental results demonstrate that NAS-optimized models significantly outperform conventional deep learning methods, showcasing the potential of NAS in improving predictive accuracy. These findings highlight the role of **automated architecture optimization** in advancing personalized medicine and accelerating drug development.

<p align="center">
    <img width="600" src="https://github.com/user-attachments/assets/64577fbf-cc25-4b08-939d-2d3fc007d71d" alt="Sample Data">
</p>

## 📄 Published Work
This research has been accepted at the **4th International Conference on Evolutionary Computing and Mobile Sustainable Networks (ICECMSN 2024)** and published in **Science Direct**.

**Title:** Neural Architecture Search-Driven Optimization of Deep Learning Models for Drug Response Prediction  
**Conference:** ICECMSN 2024  
**Publisher:** Science Direct 
**Sponser:** Elsevier 

## 🏆 Features
- **Dataset:** Uses the **PANCANCER_ANOVA dataset** from the Genomics of Drug Sensitivity in Cancer (GDSC) project.
- **NAS Search Strategies:** Implements **Random Search, Q-Learning, and Bayesian Optimization**.
- **Deep Learning Optimization:** Automates architecture selection for better predictive performance.
- **Performance Evaluation:** Compares models using **MSE, MAE, R², and RMSE** metrics.
- **Model Deployment:** Provides code for deploying the optimized model as a **Flask API**.

## 📂 Dataset
The dataset used in this study is the **PANCANCER_ANOVA** dataset, containing  **200,920 entries across 22 columns** with drug sensitivity data, genomic variations, and tissue types.

### Key features include:
- **Drug Name, Drug ID, and Target Pathway**
- **Gene Expression and Mutation Data**
- **IC50 Values and Effect Sizes**
- **False Discovery Rate (FDR) and P-values**

**Sample Data:**

<p align="center">
    <img width="1000" src="https://github.com/user-attachments/assets/c9a51ad7-411a-438c-9606-0b3d893c0676" alt="Sample Data">
</p>

## 🔬 Methodology

<p align="center">
<img width="400" src="https://github.com/user-attachments/assets/38327c3d-f957-4f47-8fca-1499f1b02893"
</p>

The workflow involves several key steps:
1. **Data Preprocessing**
   - Handling missing values
   - Feature scaling and normalization
   - Correlation analysis and feature selection
   - Dataset splitting for training, validation, and testing

**Missing Values Heatmap:**
<p align="center">
<img width="400" src="https://github.com/user-attachments/assets/9e9ff24c-61b0-4843-81ff-78f227e3b0a0">
</p>

**Correlation Matrix Analysis:**
<p align="center">
<img width="400" src="https://github.com/user-attachments/assets/c47697be-8b82-4ee1-a317-96eafec26e32"
</p>

2. **Neural Network Architecture Design**
   - Defining the search space (layers, activations, dropout rates)
   - Hyperparameter tuning (learning rate, batch size, optimizer selection)

3. **Neural Architecture Search Optimization**
   - **Random Search:** Randomly selects architectures for evaluation.
   - **Q-Learning:** Uses reinforcement learning for guided architecture search.
   - **Bayesian Optimization:** Uses probabilistic models for efficient exploration.

4. **Model Training and Evaluation**
   - Performance metrics: MSE, MAE, R², RMSE
   - Cross-validation and early stopping
   - Residual analysis for error interpretation

5. **Model Deployment**
   - Flask API for drug response prediction
   - Scalability and integration into biomedical applications

## 📊 Results

| Method                 | MSE     | R²       | MAE     | RMSE    |
|------------------------|---------|---------|---------|---------|
| Random Search         | 0.096292 | -0.003099 | 0.27427 | 0.310309 |
| Q-Learning           | **0.065483** | **0.116831** | **0.20684** | **0.255897** |
| Bayesian Optimization | 0.092374 | -0.069800 | 0.25340 | 0.303931 |

**Comparison of Performance Metrics Across Methods**

<p align="center">
    <img width="800" src="https://github.com/user-attachments/assets/4ab48cdd-f827-4ed4-9031-93037d850136" alt="Sample Data">
</p>

**Key Insights:**
- **Q-Learning performed best**, achieving the lowest error rates and highest R² score.
- **Bayesian Optimization showed moderate performance**, improving over Random Search but underperforming compared to Q-Learning.
- **Random Search was the least effective**, highlighting the need for guided optimization techniques.

## 💡Conclusion and Future Scope
This research confirms that **Q-Learning-based NAS provides the most effective neural network architectures** for drug response prediction. Future work could explore:
- **Advanced NAS techniques** (e.g., Evolutionary Algorithms, Reinforcement Learning)
- **Expanding datasets** with diverse drug compounds
- **Explainability techniques** for better interpretability

## 🚀 Installation & Usage

### Prerequisites
- Python 3.x
- TensorFlow / PyTorch
- Scikit-learn

### Setup
- Open Google Colab and load the notebook: 
- Follow the instructions in the notebook to train and evaluate the models.
- Execute each section of the notebook to preprocess data, perform NAS optimization, and evaluate model performance.

## 🛠️ Acknowledgments
- ICECMSN 2024 Conference for accepting this research.
- Science Direct for publication.
- Genomics of Drug Sensitivity in Cancer (GDSC) for the dataset.

## 📜 Citation
If you use this work, please cite:

@article{Uday2025NAS,
  author    = {Uday Kiran G, Srilakshmi V, Padmini G, Sreenidhi G, Venkata Ramana B, Preetham Reddy G J},
  title     = {Neural Architecture Search-Driven Optimization of Deep Learning Models for Drug Response Prediction},
  journal   = {Procedia Computer Science},
  volume    = {252},
  pages     = {172-181},
  year      = {2025},
  issn      = {1877-0509},
  doi       = {10.1016/j.procs.2024.12.019},
  url       = {https://www.sciencedirect.com/science/article/pii/S1877050924034513}
  }

## ✉️ Contact
For questions or collaborations, feel free to reach out:
- Name: Padmini Gudavalli
- Email: [pgudavalli2004@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/padmini-gudavalli-226245259]
