"""
Generate a sample PDF document about Machine Learning Fundamentals.

This script creates a ~12-page PDF with rich content suitable for building
a test Q&A set of 15+ diverse factual questions.

Topics covered:
1. Introduction to Machine Learning
2. Supervised Learning (classification, regression)
3. Unsupervised Learning (clustering, dimensionality reduction)
4. Neural Networks and Deep Learning
5. Model Evaluation Metrics
6. Overfitting and Regularization
7. Feature Engineering
8. Ensemble Methods
9. Natural Language Processing Basics
10. Ethics in AI
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    PageBreak,
)


def create_sample_pdf(output_path: str | Path) -> Path:
    """Generate the sample ML fundamentals PDF.

    Args:
        output_path: Where to save the generated PDF.

    Returns:
        Path to the generated PDF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=24,
        spaceAfter=30,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
        spaceBefore=20,
    )
    subheading_style = ParagraphStyle(
        "CustomSubHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=14,
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        spaceAfter=10,
    )

    story = []

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Machine Learning Fundamentals", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "A Comprehensive Guide to Core Concepts, Algorithms, and Best Practices",
        ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=14, alignment=1),
    ))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Author: Dr. Sarah Chen", body_style))
    story.append(Paragraph("Version: 2.1 — Published: March 2024", body_style))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 1: INTRODUCTION
    # =========================================================================
    story.append(Paragraph("Chapter 1: Introduction to Machine Learning", heading_style))
    story.append(Paragraph(
        "Machine learning is a subset of artificial intelligence that enables systems to learn "
        "and improve from experience without being explicitly programmed. The term was coined by "
        "Arthur Samuel in 1959, who defined it as the field of study that gives computers the "
        "ability to learn without being explicitly programmed. Today, machine learning powers "
        "applications ranging from email spam filters to self-driving cars.",
        body_style,
    ))
    story.append(Paragraph(
        "There are three main categories of machine learning: supervised learning, unsupervised "
        "learning, and reinforcement learning. In supervised learning, the algorithm learns from "
        "labeled training data to make predictions. In unsupervised learning, the algorithm finds "
        "patterns in unlabeled data. Reinforcement learning involves an agent learning through "
        "interaction with an environment to maximize cumulative reward.",
        body_style,
    ))
    story.append(Paragraph(
        "The machine learning workflow typically consists of several stages: data collection, "
        "data preprocessing, feature engineering, model selection, training, evaluation, and "
        "deployment. Each stage is critical, and poor execution at any stage can lead to suboptimal "
        "results. Data quality is often cited as the single most important factor in building "
        "successful machine learning systems — the principle of 'garbage in, garbage out' applies "
        "strongly.",
        body_style,
    ))
    story.append(Paragraph(
        "Key terminology includes: features (input variables), labels (output variables), "
        "training set (data used to train the model), test set (data used to evaluate the model), "
        "hyperparameters (model configuration settings not learned from data), and loss function "
        "(a measure of how well the model's predictions match the actual values).",
        body_style,
    ))

    # =========================================================================
    # CHAPTER 2: SUPERVISED LEARNING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 2: Supervised Learning", heading_style))
    story.append(Paragraph(
        "Supervised learning is the most common type of machine learning. The algorithm is trained "
        "on a labeled dataset, meaning each training example has an input-output pair. The goal is "
        "to learn a mapping function from inputs to outputs that generalizes well to unseen data.",
        body_style,
    ))

    story.append(Paragraph("2.1 Classification", subheading_style))
    story.append(Paragraph(
        "Classification is a supervised learning task where the output variable is a category. "
        "Examples include email spam detection (spam or not spam), image recognition (cat, dog, or "
        "bird), and medical diagnosis (disease present or absent). Common classification algorithms "
        "include Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), "
        "and k-Nearest Neighbors (k-NN).",
        body_style,
    ))
    story.append(Paragraph(
        "Logistic Regression, despite its name, is used for classification. It models the probability "
        "that an input belongs to a particular class using the sigmoid function. The decision boundary "
        "is linear, making it suitable for linearly separable data. For multi-class classification, "
        "techniques like one-vs-rest (OvR) or softmax regression are used.",
        body_style,
    ))
    story.append(Paragraph(
        "Support Vector Machines find the optimal hyperplane that maximizes the margin between "
        "classes. The kernel trick allows SVMs to handle non-linearly separable data by mapping "
        "inputs to a higher-dimensional space. Common kernels include linear, polynomial, and "
        "radial basis function (RBF). SVMs are particularly effective in high-dimensional spaces "
        "and when the number of features exceeds the number of samples.",
        body_style,
    ))

    story.append(Paragraph("2.2 Regression", subheading_style))
    story.append(Paragraph(
        "Regression is a supervised learning task where the output variable is a continuous value. "
        "Examples include predicting house prices, stock prices, and temperature forecasts. Linear "
        "Regression is the simplest form, modeling the relationship between inputs and output as a "
        "linear equation: y = mx + b. The model is trained by minimizing the Mean Squared Error (MSE) "
        "between predicted and actual values using gradient descent or the normal equation.",
        body_style,
    ))
    story.append(Paragraph(
        "Polynomial Regression extends linear regression by adding polynomial terms (x², x³, etc.) "
        "to capture non-linear relationships. However, higher-degree polynomials can lead to "
        "overfitting. Ridge Regression (L2 regularization) and Lasso Regression (L1 regularization) "
        "are extensions that add penalty terms to prevent overfitting. Lasso can also perform "
        "feature selection by driving some coefficients to exactly zero.",
        body_style,
    ))

    # =========================================================================
    # CHAPTER 3: UNSUPERVISED LEARNING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 3: Unsupervised Learning", heading_style))
    story.append(Paragraph(
        "Unsupervised learning deals with unlabeled data. The algorithm must find structure, "
        "patterns, or relationships in the data without guidance. The two main types are clustering "
        "and dimensionality reduction.",
        body_style,
    ))

    story.append(Paragraph("3.1 Clustering", subheading_style))
    story.append(Paragraph(
        "Clustering groups similar data points together. K-Means is the most widely used clustering "
        "algorithm. It works by: (1) randomly initializing K centroids, (2) assigning each data point "
        "to the nearest centroid, (3) updating centroids to the mean of assigned points, and (4) "
        "repeating steps 2-3 until convergence. The main limitation is that K must be specified in "
        "advance. The Elbow Method and Silhouette Score are commonly used to determine the optimal K.",
        body_style,
    ))
    story.append(Paragraph(
        "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an alternative "
        "that doesn't require specifying K. It groups points that are closely packed together and "
        "marks points in low-density regions as outliers. DBSCAN requires two parameters: epsilon "
        "(the maximum distance between two points to be considered neighbors) and min_samples (the "
        "minimum number of points to form a dense region). Unlike K-Means, DBSCAN can find clusters "
        "of arbitrary shape.",
        body_style,
    ))

    story.append(Paragraph("3.2 Dimensionality Reduction", subheading_style))
    story.append(Paragraph(
        "Dimensionality reduction techniques reduce the number of features while preserving "
        "important information. Principal Component Analysis (PCA) is the most common method. "
        "PCA finds the directions (principal components) of maximum variance in the data and "
        "projects the data onto these directions. The first principal component captures the most "
        "variance, the second captures the next most, and so on. PCA is useful for visualization, "
        "noise reduction, and speeding up other algorithms.",
        body_style,
    ))
    story.append(Paragraph(
        "t-SNE (t-distributed Stochastic Neighbor Embedding) is a non-linear dimensionality "
        "reduction technique primarily used for visualization. Unlike PCA, t-SNE is particularly "
        "good at preserving local structure, making it excellent for visualizing high-dimensional "
        "data in 2D or 3D. However, t-SNE is computationally expensive and results can vary "
        "between runs due to its stochastic nature. The perplexity hyperparameter controls "
        "the balance between local and global structure.",
        body_style,
    ))

    # =========================================================================
    # CHAPTER 4: NEURAL NETWORKS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 4: Neural Networks and Deep Learning", heading_style))
    story.append(Paragraph(
        "Artificial neural networks are inspired by the biological neural networks in the human "
        "brain. A neural network consists of layers of interconnected nodes (neurons). Each "
        "connection has a weight, and each neuron has a bias. The basic architecture includes an "
        "input layer, one or more hidden layers, and an output layer. Networks with more than two "
        "hidden layers are generally considered 'deep' neural networks, giving rise to the field "
        "of deep learning.",
        body_style,
    ))
    story.append(Paragraph(
        "The learning process involves forward propagation (computing predictions) and "
        "backpropagation (computing gradients and updating weights). The chain rule of calculus "
        "enables efficient gradient computation through the network layers. Common activation "
        "functions include ReLU (Rectified Linear Unit), which outputs max(0, x); sigmoid, which "
        "outputs values between 0 and 1; and tanh, which outputs values between -1 and 1. ReLU "
        "is the most popular choice for hidden layers because it helps mitigate the vanishing "
        "gradient problem.",
        body_style,
    ))
    story.append(Paragraph(
        "Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such "
        "as images. They use convolutional layers that apply learnable filters to detect features "
        "like edges, textures, and shapes. Pooling layers reduce spatial dimensions. Famous CNN "
        "architectures include LeNet-5 (1998), AlexNet (2012, which won ImageNet), VGGNet (2014), "
        "and ResNet (2015, which introduced skip connections to train very deep networks up to "
        "152 layers).",
        body_style,
    ))
    story.append(Paragraph(
        "Recurrent Neural Networks (RNNs) are designed for sequential data like text and time "
        "series. Standard RNNs suffer from the vanishing gradient problem, which makes it "
        "difficult to learn long-range dependencies. Long Short-Term Memory (LSTM) networks "
        "solve this with a gating mechanism that controls information flow. An LSTM cell has "
        "three gates: forget gate, input gate, and output gate. Gated Recurrent Units (GRUs) "
        "are a simplified version with two gates that often performs comparably to LSTMs.",
        body_style,
    ))
    story.append(Paragraph(
        "The Transformer architecture, introduced in the 2017 paper 'Attention is All You Need' "
        "by Vaswani et al., revolutionized natural language processing. Transformers use a "
        "self-attention mechanism that allows each position in the sequence to attend to all other "
        "positions, enabling parallel processing and capturing long-range dependencies more "
        "effectively than RNNs. BERT (Bidirectional Encoder Representations from Transformers) "
        "and GPT (Generative Pre-trained Transformer) are prominent Transformer-based models.",
        body_style,
    ))

    # =========================================================================
    # CHAPTER 5: EVALUATION METRICS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 5: Model Evaluation Metrics", heading_style))
    story.append(Paragraph(
        "Proper evaluation is critical for understanding model performance and making informed "
        "decisions about model selection and deployment.",
        body_style,
    ))

    story.append(Paragraph("5.1 Classification Metrics", subheading_style))
    story.append(Paragraph(
        "Accuracy is the most intuitive metric: the ratio of correct predictions to total "
        "predictions. However, accuracy can be misleading for imbalanced datasets. For example, "
        "in a dataset where 95% of samples belong to class A, a model that always predicts class A "
        "achieves 95% accuracy but is useless.",
        body_style,
    ))
    story.append(Paragraph(
        "Precision measures the ratio of true positives to all predicted positives: "
        "Precision = TP / (TP + FP). It answers 'of all positive predictions, how many were "
        "correct?' Recall (sensitivity) measures the ratio of true positives to all actual "
        "positives: Recall = TP / (TP + FN). It answers 'of all actual positives, how many "
        "were found?' The F1 Score is the harmonic mean of precision and recall: "
        "F1 = 2 * (Precision * Recall) / (Precision + Recall). It balances both metrics.",
        body_style,
    ))
    story.append(Paragraph(
        "The ROC (Receiver Operating Characteristic) curve plots True Positive Rate against "
        "False Positive Rate at various threshold settings. The AUC (Area Under the ROC Curve) "
        "provides a single number summarizing classifier performance. AUC = 1.0 indicates a "
        "perfect classifier, while AUC = 0.5 indicates random guessing. AUC is particularly "
        "useful for comparing models across different threshold settings.",
        body_style,
    ))

    story.append(Paragraph("5.2 Regression Metrics", subheading_style))
    story.append(Paragraph(
        "Mean Squared Error (MSE) is the average of squared differences between predicted and "
        "actual values. Root Mean Squared Error (RMSE) is the square root of MSE, giving errors "
        "in the same units as the target variable. Mean Absolute Error (MAE) is the average of "
        "absolute differences, which is less sensitive to outliers than MSE. R-squared (R²) "
        "measures the proportion of variance explained by the model, ranging from 0 to 1, where "
        "1 means the model perfectly explains all variance in the data.",
        body_style,
    ))

    # =========================================================================
    # CHAPTER 6: OVERFITTING AND REGULARIZATION
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 6: Overfitting and Regularization", heading_style))
    story.append(Paragraph(
        "Overfitting occurs when a model learns the training data too well, including its noise "
        "and outliers, resulting in poor generalization to new data. Signs of overfitting include "
        "high training accuracy but low test accuracy. Underfitting is the opposite — the model "
        "is too simple to capture the underlying patterns.",
        body_style,
    ))
    story.append(Paragraph(
        "The bias-variance tradeoff is a fundamental concept. Bias is the error from simplifying "
        "assumptions in the model (high bias = underfitting). Variance is the error from sensitivity "
        "to fluctuations in the training data (high variance = overfitting). The goal is to find "
        "the sweet spot that minimizes total error (bias² + variance + irreducible error).",
        body_style,
    ))
    story.append(Paragraph(
        "Regularization techniques help prevent overfitting. L1 regularization (Lasso) adds the "
        "sum of absolute weights as a penalty: Loss = Original Loss + λΣ|w|. It can drive weights "
        "to exactly zero, performing feature selection. L2 regularization (Ridge) adds the sum of "
        "squared weights: Loss = Original Loss + λΣw². It shrinks weights but rarely makes them "
        "exactly zero. Elastic Net combines both L1 and L2 penalties.",
        body_style,
    ))
    story.append(Paragraph(
        "Dropout is a regularization technique specific to neural networks, introduced by Srivastava "
        "et al. in 2014. During training, neurons are randomly 'dropped out' (set to zero) with a "
        "specified probability (typically 0.2-0.5). This prevents neurons from co-adapting and forces "
        "the network to learn more robust features. At inference time, all neurons are active but "
        "outputs are scaled accordingly.",
        body_style,
    ))
    story.append(Paragraph(
        "Cross-validation is a technique for robust model evaluation. K-fold cross-validation "
        "splits the data into K equally sized folds. The model is trained K times, each time using "
        "a different fold as the validation set and the remaining K-1 folds as the training set. "
        "The typical value is K=5 or K=10. The final performance is the average across all K runs, "
        "providing a more reliable estimate than a single train-test split.",
        body_style,
    ))

    # =========================================================================
    # CHAPTER 7: FEATURE ENGINEERING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 7: Feature Engineering", heading_style))
    story.append(Paragraph(
        "Feature engineering is the process of using domain knowledge to create, transform, or "
        "select features that make machine learning algorithms work better. It is often considered "
        "the most important step in the ML pipeline — Andrew Ng has stated that 'coming up with "
        "features is difficult, time-consuming, and requires expert knowledge.'",
        body_style,
    ))
    story.append(Paragraph(
        "Common techniques include: One-hot encoding for categorical variables (converting categories "
        "to binary columns), normalization (scaling features to [0,1] range using min-max scaling), "
        "standardization (scaling to zero mean and unit variance using z-score), log transformation "
        "(for skewed distributions), and polynomial features (creating interaction terms).",
        body_style,
    ))
    story.append(Paragraph(
        "Feature selection methods fall into three categories: filter methods (e.g., correlation "
        "analysis, chi-squared test), wrapper methods (e.g., recursive feature elimination, forward/"
        "backward selection), and embedded methods (e.g., L1 regularization, tree-based feature "
        "importance). Proper feature selection reduces overfitting, improves model interpretability, "
        "and decreases training time.",
        body_style,
    ))
    story.append(Paragraph(
        "Handling missing data is a crucial part of feature engineering. Common strategies include: "
        "dropping rows or columns with missing values (simple but loses data), imputation with mean, "
        "median, or mode (preserves data size but may introduce bias), and advanced methods like "
        "K-NN imputation or using algorithms that handle missing values natively (e.g., XGBoost). "
        "The choice depends on the proportion of missing data and whether data is missing at random.",
        body_style,
    ))

    # =========================================================================
    # CHAPTER 8: ENSEMBLE METHODS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 8: Ensemble Methods", heading_style))
    story.append(Paragraph(
        "Ensemble methods combine multiple models to produce better predictive performance than "
        "any single model alone. The key insight is that diverse models make different errors, "
        "and combining them can cancel out individual mistakes.",
        body_style,
    ))
    story.append(Paragraph(
        "Bagging (Bootstrap Aggregating) trains multiple instances of the same algorithm on "
        "different random subsets of the training data (with replacement). Predictions are "
        "aggregated by voting (classification) or averaging (regression). Random Forest is the "
        "most popular bagging method — it builds many decision trees, each trained on a random "
        "subset of data and features. Random Forests typically use sqrt(n_features) features per "
        "tree for classification and n_features/3 for regression.",
        body_style,
    ))
    story.append(Paragraph(
        "Boosting trains models sequentially, with each new model focusing on the errors made by "
        "previous models. AdaBoost adjusts sample weights, giving more weight to misclassified "
        "examples. Gradient Boosting builds trees to predict the residual errors of the ensemble. "
        "XGBoost (Extreme Gradient Boosting) is an optimized implementation that uses regularization, "
        "parallel processing, and tree pruning. It was the dominant algorithm in Kaggle competitions "
        "from 2016-2019, winning the majority of structured data competitions.",
        body_style,
    ))
    story.append(Paragraph(
        "Stacking (Stacked Generalization) trains a meta-model on the predictions of base models. "
        "The base models' predictions become features for the meta-model. For example, you might "
        "use a Random Forest, SVM, and Neural Network as base models, and a Logistic Regression "
        "as the meta-model. This approach can capture complementary strengths of different algorithms.",
        body_style,
    ))

    # =========================================================================
    # CHAPTER 9: NLP BASICS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 9: Natural Language Processing Basics", heading_style))
    story.append(Paragraph(
        "Natural Language Processing (NLP) enables machines to understand, interpret, and generate "
        "human language. It combines linguistics, computer science, and machine learning.",
        body_style,
    ))
    story.append(Paragraph(
        "Text preprocessing is the foundation of NLP. Key steps include tokenization (splitting text "
        "into words or subwords), lowercasing, removing stop words (common words like 'the', 'is', "
        "'and'), stemming (reducing words to their root form, e.g., 'running' → 'run'), and "
        "lemmatization (similar to stemming but produces valid words using vocabulary lookup).",
        body_style,
    ))
    story.append(Paragraph(
        "Text representation methods include: Bag of Words (BoW), which represents text as word "
        "frequency vectors but loses word order; TF-IDF (Term Frequency-Inverse Document Frequency), "
        "which weights words by importance across documents; and word embeddings like Word2Vec and "
        "GloVe, which represent words as dense vectors in a continuous space where similar words "
        "have similar vectors. Word2Vec has two architectures: CBOW (predicts target word from "
        "context) and Skip-gram (predicts context from target word).",
        body_style,
    ))
    story.append(Paragraph(
        "Modern NLP is dominated by pre-trained language models. BERT uses masked language modeling "
        "and next sentence prediction for pre-training, then fine-tuning for downstream tasks. "
        "GPT uses autoregressive pre-training (predicting the next token). These models capture "
        "contextual word meaning — for example, the word 'bank' in 'river bank' vs 'bank account' "
        "gets different representations. Transfer learning has made it possible to achieve "
        "state-of-the-art results with relatively small task-specific datasets.",
        body_style,
    ))

    # =========================================================================
    # CHAPTER 10: ETHICS IN AI
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Chapter 10: Ethics in AI and Machine Learning", heading_style))
    story.append(Paragraph(
        "As machine learning systems are increasingly deployed in high-stakes domains like "
        "healthcare, criminal justice, and hiring, ethical considerations have become paramount. "
        "Key ethical issues include bias, fairness, transparency, privacy, and accountability.",
        body_style,
    ))
    story.append(Paragraph(
        "Algorithmic bias occurs when ML models produce systematically unfair outcomes for certain "
        "groups. Sources of bias include: historical bias in training data (reflecting past "
        "discrimination), representation bias (underrepresentation of certain groups in data), "
        "measurement bias (using proxies that correlate with protected attributes), and aggregation "
        "bias (using a single model when subgroup differences exist). The COMPAS recidivism "
        "prediction system is a well-known example where algorithmic bias led to unfair outcomes.",
        body_style,
    ))
    story.append(Paragraph(
        "Explainability and interpretability are crucial for building trust and ensuring "
        "accountability. LIME (Local Interpretable Model-agnostic Explanations) explains individual "
        "predictions by approximating the model locally with a simple, interpretable model. SHAP "
        "(SHapley Additive exPlanations) uses game theory to assign feature importance values. "
        "These tools help stakeholders understand why a model made a particular decision.",
        body_style,
    ))
    story.append(Paragraph(
        "Data privacy is addressed through techniques like differential privacy, which adds "
        "controlled noise to data or query results to prevent individual identification. Federated "
        "learning allows models to be trained across decentralized devices without sharing raw "
        "data. The EU's General Data Protection Regulation (GDPR) includes a 'right to explanation' "
        "for automated decisions, making model interpretability a legal requirement in some contexts.",
        body_style,
    ))
    story.append(Paragraph(
        "Responsible AI frameworks recommend: conducting fairness audits throughout the ML lifecycle, "
        "documenting model decisions and limitations using model cards, engaging diverse stakeholders "
        "in system design, establishing clear accountability for AI-driven decisions, and monitoring "
        "deployed models for performance degradation and emerging biases over time.",
        body_style,
    ))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    output = create_sample_pdf(Path(__file__).parent / "sample_doc.pdf")
    print(f"Generated sample PDF: {output}")
    print(f"File size: {output.stat().st_size / 1024:.1f} KB")
