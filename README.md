# CS129-Duolingo

Retention in Gamified Learning: Analyzing Duolingo User Engagement with Machine Learning

Category: Education, Machine Learning, Learning Analytics

Member: Kaz Fukuhara (kazf28), Xinman (Yoyo) Liu (xinman), Xinyu (Teah) Shi (teah2001)

Motivation

Language learning apps like Duolingo have revolutionized the way people acquire new languages in a gamified manner, having 37.2 million active users per day. However, user engagement remains a critical challenge. In any type of learning, students start strong but their engagement and activeness drops off drastically, while others remain consistently active. This trend is also observable with Duolingo, and understanding the patterns behind student engagement can help improve personalized learning experiences, retention strategies, and overall learning outcomes.
This project aims to analyze user engagement behavior in Duolingo by clustering users into distinct engagement groups based on their learning activity. Using the 2018 Duolingo Shared Task on Second Language Acquisition Modeling (SLAM) French-to-English dataset, which consists new users’ first 30 days of behavior and performance data, we will examine meaningful engagement patterns that can inform better learning design and adaptive learning strategies within Duolingo, which can be applicable to other language learning settings. The outcome of this project will help identify different types of learners and allow the platform to optimize retention strategies.
Method

This project will employ 3 ML techniques, to segment learners based on behavioral and performance-based features extracted from Duolingo session data. The primary steps include:

Outcome Variable
- Churn time: The churn time variable is the day at which a user becomes disengaged and permanently inactive. This is calculated by finding the maximum number of days in which the user remained active, which will be categorized into 3 groups (1 - 10 days, 10 - 20 days and 21 - 30 days).

Features
- Time of Day Preference (Categorical)
- Average Correctness (Numerical)
- Response Time Variability (Numerical)
-- Variance in response time per session 
- Sentence Complexity (Numerical)
-- Has to be feature engineered
- Question Format (Categorical)
-- Most frequent learning format (reverse_translate, listening, etc.)

Softmax Classifier: By leveraging feature engineered engagement variables, the Softmax classifier assigns probabilities to each category, allowing us to assess the relative importance of different features in predicting churn time and gain insights into the key factors associated with their disengagement.

Random Forest: Similar to the Softmax classifier, we use Random Forest to determine the key factors influencing student churn time. This model helps identify the most significant features that let students quit, providing deeper insights into learning behaviors.

K-Nearest Neighbors (KNN): In addition to other classification models, we apply K-Nearest Neighbors to predict student churn time based on similar user behavior patterns. By analyzing features contributing to their disengagement, KNN helps identify which learners exhibit similar activity trends, providing insights into the factors that contribute to lower engagement.

Intended Experiments

Model Training Setup: We will use an 80-20 train-test split with 5-fold cross-validation on the training set. For each model (Softmax, Random Forest, KNN), we will perform hyperparameter tuning via grid search - optimizing learning rate and regularization for Softmax, tree parameters for Random Forest, and k-value and distance metrics for KNN.

Evaluation Approach: Model performance will be assessed using accuracy, precision, recall, and F1-score across all three churn categories. We will analyze feature importance to identify key behavioral factors influencing user engagement.

