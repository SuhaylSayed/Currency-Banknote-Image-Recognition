# Currency-Banknote-Image-Recognition
Utilizing advanced machine learning techniques, such as Convolutional Neural Networks (CNNs) and Decision Trees, developed a robust and precise image recognition model that accurately identifies Indian currency denominations, despite challenges posed by varying image quality, lighting conditions, and camera angles.

# Business Problem:

This project aims to develop an image recognition model capable of identifying currency note denominations based on visual features. The model should recognize all banknote denominations in different currency types in real-world situations, despite environmental factors such as image quality, lighting conditions, and camera angle. Models have been trained to identify six denominations of Indian Currency (Rs 10, Rs 20, Rs 50, Rs 100, Rs 200, Rs 500, and Rs 2000) using two types: Decision Trees and Convolutional Neural Networks. The approach involved training three Decision Trees with distinct feature engineering techniques and exploring two CNNs - one built from scratch and another leveraging a pre-trained model. The performance of these models and business insights are discussed in the conclusion.

# Motivation:

The project offers significant practical applications in fields such as finance, banking, and retail, where the model's ability to accurately recognize different types of currencies can make transactions faster and more reliable. Furthermore, the model will be particularly helpful for visually impaired or elderly individuals who may have trouble recognizing different types of currency. By providing a tool to make the world more accessible to the aging population, the model can directly address this challenge identified by the World Health Organization. Additionally, the model can help reduce processing costs by 80% in the financial services sector and reduce risks by accurately identifying banknotes compared to tellers.

This problem presents a unique opportunity to study and implement various machine-learning methods and models. Convolution Neural Networks and Decision Trees were explored to identify the best approach for the solution. Various data exploration and model testing techniques were employed to improve model accuracy, leading to a deeper understanding of Image Recognition, Supervised Learning methods, and machine learning as a whole.

# Dataset:

The dataset consists of 2571 images of six different currency denominations, split into six files representing each denomination. Below are sample images pulled from the dataset:

![Image samples from dataset](https://user-images.githubusercontent.com/40481691/235971927-7260103b-cee4-44e3-ba53-e1cddef04358.png)

# Summary of Findings:



**Dataset**

We developed two models to recognize Indian currency notes based on denomination: Decision Trees and Convolutional Neural Networks. Following training, we evaluated both models' classification accuracy on an unseen training dataset.

Before beginning the training process, we analyzed our dataset, which consisted of 2571 images. We carefully examined a selection of images, including their dimensions (height, width, and color channels). Our images varies significanly due to real-world factors such as image quality, lighting conditions, and camera angle. It's worth noting that many of the images depicted only partial views of the currency notes. Our analysis led us to conclude that our models would be challenged when identifying currency in various conditions, resulting in lower classification accuracy than if we had used clear images.

For example, due to variations in lighting, using the currency's color alone may not be as effective as using consistent coloring. Similarly, large differences in the background could cause the models to focus on incorrect patterns. That being said, we were confident in our dataset choice. Tt provided a diverse set of real-world images that would help our models perform well in real-world scenarios which was the goal of this project. We trained both Decision Trees and Convolutional Neural Networks models using the dataset and evaluated their classification accuracy on an unseen training dataset.

**Decision Trees**

Our initial approach involved training decision trees, even though we anticipated that they might not perform as well as CNNs. This is because CNN's  are more adept at complex image classification. Using decision trees would allow us to establish a baseline to compare the performance of our CNN models and establish their superiority.

To optimize our decision tree models, we experimented with three different feature engineering techniques. Prior to training, we divided our dataset into validation, training, and testing sets, with the testing set remaining unseen by the model. We also employed one-hot encoding to represent the labels. Additionally, we flattened the image data into one-dimensional vectors, as this is the format that the models expect for input data.



1.   SelectKBest


We first used the SelectKBest method, which selects the features with the highest score according to a specified value of k. This approach allowed us to narrow down the features used in training the decision trees and potentially improve their accuracy. After trial and error we settled on a K value of 900 is it resulted in the best model. The SelectKBest returned the following accuracy score on our most recent run: 

Train accuracy: 100.00%
Validation accuracy: 23.56%
Test accuracy: 18.75%

While we evaluated the performance of all three accuracy measures, we primarily focused on the test accuracy to assess the model's ability to classify new, unseen data. Despite consistently outperforming the 16% accuracy threshold of random guessing, the model's accuracy remained suboptimal, with scores hovering around the high teens to low twenties. Nevertheless, the SelectKBest decision tree was able to identify some correlation among the features, suggesting some potential for improvement with further refinement.

2.   PCA

PCA (Principal Component Analysis) is a dimensionality reduction technique that transforms the data into a new set of features by projecting it onto a lower-dimensional space. In our experiment, we utilized PCA to reduce the number of features in our model and potentially improve accuracy. After running multiple iterations, we found that a PCA model with the following accuracy scores performed best:

Train accuracy: 100.00%
Validation accuracy: 27.56%
Test accuracy: 28.12%

The accuracy scores of the PCA model improved compared to the SelectKBest model averaging in the high twenties. PCA decision tree was able to identify some correlation among the features and reduced the number of features used for classification. That being said the model is still remained relatively low and sub-optimal.

3.   Lasso

Lasso is a linear regression method that performs regularization by imposing a penalty on the absolute size of the regression coefficients. We applied Lasso regression to select the most significant features and improve the accuracy of our decision trees. After several iterations, we obtained the following accuracy scores:

Train accuracy: 100.00%
Validation accuracy: 38.44%
Test accuracy: 29.69%

Compared to the SelectKBest and PCA models, the Lasso model outperformed the others, achieving the highest validation and test accuracy scores averaging in the low 30s. This suggests that the Lasso decision tree was able to identify more relevant features and reduce overfitting. An interesting observation from the Lasso model is that while it had a high validation accuracy in the high 30s, its test accuracy was considerably lower. In contrast, our other models had similar validation and test accuracies. This discrepancy in the Lasso model's performance may suggest that the model was more sensitive to the specific data in the validation set and over-fitting the model to adjust to those. 



Overall, the Lasso decision tree model outperformed our other models with accuracy scores in the low thirties. Although the accuracy is still relatively low, we were pleased with the results as it demonstrated that even a simplistic model like Decision Trees could identify some patterns in the diverse image dataset. This accuracy score served as a useful baseline for assessing the improvements shown by our CNN models.


## **CNN**

### ***Custom Model***

To develop a highly accurate image recognition model, we began by carefully splitting our data into training, validation, and testing sets. We then applied normalization and rescaling to the images to ensure that pixel values ranged from 0 to 1, which simplified the learning process for our model.

Our CNN model was designed as a Sequential model consisting of three Convolutional layers followed by MaxPooling layers. The model started with a Rescaling layer at the input with no trainable parameters. The Conv2D layers had 1,792, 18,464, and 9,248 trainable parameters respectively, while the MaxPooling2D layers had none. The model concluded with two Dense layers, with 2,769,024 and 774 trainable parameters respectively. In total, our model had 2,799,302 trainable parameters.

We then compiled the model using the Adam optimizer, SparseCategoricalCrossentropy loss function, and accuracy as the evaluation metric. The Adam optimizer updated the model's parameters during training to minimize the loss function. Meanwhile, the SparseCategoricalCrossentropy loss function was utilized for multi-categorical classification, which was the case for our project. It computed the cross-entropy loss between the true class labels and the predicted class probabilities.

Our model was then trained with the training and validation datasets for 20 epochs, achieving a validation accuracy of up to 65%. Finally, we tested the model using the testing dataset. Despite the challenging nature of our dataset, with low-quality and unclear images, our model achieved an impressive accuracy of 60.511%. Overall, our approach resulted in an effective model for image recognition, significantly surpassing the accuracy of our Decision Tree baseline.



### ***Pre-Trained Model***
VGG16 is a powerful convolutional neural network architecture that was designed by the Visual Geometry Group at the University of Oxford, hence the name VGG. The network is widely used for image classification tasks, thanks to its 16 convolutional and fully connected layers. The architecture of VGG16 is characterized by the use of small 3x3 convolutional filters, which enable the network to learn more complex and abstract features. Additionally, the network features five max pooling layers, which help to reduce the spatial dimensions of the feature maps and make the model more computationally efficient.

The main purpose of VGG16 is to classify images into one of a thousand possible categories, based on the presence or absence of certain visual features in the image. This is achieved through a process called training, during which the network learns a set of weights that are optimized to accurately predict the correct category label for a given image.

After training VGG16 on our training dataset with 10 epochs, we were able to achieve an impressive 80% accuracy on our testing dataset. This result is a testament to the power and effectiveness of the VGG16 architecture in image classification tasks.

Though VGG16 outperformed our own model in terms of accuracy we were still satisfied with the performance of our model. We were able to achieve a comparable level of accuracy despite having a smaller architecture and fewer training epochs. Overall, we believe that our approach demonstrates the potential of deep learning models in image recognition tasks.

