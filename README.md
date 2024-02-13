# **AI-Driven Hotel Detection to Combat Human Trafficking**
**Yiyi Luo**


<img width="783" alt="Screenshot 2024-02-10 at 1 46 54 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/f2f482ac-e687-4ea0-a29b-227becb555c2">


## **Overview and Business Understanding**

Human trafficking is a grave violation of human rights, often leaving a digital trail in the form of photographs taken within hotel rooms. Identifying the hotels in these images is crucial for the success of trafficking investigations; however, it presents unique challenges. Investigators frequently face obstacles such as poor image quality, unconventional camera angles, and the subtle nuances of hotel room features.

This project aligns with the objectives of a notable Kaggle competition, aiming to leverage the power of machine learning to combat human trafficking by identifying hotel rooms from photographs. The competition, titled 'Hotel-ID to Combat Human Trafficking 2022 - FGVC9', presents an opportunity for participants to contribute to a crucial cause by addressing the challenges associated with hotel room identification. For more details on the competition please visit https://www.kaggle.com/competitions/hotel-id-to-combat-human-trafficking-2022-fgvc9/overview.

<img width="636" alt="Screenshot 2024-02-10 at 12 54 39 AM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/bbd1c72b-cb76-4371-9338-dc50e236825f">

## **Part I: Data Understanding and Preparation**


<img width="1200" alt="Screenshot 2024-02-11 at 5 09 43 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/a9d78d9a-bf19-4b03-befa-925ff42efce2">
<p align="center">
<img width="600" alt="Screenshot 2024-02-11 at 6 52 01 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/79a9ef5d-e68d-48fa-938a-5434c745cb03">
</p>

**Data Preparation:**

**Data Preprocessing:** Resize images, apply random red masks to them, and save the processed images to a specified directory; 
<img width="1045" alt="Screenshot 2024-02-10 at 9 19 35 AM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/09b2fd01-30f7-45d3-b9d8-7915936a764b">

**Challenge 1 - Class Diversity:** The dataset comprises a **large number of classes (1,678 distinct hotels)**, which presents a substantial challenge in terms of learning distinct features for each class.

**Challenge 2 - Class Imbalance:** There is **significant imbalance** in the dataset, with some hotels represented by a single image while others have over a thousand. This disparity can lead to overfitting and poor generalization for underrepresented classes.

**Challenge 3 - Insufficient Learning Samples:** Most hotels have **fewer than 15 images** available, which may not provide enough data for the model to effectively learn the variance within each class.

**Challenge 4 - Subtle Variations:** Hotel rooms often share similar aesthetics and structures, much like nuances in human faces, making it **difficult to distinguish between different hotels**.

**Challenge 5 - Occlusions from Privacy Masks:** The presence of **occlusion masks** (red masks) adds complexity to the task, as they can **cover significant portions** of the images, obscuring important details that are necessary for accurate classification.

**Challenge 6 - Potential Overfitting:** With a limited number of images per class, there is a risk of models overfitting to the training data, which could result in poor performance on unseen data.

**Challenge 7 - Background Noise and Variability:** The variability in lighting, decor, and camera angles within hotel room images introduces additional complexity, potentially leading to background noise that can confuse the model.

**Data Splitting:** Divide the dataset into three sets: training (70%), validation (15%), and testing (15%);

**Data Augmentation:** Define a data augmentation pipeline using Keras, incorporating random flips, rotations, zooms, contrast and brightness adjustments, resizing to 280x280 pixels, cropping back to 256x256 pixels, and translations to enhance the diversity of the training dataset;

<img width="960" alt="Screenshot 2024-02-10 at 9 08 21 AM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/88b0d520-4269-4201-94c9-3282f8d7de48">

By artificially expanding the training dataset through various transformations, augmentation helps the model to learn more generalized features, reducing the risk of overfitting and improving its ability to perform well on unseen data.

**Random Flips and Rotations:** These augmentations introduce spatial variability by flipping images horizontally or vertically and rotating them by arbitrary angles. This helps the model to recognize hotel room features regardless of their orientation, addressing unconventional camera angles and the diversity of photographic perspectives.

**Random Zooms:** By zooming in and out of images, the model learns to identify features at different scales, making it more robust to variations in image composition and focal lengths. This is particularly useful for focusing on both the macro and micro aspects of hotel room characteristics.

**Contrast and Brightness Adjustments:** Variability in lighting conditions can significantly affect the appearance of a room. Adjusting contrast and brightness helps the model to become invariant to these lighting differences, enhancing its ability to recognize rooms under diverse lighting conditions.

**Resizing and Cropping:** Resizing images to 280x280 pixels before cropping them back to 256x256 pixels introduces spatial diversity and forces the model to focus on different parts of the image. This technique is beneficial for learning from the subtle nuances of hotel room features, despite the presence of background noise and variability.

**Translations:** Shifting images horizontally or vertically simulates the effect of different camera positions, helping the model to generalize across various angles and positions within the room.

**Feature Engineering:** Analyze the images to determine if there are specific features (like bedding patterns, artwork, furniture) that could be explicitly extracted to aid in identification.

**In the initial phase of our project, due to constraints on time and computational resources, we have decided to prioritize training our models using unmodified images.** This step will allow us to establish a baseline performance and ensure that our models are learning effectively from the data in its most straightforward form. Following this, we plan to introduce a second phase of training, where the models will be further refined by including images with occlusions. 

### **Modeling**

## **Part II: Basic Keras Models**

**Build a machine learning workflow for image classification using TensorFlow and Keras:**

**1.** **Set up essential parameters**: class names, the number of classes, and the number of epochs for training. The dataset is prepared using the ImageDataGenerator for image rescaling, ensuring proper input to the neural network;
   
**2.** A versatile model creation function is defined to **construct CNN models with optional data augmentation and dropout**, making it adaptable to various scenarios. The compile_and_train_model function takes care of compiling and training the models;

**3.** **Visualize training results** by the plot_training_history function, providing plots of training and validation accuracy and loss, aiding in the interpretation of the model's performance over epochs.
   
**4.** **Implements K-Fold cross-validation** to ensure the model's robustness and generalizability. It generates training and validation data generators for each fold, trains the model, and computes the average accuracy, offering a comprehensive evaluation of the model's performance.

<img width="438" alt="Screenshot 2024-02-11 at 8 53 14 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/ed156d4e-b008-4da0-8a45-c9709b5b2bc9">

<img width="788" alt="Screenshot 2024-02-11 at 8 08 30 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/8e1cc2e4-3500-43cf-8d0f-2759ef4fc651">

<img width="5000" alt="Screenshot 2024-02-10 at 9 42 17 AM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/7d09cd9f-4e3c-44fc-a569-a618b6cf91e5">


**The limitations of a basic Keras model** in handling such a complex and nuanced task stem from its typically shallow architecture, which might lack the capacity to capture the subtle differences between highly similar hotel rooms.

## **Part III: More advanced and pre-trained convolutional neural network models**

While **VGG and Inception**, the deeper convolutional neural networks, offer improvements by delving deeper into the image structure through its multiple layers.

Given the extensive size of our dataset, comprising **22,244 images across 1,674 classes**, coupled with the constraints of **limited computational resources**, we opted for a strategic approach. We **selected a smaller subset of the dataset (the top 500 classes with the most images)** to conduct preliminary tests on various deeper CNN pre-trained models. By concentrating on **tweaking the top-performing models**, we plan to gradually make them even better. This careful way of doing things helps us deal with our limited resources and still get the best results we can.

<img width="1200" alt="Screenshot 2024-02-11 at 5 12 17 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/9f0a6c3c-02ca-48b1-b39b-b694ea3bc1ee">
<p align="center">
<img width="600" alt="Screenshot 2024-02-11 at 6 52 34 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/ea8d2f3a-f0d2-4288-b210-6f749a858a25">
</p>

**VGG-16:** Progress from a basic Keras model to the more advanced and pre-trained VGG-16 model for image classification. **VGG-16 is a powerful convolutional neural network pre-trained on the ImageNet dataset**, we are hoping to leverage deeper and more complex architectures that have been proven effective on a wide range of image recognition tasks. **VGG and Inception** represent earlier convolutional neural network (CNN) designs focusing on **depth** and **multi-scale** processing, while models like **ResNet, DenseNet, and EfficientNet** are more advanced iterations that incorporate additional mechanisms to handle the challenges of deeper network structures. 

**ResNet** introduces a novel approach to facilitate training deeper networks through skip connections;

**DenseNet** streamlines the training of deep architectures by densely connecting each layer to every other layer, ensuring maximum information flow between layers. This design not only enhances performance with a more efficient parameter usage but also helps alleviate the vanishing-gradient problem, making deep networks easier to train;

**EfficientNet** optimizes CNN scaling by uniformly increasing depth, width, and resolution with fixed coefficients, leading to state-of-the-art performance on image classification tasks. 

<img width="5500" alt="Screenshot 2024-02-11 at 7 26 20 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/93883322-d4ed-4c7e-b84d-aeabd870e0e8">


## **Part IV: Our best model-EfficientB0: more tuning and incorporating ArcFace**

**Among all the pre-trained models, EfficientNetB0 stood out; we continue to use the EfficientNetB0 architecture as a base, leveraging transfer learning, regularization, and data augmentation to enhance performance on potentially complex datasets.**

<img width="850" alt="Screenshot 2024-02-11 at 7 16 38 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/4bf5c52c-c0a3-4dfd-acf3-7dbdd2d6493c">

After rigorous experimentation with various configurations of the EfficientNetB0 model, we observed marginal enhancements in validation accuracy and a slight reduction in validation loss. However, the performance gains have plateaued, indicating a potential limitation in the capacity of the base model to learn more complex patterns or generalize further from our dataset. It suggests that we may have reached the intrinsic performance ceiling of the EfficientNetB0 architecture for our specific application. In pursuit of more substantial improvements, we have decided to transition our efforts to the ArcFace method. 


### **What is ArcFace? Why ArcFace?**
The ArcMarginProduct class, inspired by the ArcFace method, is a custom TensorFlow layer designed to **significantly enhance the discriminative power of feature embeddings** in deep learning models, particularly beneficial for classification tasks where subtle differences between classes, such as facial features or hotel rooms, are paramount. ArcFace stands out for its innovative approach of enforcing an angular margin between classes in the feature space. This technique effectively amplifies the model's sensitivity to intra-class variations and ensures a clearer separation between different classes.

<img width="1300" alt="Screenshot 2024-02-10 at 5 47 44 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/347415ed-0fea-43fa-b38f-6f777098c0f7">


By adjusting the cosine distance between feature vectors and class centers with a **scale (s) and margin (m)** parameter, ArcFace makes the model more adept at distinguishing between closely related categories. The ArcMarginProduct layer implements this by applying the angular margin in the cosine space, offering options for easy margin settings and **label smoothing (ls_eps)** to promote training stability and enhance generalization. This layer is particularly useful in scenarios demanding high discriminative capabilities, such as identifying nuanced differences in hotel room categories or in facial recognition tasks, ensuring the model learns robust and distinct features for each class, thereby improving accuracy and generalization in scenarios characterized by subtle variances.

<img width="1176" alt="Screenshot 2024-02-10 at 5 48 22 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/c4559c51-fc88-44e3-9126-676c95ded963">

#### **Reference:**
##### 1. https://arxiv.org/pdf/1801.07698.pdf (**Authors:** Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou)
##### 2. [How to train with ArcFace loss to improve model classification accuracy | by Yiwen Lai](https://yiwenlai.medium.com/how-to-train-with-arcface-loss-to-improve-model-classification-accuracy-d4035195aeb9) (**Author:** Yiwen Lai)
##### 3. https://github.com/Niellai/ObjectDetection/blob/master/10_COVID19_ArcFace.ipynb?source=post_page-----d4035195aeb9-------------------------------- (**Author:** Niel Lai)
##### 4. https://www.kaggle.com/code/hidehisaarai1213/glret21-efficientnetb0-baseline-inference/notebook (**Author:** Hidehisa Arai)
##### 5. https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py (**Author:** Lyakaap)


We **freeze all but the last three layers of EfficientNetB0** for feature extraction, apply **global average pooling**, and optionally include dropout for regularization. The ArcMarginProduct layer, which requires both feature and label inputs, is utilized to enforce an angular margin that enhances the separability between classes.

<img width="686" alt="Screenshot 2024-02-11 at 3 57 27 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/013f6446-75fe-4a05-862d-32356152faa3">

**Note:** Incorporating the ArcFace method into our EfficientNetB0 model represents a strategic pivot towards leveraging advanced techniques to enhance feature discrimination. However, the optimization of such sophisticated methods requires extensive experimentation and tuning to fully realize their potential. Given the constraints of our capstone project's timeline, we have not yet achieved the optimal configuration of the ArcFace-enhanced EfficientNetB0 model that surpasses the performance of other models. As of the submission of this minimum viable product (MVP) notebook, our exploration into fine-tuning and experimentation is ongoing. We anticipate that with additional time and iterative refinement, the integration of ArcFace will yield significant performance improvements and set a new benchmark for our model's capabilities.

## **Part V: Conclusions and Next Steps**

### **Project Conclusions:** 

The project's advancement is currently hampered by a scarcity of data. Identifying ways to augment the dataset is imperative for the continued improvement of the model's performance. **Optimal Model Selection:** EfficientNetB0 has been identified as the superior base model, delivering top-notch performance in accuracy and training time when compared with a custom-built Keras model and other advanced pre-trained models.
**ArcFace Enhancement:** The integration of ArcFace into our leading base model has shown considerable potential. Despite the time constraints of the capstone project preventing full optimization, the initial findings suggest that further tuning could yield superior results.


### **Next Steps for Progress:**

**1. Expanding Dataset Application:** Building on the success with a subset, plans are in place to extend the application to the full dataset, with the expectation of further honing the model's accuracy.

**2. Advanced Model Experimentation:** The exploration of cutting-edge models is set to continue. Ensemble methods will be investigated to amalgamate the unique advantages of each model, thereby enhancing overall efficacy.

**3. Hyperparameter Optimization:** A key focus will be to fine-tune ArcFace's hyperparameters, with ongoing experiments aimed at optimizing its performance within the model framework.

**4. Practical Deployment:** The refined model is intended to be applied to images with occlusions, specifically those from hotel environments, to assist in human trafficking investigations.

**5. User Interface Development:** Consideration is being given to creating a Streamlit-based application, which would simplify the process of uploading images for users, thereby bolstering the dataset and improving the model's precision.

**Each step taken towards optimizing the model's performance is a step towards the noble aim of rescuing more lives. The commitment to this cause remains the project's driving force.**

<img width="840" alt="Screenshot 2024-02-10 at 1 57 19 PM" src="https://github.com/Yiyi-Luo/Capstone-Project-Hotel-Detection-to-Combat-Human-Trafficking/assets/149438809/3d1a5e98-6647-4e42-a34e-9aca711c00fa">

