# Generative AI Course

## General Course Information

Generative AI explores the creation of models capable of generating new content, spanning images, text, music, and beyond. This course delves into the core concepts, methods, and applications of generative deep learning, offering both theoretical foundations and practical skills.

## Course Objectives

- Understand the foundational concepts and methods in generative deep learning.
- Implement and train various generative models, including VAEs, autoregressive models, and graph neural networks.
- Apply generative models to diverse domains, such as music generation, world modeling, and multimodal data synthesis.
- Gain practical experience in generative AI through hands-on projects and exercises.

## Learning Outcomes

By the end of this course, students should be able to:

- Explain the principles and algorithms of generative deep learning models.
- Implement and train generative models using PyTorch.
- Apply generative models to create novel content across different domains.
- Analyze and evaluate the performance of generative models for specific tasks and datasets.


## Course Content (Subject to Change)

- Introduction to Generative Modeling
- Variational Autoencoders (VAEs)
- Autoregressive Models
- Energy-Based Models
- Diffusion Models
- Graph Generative Models
- Music Generation
- Image Generation
- World Models
- Multimodal Models

## Lecture Breakdown

1. **Variational Autoencoders (VAEs)**. [Code](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/CVAE.ipynb) [Notes]()
  - [Variant versions of VAR](https://github.com/AntixK/PyTorch-VAE)      
2. **Autoregressive Models** [Intro to AR models](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/auto_regressive_models.pdf) [simple_autoregressive](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/Simple_autoregressive_gen_model.ipynb) [SimplePixelCNN](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/PixelCNN.ipynb) [PixelCNN full implementation](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/05_autoregressive/02_pixelcnn/pixelcnn.ipynb) [char rnn](https://github.com/karpathy/char-rnn) [rnn-effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
3. **Diffusion Models** -(Part I)
    * [Diffusors from sctrach](https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb) [Diffusion 1d example from sctrach](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/1d_Diffusion_Model_from_scratch.ipynb) [Diffusion Models in Pytorch](https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm.py) [Lecture](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/diffusion_models_.pdf)
    * [Tutorial  on diffusion models](https://arxiv.org/pdf/2403.18103)
    * Recall : [U-Nets](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/U_net_explained.ipynb) [U-net lecture](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/U-nets.pdf)
    * [Topological Explaination of Diffusion Models, Homotopy](https://mathematica.stackexchange.com/questions/59463/homotopy-visualization)
4. **Diffusion Models** - (Part II)
    * [Lecture](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/diffusion_models_2.pdf)
    * [Score Matching](https://colab.research.google.com/drive/1dol5AXz_oNkFZMrwpDyK6MYnOB4ayEQU?usp=sharing#scrollTo=rU0m57SJfXqb)
    * [Conditional diffusion model introduction](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)
    * [Fine Tuning diffusors](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb#scrollTo=2n9AmuTZlWLI)
    * [Training Diffusors](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=67640279-979b-490d-80fe-65673b94ae00)
    * [Stable diffusion introduction](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit3/01_stable_diffusion_introduction.ipynb#scrollTo=fx6whXJmsNG9),     [Stable diffusion](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb#scrollTo=YE7hhg5ArUu4)
    * [Stable Diffusion in pytorch](https://github.com/hkproj/pytorch-stable-diffusion/tree/main)
    * [Understanding Stable Diffusions](https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing)
    * [Stable Diffusion Video](https://www.youtube.com/watch?v=J87hffSMB60&t=615s)
    * [Diffusors course](https://huggingface.co/learn/audio-course/chapter0/introduction)
6. **Energy-based models** -
      * [Energy-based model tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html)
      * [Lecture](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/energy_models.pdf)
      * [Impainting crop](https://github.com/USTC-JialunPeng/Diverse-Structure-Inpainting)
      * Impainting [notebook](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/inpainting_noise.ipynb)
      * [simple MCMC](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/Simple_MCMC.ipynb) [MCMC good introduction video](https://www.youtube.com/watch?v=yApmR-c_hKU)
      * [Metropolis-adjusted_Langevin_algorithm](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm)
      * [Phyics of energy based models](https://physicsofebm.github.io/)
     
7. **Generative 3D Models** -
      * [Lecture](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/mesh_generation.pdf) [reference](https://arxiv.org/pdf/2301.11445), [github repo ref](https://github.com/1zb/3DShape2VecSet)
      * [Code](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/MeshAutoEncoder.ipynb)
        
           
7. **Graph Generative Models**
      * [Graph Variational AutoEncoders Lecture](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/GVAE.pdf) [Code](https://github.com/zfjsail/gae-pytorch/tree/master)
      * [Great intro to graph rep learning](https://www-cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf)
      * [Graph RNN](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/deep_graph_generation_.pdf)
   
8. **Multimodal Models (Part I)** - Fusion of Text, Image, and Audio Data
      * [A template notebook for building a multimodal](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/multimodal_model_template.ipynb)
      * [Embedding Alignment](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/Embedding_Alignment.ipynb)
      * [Introduction to clip](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/Introduction_to_clip.ipynb)
      * [Building clip](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/building_clip.ipynb)
      * [Introduction to zero shot classification with clip](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/zero_shot_classification_with_clip.ipynb)
      * [zero shot classification with clip](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/Zero_shot_classification_with_clip_part2.ipynb)
      * [Using clip in search](https://blog.lancedb.com/multi-modal-ai-made-easy-with-lancedb-clip-5aaf8801c939/)
      * [Training Clip in practice](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/Training_clip_in_practice.ipynb) [Clip-X](https://github.com/lucidrains/x-clip) [openClip](https://github.com/mlfoundations/open_clip/tree/main) [Hugging face clip model](https://huggingface.co/docs/transformers/en/model_doc/clip)
      * [Simplified DALLE from scratch](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/Simplified_DALLE.ipynb)
      * [Vision Transformer main popular github library](https://github.com/lucidrains/vit-pytorch)
      * [Lecture](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/Intro_to_MDM.pdf)
        

10. **Multimodal Models (Part II)** - Advanced Techniques and Case Studies



## Grading
This course will involve several homework assignments and a final project focused on generative AI. Attendance and participation, homework assignments, and the final project will be graded as follows:

- Attendance and Participation: 10%
- Homework Assignments: 40%
- Final Project: 50%

## Homeworks 
- [HW1](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/HW1.md) due June 1
- [HW2](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/HW2.md) [helper](https://huggingface.co/docs/diffusers/v0.18.2/en/tutorials/basic_training) due June 8
- [HW3](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/HW3_.ipynb) due June 15
- [HW4](https://github.com/USFCA-MSDS/MSDS-631-GenAI/blob/main/HW4.md) due June 26

# Course Project:

## Overview

This course will involve a final project focused on generative AI. Your final project will account for 50% of your total grade and will be completed in three tasks: the project proposal, a Jupyter notebook documenting your work, and a recorded presentation.

## Final Project

### Task 1: Project Proposal, Data Selection, and Data Description (5%) ( due June 8 )

In this task, you will submit a project proposal that includes the following components:

- **Project Proposal**: Outline the objectives and goals of your project. Describe the problem you intend to address using generative AI techniques and your overall approach.
- **Data Selection**: Detail the sources of your data and explain why these sources are suitable. Discuss any preprocessing steps required.
- **Data Description**: Provide a comprehensive description of the data, including format, size, attributes, and any inherent challenges or limitations.

### Task 2: Jupyter Notebook (35%) ( due June 27th )

You will create a comprehensive Jupyter notebook that documents your project, including the following components:

1. **Data Preprocessing (5%)**: Describe the methods and steps used to preprocess and prepare the data.
2. **Model Implementation (10%)**: Detail the architecture and implementation of your generative AI model. Include code snippets and explanations of your model choices and any modifications.
3. **Methods (5%)**: Explain the algorithms, techniques, or frameworks used in your project.
4. **Experiments and Results (10%)**: Present your experiments and results using performance metrics, visualizations, and analyses.

### Task 3: Recorded Presentation (10%) ( due June 27th )

You will deliver a recorded presentation explaining your project, including its objectives, methodology, results, and conclusions. This presentation should not exceed 10 minutes and should involve all team members.

## Generative AI Project Ideas

Here are 20 project ideas for your generative AI project:

1. **Image Generation**: Train a Generative Adversarial Network (GAN) to generate realistic images from random noise.
2. **Text-to-Image Synthesis**: Develop a model that generates images based on textual descriptions.
3. **Image-to-Image Translation**: Use a model like CycleGAN to translate images from one domain to another (e.g., transforming summer landscapes to winter landscapes).
4. **Style Transfer**: Create a model that transfers the artistic style of one image onto another while preserving the original content.
5. **Music Generation**: Train a model to compose original music pieces.
6. **Text Generation**: Develop a model to generate coherent and contextually relevant text, such as poetry, stories, or news articles.
7. **Speech Synthesis**: Build a model to generate human-like speech from text input.
8. **3D Object Generation**: Use a generative model to create 3D objects or shapes.
9. **Face Generation**: Train a model to generate realistic human faces.
10. **Super-Resolution**: Develop a model to enhance the resolution and quality of low-resolution images.
11. **Video Generation**: Create a model that generates short video clips based on given inputs.
12. **Handwriting Generation**: Train a model to produce realistic handwritten text in various styles.
13. **Chatbot Creation**: Develop a conversational AI that generates human-like responses.
14. **Fashion Design**: Use generative models to create new clothing designs.
15. **Image Inpainting**: Build a model that can fill in missing parts of an image seamlessly.
16. **Data Augmentation**: Generate synthetic data to augment a dataset for training other machine learning models.
17. **Recipe Generation**: Develop a model that creates new recipes based on ingredient lists.
18. **Game Level Generation**: Use a generative model to design new levels for video games.
19. **Code Generation**: Train a model to generate code snippets based on natural language descriptions.
20. **Art Creation**: Use generative models to create new pieces of digital art.



## Important Information

- Ensure your project is well-documented and reproducible.
- Your Jupyter notebook should be well-organized and easy to follow.
- Your recorded presentation should effectively communicate your project's objectives, methods, and results.

Good luck with your final project!

## Textbook

- "Generative Deep Learning, 2nd Edition" by David Foster. [Pytorch repo](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main)
