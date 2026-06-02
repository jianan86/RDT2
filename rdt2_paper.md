# RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization

Songming Liu * 1 Bangguo Li * 1 Kai Ma * 1 Lingxuan Wu * 1 Hengkai Tan 1 Xiao Ouyang 1 Hang Su 1 Jun Zhu 1 

# Abstract

Vision-Language-Action (VLA) models hold promise for generalist robotics but currently struggle with data scarcity, architectural inefficiencies, and the inability to generalize across different hardware platforms. We introduce RDT2, a robotic foundation model built upon a 7B parameter VLM designed to enable zero-shot deployment on novel embodiments for open-vocabulary tasks. To achieve this, we collected one of the largest open-source robotic datasets—over 10, 000 hours of demonstrations in diverse families—using an enhanced, embodiment-agnostic Universal Manipulation Interface (UMI). Our approach employs a novel three-stage training recipe that aligns discrete linguistic knowledge with continuous control via Residual Vector Quantization (RVQ), flow-matching, and distillation for realtime inference. Consequently, RDT2 becomes one of the first models that simultaneously zeroshot generalizes to unseen objects, scenes, instructions, and even robotic platforms. Besides, it outperforms state-of-the-art baselines in dexterous, long-horizon, and dynamic downstream tasks like playing table tennis. See project page for more information. 

# 1. Introduction

Vision-Language-Action (VLA) models represent a promising paradigm for achieving generalized embodied intelligence (Team et al., 2024; Kim et al., 2024; Liu et al., 2024; Black et al., 2024; Intelligence et al., 2025). They are particularly well-suited for complex manipulation tasks involving deformable objects and fluids (Ma et al., 2024), which have long been challenging for traditional control methods due to the difficulty of physical modeling and system identification (Saha & Isto, 2006; Jatavallabhula et al., 2021). However, despite several valuable trials (Ma et al., 2024), current VLA models have not replicated the broad generalization capabilities characteristic of large-scale models in other domains such as Natural Language Processing (NLP) (Achiam et al., 2023; Touvron et al., 2023; Bai et al., 2023; Guo et al., 2025). They often struggle to perform reliably when encountering novel scenes, objects, instructions, or embodiments (Ma et al., 2024), hindering real-world applications. 

Developing generalizable VLA models for robotics presents two fundamental challenges. The first is the acquisition of large-scale, diverse datasets. Traditional data collection through teleoperation (Zhao et al., 2023; Fu et al., 2024) is often prohibitively expensive and lacks variety due to the physical constraints and high cost of robotic platforms. In contrast, the Universal Manipulation Interface (UMI) (Chi et al., 2024) provides an embodiment-agnostic, handheld device that enables efficient and low-cost data collection across a multitude of real-world scenarios. The second challenge lies in designing network architectures that can effectively learn from this large-scale robot data. A key difficulty is the inherent multimodality of human-collected demonstrations (Chen et al., 2022; Chi et al., 2023). Prior approaches that model action probabilities via discretization (Brohan et al., 2022; Zitkovich et al., 2023; Kim et al., 2024) are often constrained by the resultant errors and the inefficiency of autoregressive inference. Alternative methods using diffusion models (Chen et al., 2022; Chi et al., 2025; Liu et al., 2024; Black et al., 2024) suffer from slow convergence (Pertsch et al., 2025) and a fundamental mismatch between their continuous probability distributions and the discrete counterparts of knowledge in pre-trained Vision-Language Models (VLMs). Furthermore, a significant tension exists between the growing size of these models and the real-time performance required for robotic tasks. While some distillation techniques have been explored (Chen et al., 2023; Wang et al., 2024b; Prasad et al., 2024), the development of practical methods for large-scale VLA models remains an open problem. 

Furthermore, VLA models confront a significant limitation for cross-embodiment deployment. Due to variations in physical characteristics across robotic platforms, models trained on one embodiment exhibit poor generalization when transferred to another. While some methods attempt to unify data from different embodiments into a common embedding space (Team et al., 2024; Liu et al., 2024; Wang et al., 2024a; Yang et al., 2024), they still fall short of enabling zero-shot deployment on novel platforms. Consequently, adapting a VLA model to a new robot often necessitates hundreds of hours of data collection and fine-tuning. This substantial cost not only impedes the reproducibility and widespread applicability of VLA research but also curtails the overall progress of the field (Khazatsky et al., 2024; Atreya et al., 2025; Mirchandani et al., 2025). 

To address the aforementioned challenges, we introduce RDT2, one of the first robotic foundation models for zeroshot deployment on novel embodiments, which can handle open-vocabulary tasks. RDT2 is built upon a 7B pretrained VLM, Qwen2.5-VL (Bai et al., 2025), with specialized action heads and three-stage training strategies for learning from large-scale robotic data. For fast convergence, in Stage 1, we encode the continuous robot actions into discrete tokens with Residual Vector Quantization (RVQ) (Van Den Oord et al., 2017; Esser et al., 2021; Lee et al., 2022) and then train the VLM by minimizing the cross-entropy loss. This also avoids destroying the knowledge stored in the form of discrete probabilities during pre-training. In Stage 2, for expressiveness and efficiency, we employ an action expert to model continuous probability and train it with the flow-matching loss. In Stage 3, we propose a simple yet effective distillation loss and distill the action expert into a single-step generator, achieving ultra-fast inference speed. 

Based on the above methods, we were able to train our model, RDT2, on one of the largest open-source UMI datasets, comprising over 10, 000 hours of human demonstrations. This large-scale data collection was made possible by redesigning the UMI hardware with higher-strength materials and high-precision tracking methods to ensure reliability. We fabricated approximately 100 of these enhanced devices and deployed them across more than 100 real-world household environments to capture a diverse range of manipulation tasks. In our experiments, we first evaluated RDT2’s zero-shot generalizability across unseen objects, scenes, instructions, and even embodiments. Because UMI provides an embodiment-agnostic physical interface, coupled with large-scale pretraining, RDT2 became one of the first models to achieve combined generalization of the four factors for open-vocabulary tasks. Through experiments with four different model sizes, we discovered that simultaneously scaling up model parameters and data scale yields consistent and predictable performance gains. Besides, our fine-tuning experiments showed that RDT2 outperformed state-of-the-art baselines such as $\pi _ { 0 } – \mathrm { F A S T }$ (Pertsch et al., 2025) and $\pi _ { 0 . 5 }$ (Intelligence et al., 2025) on challenging tasks involving deformable objects, dexterity, long horizons, and high dynamics, such as playing table tennis. Finally, extensive ablation studies demonstrated the effectiveness of the adopted training strategy and design choices. 

# 2. Related Work

Data Pyramid for Robotics. The landscape of data for robot learning can be conceptualized as a pyramid (Bjorck et al., 2025). At the apex resides teleoperation data, which, gathered via systems like VR (Khazatsky et al., 2024; Cheng et al., 2024; Chen et al., 2025a) or master-slave arms (Zhao et al., 2023; Fu et al., 2024; Aldaco et al., 2024), offers the highest fidelity but is also the most expensive to acquire. Its collection is typically confined to structured laboratory settings (Walke et al., 2023; Fang et al., 2023; Khazatsky et al., 2024; O’Neill et al., 2024; Wu et al., 2024), creating a distributional gap between the training data and real-world applications. Occupying the middle tier is simulation data (Wang et al., 2023; Li et al., 2023; Mu et al., 2024; Chen et al., 2025b); it is inexpensive and scalable but plagued by a significant sim-to-real gap, and the challenge of generating diverse, interactive, and realistic scenarios remains an open problem (Nasiriany et al., 2024; Ren et al., 2024; Zhang et al., 2025). At the base lies the vast repository of internet videos (Ye et al., 2024; Yang et al., 2025; Luo et al., 2025; Feng et al., 2025). Although abundant, this data is unstructured and noisy, and most importantly, lacks the explicit action labels required for the supervised policy training (McCarthy et al., 2025). 

Imitation Learning Models. Previous models can be broadly classified by their strategy for generalization. A significant body of work focuses on small-scale models (Pari et al., 2021; Florence et al., 2022; Shafiullah et al., 2022; Jang et al., 2022), such as Diffusion Policy (Chen et al., 2022; Chi et al., 2023) and ACT (Zhao et al., 2023), which are typically trained on a per-task basis. While proficient within their specific domains, these models inherently lack the capacity to generalize across diverse tasks or embodiments. To address the cross-embodiment challenge, another line of research leverages the UMI (Chi et al., 2024; Xu et al., 2025) to collect embodiment-agnostic data. However, the limited scale of these datasets constrains the resulting models’ performance on open-vocabulary tasks. More recently, the field has seen the emergence of large models like OpenVLA (Kim et al., 2024), RDT-1B (Liu et al., 2024), $\pi _ { 0 }$ (Black et al., 2024), and $\pi _ { 0 . 5 }$ (Intelligence et al., 2025). However, since the dataset relies on specific robots, they fall short of embodiment transferability without fine-tuning. 

# 3. Problem Formulation and Challenges

We consider the bimanual manipulation task in the setting of language-conditioned imitation learning for VLA models, which is well-established in the field of robot learning (Stepputtis et al., 2020; Zhou et al., 2023). Formally, the task is modeled as a sequential decision-making process. Let ℓ denote the a free-form language instruction describing the task. At each time step t, an agent is required to take an action chunk $\mathbf { A } _ { t } : = ( \mathbf { a } _ { t } , \dots , \mathbf { a } _ { t + T _ { a } } )$ sampled from $p ( \mathbf { A } _ { t } \mid \boldsymbol { \ell } , \mathbf { o } _ { t } )$ , where $\mathbf { a } _ { t } \in \mathbb { R } ^ { d }$ is the d-dimensional action taken at $t , T _ { a }$ is the chunk size (Zhao et al., 2023), and $\mathbf { o } _ { t }$ is the RGB observation that the agent accept at t. Here, we assume that $\mathbf { o } _ { t }$ already contains all the information needed to make a decision, without considering historical observations $\{ \mathbf { o } _ { i } \ | \ i \ < \ t \}$ . To obtain a feasible agent, we train a VLA model to learn the distribution p $( \mathbf { A } _ { t } \ | \ \ell , \mathbf { o } _ { t } )$ from a demonstration dataset of human experts $\mathcal { D } : = \{ ( \ell ^ { ( i ) } , \mathbf { o } _ { t } ^ { ( i ) } , \mathbf { A } _ { t } ^ { ( i ) } ) ~ | ~ 0 \leq t < T ^ { ( i ) } , 1 \leq i \leq N \}$ , where $T ^ { ( i ) }$ is the i-th trajectory length and N is the number of total trajectories. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/30fc9160ca65ad21176a8415741fe1f6794eda43a56339183efc3ad0227b0a13.jpg)



(a) Our Re-Designed UMI


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/57b82ecfd64ade2671c5ddf0de5591c847663f9c9f14e622c0bcf764d24cd752.jpg)



(b) Easy Deployment


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/f5c0b4e9935488d663ed9bf0cee237278b4a5d0160439c3d2698a585d2d0f084.jpg)



(c) Cross-Embodiment Transfer



Figure 1: Illustration of our UMI solution. We re-designed the UMI hardware for better consistency and reliability in large-scale data collection. As long as the same model of camera and gripper are installed, the policy trained on the data collected by our UMIs can be zero-shot transferred to various robot arms.


A given manipulation task is defined by the composition of several key elements: the manipulated object, the operational scene, the natural language instruction from users, and the robotic embodiment. Any finite training dataset can only cover a sparse subset of the vast combinatorial space spanned by these elements. A practical VLA model must therefore generalize to unseen compositions of objects, scenes, instructions, and even novel embodiments during deployment. Achieving this compositional generalization, which is crucial for real-world applications, presents two fundamental challenges: 

Challenge of Scaling Up Robotic Data. It is well-studied in natural language processing and computer vision that increasing dataset scale and diversity improves model generalization (Kaplan et al., 2020; Zhai et al., 2022). However, applying this principle to robotics via teleoperation is currently impractical. The high cost of robotic hardware makes parallel data acquisition, and thus large-scale data collection, prohibitively expensive. Furthermore, the lack of portability of these systems constrains data acquisition primarily to laboratory or factory settings, severely limiting the diversity and real-world relevance of the collected data. This data scarcity is exacerbated by hardware heterogeneity, as data collected on one robotic platform is often incompatible with others, creating isolated and non-interoperable datasets. 

Challenge of Network Architecture. According to previous studies (Chen et al., 2022; Chi et al., 2023), human data exhibits significant multimodality, necessitating models that learn a distribution over actions rather than a deterministic mapping. This leads to a choice between discrete and continuous action representations, each with distinct tradeoffs. Discrete methods align naturally with the probabilistic outputs of the pretrained VLM, but they suffer from quantization errors and the inefficiency of autoregressive sampling. Conversely, continuous approaches like diffusion models offer more efficient sampling but are hampered by slower training convergence (Pertsch et al., 2025) and risk corrupting the discrete knowledge within the VLM (Deng et al., 2025). A critical challenge, therefore, is to synthesize the advantages of both paradigms. What is more, the real-time performance demanded by robotic tasks makes the efficient deployment of large-scale VLAs a formidable obstacle. 

# 4. Hardware and Dataset

To address the data-scaling challenge, we employ UMI (Chi et al., 2024), a portable framework facilitating scalable, in-the-wild data collection. UMI records the 6-DoF endeffector pose and the gripper width using a hand-held device with vision and a tracker. When installing a physically consistent gripper, policies learned from UMI data can be deployed on diverse robotic arms as both vision and structure gaps are minimized across embodiments. Fig. 1 illustrates our UMI solution from design to deployment. 

Re-Designing UMI. However, the original UMI hardware lacks the reliability requisite for large-scale in-the-wild collection. To address this, we re-engineered the whole system (Fig. 1a) to maximize structural rigidity, ensure drift-free infrared tracking, and enhance manipulation dexterity in cluttered environments. As shown in Tab. 1, these modifications resolve critical pose inconsistencies and limitations of reachability, yielding significantly improved data fidelity; detailed hardware specifications are provided in App. A. 


Table 1: Comparison between the original UMI and our redesigned hardware.


<table><tr><td>Specification</td><td>Naive UMI</td><td>Our UMI</td><td>Advantage</td></tr><tr><td>Fabrication</td><td>3D Printing (PLA / PETG)</td><td>CNC (nylon 66 &amp; glass fiber)</td><td rowspan="2">Higher stiffness, better machining accuracy and consistency; suitable for long-term, high-frequency data collectionBetter tracking precision for the end-effector 6D pose; more robust to high-speed motion, texture-less backgrounds, and transparent backgrounds</td></tr><tr><td>Tracking</td><td>SLAM</td><td>Infrared Light</td></tr><tr><td>End-Effector</td><td>Parallel Jaws</td><td>Linkage Gripper</td><td>More compact structure; improved dexterity and accessibility in tight clearances or clutter</td></tr></table>

UMI Dataset at Scale. Enabled by these hardware advancements, we curate one of the largest open-source UMI datasets to date, comprising over 10, 000 hours of manipulation data spanning more than 100 households. Captured entirely in the wild, our dataset encapsulates a vast distribution of unstructured environments and complex human behaviors, providing a robust substrate for generalist policy learning; we elaborate on the dataset details in App. B. 

# 5. Model and Training Pipeline

We introduce RDT2, a VLA model trained through a threestage pipeline as shown in Fig. 2. In Stage 1 (Sec. 5.1), we discretize the continuous action space into tokens using RVQ and train the VLM backbone via a standard crossentropy loss. Subsequently, in Stage 2 (Sec. 5.2), we freeze the VLM backbone and train a diffusion-based action expert, leveraging a flow-matching loss to generate continuous actions. This hybrid approach harnesses the benefits of both discretization and diffusion, effectively addressing the challenge of modeling multimodal action distributions. Finally, to resolve the real-time challenge for robotic tasks, the Stage 3 (Sec. 5.3) involves distilling the multi-step action expert into an efficient, single-step generator, thereby enabling rapid inference for our large-scale VLA model. We refer to App. C for hyperparameter and training details. 

# 5.1. Stage 1

As previously discussed, diffusion models present two primary issues for VLA training: slow convergence and the degradation of discrete probability knowledge within pretrained VLMs. To mitigate these problems, we opt to first pretrain the VLM backbone using a cross-entropy loss before diffusion training, which is consistent with its original training objective. This helps us effectively preserve the model’s valuable pretrained knowledge, which additionally benefits from our add-ons of vision-language data during training. As illustrated in Fig. 6, our experiments confirm that this discretized pretraining phase significantly accelerates the convergence of the VLA model compared to training directly with the diffusion loss from the outset. In the following, we elaborate on the training details. 

RVQ Tokenizer. To facilitate the cross-entropy training, we employ the RVQ (Van Den Oord et al., 2017; Esser et al., 2021; Lee et al., 2022) for discretization due to its high compression efficiency. Specifically, we first encode the continuous action chunk $\mathbf { A } _ { t } \in \mathbb { R } ^ { T _ { a } \times d }$ with a 1D temproal convolutional nerual networks (CNNs) ϕenc into n latents of C dimensions, denoted by $\{ \mathbf { z } _ { i } \in \mathbb { R } ^ { C } \} _ { i = 1 } ^ { n } = \phi _ { \mathrm { e n c } } ( \mathbf { A } _ { t } )$ . For each $\mathbf { z } _ { i } , 1 \leq i \leq n$ , starting with $\mathbf { r } _ { 0 } ^ { i } = \mathbf { z } _ { i }$ , we quantize it via an iterative process of depth m: 

$$
k _ {j} ^ {i} = \arg \min _ {1 \leq k \leq K} \| \mathbf {r} _ {j - 1} ^ {i} - \mathbf {e} _ {j} (k) \| _ {2} ^ {2}, \tag {1}
$$

$$
\mathbf {r} _ {j} ^ {i} = \mathbf {r} _ {j - 1} ^ {i} - \mathbf {e} _ {j} (k _ {j} ^ {i}),
$$

for $j = 1 , \ldots , m$ , where $\mathbf { e } _ { j } \in \mathbb { R } ^ { K \times C }$ is the learnable codebook of size K at depth j. As a result, $\{ k _ { 1 } ^ { i } , \ldots , k _ { m } ^ { i } \} _ { i = 1 } ^ { n }$ will be the token index for At and $\begin{array} { r } { \phi _ { \mathrm { d e c } } \big ( \{ \sum _ { j = 1 } ^ { m } { \bf e } _ { j } ( k _ { j } ^ { i } ) \} _ { i = 1 } ^ { n } \big ) } \end{array}$ t  dec i i=1  will be the quantization result, $\hat { \mathbf { A } } _ { t } : = \phi _ { \mathrm { d e c } } ( \{ \hat { \mathbf { z } } _ { i } \} _ { i = 1 } ^ { n } ) =$ where $\phi _ { \mathrm { d e c } } ( \cdot )$ is the reverse 1D CNN decoder. We minimize the following loss to train the tokenizer: 

$$
\mathcal {L} _ {\mathrm{vq}} := \mathbb {E} _ {\{\cdot , \cdot , \mathbf {A} _ {t} \} \sim \mathcal {D}, 1 \leq i \leq n} \left[ \| \mathbf {A} _ {t} - \hat {\mathbf {A}} _ {t} \| _ {2} ^ {2} \right. \tag {2}
$$

$$
\left. + \left\| \operatorname{sg} (\mathbf {z} _ {i}) - \hat {\mathbf {z}} _ {i} \right\| _ {2} ^ {2} + \beta \| \mathbf {z} _ {i} - \operatorname{sg} (\hat {\mathbf {z}} _ {i}) \| _ {2} ^ {2} \right].
$$

To mitigate the notorious codebook collapse, we have taken several measures during RVQ training, including lower codebook dimension (Yu et al., 2021), replacing the Euclidean distance with cosine similarity in Eq. (1) (Yu et al., 2021), smoothing codebook updates via exponential moving average (EMA) (Razavi et al., 2019), and restarting inactive codebook entries every fixed period (Zeghidour et al., 2021). As shown in Fig. 8, at the same level of quantization errors, our RVQ can compress action chunks into fewer tokens, which could greatly accelerate the large VLA’s convergence. 

Model Details. We selected the 7B Qwen2.5-VL (Bai et al., 2025) as our VLA backbone, leveraging its extensive pre-training on large-scale vision-language corpora. We project various modalities to a unified latent space for learning: vision and language by Qwen encoder, actions by our RVQ model. We reserved the 1024 least frequent entries in the vocabulary to represent these action tokens. The VLA model was trained for 128K iterations on a composite dataset of our UMI dataset and a small subset of visionlanguage data, using a next-token prediction objective. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/92ac58e14ebdd05a21cceba4815272c31a9eaa2fbd920ad49b03b452b30b8834.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/53f64d12defe4e833d9c978e23c91c030e762df20465a00e54704d5120434f50.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/d4303e629c23735fe7a32bd7870911387cca28b73beacb299ed710ddc44645ea.jpg)



Figure 2: A three-stage pipeline for training RDT2. In Stage 1, we pre-train a 7B VLM backbone with discretized action data for vision-language reasoning capabilities. Then, in Stage 2, we train a small diffusion action expert to generate continuous actions efficiently. For highly dynamic tasks, we introduce a third stage that distills the diffusion policy into a one-step generator, thereby enabling extremely rapid inference speed.


# 5.2. Stage 2

To enhance inference efficiency beyond that of autoregressive models, we introduce a second stage of training. In this stage, we freeze the pretrained VLA backbone from Stage 1 and train a dedicated action expert. This expert, a 400M parameter variant of RDT-1B (Liu et al., 2024), is optimized for speed by substituting Multi-Head Attention (MHA) (Vaswani et al., 2017) with Grouped Query Attention (GQA) (Ainslie et al., 2023). It generates continuous actions through a diffusion process, which is conditioned on natural language and image representations encoded by the frozen VLA backbone. Specifically, the action expert leverages cross-attention to incorporate the latent features from each layer of the VLA backbone. 

Flow-Matching Training. The action expert is supervised by a flow-matching loss (Lipman et al., 2022): 

$$
\mathcal {L} _ {\text { expert }} (\theta) := \mathbb {E} _ {\{\ell , \mathbf {o} _ {t}, \mathbf {A} _ {t} \} \sim \mathcal {D}, \tau \sim \mathcal {U} (0, 1)} [ \| \tag {3}
$$

$$
\left. \mathbf {v} _ {\theta} (\tau , \mathbf {A} _ {t} ^ {\tau}, \mathrm{VLA} (\ell , \mathbf {o} _ {t})) - \mathbf {u} (\mathbf {A} _ {t} ^ {\tau} | \mathbf {A} _ {t}) \| _ {2} ^ {2} \right],
$$

where τ is the flow-matching time step, $\mathbf { v } _ { \theta } ( \cdot )$ is the denoising network with trainable parameters θ, and $\mathrm { V L A } ( \cdot )$ is the frozen VLA backbone. Here, we denote the noisy action chunk by $\mathbf { A } _ { t } ^ { \tau } : = ( 1 - \tau ) \boldsymbol { \epsilon } + \tau \mathbf { A } _ { t }$ , where $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is a random Gaussian noise. The ground-truth velocity is given by: ${ \bf u } ( { \bf A } _ { t } ^ { \tau } \ | \ { \bf A } _ { t } ) : = { \bf A } _ { t } - \epsilon$ . During inference, we first sample a Gaussian noise vector: $\mathbf { A } _ { t } ^ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and then denoise it to a clean action chunk: 

$$
\mathbf {A} _ {t} ^ {\tau + \delta \tau} = \mathbf {A} _ {t} ^ {\tau} + \delta \tau \cdot \mathbf {v} _ {\theta} (\tau , \mathbf {A} _ {t} ^ {\tau}, \mathrm{VLA} (\ell , \mathbf {o} _ {t})), \tag {4}
$$

from $\tau = 0 \ t \mathbf { o } \ \tau = 1$ . In practice, we set the step size $\tau = 0 . 2$ , corresponding to 5 integration steps. Besides, we only calculate VLA(·) once since it stays invariant during integration. In our experiment, we randomly initialized the action expert and trained it for 66K iterations on our UMI dataset, with the VLA backbone (trained in Stage 1) frozen. 

# 5.3. Stage 3

The action generation process, as formulated in Eq. (4), necessitates five sequential forward passes through the denoising network for each action chunk, which imposes a considerable inference overhead. This latency presents a practical bottleneck for tasks with high dynamic requirements, such as playing table tennis. To overcome this limitation, we employ diffusion distillation (Salimans & Ho, 2022; Chen et al., 2023) to convert the expert policy trained in Stage 2 into a single-step generator. As illustrated in Fig. 7, this technique drastically reduces model latency, enabling our large-scale VLA to achieve a significantly faster inference speed than much smaller models. 

Diffusion Distillation. For highly dynamic tasks, we distill the action expert into a single-step generator with parameters $\theta ^ { \prime }$ using the following regression objective: 

$$
\mathcal {L} _ {\text { distill }} \left(\theta^ {\prime}\right) := \mathbb {E} _ {\{\ell , \mathbf {o} _ {t}, \cdot \}} \sim_ {\mathcal {D}, \mathbf {A} _ {t} ^ {0}} \sim_ {\mathcal {N} (\mathbf {0}, \mathbf {I})} [ \| \tag {5}
$$

$$
\left. \mathcal {F} (\mathbf {A} _ {t} ^ {0}, \ell , \mathbf {o} _ {t}; \theta) - G (\mathbf {A} _ {t} ^ {0}, \ell , \mathbf {o} _ {t}; \theta^ {\prime}) \| _ {2} ^ {2} \right],
$$

where $\mathcal F ( \cdot )$ denotes the generation process in Eq. (4) and $G ( \mathbf { A } _ { t } ^ { 0 } , \ell , \mathbf { o } _ { t } ; \theta ^ { \prime } ) : = \mathbf { A } _ { t } ^ { 0 } + \mathbf { v } _ { \theta ^ { \prime } } ( 0 , \mathbf { A } _ { t } ^ { 0 } , { \mathrm { V L A } } ( \ell , \mathbf { o } _ { t } ) )$ is the target single-step generator. It is noted that θ has been pretrained in Stage 2 and stays frozen in this stage, $\mathrm { V L A } ( \cdot )$ is also frozen, and $\theta ^ { \prime }$ is trainable, initialized from θ. Unlike previous distillation practices, $\mathcal F ( \cdot )$ is computed onthe-fly during training, rather than pre-generated during data preparation. This approach offers a compelling advantage with acceptable computational overhead. Generating low-dimensional actions is remarkably efficient (with a few integration steps), in stark contrast to the image or video generation requiring up to hundreds of steps. At the same time, it yields a significant benefit by substantially reducing the risk of the distilled policy overfitting to the pre-generated data, a common pitfall in regression-based distillation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/b23d3d958306f1a0f2d937d78426ed1ba5e15f6ac41b8f7585026543dc60871c.jpg)



Figure 3: Results of zero-shot experiments of RDT2. The error bar represents the standard error.


# 6. Experiments

This paper aims to achieve superior generalizability for robotic models by increasing the quantity and diversity of data, which will be rigorously verified in this section. While it is common practice to fine-tune VLAs before evaluation, we want to zero-shot test our RDT2 under “4U” setting — Unseen embodiment, Unseen scene, Unseen object, and Unseen instruction. For quantitative evaluation, we will conduct repeated experiments up to 1000 trials to ensure sufficiently low variance (see Fig. 4), which was previously lacking in many studies but is crucial for the reliability of results. To be specific, our experiments answer the following questions (see App. D for experimental details): 

• Q1: Can RDT2 effectively generalize to unseen embodiments, objects, scenes, and instructions, which is impractical for previous VLAs? 

• Q2: What is the scaling law of RDT2’s generalizability with respect to training data and model size? 

• Q3: How does RDT2 compare to other VLAs, in terms of fine-tuning experiments on challenging dexterous, dynamic, or long-horizon tasks? 

• Q4: How does each component of our training strategy contribute to the performance of RDT2? 

# 6.1. Zero-Shot Experiments

To answer Q1, without any fine-tuning, we deployed RDT2 on unseen embodiments and evaluated it on tasks with unseen objects, scenes, and instructions. The model was pretrained solely on UMI human data and vision-language pairs, without any robotic data. We considered simple openvocabulary tasks: pick up an object specified in free-form language, pick up a specified object and place it in a specified location, wipe a table with any cloth, press any button, and shake any object. The specific task settings are described in Fig. 15. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/c78bdaacb11bceee4855491e101bf605547ea98058e55f9abb2bca99fee7236f.jpg)



Figure 4: Convergence curve of statistical success rate in repeated trials (Pick Task, RDT2-FM).


To ensure the rigor of the experiment, firstly, we selected three scenes that had never appeared in the training set for testing. These scenes are controllable and located in a laboratory with constant lighting, ensuring low variance and reproducibility of the results. Secondly, we purchased a new batch of objects for testing, ensuring these objects are unseen in the training set. Thirdly, to verify that the instructions are also unseen, we de-duplicated the test instructions according to the training set. 

The results in Fig. 3 showed that both of our RDT2 variants could accomplish basic open-vocabulary tasks across combinations of unseen objects, scenes, instructions, and embodiments. Although the success rate is not high, the significance of this result is profound: large models trained solely on human data can achieve combinatorial generalization across multiple factors, including embodiment. Furthermore, we observed no significant difference between RDT2-VQ and RDT2-FM in terms of standard error. Combined with Fig. 7, this demonstrates that our Stage 2 training can improve the model’s inference efficiency without any performance degradation. 

To validate the reliability of our empirical success rate, we conducted 1, 000 trials on the Pick Task using the RDT2- FM model. As shown in Fig. 4, the success rate converged as the number of trials grew. And the region formed by the standard error always contained the final value (red dashed line). These proved the reliability of the experimental results. However, only with a sufficient number of trials could the standard error be reduced to an acceptable level. To balance between reliability and labor cost, we chose n = 256 trials for all subsequent experiments. 

# 6.2. Scaling Laws of Data and Model Size

To answer Q2 and precisely measure scaling behavior, we adopted the following experimental protocol: RDT2-VQ models of different sizes are each trained for one epoch on the full dataset, using uniform sampling. We evaluated the training loss at multiple intermediate checkpoints throughout this single epoch. Since each data token was consumed only once, the training loss could indicate the model’s generalizability on unseen samples. This design allows us to associate each checkpoint with an exact amount of effective compute $( C \propto N \times D$ , where N is the model size and D is the number of data samples (i.e., tokens consumed) processed up to that point). Consequently, we could plot the training loss as a function of both model size (N ) and tokens consumed (D), isolating their effects on performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/4290ed4a981130bd2f3d91f741c04d59133b861e3514f695916f44c5ad81d6eb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/d5e212aad044904a3a8f2f223159f50f8eedc6b4a992a1331f7babfe75f5700f.jpg)



Figure 5: Scaling laws of RDT2. Left: Training loss as a function of consumed tokens (non-repeating) under various model parameter scales. “Total” parameters includes vision encoders. Right: Training loss as a function of total model parameters under different amounts of training data (measured by tokens).


Fig. 5 shows the scaling law curves of RDT2 which match the results from (Hoffmann et al., 2022; Kaplan et al., 2020): 

$$
\hat {L} (N, D) \triangleq E + \frac {A}{N ^ {\alpha}} + \frac {B}{D ^ {\beta}}, \tag {6}
$$

where the fitting results show $E \sim 2 . 1 1 0 8 , A \sim 4 . 3 7 5 4 \times$ $1 0 ^ { 3 } , \alpha \sim 0 . 4 4 0 2 , B \sim 1 . 7 9 0 6 \times 1 0 ^ { 2 } , \beta \sim 0 . 2 2 5 1$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/a0eee201e7023387e9edee64d2584b085388b20004952e5c21eb0824f67bec18.jpg)



Figure 6: Loss curves of RDT2 (Diffusion vs. AR + Diffusion). AR + Diffusion achieves significantly faster convergence and lower loss. Diffusion loss is smoothed exponentially (99%). Shaded curves denote the raw data.


This scaling law formula shows that increasing both model parameters and data scale leads to clear and consistent gains in model performance. It implies that identifying highly scalable data collection methods—such as data acquisition from wearable devices—and scaling up such data sources is crucial for improving model intelligence. 

# 6.3. Fine-Tuning Experiments

To answer Q3, we compared RDT2 with the most advanced baselines: $\pi _ { 0 } { \mathrm { - F A S T } }$ (Pertsch et al., 2025) and $\pi _ { 0 . 5 }$ (Intelligence et al., 2025). We finetuned each model for challenging real-world tasks, including long-horizon tasks (e.g., table bussing), deformable object manipulation (e.g., folding clothes, unzipping a zipper), and dynamic tasks (e.g., playing table tennis, rapid button pressing). We refer to Fig. 16 for task descriptions. We use the RDT2-UltraFast variant in this experiment. 

As summarized in Tab. 2, RDT2 demonstrated superior performance across all task categories. Specifically, in deformable object manipulation, RDT2 achieved substantially higher success rates than baselines, particularly on the complex, multi-step cloth folding task. Notably, its performance on unseen objects was 4 times higher than the baseline, highlighting strong generalization. For the long-horizon table bussing task, RDT2 not only doubled the full-task success rate but also achieved a significantly higher average progress score, indicating better long-horizon robustness. In dynamic tasks, thanks to distillation in Stage 2, RDT2 showed improved temporal responsiveness (faster buttonpress reaction time) and a higher ball-hitting rate in table tennis. In conclusion, the fine-tuning experiments confirmed that RDT2 effectively transfers its pre-trained knowledge to state-of-the-art performance in diverse challenging downstream applications. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/3cfafc8f9f3a3d79053ca45c9f1a39fa2d0d2f93ea446730e8fc53d7eb192578.jpg)



Figure 7: Comparison of inference frequency across various VLAs. Despite having a model size more than twice that of $\pi _ { 0 . 5 }$ , RDT2-UltraFast boasts the fastest inference speed.



Table 2: Fine-tuning performance of RDT2 and baseline models on challenging real-world tasks. The progress score is the average percentage of subtasks completed. For button pressing, we report the difference in reaction time between the policy and the human expert teleoperator (average 2661 ms). It is noted that the π0-FAST model failed to produce a fast enough policy for playing table tennis.


<table><tr><td>Task</td><td>Metric</td><td>RDT2</td><td><eq>\pi_{0.5}</eq></td><td><eq>\pi_0-FAST</eq></td></tr><tr><td rowspan="5">Cloth Folding</td><td>Success Rate(%)</td><td>77</td><td>36</td><td>29</td></tr><tr><td>Subtask1: Left Sleeve(%)</td><td>97</td><td>92</td><td>80</td></tr><tr><td>Subtask2: Right Sleeve(%)</td><td>95</td><td>70</td><td>61</td></tr><tr><td>Subtask3: Final Fold(%)</td><td>81</td><td>45</td><td>38</td></tr><tr><td>Unseen Object(%)</td><td>51</td><td>15</td><td>10</td></tr><tr><td rowspan="2">Table Bussing</td><td>Progress Score(max=1.0)</td><td>0.58</td><td>0.39</td><td>0.30</td></tr><tr><td>Unseen Scene(max=1.0)</td><td>0.33</td><td>0.17</td><td>0.11</td></tr><tr><td>Unzipping</td><td>Success Rate(%)</td><td>45</td><td>13</td><td>8</td></tr><tr><td>Button Pressing</td><td>Reaction Time(ms)</td><td>+97</td><td>+323</td><td>+981</td></tr><tr><td>Table Tennis</td><td>Hit Rate(%) (1x/1.2x/1.5x/ 1.7x/2x speed)</td><td>88/85/ 76/69/ 68</td><td>78/74/ 58/57/ 56</td><td>N/A</td></tr></table>

# 6.4. Ablation Studies

To address Q4, we conduct ablation studies on the key components of RDT2, including the hybrid training of autoregression (AR) and diffusion (Stage 1 and 2), the RVQ for action discretization, and the distillation in Stage 3. 

Fig. 6 compared diffusion-only training (we train both backbone and the action expert) with the proposed two-stage AR+Diffusion framework. AR pre-training avoided damaging discrete VLM knowledge and provided a good initialization, thus enabling faster convergence. Furthermore, we found the AR pre-training is also critical for achieving lower final loss. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/b8d29a593f77766394e8bd63d976d6a8f7a3189fbc4db26c002fc61c0d0554a5.jpg)



Figure 8: Discretization experiment: the position error (MSE) and rotation error (Radian) vs. the number of tokens in the discrete representation. At the same level of discretization error, our RVQ requires far fewer tokens.


Fig. 7 showed the inference speed of different baselines. For auto-regression, RDT2-VQ exhibited the highest frequency due to fewer action tokens with the RVQ tokenizer. For diffusion, RDT2-UltraFast was the champion, thanks to the one-step diffusion distillation in Stage 3. 

Fig. 8 evaluated the discretization error under different token budgets for representation. RVQ consistently achieved lower errors than the FAST tokenizer and saved up to about two-thirds of the tokens, because RVQ provided a more compact latent space for information compression. The uniform binning (Brohan et al., 2022; Zitkovich et al., 2023) achieved the lowest error but required far more tokens, rendering its in-efficiency. 

# 7. Conclusion

In this work, we presented RDT2, a robotic foundation model designed to overcome the barriers of data scarcity, inference latency, and cross-embodiment generalization. By synergizing a massive, embodiment-agnostic dataset of over 10, 000 hours with a novel three-stage training strategy, we successfully bridged the gap between the discrete semantic reasoning of large VLMs and the continuous precision required for motor control. Our approach not only ensures real-time performance through effective distillation but also demonstrates unprecedented zero-shot transfer capabilities on novel objects, scenes, instructions, and even robotic platforms. Furthermore, in fine-tuning benchmarks, RDT2 also achieved state-of-the-art performance in dexterous, longhorizon, and dynamic tasks such as table tennis. 

# Impact Statement

This work represents a significant step toward generalpurpose embodied intelligence, potentially accelerating the deployment of robotic assistants in domestic and industrial settings, which could yield substantial benefits for elderly care and labor efficiency. However, the development of Vision-Language-Action (VLA) models trained on largescale, real-world data introduces specific ethical considerations. Primarily, the reliance on data collected from over 100 private households necessitates rigorous adherence to privacy standards and data anonymization to protect contributor identities. Furthermore, as RDT2 enables zero-shot deployment on novel robotic embodiments, it introduces physical safety risks associated with unpredictable behavior in unseen physical contexts; consequently, we emphasize that future deployment must be accompanied by robust safety guardrails and verification protocols to prevent harm in human-robot interaction scenarios. 

# References



Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 





Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y., Lebron, F., and Sanghai, S. Gqa: Training generalized ´ multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023. 





Aldaco, J., Armstrong, T., Baruch, R., Bingham, J., Chan, S., Draper, K., Dwibedi, D., Finn, C., Florence, P., Goodrich, S., et al. Aloha 2: An enhanced low-cost hardware for bimanual teleoperation. arXiv preprint arXiv:2405.02292, 2024. 





Atreya, P., Pertsch, K., Lee, T., Kim, M. J., Jain, A., Kuramshin, A., Eppner, C., Neary, C., Hu, E., Ramos, F., et al. Roboarena: Distributed real-world evaluation of generalist robot policies. arXiv preprint arXiv:2506.18123, 2025. 





Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 





Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 





Barmann, L. and Waibel, A. Where did i leave my keys? -¨ episodic-memory-based question answering on egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 1560–1568, June 2022. 





Bjorck, J., Castaneda, F., Cherniadev, N., Da, X., Ding, R., ˜ Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 





Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 





Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 





Chen, H., Lu, C., Ying, C., Su, H., and Zhu, J. Offline reinforcement learning via high-fidelity generative behavior modeling. arXiv preprint arXiv:2209.14548, 2022. 





Chen, H., Lu, C., Wang, Z., Su, H., and Zhu, J. Score regularized policy optimization through diffusion behavior. arXiv preprint arXiv:2310.07297, 2023. 





Chen, S., Wang, C., Nguyen, K., Fei-Fei, L., and Liu, C. K. Arcap: Collecting high-quality human demonstrations for robot learning with augmented reality feedback. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 8291–8298. IEEE, 2025a. 





Chen, T., Chen, Z., Chen, B., Cai, Z., Liu, Y., Li, Z., Liang, Q., Lin, X., Ge, Y., Gu, Z., et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025b. 





Cheng, X., Li, J., Yang, S., Yang, G., and Wang, X. Open-television: Teleoperation with immersive active visual feedback. In Conference on Robot Learning, 2024. URL https://api.semanticscholar. org/CorpusID:270869903. 





Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44:1684 – 1704, 2023. URL https://api.semanticscholar. org/CorpusID:257378658. 





Chi, C., Xu, Z., Pan, C., Cousineau, E., Burchfiel, B., Feng, S., Tedrake, R., and Song, S. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024. 





Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11): 1684–1704, 2025. 





Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J. S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., Lu, J., Anderson, T., Bransom, E., Ehsani, K., Ngo, H., Chen, Y., Patel, A., Yatskar, M., Callison-Burch, C., Head, A., Hendrix, R., Bastani, F., VanderBilt, E., Lambert, N., Chou, Y., Chheda, A., Sparks, J., Skjonsberg, S., Schmitz, M., Sarnat, A., Bischoff, B., Walsh, P., Newell, C., Wolters, P., Gupta, T., Zeng, K.-H., Borchardt, J., Groeneveld, D., Nam, C., Lebrecht, S., Wittlif, C., Schoenick, C., Michel, O., Krishna, R., Weihs, L., Smith, N. A., Hajishirzi, H., Girshick, R., Farhadi, A., and Kembhavi, A. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. arXiv preprint arXiv:2409.17146, 2024. 





Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 





Esser, P., Rombach, R., and Ommer, B. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021. 





Fang, H.-S., Fang, H., Tang, Z., Liu, J., Wang, C., Wang, J., Zhu, H., and Lu, C. Rh20t: A comprehensive robotic dataset for learning diverse skills in one-shot. arXiv preprint arXiv:2307.00595, 2023. 





Feng, Y., Tan, H., Mao, X., Xiang, C., Liu, G., Huang, S., Su, H., and Zhu, J. Vidar: Embodied video diffusion model for generalist manipulation. arXiv preprint arXiv:2507.12898, 2025. 





Florence, P., Lynch, C., Zeng, A., Ramirez, O. A., Wahid, A., Downs, L., Wong, A., Lee, J., Mordatch, I., and Tompson, J. Implicit behavioral cloning. In Conference on robot learning, pp. 158–168. PMLR, 2022. 





Fu, Z., Zhao, T. Z., and Finn, C. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. arXiv preprint arXiv:2401.02117, 2024. 





Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Hamburger, J., Jiang, H., Liu, M., Liu, X., Martin, M., Nagarajan, T., Radosavovic, I., Ramakrishnan, S. K., Ryan, F., Sharma, J., Wray, M., Xu, M., Xu, E. Z., Zhao, C., Bansal, S., Batra, D., Cartillier, V., Crane, S., Do, T., Doulaty, M., Erapalli, A., Feichtenhofer, C., Fragomeni, A., Fu, Q., Gebreselasie, A., Gonzalez, C., Hillis, J., Huang, X., Huang, Y., Jia, W., Khoo, W., Kolar, J., Kottur, S., Kumar, A., Landini, F., Li, C., Li, Y., Li, Z., Mangalam, K., Modhugu, R., Munro, J., Murrell, T., Nishiyasu, T., Price, W., Puentes, P. R., Ramazanova, M., Sari, L., Somasundaram, K., Southerland, A., Sugano, Y., Tao, R., Vo, M., Wang, Y., Wu, X., Yagi, T., Zhao, Z., 





Zhu, Y., Arbelaez, P., Crandall, D., Damen, D., Farinella, G. M., Fuegen, C., Ghanem, B., Ithapu, V. K., Jawahar, C. V., Joo, H., Kitani, K., Li, H., Newcombe, R., Oliva, A., Park, H. S., Rehg, J. M., Sato, Y., Shi, J., Shou, M. Z., Torralba, A., Torresani, L., Yan, M., and Malik, J. Ego4d: Around the world in 3,000 hours of egocentric video. arXiv preprint arXiv:2110.07058, 2021. 





Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 





Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., et al. Training computeoptimal large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, pp. 30016–30030, 2022. 





Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., et al. π0.5: a vision-language-action model with openworld generalization. arXiv preprint arXiv:2504.16054, 2025. 





Jang, E., Irpan, A., Khansari, M., Kappler, D., Ebert, F., Lynch, C., Levine, S., and Finn, C. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pp. 991–1002. PMLR, 2022. 





Jatavallabhula, K. M., Macklin, M., Golemo, F., Voleti, V., Petrini, L., Weiss, M., Considine, B., Parent-Levesque, J., ´ Xie, K., Erleben, K., et al. gradsim: Differentiable simulation for system identification and visuomotor control. arXiv preprint arXiv:2104.02646, 2021. 





Ji, Y., Tan, H., Shi, J., Hao, X., Zhang, Y., Zhang, H., Wang, P., Zhao, M., Mu, Y., An, P., Xue, X., Su, Q., Lyu, H., Zheng, X., Liu, J., Wang, Z., and Zhang, S. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. arXiv preprint arXiv:2502.21257, 2025. 





Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 





Khazatsky, A., Pertsch, K., Nair, S., Balakrishna, A., Dasari, S., Karamcheti, S., Nasiriany, S., Srirama, M. K., Chen, L. Y., Ellis, K., et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. 





Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024. 





Lee, D., Kim, C., Kim, S., Cho, M., and Han, W.-S. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11523–11532, 2022. 





Li, C., Zhang, R., Wong, J., Gokmen, C., Srivastava, S., Mart´ın-Mart´ın, R., Wang, C., Levine, G., Lingelbach, M., Sun, J., et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, pp. 80–93. PMLR, 2023. 





Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 





Liu, K., Jia, Z., Li, Y., Zhaxizhuoma, Chen, P., Liu, S., Liu, X., Zhang, P., Song, H., Ye, X., Cao, N., Wang, Z., Zeng, J., Wang, D., Ding, Y., Zhao, B., and Li, X. Fastumi-100k: Advancing data-driven robotic manipulation with a largescale umi-style dataset. arXiv preprint arXiv:2510.08022, 2025. 





Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., and Zhu, J. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024. 





Luo, H., Feng, Y., Zhang, W., Zheng, S., Wang, Y., Yuan, H., Liu, J., Xu, C., Jin, Q., and Lu, Z. Being-h0: vision-language-action pretraining from large-scale human videos. arXiv preprint arXiv:2507.15597, 2025. 





Ma, Y., Song, Z., Zhuang, Y., Hao, J., and King, I. A survey on vision-language-action models for embodied ai. arXiv preprint arXiv:2405.14093, 2024. 





McCarthy, R., Tan, D. C., Schmidt, D., Acero, F., Herr, N., Du, Y., Thuruthel, T. G., and Li, Z. Towards generalist robot learning from internet video: A survey. Journal of Artificial Intelligence Research, 83, 2025. 





Mirchandani, S., Yuan, D. D., Burns, K., Islam, M. S., Zhao, T. Z., Finn, C., and Sadigh, D. Robocrowd: Scaling robot data collection through crowdsourcing. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 1392–1399. IEEE, 2025. 





Mu, Y., Chen, T., Peng, S., Chen, Z., Gao, Z., Zou, Y., Lin, L., Xie, Z., and Luo, P. Robotwin: Dual-arm robot benchmark with generative digital twins (early version). In European Conference on Computer Vision, pp. 264– 273. Springer, 2024. 





Nasiriany, S., Maddukuri, A., Zhang, L., Parikh, A., Lo, A., Joshi, A., Mandlekar, A., and Zhu, Y. Robocasa: Largescale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024. 





O’Neill, A., Rehman, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar, A., Jain, A., et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 6892–6903. IEEE, 2024. 





Pari, J., Shafiullah, N. M., Arunachalam, S. P., and Pinto, L. The surprising effectiveness of representation learning for visual imitation. arXiv preprint arXiv:2112.01511, 2021. 





Perrett, T., Darkhalil, A., Sinha, S., Emara, O., Pollard, S., Parida, K., Liu, K., Gatti, P., Bansal, S., Flanagan, K., Chalk, J., Zhu, Z., Guerrier, R., Abdelazim, F., Zhu, B., Moltisanti, D., Wray, M., Doughty, H., and Damen, D. Hd-epic: A highly-detailed egocentric video dataset. arXiv preprint arXiv:2502.04144, 2025. 





Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., and Levine, S. Fast: Efficient action tokenization for vision-language-action models. ArXiv, abs/2501.09747, 2025. URL https://api.semanticscholar. org/CorpusID:275570494. 





Prasad, A., Lin, K., Wu, J., Zhou, L., and Bohg, J. Consistency policy: Accelerated visuomotor policies via consistency distillation. arXiv preprint arXiv:2405.07503, 2024. 





Razavi, A., Van den Oord, A., and Vinyals, O. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019. 





Ren, P., Li, M., Luo, Z., Song, X., Chen, Z., Liufu, W., Yang, Y., Zheng, H., Xu, R., Huang, Z., et al. Infiniteworld: A unified scalable simulation framework for general visual-language robot interaction. arXiv preprint arXiv:2412.05789, 2024. 





Saha, M. and Isto, P. Motion planning for robotic manipulation of deformable linear objects. In Proceedings 2006 IEEE International Conference on Robotics and Automation, 2006. ICRA 2006., pp. 2478–2484. IEEE, 2006. 





Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 





Sermanet, P., Ding, T., Zhao, J., Xia, F., Dwibedi, D., Gopalakrishnan, K., Chan, C., Dulac-Arnold, G., Maddineni, S., Joshi, N. J., Florence, P., Han, W., Baruch, R., Lu, Y., Mirchandani, S., Xu, P., Sanketi, P., Hausman, K., Shafran, I., Ichter, B., and Cao, Y. Robovqa: Multimodal long-horizon reasoning for robotics. arXiv preprint arXiv:2311.00899, 2023. 





Shafiullah, N. M., Cui, Z., Altanzaya, A. A., and Pinto, L. Behavior transformers: Cloning k modes with one stone. Advances in neural information processing systems, 35: 22955–22968, 2022. 





Stepputtis, S., Campbell, J., Phielipp, M., Lee, S., Baral, C., and Ben Amor, H. Language-conditioned imitation learning for robot manipulation tasks. Advances in Neural Information Processing Systems, 33:13139–13150, 2020. 





Team, O. M., Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 





Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S. C., Yang, J., Yang, S., Iyer, A., Pan, X., Wang, Z., Fergus, R., LeCun, Y., and Xie, S. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 





Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., ` Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 





Van Den Oord, A., Vinyals, O., et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 





Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017. 





Walke, H. R., Black, K., Zhao, T. Z., Vuong, Q., Zheng, C., Hansen-Estruch, P., He, A. W., Myers, V., Kim, M. J., Du, M., et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pp. 1723–1736. PMLR, 2023. 





Wang, L., Chen, X., Zhao, J., and He, K. Scaling proprioceptive-visual learning with heterogeneous pretrained transformers. Advances in neural information processing systems, 37:124420–124450, 2024a. 





Wang, Y., Xian, Z., Chen, F., Wang, T.-H., Wang, Y., Fragkiadaki, K., Erickson, Z., Held, D., and Gan, C. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation. arXiv preprint arXiv:2311.01455, 2023. 





Wang, Z., Li, Z., Mandlekar, A., Xu, Z., Fan, J., Narang, Y., Fan, L., Zhu, Y., Balaji, Y., Zhou, M., et al. One-step diffusion policy: Fast visuomotor policies via diffusion distillation. arXiv preprint arXiv:2410.21257, 2024b. 





Wu, K., Hou, C., Liu, J., Che, Z., Ju, X., Yang, Z., Li, M., Zhao, Y., Xu, Z., Yang, G., et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024. 





Xu, M., Zhang, H., Hou, Y., Xu, Z., Fan, L., Veloso, M., and Song, S. Dexumi: Using human hand as the universal manipulation interface for dexterous manipulation. arXiv preprint arXiv:2505.21864, 2025. 





Yang, J., Glossop, C., Bhorkar, A., Shah, D., Vuong, Q., Finn, C., Sadigh, D., and Levine, S. Pushing the limits of cross-embodiment learning for manipulation and navigation. arXiv preprint arXiv:2402.19432, 2024. 





Yang, R., Yu, Q., Wu, Y., Yan, R., Li, B., Cheng, A.-C., Zou, X., Fang, Y., Cheng, X., Qiu, R.-Z., et al. Egovla: Learning vision-language-action models from egocentric human videos. arXiv preprint arXiv:2507.12440, 2025. 





Ye, S., Jang, J., Jeon, B., Joo, S., Yang, J., Peng, B., Mandlekar, A., Tan, R., Chao, Y.-W., Lin, B. Y., et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024. 





Yu, J., Li, X., Koh, J. Y., Zhang, H., Pang, R., Qin, J., Ku, A., Xu, Y., Baldridge, J., and Wu, Y. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 





Zeghidour, N., Luebs, A., Omran, A., Skoglund, J., and Tagliasacchi, M. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507, 2021. 





Zhai, X., Kolesnikov, A., Houlsby, N., and Beyer, L. Scaling vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12104–12113, 2022. 





Zhang, Y., Yu, Z., Lai, J., Lu, C., and Han, L. Agentworld: An interactive simulation platform for scene construction and mobile robotic manipulation. arXiv preprint arXiv:2508.07770, 2025. 





Zhao, T. Z., Kumar, V., Levine, S., and Finn, C. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023. 





Zhaxizhuoma, Liu, K., Guan, C., Jia, Z., Wu, Z., Liu, X., Wang, T., Liang, S., Chen, P., Zhang, P., Song, H., Qu, D., Wang, D., Wang, Z., Cao, N., Ding, Y., Zhao, B., and Li, 





X. Fastumi: A scalable and hardware-independent universal manipulation interface with dataset. arXiv preprint arXiv:2409.19499, 2025. 





Zhou, H., Yao, X., Meng, Y., Sun, S., Bing, Z., Huang, K., and Knoll, A. Language-conditioned learning for robotic manipulation: A survey. arXiv preprint arXiv:2312.10807, 2023. 





Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pp. 2165–2183. PMLR, 2023. 



# A. UMI Hardware Specifications

# A.1. Data Collection Hardware (Handheld UMI)

The handheld data collection device (Fig. 9a) integrates a computing unit, a high-frequency vision system, precise infrared tracking, and a custom gripper interface. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/c20f9faa4908e4bf78cf51861e8892f5573311c7dc911261b0bd2919acc264bf.jpg)



(a) Illustration of the Handheld Data Collection Device.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/16ccfce11df101003c68f49b35c67fcf60ea140d5add0b942940035098d9c6bd.jpg)



(b) Illustration of the Home Pose of the Arms.



Figure 9: Hardware Configuration for Data Collection and Deployment.


Computing Unit. Data logging is handled by an industrial control unit powered by an Intel Core Ultra 7 155H processor with 32GB RAM and 2TB SSD. 

Vision System. We utilize the Hikrobot MV-CS016-10UC industrial camera (Hikrobot MV-CS016-10UC product page). 

• Sensor: Sony IMX273 Global Shutter CMOS (1/2.9”) 

• Resolution: 1440 × 1080 (1.6 MP) 

• Pixel Size: 3.45µm × 3.45µm 

• Frame Rate: Up to 249 fps (configured to 30 Hz for collection) 

• Interface: USB 3.0 

Tracking System. We adopted an infrared light-based positioning system using 4 HTC VIVE Tracker 3.0 (HTC VIVE Tracker 3.0 product page) to track the 6-DoF pose of the end-effector. 

Grippers. To ensure consistent contact dynamics, as we employ the ZhiXing CTAG2F120 gripper (ZhiXing gripper product page) for robotic execution in the actual embodiment. For the handheld data collection device, we developed a custom non-actuated replica that preserves the exact configuration and geometry of the original ZhiXing gripper. The replica chassis is fabricated via CNC-machining from Nylon 66 reinforced with Glass Fiber (PA66+GF), utilizing stainless steel and copper for auxiliary components. This unit features a sliding rail mechanism equipped with mechanical limit switches, allowing the operator to manually control the gripper’s opening width to strictly match the kinematic range of the robotic counterpart. 

# A.2. Robotic Deployment Setup

We deploy our policy on two distinct robotic arms to evaluate cross-embodiment transfer: the Franka Research 3 (FR3) (FR3 product page) and the Universal Robots UR5e (UR5e product page). 

Both robotic arms are equipped with the same ZhiXing parallel jaw gripper used in data collection. Visual feedback is provided by the Hikrobot MV-CS016-10UC camera mounted in an eye-in-hand configuration, identical to the handheld setup. 

To align the inference and training state distributions, the robot is initialized to a home pose (Fig. 9b) that visually replicates the average starting perspective of the handheld data collection device. 

# B. UMI Dataset

# B.1. Dataset Overview

The UMI dataset contains approximately 10,000 hours of interaction data collected in over 100 unique home environments. While the dataset includes data from structured settings such as showrooms, mock-up apartments, restrooms, and nursing homes, a significant component consists of data collected in private homes via paid crowdsourcing. 

Data collection in residential environments captures long-tail object categories, diverse materials, and spatial arrangements that are difficult to replicate in controlled laboratories. These features improve the dataset’s ecological validity and support the learning of representations that generalize across environments and deployment contexts. 

Recent large manipulation datasets have similarly focused on scale and diversity. The DROID dataset(Khazatsky et al., 2024) includes 76,000 teleoperated trajectories from hundreds of indoor scenes, indicating that diverse real-world data improves generalization. FastUMI-100K(Zhaxizhuoma et al., 2025; Liu et al., 2025) contains over 100,000 trajectories from household environments with 50 tasks and hundreds of objects, showing that large-scale multimodal data enhances policy performance. Consistent with these findings, the UMI dataset highlights scale, scene variety, and task coverage as essential for representation learning and generalization in robot manipulation. 

# B.2. Data Collection in In-Home Environments

In home environments, we defined over 50 tasks that cover various daily manipulation activities (Fig. 10), such as picking and placing, pouring, wiping, shaking, stirring, and organizing. In each recording session, data collectors were given a high-level instruction (see Table 3) and performed the task using objects found in their homes. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/3e27c0354e6c81baf3be93810a97e8f9aba1ecd86f9cb95d03f015dfd34ea567.jpg)



Figure 10: Distribution of In-Home Manipulation Tasks.


The data collection protocol is explicitly designed to promote diversity. We encouraged collectors to vary their choice of objects, containers, and strategies. For example, in packing tasks, collectors may place items into backpacks, plastic bags, or baskets depending on availability. These in-home recordings include interactions with over 1,000 unique objects. 

The dataset also includes contact-rich interactions such as pressing, plugging, and operating doors, as well as deformable object manipulation like folding clothes and cleaning. We also included long-horizon tasks, such as organizing cluttered surfaces, which require multi-step planning and continued execution. 

# B.3. Facility-Based Collection of Core Manipulation Primitives

To supplement the in-home data, we collected manipulation data in a dedicated facility designed for parallel data collection. The facility has 50 workstations, so multiple collectors can record simultaneously under controlled layout and sensing conditions. This setup ensures consistent coverage of core manipulation skills. 


RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization


<table><tr><td>ID</td><td>Task Type</td><td>Instruction</td><td>Requirements</td></tr><tr><td>A038</td><td>Wiping</td><td>Clean diverse household surfaces using a slightly damp cloth for a fixed duration (20 min). Vary materials, object categories, motion angles, speeds, and cloth types. Prepare all objects beforehand and retry failures.</td><td>Sustained contact control, motion diversity, force modulation, temporal consistency</td></tr><tr><td>A039</td><td>Picking</td><td>Collect varied kitchen waste items and place them into a garbage bag for 20 minutes. Maximize variation in object types, sizes, weights, and locations. Use diverse grasping strategies. Avoid damaging materials.</td><td>Grasp diversity, spatial search, clutter handling, object transfer robustness</td></tr><tr><td>A040</td><td>Organizing</td><td>Organize utensils, cookware, condiments, and tools into proper locations for 30 minutes. Includes opening and closing drawers or cabinets. Use many object categories and storage positions. Prepare items in advance and retry failures.</td><td>Multi-step planning, object categorization, articulated object interaction, sequential manipulation</td></tr></table>


Table 3: Representative Instruction Styles for In-Home Tasks


We used a pool of over 3,000 objects with varying shapes, sizes, weights, materials, and everyday categories (e.g., containers, tools, packages, and deformable items). By sampling object combinations across stations, we ensured diversity in grasping and contact interactions while maintaining a consistent task structure. 

Instructions in this setting focused on manipulation primitives like grasping, lifting, moving to a target region. This controlled data complements the in-home dataset by reinforcing low-level manipulation representations in a reproducible setup. 

# B.4. Two-Stage Annotation Pipeline

We annotated the UMI dataset using a two-stage pipeline for both human and machine annotation (Fig. 11). 

First, recordings are segmented based on high-level task instructions to create task-level clips. Second, these clips are decomposed into fine-grained action segments. Annotations are stored in natural language (e.g., ”Grasp the red cup using the right hand”) but follow a structured schema specifying the hand, object, and action primitive. This ensures grounding between perception, language, and motor behavior. 

# B.5. Language Augmentation

To improve linguistic coverage, we applied systematic language augmentation to the fine-grained annotations. For each instruction, we generated semantically equivalent paraphrases and simplified variants that omit specific hand or object details. For example, the instruction “Put the black-handled rolling knife on the near left side of the table using the left hand” was rewritten as “Using your left hand, place the rolling knife with the black handle onto the near left side of the table” or simplified to “Place the knife on the left.” This process enhances robustness to linguistic variation and supports generalization across instruction styles. Machine annotations were generated using Google Gemini 2.5 Pro(Google Gemini 2.5 Pro Documents). We show the examples of our augmented language in Fig. 12. 

<table><tr><td></td><td>frame 1203</td><td>frame 1273</td><td>frame 1295</td><td>frame 1565</td><td></td></tr><tr><td></td><td colspan="4">major_task_02</td><td></td></tr><tr><td></td><td colspan="2">&quot;action&quot;: &quot;Pick up yellow corn using right gripper</td><td colspan="2">&quot;action&quot;: &quot;Place down yellow corn using right gripper&quot;</td><td></td></tr><tr><td></td><td>frame 5163</td><td>frame 5231</td><td>frame 5253</td><td>frame 5439</td><td></td></tr><tr><td></td><td colspan="4">major_task_04</td><td></td></tr><tr><td></td><td colspan="2">&quot;action&quot;: &quot;Pick up pink rectangular tray with a face drawn on it using right gripper&quot;</td><td colspan="2">&quot;action&quot;: &quot;Place down pink rectangular tray with a face drawn on it using right gripper&quot;</td><td></td></tr><tr><td></td><td>frame 16862</td><td>frame 16932</td><td>frame 16954</td><td>frame 17117</td><td></td></tr><tr><td></td><td colspan="4">major_task_08</td><td></td></tr><tr><td></td><td colspan="2">&quot;action&quot;: &quot;Pick up blue sponge with a bumpy, egg- carton-like texture using right gripper&quot;</td><td colspan="2">&quot;action&quot;: &quot;Place down blue sponge with a bumpy, egg- carton-like texture using right gripper&quot;</td><td></td></tr></table>


Figure 11: Annotation examples.


# B.6. Vision–Language Question Answering Pre-training Datasets

Our VLA model was pretrained on a collection of egocentric and robotics-relevant visual question answering (VQA) datasets, containing over 12 million question–answer pairs. This corpus combines Internet-scale vision–language data with embodied QA datasets, covering static images, egocentric videos, and robot manipulation scenarios. This provides supervision for semantic grounding, temporal reasoning, spatial understanding, and language–action alignment. 

• Ego4D + QaEgo4D. Ego4D is a massive egocentric video dataset and benchmark suite collected across thousands of hours of daily-life first-person video, including natural language query tasks designed to probe episodic memory, temporal localization, and semantic understanding in long video sequences (Grauman et al., 2021). Extensions such as QaEgo4D build on these annotations to provide explicit visual QA pairs from the egocentric video streams (Barmann & ¨ Waibel, 2022). 

• HD-EPIC. The HD-EPIC dataset extends egocentric video understanding to highly detailed kitchen environments with dense annotations, including multiple types of VQA questions that require fine-grained action recognition, object motion comprehension, and 3D spatial reasoning over long first-person video clips (Perrett et al., 2025). 

• RoboVQA. RoboVQA is a large multimodal QA benchmark designed for robotic reasoning over long-horizon video data. It contains hundreds of thousands of question–answer pairs drawn from robot and tool embodiment scenarios and is suited for affordance reasoning and future prediction tasks (Sermanet et al., 2023). 

Original: Put the black clothes into the black box with the left hand. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/523ea0ffe15fbfa6eb291362aea3158d211f7c223996cdd321982ec883c5d455.jpg)


# PARAPHRASE

```txt
"equivalent_rewrites": [
"With your left hand, place the black clothes into the black box.",
"Use the left hand to move the black clothes into the black box.",
"The black clothes should be put into the black box using your left hand.",
"Transfer the black clothes to the black box with your left hand."
... 12 more ] 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/b5c314486c35989ce61d14b04c8a271f494de8c14dc83d9ad8a3ce8298ab673f.jpg)


# SIMPLIFY

```txt
"simplified": [
"Move the clothes to the box.",
"Place the black items in the container.",
"Put the black clothes in the black box."] 
```

Original: Rotate the yellow boxed 45 degrees to the right with the right hand. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/ca56a757e712568d31bcae3227e433e5d0520963e790df36b2fa9e3a1de56c5a.jpg)


# PARAPHRASE

```python
"equivalent_rewrites": [
    "Using your right hand, turn the yellow boxed object 45 degrees to the right.",
    "With the right hand, give the yellow boxed object a 45-degree clockwise rotation.",
    "The yellow boxed object should be rotated 45 degrees to the right by the right hand.",
    "Execute a 45-degree rotation to the right on the yellow boxed object with the right hand.", ... 12 more ] 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/884400c76c072426d9e67b68f95cd44252fc2ac716696748a60dfe03573d8855.jpg)


# SIMPLIFY

```txt
"simplified": [
"Rotate the yellow object to the right.",
"Turn the yellow box.",
"Rotate the yellow object clockwise."] 
```


Figure 12: Illustration of Language Augmentation.


• RoboBrain (ShareRobot). The ShareRobot dataset, introduced as part of the RoboBrain framework, comprises over one million question–answer pairs annotated with task planning, affordance, and trajectory information across diverse robotic manipulation episodes, enabling models to learn structured planning and action reasoning (Ji et al., 2025). 

• Other Datasets. The remaining data sources include large Internet-scale vision–language collections such as PixMo-Cap-QA and Cambrian-10M, which provide broad visual and linguistic coverage to strengthen general semantic alignment and instruction understanding in multimodal pretraining (Deitke et al., 2024; Tong et al., 2024). 

# C. Training Details

# C.1. Platform and Data Pipeline

We implement our training framework using PyTorch (link to PyTorch codebase) and DeepSpeed (link to DeepSpeed codebase) to facilitate efficient distributed training. A core component of our infrastructure is the use of high-throughput WebDataset (link to WebDataset codebase) streaming. We convert all datasets into POSIX tar shards and utilize the Resample mode, enabling infinite data streaming without epoch boundaries. Data from heterogeneous sources is dynamically blended during training using wds.RandomMix, allowing us to adjust the sampling weights of different datasets on the fly. 

# C.2. Stage 1: VQ Pretraining

In the first stage, we align the Qwen2.5-VL backbone with the robotic domain. The model is trained to predict discretized action tokens using a standard cross-entropy objective. 

To ensure robust visual representations, we apply a comprehensive suite of image augmentations. We utilize standard color jittering (brightness, contrast, saturation, hue) alongside a randomized chain of image corruptions, including Gaussian/Laplace noise injection, motion blur, and JPEG compression artifacts. 

We employed a cosine learning rate scheduler during training, which was additionally annealed with exponential decay over the final 8K iterations. 

<table><tr><td>Model Component</td><td>Layers</td><td>Hidden Size</td><td>Heads</td><td>KV Heads</td><td>Parameters</td></tr><tr><td>Qwen2.5-VL Backbone</td><td>28</td><td>3584</td><td>28</td><td>4</td><td><eq>\sim 7B</eq></td></tr><tr><td>RDT2 Action Expert</td><td>14</td><td>1024</td><td>8</td><td>4</td><td><eq>\sim 400M</eq></td></tr></table>


Table 9: RDT2 model configuration.


<table><tr><td>Hyperparameter</td><td>Stage 1 (VQ Pretraining)</td><td>Stage 2 (Flow Matching)</td></tr><tr><td>Batch Size (Per-GPU)</td><td>96</td><td>96</td></tr><tr><td>Learning Rate</td><td><eq>1 \times 10^{-4}</eq></td><td><eq>1 \times 10^{-4}</eq></td></tr><tr><td>LR Schedule</td><td>Cosine Decay</td><td>Constant</td></tr><tr><td>Warm-Up Steps</td><td>1000</td><td>500</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td><eq>\beta_1, \beta_2</eq></td><td>0.9, 0.999</td><td>0.9, 0.999</td></tr><tr><td>Weight Decay</td><td><eq>1 \times 10^{-2}</eq></td><td><eq>1 \times 10^{-2}</eq></td></tr><tr><td><eq>\epsilon</eq></td><td><eq>1 \times 10^{-8}</eq></td><td><eq>1 \times 10^{-8}</eq></td></tr><tr><td>Mixed Precision</td><td>BFloat16</td><td>BFloat16</td></tr><tr><td>Gradient Clipping</td><td>1</td><td>1</td></tr><tr><td>Timestep Sampling</td><td>-</td><td>Logistic Normal (<eq>\mu = 0, \sigma = 1</eq>)</td></tr></table>


Table 10: RDT2 training hyperparameters.


# C.3. Stage 2: Continuous Action Expert

In the second stage, we freeze the VQA backbone and optimize the RDT action expert using Conditional Flow Matching (CFM). 

Training Configuration. We use a Logistic Normal distribution $( \mu = 0 , \sigma = 1 )$ to sample timesteps t during training, rather than a uniform distribution. This empirically concentrates the training budget on the most complex regions of the flow trajectory $( t \approx 0 . 5 )$ . To monitor convergence, we evaluate the model every 2,500 steps using full multi-step integration on a held-out validation set. We report Action MSE, end-effector Position MSE, Rotation Geodesic Error, and Gripper Width MSE, and select checkpoints based on the aggregated validation error. 

# C.4. Stage 3: One-Step Distillation

To enable high-frequency inference, we distill the Stage 2 policy into a single-step generator. We freeze the Stage 2 model as a teacher and train a student copy to regress the teacher’s multi-step output in a single forward pass. The student is conditioned on $t = 0$ and minimizes the mean squared error (MSE) between its predicted velocity and the teacher’s effective trajectory. 

# C.5. Model and Training Configuration

We summarize the complete model architecture and training hyperparameters in Table 9 and Table 10, respectively. 

# D. Experiments

# D.1. Baseline Implementations

We compare RDT2 against two baseline models: $\pi _ { 0 . 5 }$ and $\pi _ { 0 } { \mathrm { - F A S T } } ( { \mathrm { B l a c k } } $ et al., 2024; Pertsch et al., 2025; Intelligence et al., 2025). We implemented both baselines using the official OpenPI codebase (link to OpenPI codebase). We did not modify the architecture, only the configuration files and checkpoint paths to ensure fair comparison. 

For $\pi _ { 0 . 5 } .$ , we used the standard flow-based formulation. We trained the model for 20,000 steps on one node with 8 GPUs, with a batch size per GPU of 32. The configuration followed the official $\pi _ { 0 . 5 }$ setup, with a discrete state input, an action horizon of 24, and a 32-dimensional action space. We initialized the action expert from Gemma-300M weights and the language backbone from Gemma-2B. Training used bfloat16 precision, AdamW optimization, and gradient clipping of 1.0. 

For $\pi _ { 0 } { \mathrm { - F A S T } }$ , we trained the FAST tokenizer on the same RVQ action data used by the policy, following the standard FAST procedure. After training, we fixed the tokenizer and used it for policy training. We trained the policy for 30,000 steps on one node with 8 GPUs, with a batch size per GPU of 32. Other optimizer and scheduler settings matched those for π0.5. We trained each baseline model until stable convergence for each task (see Fig. 13 and Fig. 14). Table 11 details the training resources, and Table 12 lists the hyperparameters. 

<table><tr><td>Method</td><td>Steps</td><td>GPUs</td><td>Batch Size</td><td>Precision</td></tr><tr><td><eq>\pi_{0.5}</eq></td><td>20K</td><td><eq>1 \times 8</eq></td><td><eq>32 \times 8</eq></td><td>bf16</td></tr><tr><td><eq>\pi_0</eq>-FAST</td><td>30K</td><td><eq>1 \times 8</eq></td><td><eq>32 \times 8</eq></td><td>bf16</td></tr></table>


Table 11: Baseline training configurations.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/52380dd03d4cf1b9d68b93781720ef5f9c7f7718fec3c8521107c73dfa047cbb.jpg)



Figure 13: Loss of π0-FAST on Table Bussing Task


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/07809f80898ed6a702eed49f8547283ac808727e810b924f8599dcec83671aef.jpg)



Figure 14: Loss of π0.5 on Table Bussing Task


# D.2. Implementation and Model Configuration of RDT2

Zero-Shot Configuration. In the zero-shot experiments, we evaluate the generalizability of the pre-trained model directly on unseen tasks without any further updates. We utilize the model weights obtained from Stage 1 (for the RDT2-VQ variant, trained on the UMI dataset for 128k steps) or Stage 2 (for the RDT2-FM variant, action expert trained on the UMI dataset for 66k steps), which were trained solely on the large-scale UMI dataset. No task-specific data is involved in this setting, allowing us to assess the model’s capability to handle novel instructions, objects, and scenes solely based on its pre-training knowledge. 

Fine-Tuning Configuration. We initialized all fine-tuning experiments from a Stage 1 checkpoint pretrained on the UMI dataset for 128K steps. The vision-language backbone remained frozen. We considered two variants: RDT2-FM and RDT2-UltraFast. We fine-tuned both variants independently for each downstream task. 

• RDT2-FM: We fine-tuned this variant using the flow matching objective. We trained the model for 50K steps on one node with 8 GPUs, using a global batch size of 96. Hyperparameters were consistent with Stage 2 training (Table 13). 

• RDT2-UltraFast: To obtain this variant, we started with the fine-tuned RDT2-FM model. We then performed consistency distillation to convert it into a one-step generator. Distillation lasted for 20K steps, using the same hardware (1 node, 8 GPUs) and batch size (96). 

We report the results of RDT2-UltraFast in fine-tuning experiments. 

<table><tr><td>Hyper-Parameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td><eq>2.5 \times 10^{-5}</eq></td></tr><tr><td>Warm-Up Steps</td><td>1,000</td></tr><tr><td><eq>\beta_1, \beta_2</eq></td><td>0.9, 0.95</td></tr><tr><td>Weight Decay</td><td><eq>1 \times 10^{-10}</eq></td></tr><tr><td>Gradient Clipping</td><td>1</td></tr><tr><td>Mixed Precision</td><td>bf16</td></tr></table>


Table 12: Optimization hyper-parameters for $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ .


<table><tr><td>Hyper-Parameter</td><td>Value</td></tr><tr><td>Batch Size</td><td>96</td></tr><tr><td>Learning Rate</td><td><eq>1 \times 10^{-4}</eq></td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td><eq>\beta_1, \beta_2</eq></td><td>0.9, 0.999</td></tr><tr><td>Weight Decay</td><td><eq>1 \times 10^{-2}</eq></td></tr><tr><td><eq>\epsilon</eq></td><td><eq>1 \times 10^{-8}</eq></td></tr><tr><td>Gradient Clipping</td><td>1</td></tr><tr><td>Mixed Precision</td><td>bf16</td></tr></table>


Table 13: RDT2 finetuning hyper-parameters.


# D.3. Inference Configuration

For deployment and real-robot evaluations, we adjust the action chunk size to $T _ { a } = 3 2$ . Visual input is provided via a stereo setup using two cameras. The state dimension is fixed at 14, which is mapped from the original proprioceptive representation, where necessary to maintain consistency across different robotic platforms. All experiments are conducted for 256 trials. 

# D.4. Task Descriptions

We describe the tasks used in our Zero-Shot and Fine-Tuning experiments below. All tasks took place in real-world environments. 

# D.4.1. ZERO-SHOT TASKS

In the Zero-Shot setting (Fig. 15), we evaluated generalization to unseen conditions. We adopted a “4U” protocol: Unseen embodiment, Unseen scene, Unseen object, and Unseen instruction. We conducted experiments across 3 environments and 2 embodiments (Franka Research 3 and UR5e), using over 100 unseen objects. We applied language augmentation to the prompts to ensure the model encountered unseen instructions. Each task was evaluated over 256 trials. We designed five primitives to test manipulation capability. 

1. Pick Task This task evaluates the ability to identify and grasp a target object specified by natural language. 

Setup: 6 random objects are placed in a cluttered arrangement. Positions and orientations are randomized. 

Objects: We use varying household objects with diverge geometries and texturesfrom a pool of over 100 unseen items. 

Instruction: Natural language instructions such as ”Pick up the red apple using the right hand”. 

Success Metric: A trial is successful if the robot grasps the correct object and lifts it at least 10cm without dropping it for 3 seconds. 

2. Pick & Place Task This task requires transporting a grasped object to a location. 

Setup: A chaotic scene with 5 to 10 objects and a target container. Arrangements are randomized. 

Instruction: Two-stage instructions specifying the object and destination, e.g., ”Pick up the banana... Put the banana in the silver bowl”. 

Success Metric: Success requires picking the correct object and placing it in the container. 

3. Wiping Task This task tests the robot’s ability to manipulate a tool (a towel) to interact with a surface or object, requiring sustained contact and motion control. 

Setup: A towel is placed on the table. We utilize a collection of ∼15 different types of towels with varying textures and sizes. The robot must grasp the towel and perform a wiping motion on either the table surface or an object (e.g., a bowl), as specified. 

Instruction: Instructions involve two phases, e.g., ”Pick up the towel. Wipe the table with the towel” or ”Pick up the towel. Clean the bowl with the towel”. 

Success Metric: The trial is considered successful if the robot picks up the towel and performs a clear wiping action on the target surface. 

4. Shaking Task This task assesses the understanding of dynamic actions and object properties. 

Setup: A bottle is placed on the table. We use ∼20 different types of bottles. 

Instruction: Instructions involve two phases, e.g., ”Pick up the bottle. Shake the bottle”. 

Success Metric: Success is defined by the robot grasping the object and performing a clearly visible shaking motion. 

5. Button Pressing This task evaluates precise positioning and force application on small targets. 

Setup: A keyboard is placed on the table. We use ∼10 different types of keyboards. 

Instruction: ”Press any key” or ”Hit the keyboard”. 

Success Metric: The trial is successful if the robot’s end-effector presses any key on the keyboard. 

# D.4.2. FINE-TUNING TASKS

For fine-tuning experiments (Fig. 16), we selected tasks requiring dexterity, long-horizon planning, or dynamic control. We fine-tuned the model on 200 demonstrations for each task. 

1. Cloth Folding This is a highly challenging deformable object manipulation task involving a sequence of precise actions. 

Task Goal: Fold a long-sleeved shirt placed flat on the table into a compact square. 

Procedure: The task implies three distinct subtasks: 1) Left Sleeve: Fold the left sleeve inwards; 2) Right Sleeve: Fold the right sleeve inwards; 3) Final Fold: Fold the bottom of the shirt upwards to the neck. 

Evaluation: The total success rate is calculated based on the successful completion of all three subtasks. 

Unseen Object Setting: We evaluate on 3 different types of unseen shirts featuring varying colors, textures, and sizes compared to the fine-tuning data to test generalization. 

2. Table Bussing (Long-Horizon) A task requiring sequential manipulation of multiple objects to prepare a dining setup. 

Task Goal: return all items (e.g., a plate, a cup, and cutlery) from random initial positions to the correct locations to form a dining setup. 

Complexity: This is a long-horizon task that requires the robot to plan and execute multiple pick-and-place actions in sequence. The order of operations may vary, and the robot must handle clutter. 

Metric - Progress Score: We use a Progress Score to evaluate performance. Each item picked up and successfully placed into the designated place contributes 0.2 points to the score. Points are only awarded if the item is correctly placed in the designated location. 

Unseen Scene: We modify the table background and lighting conditions, testing on 2 distinct unseen scenes to evaluate robustness to visual domain shifts. 

3. Unzipping Another deformable object task requiring fine motor skills. 

Task Goal: Unzip a zipper on a bag. 

Setup: The task involves bimanual manipulation where both the left and right hands participate. The robot must grasp the small fabric doll attached to the zipper tab and pull it along a specific trajectory. 

Success : A trial is considered successful only if the zipper is successfully unzipped. 

4. Button Pressing (Dynamic) Unlike the static zero-shot version, this task focuses on reaction speed. 

Setup: A keyboard is placed on the table, and a screen is positioned in front of the robot. The screen turns green at a random time. 

Task Goal: The robot must press any key on the keyboard as quickly as possible after the screen turns green. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/7e039d4bec74abc44039b6465b1f383905a1c59ba7122f91a2c1bcd102696a32.jpg)


Pick up <object> with the right hand. 

Your objective is to use the right hand to pick up <object>. 

The <object> should be picked up by the right hand. 

Your right hand should be utilized to grasp <object>. 

Figure 15: Demonstrations of zero-shot experiments of RDT2. 

Metric - Reaction Time: We measure the time interval between the screen turning green and the key pressing. If the robot presses a key before the screen turns green, the trial is not counted as a success. 

5. Table Tennis A highly dynamic task requiring rapid visual processing and motion generation. 

Setup: A ball launcher shoots a ping-pong ball towards the robot. The robot holds a paddle. 

Task Goal: The robot must intercept and hit the moving ball. 

Metric - Hit Rate: The percentage of balls successfully hit by the paddle. This tasks effectively benchmarks the inference latency and control frequency of the policy, as slow models will consistently miss the ball. 

# D.5. Implementation Details of Ablation Studies

To test our hybrid training strategy (Stage 1 + Stage 2), we compared it against training the Action Expert from scratch with only Flow Matching. 


Task 1: Cloth Folding


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/9c2bceb576374696ed152fe9896500e833a82dc2435921276579abcd299add09.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/b35294ee04bd824f2e5ad7697ade3e0fc8a31afa13c14ef2b9e9ff600e5e0edd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/6c185f1f6b785799f800de0ca4fa1b28c9b7868dfeff21e0a2d06ceed7713edf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/8eb5d5fb1e6171f84228c3a0e31a7feba925cf288bacb56ff13db9b6e09b7eb1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/785287aa526b6c544d45cff12e8413e78668a269a6cc6ad6ff35f6628d1908a9.jpg)



Task 2: Table Bussing


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/02053d2dc0a14724ff71d437607e598cd1ba15036e9e40eb98cc1440de942b19.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/c1cea57debd5666907a2098ab95495f22c2a761a8403649bf20a0335a6bbae42.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/ed8d53a9ac0285876e1205f67db754d667d654da45e76415070eff1e244c497e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/7968ea54ff8071c25a5ba886ef740558d0a195fe31324375d9d1d1cac8a3dd90.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/b028fa7ac9befbf7782e44e7df9a795ec27db3538e29a72188bef0db06de51ee.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/ee543284c560973501afbead85dbe9d40dd46e114db3deb3a00d8ef6bebd817a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/bb281b134acb5fc9a7d87f27ea9ece37f8772bb181cd73c9b7267e2e2b55d5ad.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/21d6c0de2856d2e7d968e9849ac2b37f3102a9d862af7de5720ea380ba8d28fe.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/72234631c55bf05e5a2bcf01fbe5d7cb32ea7ea4dc94aa66d238e0bd3dbedb2a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/4d4c8415957907a6a4cf6bcd3a15a228fd54fcc242453c405332b2636db1056b.jpg)



Task 3: Unzipping


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/f03ac4f78d463ff8a03dfcdd0b4d246fae8792ed411cf5fd148327b8b63971c1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/f237c35ce60f0ab183740c5d6c27de413d2c562c8773581e91e0b33d612b0c94.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/bc6cf849e6679eee48a75c12492c08eff954b3d519fed4cf068f47f9a6a9fe92.jpg)



Task 4: Button Pressing


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/472f6a3b87140c9e658ba4cc7f2d6b619b208c07a4793b4507ac5df1694b9bde.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/fa2cc1fdaad7e28fbfbeb084d7678063d54b21834fcc22e9abc5fd4fdad926bc.jpg)



Task 5: Table Tennis


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/5d220583911aaa9a7fc0854baf736e768f37eada82ddb92da6a076bae02d0e6c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/666ba8ee2a00553b9a87d498740be583e08e6f4e6e4642cdc4a8b3afff74dde8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/b634edfbdeebcdfcfc347e36722768044a0631bb3bac81ef7dc180cf1198bf50.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/0a39fb2b848ae5bd7127b8e0d12103c44da06ef5d328ea862a477109c1cd42cf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/339e72ef-adbb-44b1-b6bb-9b79e5c24840/cce494d1fbfd3bb1253644d46bb3b932c655e760ced604886b12452f1854c1df.jpg)



Figure 16: Demonstrations of fine-tuning experiments of RDT2.


Hybrid Training (Stage 1: AR Pre-training). We used the standard Cross-Entropy objective to pre-train the model, minimizing the negative log-likelihood of discretized action tokens: 

$$
\mathcal {L} _ {\mathrm{AR}} = - \sum_ {t = 1} ^ {T} \log P (a _ {t} | a _ {<   t}, V, L), \tag {7}
$$

where $a _ { t }$ is the action token at step t, and V, L are visual and language inputs. Configuration followed Table 10. We trained for 128K steps on 7 nodes (8 GPUs each). We used a batch size per GPU of 96 with gradient accumulation, for a global batch size of 5,376. 

Hybrid Training (Stage 2: Diffusion Fine-tuning). We froze the vision-language backbone and fine-tuned the Action Expert with Conditional Flow Matching (CFM). We minimized the flow matching loss: 

$$
\mathcal {L} _ {\mathrm{CFM}} = \mathbb {E} _ {t, x _ {0}, x _ {1}} \| v _ {t} (x _ {t}) - (x _ {1} - x _ {0}) \| ^ {2}. \tag {8}
$$

Parameters are in Tab. 10. We trained for 66K steps on 7 nodes (8 GPUs each) with a batch size per GPU of 96, for a global batch size of 2,304. 

Diffusion from Scratch. We trained the entire model (including the VLM backbone) from scratch using the same CFM loss. To match the scale of the hybrid training, we trained for 187K steps on 7 nodes (8 GPUs each). We used a batch size per GPU of 64 (via gradient accumulation), for a global batch size of 3,584. Other hyperparameters matched the standard Flow Matching configuration. 