





# **An Intelligent Academic Assistant built using Fine-Tuning and RAG**
**Authors**


Sunrit Pal (24N0456) \\
Sarvesh Maurya (24N0465)








# Abstract

This project aims to develop an intelligent academic assistant capable of answering queries related to IIT Bombay-specific information such as academics, admissions, hostel policies, and placement guidelines. The system leverages a fine-tuned Mistral-7B-Instruct model enhanced using parameter-efficient fine-tuning techniques such as Low-Rank Adaptation (LoRA) and 4-bit quantization.Initially, the LLaMA-2 7B model was considered for fine-tuning. However, due to severe computational constraints including GPU memory overflow and RAM instability during training, it was not feasible to use LLaMA-2 in our setup. Therefore, the Mistral-7B-Instruct model was adopted as a more memory-efficient alternative that enabled stable training with quantization techniques.

In addition, the system incorporates Retrieval-Augmented Generation (RAG), where relevant institutional documents are retrieved dynamically from an external knowledge base. This hybrid approach combines contextual understanding with factual grounding, improving accuracy and reducing hallucinations.


# Introduction

Large Language Models (LLMs) have significantly advanced natural language processing by enabling machines to generate coherent and contextually relevant text. However, these models are typically trained on general-purpose datasets and lack the ability to accurately respond to domain-specific queries.

In academic institutions such as IIT Bombay, students frequently require information related to academic rules, hostel allocation, admissions, and placement policies. Existing systems are inefficient as they rely on static documents or manual query resolution.

While fine-tuning improves domain understanding, it does not guarantee factual correctness or up-to-date information. To address this limitation, this project integrates a Retrieval-Augmented Generation (RAG) framework.


## Model Selection Justification


Initially, the LLaMA-2 7B Chat model was selected due to its strong performance in conversational tasks. However, during implementation, multiple practical challenges were encountered:


    - GPU memory overflow (Out-of-Memory errors)
    - RAM crashes during training
    - inability to sustain training with limited hardware


Even after applying optimization techniques such as gradient accumulation and reduced batch sizes, the model could not be trained reliably.

As a result, the Mistral-7B-Instruct model was chosen as an alternative. This model provides:


    - Better memory efficiency
    - Compatibility with 4-bit quantization
    - Stable training under limited resources


Thus, Mistral-7B-Instruct was used as the base model for fine-tuning in this project.


# Problem Statement


The rapid advancement of Large Language Models (LLMs) has enabled the development of conversational systems capable of generating fluent and contextually relevant responses. However, these models are typically trained on large-scale general-purpose corpora and lack the ability to accurately capture domain-specific knowledge required for specialized applications such as institutional query answering.

In the context of academic institutions like IIT Bombay, users frequently seek precise and policy-driven information related to academic regulations, hostel allocation, admission procedures, and placement guidelines. General-purpose LLMs often fail to provide reliable responses in such settings due to the absence of institution-specific training data. As a result, their outputs tend to be vague, inconsistent, or factually incorrect.

Fine-tuning a pre-trained LLM on domain-specific data improves contextual understanding and enables the model to learn relevant patterns. However, fine-tuning alone introduces several key limitations:


    - **Static Knowledge Representation:** The model stores knowledge within its parameters, making it difficult to update information dynamically when institutional policies change.
    
    - **Hallucination:** The model may generate plausible but incorrect responses when the required information is not sufficiently represented in the training data.
    
    - **Incomplete Coverage:** It is impractical to include all possible institutional documents and edge cases in the training dataset.
    
    - **Computational Constraints:** Full fine-tuning is resource-intensive, requiring parameter-efficient approaches such as LoRA and quantization.


To address these limitations, Retrieval-Augmented Generation (RAG) introduces a non-parametric memory component, where relevant information is retrieved dynamically from an external knowledge base during inference. This enables the system to provide factually grounded responses without relying solely on learned parameters.

However, retrieval-based systems alone may lack strong reasoning capabilities and depend heavily on retrieval quality. Therefore, a hybrid approach that combines fine-tuning with retrieval is required.

**Thus, the core problem addressed in this project is to design a hybrid AI system that integrates parameter-efficient fine-tuning with retrieval-based augmentation to achieve both contextual understanding and factual accuracy in domain-specific question answering under computational constraints.**


# Objectives


The primary objective of this project is to develop a domain-specific academic assistant capable of accurately answering queries related to IIT Bombay by leveraging a hybrid architecture that combines fine-tuning and retrieval mechanisms.

The specific objectives are as follows:


    - **Develop a Domain-Specific Language Model:** Fine-tune the Mistral-7B-Instruct model on structured IIT Bombay datasets to capture institutional terminology, policies, and query patterns.
    
    - **Implement Parameter-Efficient Fine-Tuning:** Utilize Low-Rank Adaptation (LoRA) along with 4-bit quantization to enable efficient training with reduced computational requirements.
    
    - **Design a Retrieval-Augmented Generation (RAG) Pipeline:** Construct an external knowledge base by processing institutional documents into embeddings and storing them in a vector database for semantic retrieval.
    
    - **Integrate Retrieval with Generation:** Retrieve relevant document chunks based on user queries and incorporate them into the model input to generate contextually grounded responses.
    
    - **Improve Factual Accuracy and Reduce Hallucination:** Ensure that generated responses are supported by retrieved evidence, thereby minimizing incorrect or fabricated outputs.
    
    - **Enable Scalability and Updatability:** Design the system such that new institutional data can be added without requiring re-training of the model.
    
    - **Evaluate System Performance:** Assess the system based on qualitative metrics such as response accuracy, contextual relevance, and factual correctness.


**Overall, the objective is to build a robust hybrid AI system that combines the reasoning capability of fine-tuned language models with the factual reliability of retrieval-based methods for real-world academic assistance.**


# System Overview


The proposed system is a hybrid architecture that integrates a fine-tuned large language model with a retrieval-based augmentation mechanism to achieve both contextual reasoning and factual grounding.

The system consists of two main components:


    - **Parametric Module:** A Mistral-7B-Instruct model fine-tuned using Low-Rank Adaptation (LoRA) with 4-bit quantization. This module encodes linguistic structure, domain-specific patterns, and response generation capability within its parameters.
    
    - **Non-Parametric Module:** A retrieval system that maintains an external knowledge base in the form of dense vector embeddings. This module enables semantic search over document chunks using similarity-based retrieval.


At inference time, given a query $q$, the system first computes its embedding:

\[
e_q = f(q)
\]

The query embedding is used to retrieve relevant document embeddings $\{e_1, e_2, ..., e_n\}$ using cosine similarity:

\[
\text{sim}(e_q, e_i) = \frac{e_q \cdot e_i}{\|e_q\| \|e_i\|}
\]

The top-$k$ most relevant context chunks are selected:

\[
C = \text{TopK}(\text{sim}(e_q, e_i))
\]

The retrieved context is concatenated with the original query to form the augmented input:

\[
x = [q; C]
\]

The fine-tuned model generates the final response conditioned on this augmented input:

\[
y = P_\theta(y \mid x)
\]

where $\theta$ denotes the parameters of the fine-tuned model.

The parametric module captures generalized patterns and reasoning ability, while the non-parametric module ensures access to relevant external information. This combination enables improved factual consistency, reduced hallucination, and adaptability without retraining.

The overall system is designed to be modular, allowing independent updates to the retrieval database while maintaining a fixed fine-tuned model.


# Proposed Methodology


The proposed methodology follows a hybrid design that integrates parameter-efficient fine-tuning with retrieval-augmented generation. The pipeline is divided into two complementary phases: (i) model adaptation through fine-tuning, and (ii) knowledge augmentation through retrieval at inference time.


## Overall Pipeline


The complete system pipeline can be represented as:

\[
\begin{aligned}
\text{Dataset} &\rightarrow \text{Preprocessing} \rightarrow \text{Tokenization} \rightarrow \text{Model Loading} \\
&\rightarrow \text{LoRA Fine-Tuning} \rightarrow \text{Knowledge Base Construction} \\
&\rightarrow \text{Retrieval} \rightarrow \text{Augmented Inference}
\end{aligned}
\]

The first stage focuses on adapting the base model to the domain, while the second stage introduces dynamic retrieval to improve factual accuracy.


## Dataset Preparation


The dataset used for fine-tuning was constructed from IIT Bombay-related academic information such as policies, rules, and guidelines.

Initially, raw PDF documents containing institutional information were directly considered for training. However, this approach resulted in poor model performance. The model struggled to learn meaningful patterns due to:


    - Unstructured and noisy text present in PDFs
    - Lack of clear supervision signals
    - Increased hallucinations in generated responses


The model often generated incorrect or vague answers because it could not clearly distinguish between questions and relevant answers when trained on raw textual data.

To address this issue, the dataset construction approach was revised.


### Q\&A Based Dataset Construction


Instead of using raw documents, the dataset was manually curated in the form of structured Question-Answer pairs:

\[
D = \{(q_i, a_i)\}_{i=1}^{N}
\]

where:

    - $q_i$ represents a user query (question)
    - $a_i$ represents the corresponding correct response (answer)


Multiple such pairs were manually created based on IIT Bombay-related information to ensure high-quality supervision.


### JSON Dataset Format


The Q\&A pairs were stored in JSON format for efficient processing and compatibility with the training pipeline.

A typical JSON entry is structured as:

```
{
  "question": "What is CPI?",
  "answer": "CPI is the cumulative performance index..."
}
```

This structured format allows easy loading and preprocessing during fine-tuning.


### Instruction Formatting for Training


Although the dataset consists only of $(q, a)$ pairs, it is converted into an instruction-based format before training to match the expected input structure of the Mistral model.

Each pair is transformed as:

```
<s>[INST] question [/INST] answer
```

This ensures that:

    - The model interprets the question as an instruction
    - The answer is treated as the desired output



### Why This Approach Works Better


This revised dataset construction approach provides several advantages:


    - **Clear Supervision:** Direct mapping between question and answer
    - **Reduced Hallucination:** Model is guided with correct responses
    - **Better Learning:** Structured data improves training efficiency
    - **Higher Accuracy:** Model learns domain-specific patterns effectively


Thus, transitioning from raw PDF-based training to structured JSON-based Q\&A training significantly improved the performance and reliability of the model.


### Dataset Aggregation


All JSON files containing question-answer pairs are loaded and combined into a single dataset. Each file contributes a set of $(q, a)$ pairs, forming a unified training dataset.


### Token Creation


After aggregation, a custom tokenization function is applied. Padding is used to ensure uniform sequence lengths across all training samples, which improves computational efficiency and stability during training.


### Token Sequence Chunking


After tokenization, the dataset is further divided into smaller token sequences (chunks) to ensure that each input fits within the model's maximum context length. This step is important to prevent overflow issues and to maintain efficient training.

Thus, the training pipeline can be viewed as:

\[
(q, a) \rightarrow \text{tokens} \rightarrow \text{token chunks}
\]


## Tokenization


The textual data is converted into token sequences using a tokenizer compatible with the base model. Given a text sequence $T$, tokenization maps it into a sequence of tokens:

\[
T \rightarrow \{t_1, t_2, ..., t_n\}
\]

Padding and truncation are applied to ensure fixed sequence length constraints during training.

### Tokenizer Details


The tokenizer used in this system is a subword tokenizer associated with the Mistral-7B-Instruct model. Instead of splitting text into full words, it decomposes words into smaller subword units. This allows the model to handle rare and unseen words effectively while maintaining a compact vocabulary.

A key limitation of the Mistral tokenizer is that it does not provide a predefined padding token. To address this, the end-of-sequence token is reused as the padding token:

\[
\text{pad\_token} = \text{eos\_token}
\]

Padding is necessary to ensure that all token sequences have equal length, which enables efficient batch processing during training.


## Model Loading and Quantization


### Hugging Face Authentication


To access the Mistral-7B-Instruct model, a Hugging Face access token is generated. This token is used to authenticate requests made to the Hugging Face servers. Upon verification of the token, access to the model weights and tokenizer files is granted.


### Model Loading Mechanism


The model is loaded using the \texttt{AutoModelForCausalLM} class. This class is specifically designed for autoregressive language modeling, where the model predicts the next token based on previously generated tokens:

\[
P(y) = \prod_{t=1}^{T} P(y_t \mid y_{<t})
\]

The base model is initialized using Mistral-7B-Instruct. To reduce memory requirements, the model is loaded using 4-bit quantization. This compresses the weight matrices while preserving representational capacity, enabling efficient training on limited hardware.

### 4-bit Quantization


To reduce memory consumption, the model is loaded using 4-bit quantization. This compresses the model weights significantly while preserving performance. Computation is performed in float16 precision to ensure numerical stability.

This allows large models such as Mistral-7B to be trained and deployed even under limited GPU memory constraints.


## LoRA Fine-Tuning


Fine-tuning is performed using Low-Rank Adaptation (LoRA), where trainable low-rank matrices are injected into selected layers of the model. Instead of updating the full weight matrix $W$, LoRA modifies it as:

\[
W' = W + BA
\]

where $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ with $r \ll d$. This reduces the number of trainable parameters while enabling effective domain adaptation.

LoRA is applied primarily to attention projection layers, allowing the model to adapt its internal representations without full retraining.

### LoRA Configuration


In this project, the LoRA rank is set to $r = 32$, and the scaling factor $\alpha = 64$. The scaling factor controls the contribution of the low-rank update:

\[
W' = W + \frac{\alpha}{r} BA
\]


### Target Modules


LoRA is applied to the following modules:

    - Query projection ($q\_proj$)
    - Key projection ($k\_proj$)
    - Value projection ($v\_proj$)
    - Output projection ($o\_proj$)
    - Feedforward layers ($up\_proj$, $down\_proj$)
    - Gate projection ($gate\_proj$)


These modules play a crucial role in attention and transformation of representations.


### PEFT Framework


Parameter Efficient Fine-Tuning (PEFT) enables training by updating only the LoRA parameters while keeping the base model weights frozen. This reduces computational cost and memory usage significantly.

## Training Strategy


The model is trained using Supervised Fine-Tuning (SFT), where the objective is to minimize the cross-entropy loss between predicted and target tokens:

\[
\mathcal{L} = - \sum_{t=1}^{T} y_t \log(\hat{y}_t)
\]

Optimization is performed using the AdamW optimizer with gradient accumulation to handle memory constraints.


## Knowledge Base Construction


For retrieval, the document corpus is segmented into smaller chunks:

\[
D = \{d_1, d_2, ..., d_n\}
\]

Each chunk is transformed into a dense embedding using an embedding function:

\[
e_i = f(d_i)
\]


### Consistency of Embedding Space


The same embedding model (MiniLM-L6-v2) is used both during vector database creation and during query processing. Specifically:


    - During database creation, each document chunk $d_i$ is converted into an embedding:
    \[
    e_i = f(d_i)
    \]

    - During inference, the user query $q$ is also converted into an embedding:
    \[
    e_q = f(q)
    \]



## Dense Semantic Embeddings


The embedding model used in this system generates **dense semantic embeddings**. 
Unlike traditional representations such as one-hot encoding, which are sparse and high-dimensional 
with most values being zero, dense embeddings represent text in a compact continuous vector space.

These embeddings capture the *semantic meaning* of text rather than just exact word presence. 
As a result, semantically similar sentences are mapped to nearby points in the vector space. 
This property is crucial for retrieval tasks, as it allows similarity measures such as cosine similarity 
to effectively identify relevant documents even when there is no exact keyword match.

In contrast, sparse encodings like one-hot vectors are inefficient in terms of storage and fail to capture 
relationships between words, making them unsuitable for modern retrieval-based systems such as RAG.

Using the same embedding model ensures that both $e_q$ and $e_i$ lie in the same vector space. This is essential because similarity measures such as cosine similarity are only meaningful when vectors are generated from the same embedding function.

If different models were used, the embeddings would lie in different spaces, making similarity comparisons invalid and degrading retrieval performance.

These embeddings are stored in a vector database to enable efficient similarity search.

### Vector Database Structure


Each entry in the vector database consists of:

    - Embedding vector
    - Corresponding text chunk
    - Metadata (such as source document)



### Vector Database Representation


The vector database can be conceptualized as a structured table:

\begin{center}
\begin{tabular}{|c|c|c|}
\hline
Embedding Vector & Text Chunk & Metadata \\
\hline
$e_1$ & $d_1$ & $m_1$ \\
$e_2$ & $d_2$ & $m_2$ \\
\hline
\end{tabular}
\end{center}


### Vector Database Creation Pipeline


The construction process involves:

    - Reading all PDF documents
    - Extracting text from each document
    - Combining text into a single dataset
    - Splitting text into smaller chunks
    - Converting each chunk into embeddings using MiniLM-L6-v2
    - Storing embeddings in Chroma vector database



### Embedding Model


The MiniLM-L6-v2 model is used to convert text into vector embeddings that capture semantic meaning.


### Embedding Consistency


The same embedding model is used for both document chunks and user queries to ensure they lie in the same vector space. This is essential for meaningful similarity computation.


## Retrieval Mechanism


Given a query $q$, its embedding is computed:

\[
e_q = f(q)
\]

Similarity between the query and document embeddings is computed using cosine similarity:

\[
\text{sim}(e_q, e_i) = \frac{e_q \cdot e_i}{\|e_q\| \|e_i\|}
\]


### Similarity-Based Retrieval


The query embedding is compared with all stored embeddings using cosine similarity. The results are ranked based on similarity scores.


### Top-K Selection


The top-$k$ most similar document chunks are selected and used as context for the language model.


## Context Augmentation and Generation


The retrieved context $C$ is concatenated with the original query:

\[
x = [q; C]
\]

This augmented input is passed to the fine-tuned model, which generates the response:

\[
y = P_\theta(y \mid x)
\]


## Inference Pipeline


During inference, the model generates tokens autoregressively:

\[
P(y) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x)
\]

The inclusion of retrieved context ensures that generation is grounded in relevant external information.


## Integration of Fine-Tuning and Retrieval


Fine-tuning enables the model to learn domain-specific semantics and response structures, while retrieval provides dynamic access to factual information. The integration of these components results in a system that balances generalization and accuracy, ensuring both coherent and reliable outputs.




# Evaluation Strategy


The evaluation of the proposed Retrieval-Augmented Generation (RAG) system is focused on measuring both the quality of generated answers and the effectiveness of the retrieval mechanism. Since the system integrates a retrieval module with a generative language model, it is important to evaluate both components independently as well as jointly.

In this work, evaluation is conducted using two primary quantitative metrics:


    - **Answer Similarity (Semantic Similarity)**
    - **Retrieval Similarity (Average Cosine Similarity)**


These metrics are based on dense vector embeddings and cosine similarity, allowing comparison based on semantic meaning rather than exact textual matching.


## Embedding-Based Evaluation


All textual inputs (queries, generated answers, and ground-truth answers) are converted into dense vector representations using a pre-trained sentence embedding model. These embeddings capture semantic meaning in a high-dimensional vector space.

Let:

    - $e_q$ = embedding of the query
    - $e_p$ = embedding of the generated (predicted) answer
    - $e_g$ = embedding of the ground-truth answer
    - $e_i$ = embedding of the $i^{th}$ retrieved document chunk


Each embedding is a real-valued vector in $\mathbb{R}^d$.


## Cosine Similarity


To measure similarity between two embeddings, cosine similarity is used. It computes the cosine of the angle between two vectors and is defined as:

\[
\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}
\]

where:

    - $a \cdot b$ is the dot product of the vectors
    - $\|a\|$ and $\|b\|$ are the Euclidean norms


The value of cosine similarity lies in the range $[-1, 1]$, where:

    - 1 indicates identical semantic meaning
    - 0 indicates no semantic similarity
    - values closer to 1 indicate stronger similarity



## Answer Similarity (Generation Evaluation)


The quality of generated responses is evaluated by comparing the generated answer with the ground-truth answer using cosine similarity between their embeddings.

\[
\text{AnswerSim} = \text{sim}(e_p, e_g)
\]

This metric measures how semantically close the generated response is to the expected answer. Unlike exact match metrics, this approach accounts for paraphrasing and variation in wording.

A higher Answer Similarity score indicates that the generated answer captures the correct meaning, even if the phrasing differs.

**Interpretation:**

    - High score ($>0.8$): Highly accurate answer
    - Moderate score ($0.6 - 0.8$): Partially correct or verbose answer
    - Low score ($<0.5$): Incorrect or irrelevant answer



## Retrieval Similarity (Retrieval Evaluation)


The performance of the retrieval module is evaluated by measuring how relevant the retrieved document chunks are with respect to the query.

For a given query $q$, let the top-$k$ retrieved chunks be $C = \{d_1, d_2, ..., d_k\}$. The average cosine similarity between the query embedding and each retrieved chunk embedding is computed as:

\[
\text{AvgSim} = \frac{1}{k} \sum_{i=1}^{k} \text{sim}(e_q, e_i)
\]

This metric evaluates whether the retrieval system is successfully identifying semantically relevant context.

**Interpretation:**

    - High score ($>0.7$): Highly relevant retrieval
    - Moderate score ($0.6 - 0.7$): Acceptable retrieval quality
    - Low score ($<0.5$): Poor retrieval performance



## Evaluation Procedure


The evaluation process is conducted as follows:


    - A query $q$ is provided as input to the system.
    - The retrieval module returns top-$k$ relevant document chunks.
    - The generative model produces an answer based on the retrieved context.
    - The generated answer is compared with the ground-truth answer using cosine similarity.
    - The relevance of retrieved chunks is evaluated using average cosine similarity with respect to the query.



## Summary


The evaluation framework focuses on two key aspects:

    - **Generation Quality:** Measured using Answer Similarity
    - **Retrieval Effectiveness:** Measured using Average Retrieval Similarity


This approach ensures that the system is evaluated not only on its ability to generate meaningful responses, but also on its ability to retrieve relevant supporting information. By using embedding-based similarity measures, the evaluation captures semantic correctness rather than relying on exact word matching.


# Expected Outcomes


The proposed hybrid system is expected to demonstrate significant improvements over a standalone fine-tuned language model by effectively combining parametric learning with retrieval-based augmentation.


## Improved Response Accuracy


By incorporating relevant retrieved context into the input, the system is expected to generate responses that are more accurate and aligned with the underlying knowledge base. The conditional generation process:

\[
y = P_\theta(y \mid x), \quad x = [q; C]
\]

ensures that the output is influenced not only by learned parameters $\theta$ but also by retrieved evidence $C$.


## Reduction in Hallucination


The retrieval mechanism provides explicit grounding information, which constrains the model's generation space. As a result, the probability of generating unsupported or fabricated content is expected to decrease. This leads to a lower hallucination rate compared to purely fine-tuned models.


## Enhanced Contextual Relevance


The use of semantic retrieval ensures that the augmented input contains query-relevant information. This improves the alignment between the query and the generated response, resulting in higher contextual relevance.


## Scalability and Updatability


Unlike fine-tuned models where knowledge is embedded in parameters, the external knowledge base can be updated independently. New information can be incorporated by adding new document embeddings without modifying model parameters:

\[
\{e_1, e_2, ..., e_n\} \rightarrow \{e_1, e_2, ..., e_n, e_{n+1}\}
\]

This enables efficient scalability and adaptability to evolving data.


## Computational Efficiency


The use of LoRA and 4-bit quantization significantly reduces the number of trainable parameters and memory footprint. This allows efficient training and deployment under limited computational resources while maintaining model performance.


## Robust End-to-End Performance


The integration of retrieval and generation is expected to yield a system that balances reasoning and factual correctness. The final output distribution:

\[
P(y) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x)
\]

reflects both learned patterns and retrieved knowledge, resulting in coherent, accurate, and context-aware responses.

**Overall, the system is expected to provide reliable, grounded, and scalable question answering performance, addressing key limitations of both standalone fine-tuned models and retrieval-only systems.**

# User Interface Implementation


A user-friendly interface is developed using Gradio, allowing students to interact with the system through a simple input-output interface. Users can enter queries, and the system processes them using the RAG pipeline to generate accurate responses.


# Applications


The proposed hybrid system can be applied in multiple domain-specific and real-time assistance scenarios:


    - **Academic Assistant:** Automated answering of queries related to policies, rules, and procedures with context-aware responses.
    
    - **Institutional Chatbot:** Deployment as a conversational interface for handling student and administrative queries.
    
    - **Document Query System:** Semantic search and question answering over large document collections using retrieval and generation.
    
    - **Knowledge Management Systems:** Integration into systems requiring structured access to domain-specific information.



# Limitations


Despite its advantages, the system has several limitations:


    - **Retrieval Dependency:** The quality of generated responses is highly dependent on the relevance of retrieved document chunks.
    
    - **Embedding Limitations:** Semantic embeddings may fail to capture nuanced differences in queries, leading to suboptimal retrieval.
    
    - **Context Length Constraint:** The number of retrieved chunks is limited by the model's maximum input token length.
    
    - **Computational Overhead:** Retrieval and embedding operations introduce additional latency during inference.
    
    - **Data Coverage:** The system can only retrieve and generate information present within the knowledge base; unseen or missing data cannot be inferred reliably.



# Future Work


Although the proposed RAG-based academic assistant demonstrates promising performance, there are several directions for future improvement.

One major limitation of the current system is the limited size and diversity of the training and retrieval data. The existing dataset primarily contains manually curated question-answer pairs focused on specific IIT Bombay-related topics. Increasing the volume of data and incorporating more diverse query patterns, institutional documents, and real-world conversational examples can significantly improve the robustness and generalization capability of the system.

Future work can also focus on expanding the knowledge base with additional academic policies, department-specific information, placement reports, hostel notices, and frequently asked student queries. A more diverse dataset would help the model generate more relevant, context-aware, and accurate responses for a wider range of user questions.

In addition, retrieval quality can be improved using advanced embedding models and better chunking strategies. Further optimization of the RAG pipeline may help reduce hallucinations and improve factual grounding.

The system can also be extended into a fully scalable institutional assistant with continuous knowledge base updates, multilingual support, and deployment for real-time student interaction.


# Conclusion


This work presents a hybrid approach for domain-specific question answering by integrating parameter-efficient fine-tuning with Retrieval-Augmented Generation (RAG). The fine-tuned Mistral-7B-Instruct model, adapted using LoRA and 4-bit quantization, captures domain-specific language patterns and enables efficient learning under resource constraints.

However, relying solely on fine-tuning leads to limitations such as static knowledge representation and hallucination. To address this, a retrieval mechanism is incorporated, allowing the system to dynamically access relevant information from an external knowledge base at inference time. The augmented input formulation:

\[
x = [q; C]
\]

ensures that the generated response is conditioned on both the query and retrieved context, resulting in factually grounded outputs.

The integration of parametric and non-parametric components provides a balanced architecture, where the model retains strong reasoning capabilities while improving factual accuracy and adaptability. The system also supports scalability, as updates to the knowledge base do not require re-training of the model.

Overall, the proposed framework demonstrates a practical and efficient solution for building reliable domain-specific assistants by combining fine-tuning and retrieval-based augmentation.


