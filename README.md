# Medical Doctor Agent

https://github.com/user-attachments/assets/89bb706b-3430-44d6-8912-6378edeb94d9

> This repository contains my team's final project, with a grade of **97/100**, for the **Reinforcement Learning** subject at **University of Technology Sydney** (UTS) taught by [Assoc. Prof. Nabin Sharma](https://profiles.uts.edu.au/Nabin.Sharma).

## I. Introduction

Clinical question-answering requires verifiable reasoning and machine-readable outputs, but general-purpose LLMs often produce unstructured rationales or fragile answers. We introduce a _two-stage post-training pipeline_ that transforms small LMs into structured medical reasoners:

-   First, **Supervised Fine-Tuning (SFT)** trains the response grammar, reasoning within `<THINK>…</THINK>` followed by a final medical decision in `<ANSWER>…</ANSWER>`.
-   Next, we implement **Group Relative Policy Optimization** ([GRPO](https://arxiv.org/pdf/2402.03300)) with a [multi-reward setup](#III-multi-reward-system) that simultaneously optimizes: **(i)** strict format adherence, **(ii)** partial credit for format, and **(iii)** semantic answer correctness through an [LLM verifier](https://huggingface.co/FreedomIntelligence/medical_o1_verifier_3B) that manages clinical aliases and wording differences.

We utilize LoRA for efficient parameter updates and a length-independent **Dr. GRPO** objective to prevent reward-length coupling. Evaluated on [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) (n=1,273) and [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) (n=4,183), our top model (**Qwen3-1.7B-Instruct** + [GRPO](https://arxiv.org/pdf/2402.03300)) attains 49.41% and 46.07% exact-match accuracy, respectively, with nearly 100% format compliance; [GRPO](https://arxiv.org/pdf/2402.03300) also surpasses [PPO](https://arxiv.org/abs/1707.06347) on both datasets. These findings demonstrate that verifier-guided, multi-signal [GRPO](https://arxiv.org/pdf/2402.03300) consistently enhances factual accuracy while ensuring outputs are interpretable and conform to templates, offering a practical route toward reliable, compact medical reasoning systems.

## II. Proposed Solution

![](./images/solution.png)

The models will be fine-tuned to produce structured outputs with a reasoning section wrapped in `<THINK>` tags for step-by-step logic, followed by a precise medical answer in `<SOLUTION>` tags. We designed a two-stage pipeline here, **Supervised Fine-Tuning (SFT)** followed by **Reinforcement Learning (RL)**, to transform LLMs into structured medical reasoners:

-   **Phase 1 - SFT**: The goal here is not to teach the model to be a medical solver yet. It's to teach the model the grammar of our desired output. Here, we used a [dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) of multiple medical problems with high-quality reasoning traces, formatted with our custom `<THINK>` and `<SOLUTION>` tags. This forces the model to learn the structural template we defined.
-   **Phase 2 - RL**: This is where we refine the logic using **RL**. Now that the model already knows how to structure its response, we then use [GRPO](https://arxiv.org/pdf/2402.03300) and this [medical questions dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem) to teach it how to reason accurately and arrive at the correct medical answer.

[GRPO](https://arxiv.org/pdf/2402.03300) is an SOTA RL technique designed to overcome key limitations of the traditional [PPO](https://arxiv.org/abs/1707.06347). Specifically, [PPO](https://arxiv.org/abs/1707.06347) can suffer from high memory overhead due to its reliance on value network and instability in value function estimation. [GRPO](https://arxiv.org/pdf/2402.03300) addresses these issues by eliminating the need for a learned value function, instead using **group-relative advantage estimation** across multiple responses. This not only reduces computational cost but also improves training stability and scalability

Another drawback of [PPO](https://arxiv.org/abs/1707.06347) is that it also relies on a reward model that assigns **an absolute score to a generation**. There are 2 problems with this:

-   First, the reward model can rely on human judgments that usually lack explicit criteria and require expensive human annotation.
-   Second, it can be unstable as the LLM might learn to **hack** the reward. For example, it can generate very long completion if the length is correlated with a higher score. The solution here is to define a list of smaller verifiable rewards, not a final all consuming singular one.

With [GRPO](https://arxiv.org/pdf/2402.03300), we already generated **a group of responses** for each prompt right? Instead of scoring each one in isolation, we evaluate them relative to each other with our [multi-reward system](#III-multi-reward-system):

-   This **Reinforcement Learning with Verifiable Rewards** will allow us to further eliminate the need for a reward model and replace subjective human evaluation with reliable, objective signals.
-   This relative comparison is far more stable and directly optimizes for what we want: better reasoning, not just a higher score.

## III. Multi-Reward System

![](./images/rewards.png)

Our core innovation is this multi-reward design. A single reward is not enough to capture the nuances of good medical reasoning. We designed a **panel of 4 expert judges** working in parallel, each evaluating the model's output from a different perspective:

1.  The first is the **Strict Formatter** which strictly evaluate format compliance to enforce the structure. It gives a large reward only if the entire response perfectly adheres to our `THINK` and `ANSWER` structure.

2.  The second is the **Partial Formatter** giving partial credit for incomplete tags. If the model messes up the full structure, but for example, still includes the `</THINK>` tag correctly, it still gets a small amount of credit.

3.  The third, also the **most important one**. It will check if the answer in the `<ANSWER>` tag is correct or not. Given the prevalence of aliases in the medical domain, exact matching methods, which commonly applied in mathematics, will be impractical here. Instead, as suggested by [HuatuoGPT-o1](https://arxiv.org/pdf/2412.18925), we use an [LLM verifier](https://huggingface.co/FreedomIntelligence/medical_o1_verifier_3B) here and prompt it to perform validation, returning a probability of how close the prediction aligns with the ground-truth answer. We designed this function to be sophisticated, giving full marks for an high probability, partial credit for close approximations, and a heavy penalty for wrong answers to avoid overconfidence.

By combining these 3 signals, we can prevent over-optimization on 1 aspect, which can lead to reward hacking problem. The [GRPO](https://arxiv.org/pdf/2402.03300)'s group-relative policy can navigate the complex trade-offs between formatting, correctness, readability, and optimizes by ranking completions, leading to a much more capable and reliable reasoning model.

## IV. GRPO Objective Improvement

![](./images/grpo.png)

Compared to the original formulation in the [DeepSeekMath](https://arxiv.org/pdf/2402.03300) paper, we followed [Hugging Face's GRPO guideline](https://huggingface.co/docs/trl/main/en/grpo_trainer#computing-the-loss) and made some further improvements to the [GRPO](https://arxiv.org/pdf/2402.03300) objective for more efficient training:

-   First, we can calculate the _mean_ at the _group_ and the _std_ at the _batch_ level. This scaling strategy enables more robust reward shaping, as evident by this [paper](https://huggingface.co/papers/2508.08221).
-   Second, we didn't use the **KL divergence** term, as motivated by several recent studies, which showed that **KL** term is not essential for training with [GRPO](https://arxiv.org/pdf/2402.03300). Therefore, it has become a common practice to exclude it.
-   Lastly, this [paper](https://huggingface.co/papers/2503.20783) has demonstrated that the initial [GRPO](https://arxiv.org/pdf/2402.03300) formulation introduces a **response length bias**. To solve that, they proposed dividing by a **constant generation budget** instead of the sequence length, so we employ this [Dr.GRPO](https://huggingface.co/papers/2503.20783) loss here to further enhances stability by preventing the model from being biased towards longer or shorter answers, focusing purely on the quality of the content.

## V. Experimental Results

![](./images/results.png)

👉 You can refer to our [slides](./slides.pdf) and [full report](./report.pdf) for more details on the methodology and results analysis.

## VI. References

**1. [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)**:

-   We adapted many ideas from their work on medical reasoning with LLMs.
-   We used their [PPO](https://arxiv.org/abs/1707.06347) approach as a baseline to compare against our [GRPO](https://arxiv.org/pdf/2402.03300) solution. Note that, we used smaller model here due to computational constraints on Colab Pro Environment.

**2. Hugging Face's Cookbook**:

-   [GRPO Trainer Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer#computing-the-loss).
-   [Post training an LLM for reasoning with GRPO in TRL](https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl).
-   [HuatuoGPT-o1 Medical RAG and Reasoning](https://huggingface.co/learn/cookbook/medical_rag_and_reasoning): We followed this to build our demo with RAG capabilities.

**3. Unsloth Documentation**:

> We mainly used [Unsloth](https://docs.unsloth.ai/) to implement our [GRPO](https://arxiv.org/pdf/2402.03300) training.

-   [Reinforcement Learning (RL) Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide).

-   [GRPO (Reasoning RL) notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks): We learned a lot from these notebooks.
