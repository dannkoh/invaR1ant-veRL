<h1 align="center">invaR1ant-veRL</h1>
<p align="center"><em>A customized fork of veRL with specialized reinforcement learning tools for large language models</em></p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#published-results">Published Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#custom-modifications">Custom Modifications</a> •
  <a href="#usage">Usage</a> •
  <a href="#references">References</a>
</p>

<hr />

<h2 id="overview" align="center">Overview</h2>

<p>
  <strong>invaR1ant-veRL</strong> leverages the robust reinforcement learning library originally developed in <a href="https://github.com/volcengine/verl">veRL</a>. This fork retains the powerful features of the original project while introducing custom modifications specific to invaR1ant's goals.
</p>

<p>
  This framework provides a flexible and efficient RL training infrastructure for large language models, with specialized hooks and integrations tailored for project-specific needs. By building on veRL's comprehensive foundation, invaR1ant-veRL enables advanced RL techniques for LLM fine-tuning while adding custom components.
</p>

<h3 align="center">Highlights</h3>

<ul>
  <li>Forked and customized version of the original veRL library</li>
  <li>Retains all core veRL features including FSDP, Megatron-LM, and vLLM support</li>
  <li>Easily extendable RL algorithms and modular APIs for integrating various LLM frameworks</li>
  <li>Custom modifications aimed at enhancing project-specific functionality</li>
  <li>Streamlined workflows for RL-based LLM training</li>
</ul>

<hr />

<h2 id="key-features" align="center">Key Features</h2>

<p>invaR1ant-veRL inherits and extends the powerful features of the original veRL project:</p>

<table align="center">
  <tr>
    <td align="center"><strong>Distributed Training</strong></td>
    <td>FSDP and Megatron-LM support for efficient large-scale training</td>
  </tr>
  <tr>
    <td align="center"><strong>Modular Architecture</strong></td>
    <td>Easily plug in different components (models, datasets, reward functions)</td>
  </tr>
  <tr>
    <td align="center"><strong>vLLM Integration</strong></td>
    <td>High-throughput LLM inference for RL feedback loops</td>
  </tr>
  <tr>
    <td align="center"><strong>Flexible RL Algorithms</strong></td>
    <td>Support for PPO, DPO, and other reinforcement learning approaches</td>
  </tr>
  <tr>
    <td align="center"><strong>Custom Reward Functions</strong></td>
    <td>Enhanced with invaR1ant-specific reward mechanisms</td>
  </tr>
</table>

<p align="center">
  For a complete list of veRL's features, please refer to the <a href="https://verl.readthedocs.io">original documentation</a>.
</p>

<hr />

<h2 id="published-results" align="center">Published Results</h2>

<p>The <strong>invaR1ant</strong> project has released the following models and datasets, built upon the <a href="https://huggingface.co/Qwen/Qwen2.5-3B">Qwen 2.5</a> foundation models:</p>

<h3 align="center">Base Models</h3>

<table align="center">
  <tr>
    <th>Model</th>
    <th>Description</th>
    <th>Link</th>
  </tr>
  <tr>
    <td><strong>Qwen2.5-3B</strong></td>
    <td>Original foundation model developed by the Qwen team, featuring 3 billion parameters with strong multilingual capabilities and reasoning skills</td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B">HuggingFace Hub</a></td>
  </tr>
  <tr>
    <td><strong>Qwen2.5-3B-Instruct</strong></td>
    <td>Instruction-tuned variant of Qwen2.5-3B optimized for following user instructions and conversations</td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct">HuggingFace Hub</a></td>
  </tr>
</table>

<h3 align="center">invaR1ant Models</h3>

<table align="center">
  <tr>
    <th>Model</th>
    <th>Description</th>
    <th>Link</th>
  </tr>
  <tr>
    <td><strong>invaR1ant-1.0</strong></td>
    <td>Further refined Qwen2.5-3B base model trained with the invaR1ant-veRL framework using custom reward functions. Enhances logical consistency and reasoning capabilities while retaining the original model's strengths.</td>
    <td><a href="https://huggingface.co/dannkoh/invaR1ant-1.0">HuggingFace Hub</a></td>
  </tr>
  <tr>
    <td><strong>invaR1ant-1.0-Instruct</strong></td>
    <td>Instruction-tuned variant of invaR1ant-1.0 optimized with RLHF techniques from veRL. Delivers improved instruction following, more consistent reasoning, and enhanced capabilities for complex tasks.</td>
    <td><a href="https://huggingface.co/dannkoh/invaR1ant-1.0-Instruct">HuggingFace Hub</a></td>
  </tr>
</table>

<h3 align="center">Key Enhancements over Qwen2.5</h3>

<ul align="center" style="display: inline-block; text-align: left;">
  <li>Customized reward models specifically designed for logical consistency</li>
  <li>Additional fine-tuning on specialized datasets targeting reasoning capabilities</li>
  <li>Improved performance on tasks requiring multi-step logical inference</li>
  <li>Reduced tendency to generate incorrect or inconsistent statements</li>
</ul>

<h3 align="center">Datasets</h3>

<table align="center">
  <tr>
    <th>Dataset</th>
    <th>Description</th>
    <th>Link</th>
  </tr>
  <tr>
    <td><strong>invaR1ant-easy</strong></td>
    <td>Specialized dataset for training v1.0 invaR1ant models, containing carefully crafted examples to enhance the model's reasoning capabilities while maintaining factual accuracy</td>
    <td><a href="https://huggingface.co/datasets/dannkoh/invaR1ant-easy">HuggingFace Hub</a></td>
  </tr>
</table>

<hr />

<h2 id="installation" align="center">Installation</h2>

<h3 align="center">Prerequisites</h3>

<p>Before running the <code>requirements.txt</code> file, you may need to install several dependencies manually to avoid compilation errors, here are some that helped me:</p>

<h4>1. PyTorch</h4>

```bash
pip install torch torchvision torchaudio
```
<p><a href="https://pytorch.org/get-started/locally/">Torch installation guide</a></p>
<p><em>Note: Tested with CUDA 12.4, but other versions may work as well.</em></p>

<h4>2. Flash Attention</h4>

```bash
pip install flash-attn --no-build-isolation
```
<p><a href="https://github.com/Dao-AILab/flash-attention">Flash-Attn installation guide</a></p>
<p><em>Note: The <code>--no-build-isolation</code> flag is necessary as standard installation may fail.</em></p>

<h4>3. XGrammar (From Source)</h4>

```bash
##########################DEPENDENCIES############################
# install Python dependency
pip install ninja pybind11 torch
##################################################################

# 1. clone from GitHub
git clone --recursive https://github.com/mlc-ai/xgrammar.git && cd xgrammar
# 2. build XGrammar core and Python bindings
mkdir build && cd build
cmake .. -G Ninja
ninja
# 3. install the Python package
cd ../python
pip install .
```
<p><a href="https://xgrammar.mlc.ai/docs/start/install">XGrammar installation guide</a></p>

<h4>4. XFormers</h4>

```bash
pip install -U xformers
```
<p><a href="https://github.com/facebookresearch/xformers">XFormers installation guide</a></p>

<h4>5. VLLM</h4>

```bash
pip install vllm==0.7.3
```
<p><a href="https://docs.vllm.ai/en/stable/getting_started/installation/gpu/index.html">VLLM installation guide</a></p>
<p><em>Note: Installation issues may occur with <code>xformers-0.0.28.post3.tar.gz</code> in certain environments.</em></p>

<h4>6. Remaining Dependencies</h4>

```bash
pip install -r requirements.txt
```

<hr />

<h2 id="project-structure" align="center">Project Structure</h2>

<pre>
invaR1ant-veRL/
├── examples/              # Example scripts and notebooks demonstrating usage
├── verl/                  # Core veRL library files with custom modifications
│   ├── algorithms/        # Reinforcement learning algorithm implementations
│   ├── models/            # Model architectures and wrappers
│   ├── trainers/          # Training infrastructure code
│   ├── utils/             # Utility functions and helpers
│   │   ├── dataset/
│   │   │   └── rl_dataset.py   # Custom dataset implementation
│   │   └── reward_score/
│   │       └── custom.py       # Custom reward function logic
│   └── inference/         # Inference components and wrappers
├── data/                  # Data files and datasets (some excluded due to size)
├── scripts/               # Utility scripts for running and managing the project
├── configs/               # Configuration files for various experiments
└── README.md              # This file
</pre>

<hr />

<h2 id="custom-modifications" align="center">Custom Modifications</h2>

<p>
  This fork contains several key differences from the original veRL project:
</p>

<table align="center">
  <tr>
    <th>Component</th>
    <th>Modification</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td><code>/verl/utils/reward_score/custom.py</code></td>
    <td>Custom reward function logic</td>
    <td>Project-specific reward mechanisms for RL fine-tuning</td>
  </tr>
  <tr>
    <td><code>/verl/utils/dataset/rl_dataset.py</code></td>
    <td>Removal of chat template</td>
    <td>Adapting dataset handling for invaR1ant's specialized use cases</td>
  </tr>
</table>

<p>
  These modifications are designed to enhance the framework's capabilities for invaR1ant's specific requirements while maintaining compatibility with veRL's core architecture.
</p>

<hr />

<h2 id="usage" align="center">Usage</h2>

<h3>Example Scripts</h3>

<p>Check the <code>/examples</code> directory for more comprehensive usage examples:</p>

<ul>
  <li><code>examples/grpo_trainer/</code>: GRPO Training Examples</li>
  <li><code>examples/ppo_trainer/</code>: PPO training Examples</li>
  <li><code>scripts/</code>: Scripts used for this project</li>
</ul>

<h3>Using Published Models</h3>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model
model_name = "dannkoh/invaR1ant-1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Or load the instruction-tuned model
instruct_model_name = "dannkoh/invaR1ant-1.0-Instruct"
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_name)
instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)

# Generate text with the model
inputs = tokenizer("Generate a creative story about AI:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

<hr />

<h2 id="references" align="center">References</h2>

<h3>Referencing the Original veRL Project</h3>

<p>
  For any original features, news updates, and community contributions not specific to this fork, please check out the <a href="https://github.com/volcengine/verl">veRL official repository</a>. The original project handles:
</p>

<ul>
  <li>Release news and announcements</li>
  <li>Official performance tuning guides</li>
  <li>Citation and acknowledgment information</li>
</ul>

<hr />

<h2 id="citation" align="center">Citation</h2>

<p>
  If you find this project helpful, please cite:
</p>
<pre><code>Paper releasing soon!
</code></pre>

<hr />

<h2 id="author" align="center">Author</h2>

<div align="center">
  <table>
    <tr>
      <td align="right"><strong>Name:</strong></td>
      <td>Daniel Koh</td>
    </tr>
    <tr>
      <td align="right"><strong>LinkedIn:</strong></td>
      <td><a href="https://uk.linkedin.com/in/dankoh02">https://uk.linkedin.com/in/dankoh02</a></td>
    </tr>
    <tr>
      <td align="right"><strong>University Email:</strong></td>
      <td><a href="mailto:daniel.koh@student.manchester.ac.uk">daniel.koh@student.manchester.ac.uk</a></td>
    </tr>
    <tr>
      <td align="right"><strong>Personal Email:</strong></td>
      <td><a href="mailto:danielkoh03120207@gmail.com">danielkoh03120207@gmail.com</a></td>
    </tr>
    <tr>
      <td align="right"><strong>GitHub:</strong></td>
      <td><a href="https://github.com/dannkoh">https://github.com/dannkoh</a></td>
    </tr>
  </table>
</div>
