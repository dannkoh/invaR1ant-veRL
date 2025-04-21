<h1 align="center">WARP-veRL</h1>
<p align="center"><em>Worst-case Asymptotic Reasoner for Path constraints (WARP) – a fork of veRL with custom reinforcement learning for LLMs</em></p>

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
  <strong>WARP-veRL</strong> leverages the robust reinforcement learning library originally developed in <a href="https://github.com/volcengine/verl">veRL</a>. This fork extends the original project to enable worst-case logical reasoning over path constraints.
</p>

<p>
  The framework provides a flexible and efficient RL training infrastructure for large language models, with specialized hooks for SMT-based verification and custom reward scoring tailored to the WARP-easy dataset.
</p>

<h3 align="center">Highlights</h3>

<ul>
  <li>Forked and customized version of the original veRL library</li>
  <li>Retains all core veRL features including FSDP, Megatron-LM, and vLLM support</li>
  <li>Easily extendable RL algorithms and modular APIs for integrating various LLM frameworks</li>
  <li>Custom modifications aimed at worst-case path-constraint reasoning</li>
  <li>Streamlined workflows for RL-based LLM training with WARP-easy dataset</li>
</ul>

<hr />

<h2 id="key-features" align="center">Key Features</h2>

<p>WARP-veRL inherits and extends the powerful features of the original veRL project:</p>

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
    <td>Support for PPO, GRPO, and other reinforcement learning approaches</td>
  </tr>
  <tr>
    <td align="center"><strong>Custom Reward Functions</strong></td>
    <td>Enhanced with WARP-specific SMT-based reward mechanisms</td>
  </tr>
</table>

<p align="center">
  For a complete list of veRL's features, please refer to the <a href="https://verl.readthedocs.io">original documentation</a>.
</p>

<hr />

<h2 id="published-results" align="center">Published Results</h2>

<p>The <strong>WARP</strong> project has released the following models and datasets, built upon the <a href="https://huggingface.co/Qwen/Qwen2.5-3B">Qwen 2.5</a> foundation models:</p>

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

<h3 align="center">WARP Models</h3>

<table align="center">
  <tr>
    <th>Model</th>
    <th>Description</th>
    <th>Link</th>
  </tr>
  <tr>
    <td><strong>WARP-1.0</strong></td>
    <td>Qwen2.5-3B fine-tuned with WARP-veRL for worst-case logical reasoning over path constraints</td>
    <td><strong>TO BE RELEASED</strong></td>
  </tr>
</table>

<h3 align="center">Key Enhancements over Qwen2.5</h3>

<ul align="center" style="display: inline-block; text-align: left;">
  <li>Customized reward models specifically designed for SMT-based equivalence checking</li>
  <li>Additional fine-tuning on the WARP-easy dataset targeting path-constraint reasoning</li>
  <li>Improved performance on tasks requiring worst-case logical inference</li>
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
    <td><strong>WARP-easy</strong></td>
    <td>Easy-mode dataset for WARP-1.0, focusing on path-constraint verification examples</td>
    <td><a href="https://huggingface.co/datasets/dannkoh/WARP-easy">HuggingFace Hub</a></td>
  </tr>
</table>

<hr />

<h2 id="installation" align="center">Installation</h2>

<h3 align="center">Prerequisites</h3>

<p>WARP-veRL builds on veRL's framework, so the simplest approach is to follow the official veRL installation instructions:</p>

<ol>
  <li>Follow the <a href="https://verl.readthedocs.io/en/latest/start/install.html">veRL installation guide</a> to set up all core dependencies</li>
  <li>Install our additional dependency:</li>
</ol>

```bash
git clone https://github.com/dannkoh/WARP-veRL.git
cd WARP-veRL
pip install -r requirements.txt
```

<p><em>Note: If you encounter any installation issues, please refer to the troubleshooting section in the <a href="https://verl.readthedocs.io">veRL documentation</a>.</em></p>

<hr />

<h2 id="project-structure" align="center">Project Structure</h2>

<pre>
WARP-veRL/
├── examples/              # Example scripts and notebooks demonstrating usage
├── verl/                  # Core veRL library files with custom modifications
│   ├── algorithms/        # Reinforcement learning algorithm implementations
│   ├── models/            # Model architectures and wrappers
│   ├── trainers/          # Training infrastructure code
│   ├── utils/             # Utility functions and helpers
│   │   ├── dataset/
│   │   │   └── rl_dataset.py   # Custom dataset implementation for use-case
│   │   └── reward_score/
│   │       └── custom.py       # Custom reward function logic (SMT-based)
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
    <td>SMT-based logical equivalence checking for path constraints</td>
  </tr>
  <tr>
    <td><code>/verl/utils/dataset/rl_dataset.py</code></td>
    <td>Dataset loader adapted for WARP-easy format</td>
    <td>Specialized path-constraint examples</td>
  </tr>
</table>

<hr />

<h2 id="usage" align="center">Usage</h2>

<h3>Example Scripts</h3>

<p>Check the <code>/examples</code> directory for more comprehensive usage examples:</p>

<ul>
  <li><code>examples/grpo_trainer/</code>: GRPO Training Examples</li>
  <li><code>examples/ppo_trainer/</code>: PPO Training Examples</li>
  <li><code>scripts/</code>: Utility scripts for this project</li>
</ul>

<h3>Using Published Models</h3>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the WARP model
model_name = "dannkoh/WARP-1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate text with the model
inputs = tokenizer("Check path constraints φ ⇒ ψ:", return_tensors="pt")
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

