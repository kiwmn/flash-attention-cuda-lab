# Optimization steps

Each series has independent, consecutive step numbers and includes its final kernel.
Additional branches state their base step explicitly.
The same final sources are available in [kernels](../kernels/README.md).

- [Standard attention on RTX 3090](standard/README.md)
- [Standard attention on A100](a100/README.md)
- [Causal attention on RTX 3090](causal/README.md)
- [Paged attention on RTX 3090](paged/README.md)

Pass the scenario and local step number to the Python interface to run a particular implementation.
