# Research domains

This directory is organized by research ownership rather than by file type.

| Directory | Owner |
| --- | --- |
| [`weight-vae/`](weight-vae/) | Weight-distribution VAE, CELO, latent optimization, and reparameterization research |
| [`cfm/`](cfm/) | Categorical Flow Maps, Semicat/BD3LMS baselines, TinyStories evaluation, and paper |
| [`lsdl/`](lsdl/) | LSDL/LFQ notebook work and its regression evidence |
| [`unclassified/`](unclassified/) | Independent or unclear work with no demonstrated dependency on the three main programs |
| [`shared/`](shared/) | Cross-project scripts, tests, docs, and configuration |

All canonical directories except the environment-owned `workflow_data` mount
live below `projects/`. Tightly coupled Weight VAE components move together in
`weight-vae/workspace/`, preserving their relative layout. Former root paths
are compatibility links. Shared storage lives under `shared/storage/`. The
authoritative mapping is [`manifest.json`](manifest.json).
