# Probe Preconditioner Dump

This note dumps the current implementation of the explicit probe-metric preconditioner used in the E4 debug / direction-match experiments.

Relevant code:

- `post_train_research/loss_landscape_analysis/flow_preconditioning/probe_geometry.py`
- `post_train_research/loss_landscape_analysis/flow_preconditioning/e4_debug.py`
- `post_train_research/loss_landscape_analysis/flow_preconditioning/e4_direction_match.py`

## Probe Definition

For E4, the probe is a scale-normalized residual plus an optional damped parameter component:

```python
class ResidualThetaProbe:
    def __call__(self, theta):
        residual = residual_fn(theta) / residual_rms
        damped = rho * theta / theta_rms
        return torch.cat([residual.reshape(-1), damped.reshape(-1)], dim=0)
```

Mathematically:

```text
P(theta) = [ r(theta) / s_r, rho * theta / s_theta ]
```

where:

```text
r(theta) = h_theta(X_probe) - y_probe
s_r      = RMS residual over the flow-training pool
s_theta  = RMS theta over the flow-training pool
```

For `rho = 0`, the probe is residual-only.

## Probe Metric

At a parameter vector `theta`, compute the exact Jacobian of the probe:

```text
J_P(theta) = d P(theta) / d theta
```

Then the pullback metric is:

```text
G_P(theta) = J_P(theta)^T J_P(theta)
```

Code path:

```python
jacobian = probe_jacobians_for_theta(probe, theta.reshape(1, -1), create_graph=False)
G = jacobian.squeeze(0).T @ jacobian.squeeze(0)
```

For batched direction targets:

```python
jacobians = probe_jacobians_for_theta(probe, theta_batch, create_graph=False)
probe_metric, trace_g, _ = metric_tensors_from_jacobians(jacobians)
```

Shapes for E4:

```text
theta:        [25]
J_P:          [probe_dim, 25]
G_P:          [25, 25]
grad:         [25]
p_metric:     [25]
```

## Preconditioned Direction

The explicit probe-metric preconditioner direction is:

```text
p_metric = (G_P(theta) + lambda I)^(-1) g
```

where:

```text
g = grad_theta L_train(theta)
```

The optimizer step uses this as a descent direction:

```text
theta_next = theta - lr * p_metric
```

So `p_metric` is not the signed update itself; the update is `-lr * p_metric`.

## Damping Variants

There are two damping variants in the repo.

### Fixed Damping Debug Path

Used by `run_probe_metric_curve` and older alignment diagnostics in `e4_debug.py`:

```python
p_metric = torch.linalg.solve(G_P + damping * I, grad)
theta = theta - lr * p_metric
```

This is configured as a literal scalar damping value.

### Scaled Damping Direction-Match Path

Used by `compute_direction_targets` and `run_scaled_probe_metric_curve` in `e4_direction_match.py`:

```python
damping = metric_alpha * trace(G_P) / dim + eps
p_metric = torch.linalg.solve(G_P + damping * I, grad)
```

This is the current preferred target for NF direction matching because damping scales with the local metric magnitude.

## Batched Target Precompute

`compute_direction_targets(...)` precomputes direction targets independent of NF parameters:

```python
theta = theta_samples.detach()
grad = _train_gradients(train_loss_fn, theta)
jacobians = probe_jacobians_for_theta(probe, theta, create_graph=False).detach()
probe_metric, trace_g, _ = metric_tensors_from_jacobians(jacobians)

dim = theta.shape[1]
damping = metric_alpha * trace_g / dim + eps
eye = torch.eye(dim).expand(theta.shape[0], dim, dim)

metric_preconditioner = torch.linalg.solve(
    probe_metric + damping.reshape(-1, 1, 1) * eye,
    eye,
)
p_metric = torch.bmm(metric_preconditioner, grad.reshape(-1, dim, 1)).reshape(-1, dim)
```

Saved target tensors:

```text
theta
grad
probe_metric = G_P
metric_preconditioner = (G_P + lambda I)^(-1)
p_metric
damping
rho
metric_alpha
split
```

## NF Preconditioner Compared Against Probe Preconditioner

The NF-induced local preconditioner is:

```text
p_NF = J_inv J_inv^T g
```

where:

```text
u = flow(theta)
J_inv = d flow^{-1}(u) / d u
```

Code sketch from `e4_debug.py`:

```python
u = flow(theta[None])[0].reshape(-1)

def inverse_at_u(u):
    return flow.inverse(u.unsqueeze(0))[0].squeeze(0)

J_inv = exact_jacobians(inverse_at_u, u.reshape(1, -1), create_graph=False).squeeze(0)
p_nf = J_inv @ (J_inv.T @ grad)
```

Direction-match training compares `p_NF` against the precomputed `p_metric`.

## Diagnostics

The alignment diagnostics report:

```text
cos(p_NF, p_metric)
||p_metric|| / ||p_NF||
cos(g, p_metric)
cos(g, p_NF)
```

For actual probe-metric optimizer trajectories, the self-consistency check is:

```text
cos(theta_t - theta_{t+1}, p_metric) ~= 1
```

because:

```text
theta_t - theta_{t+1} = lr * p_metric
```

