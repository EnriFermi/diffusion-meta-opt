# Loss Landscape Metrics

This document defines the metrics used for 2D loss landscape slices in
`mnist_one_layer_vit_loss_landscape_latent_vs_raw.ipynb`.

Let the loss at a point in the 2D slice be:

$$
L(\alpha, \beta)
$$

The base point is:

$$
L_0 = L(0, 0)
$$

The loss increase relative to the base point is:

$$
\Delta L(\alpha, \beta)
=
L(\alpha, \beta) - L_0
$$

For a radius $\rho$, define the disk:

$$
D_\rho
=
\left\{
(\alpha, \beta)
:
\alpha^2 + \beta^2 \le \rho^2
\right\}
$$

## Sharpness: $S_\rho$

The sharpness metric is the maximum loss increase inside the disk:

$$
S_\rho
=
\max_{(\alpha,\beta) \in D_\rho}
\left[
L(\alpha,\beta) - L(0,0)
\right]
$$

Equivalently:

$$
S_\rho
=
\max_{(\alpha,\beta) \in D_\rho}
\Delta L(\alpha,\beta)
$$

## Mean Loss Increase: $M_\rho$

The mean loss increase metric is the average loss increase inside the disk:

$$
M_\rho
=
\frac{1}{|D_\rho|}
\int_{D_\rho}
\left[
L(\alpha,\beta) - L(0,0)
\right]
d\alpha\,d\beta
$$

Equivalently:

$$
M_\rho
=
\frac{1}{|D_\rho|}
\int_{D_\rho}
\Delta L(\alpha,\beta)
d\alpha\,d\beta
$$

In the notebook this is estimated on a finite grid, not by continuous
integration.

## Good Area: $A_{\tau,\rho}$

For a tolerance $\tau$, the good area is the area inside the disk where the
loss is at most $\tau$ above the base loss:

$$
A_{\tau,\rho}
=
\operatorname{Area}
\left(
\left\{
(\alpha,\beta) \in D_\rho
:
L(\alpha,\beta) \le L(0,0) + \tau
\right\}
\right)
$$

Equivalently:

$$
A_{\tau,\rho}
=
\operatorname{Area}
\left(
\left\{
(\alpha,\beta) \in D_\rho
:
\Delta L(\alpha,\beta) \le \tau
\right\}
\right)
$$

## Good Area Fraction: $A^{\mathrm{frac}}_{\tau,\rho}$

The good area fraction is the fraction of the disk where the loss is at most
$\tau$ above the base loss:

$$
A^{\mathrm{frac}}_{\tau,\rho}
=
\frac{
\operatorname{Area}
\left(
\left\{
(\alpha,\beta) \in D_\rho
:
L(\alpha,\beta) \le L(0,0) + \tau
\right\}
\right)
}{
\operatorname{Area}(D_\rho)
}
$$

Equivalently:

$$
A^{\mathrm{frac}}_{\tau,\rho}
=
\frac{
\operatorname{Area}
\left(
\left\{
(\alpha,\beta) \in D_\rho
:
\Delta L(\alpha,\beta) \le \tau
\right\}
\right)
}{
\operatorname{Area}(D_\rho)
}
$$

## Discrete Grid Version

The notebook evaluates the landscape on a finite grid:

$$
\left\{
(\alpha_i, \beta_j)
\right\}_{i,j}
$$

The grid disk is:

$$
D_\rho^{\mathrm{grid}}
=
\left\{
(\alpha_i,\beta_j)
:
\alpha_i^2 + \beta_j^2 \le \rho^2
\right\}
$$

The discrete sharpness estimate is:

$$
S_\rho
=
\max_{(\alpha_i,\beta_j) \in D_\rho^{\mathrm{grid}}}
\Delta L(\alpha_i,\beta_j)
$$

The discrete mean estimate is:

$$
M_\rho
=
\frac{1}{|D_\rho^{\mathrm{grid}}|}
\sum_{(\alpha_i,\beta_j) \in D_\rho^{\mathrm{grid}}}
\Delta L(\alpha_i,\beta_j)
$$

The discrete good area fraction is:

$$
A^{\mathrm{frac}}_{\tau,\rho}
=
\frac{
\sum_{(\alpha_i,\beta_j) \in D_\rho^{\mathrm{grid}}}
\mathbf{1}
\left[
L(\alpha_i,\beta_j) \le L(0,0) + \tau
\right]
}{
|D_\rho^{\mathrm{grid}}|
}
$$

Equivalently:

$$
A^{\mathrm{frac}}_{\tau,\rho}
=
\frac{
\sum_{(\alpha_i,\beta_j) \in D_\rho^{\mathrm{grid}}}
\mathbf{1}
\left[
\Delta L(\alpha_i,\beta_j) \le \tau
\right]
}{
|D_\rho^{\mathrm{grid}}|
}
$$

The discrete area estimate uses the grid cell area:

$$
\Delta A = \Delta\alpha \Delta\beta
$$

Then:

$$
A_{\tau,\rho}
=
\Delta\alpha \Delta\beta
\sum_{(\alpha_i,\beta_j) \in D_\rho^{\mathrm{grid}}}
\mathbf{1}
\left[
L(\alpha_i,\beta_j) \le L(0,0) + \tau
\right]
$$

Equivalently:

$$
A_{\tau,\rho}
=
\Delta\alpha \Delta\beta
\sum_{(\alpha_i,\beta_j) \in D_\rho^{\mathrm{grid}}}
\mathbf{1}
\left[
\Delta L(\alpha_i,\beta_j) \le \tau
\right]
$$

## Interpretation

- $S_\rho$ measures the worst loss increase inside radius $\rho$.
- $M_\rho$ measures the average loss increase inside radius $\rho$.
- $A_{\tau,\rho}$ measures the absolute area of points that remain within
  $\tau$ of the base loss.
- $A^{\mathrm{frac}}_{\tau,\rho}$ measures the fraction of the disk that
  remains within $\tau$ of the base loss.

