# DiT Implementation Issues

DiT = Diffusion Transformer (Peebles and Xie, 2023).

**These issues have been fixed.**

The model trains and samples correctly. But two design choices make it weaker than DiT (sections 1 and 2). Sections 3 to 5 are other issues that do not make the model weaker.

# Issues That Make the Model Weaker

## 1. The adaLN conditioning layer is linear

adaLN = adaptive Layer Normalization.

### My code (`DiffusionTransformerLayer`)

```python
self.mlp = MLP([dim, 6*dim], final_init=True)   # one nn.Linear, no activation
...
cond = self.mlp(c)   # c = time_embedding + y_embedding
```

### The problem

`MLP([dim, 6*dim])` is one `nn.Linear`. It has no activation. A linear map splits over a sum:

```
mlp(emb_t + emb_y) = A·emb_t + A·emb_y + b
```

Thus, each scale, shift, and gate value is:

- one part that only `t` controls, plus
- one part that only `y` controls.

The two parts only add. They cannot interact. The effect of the label `y` is the same at all times `t`.

### Why this is bad

The best effect of the label changes with `t`:

- Near `t = 0`, `x` is almost pure noise. The label must control the velocity a lot.
- Near `t = 1`, `x` already shows the digit. The label must control the velocity less.

The model can learn some interaction through `x`, because the rest of the network is nonlinear. But the conditioning path itself cannot learn it.

DiT also puts an MLP (Multi-Layer Perceptron) after the time frequency embedding. My model sends the Fourier features directly to the layers.

### Fix

Apply SiLU before the linear layer, as DiT does. Keep the zero init on the linear layer (adaLN-Zero).

```python
self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))
nn.init.zeros_(self.mlp[-1].weight)
nn.init.zeros_(self.mlp[-1].bias)
```

`SiLU(emb_t + emb_y)` does not split into a `t` part plus a `y` part. Thus `t` and `y` can interact.

## 2. The 1×1 convolution in `Depatchifier` has no effect

### My code (`Depatchifier`)

```python
self.linear = nn.Linear(dim, final_dim*patch_size*patch_size)
self.conv = nn.Conv2d(final_dim, c_out, 1, 1)
```

### The problem

There is no activation between `self.linear` and `self.conv`. A 1×1 convolution is a linear map on each pixel. Two linear maps in series make one linear map:

```
conv(linear(token)) = W·token + b      # W has shape (c_out*p*p, dim)
```

Thus:

- `final_dim` does not give more capacity. It only adds parameters.
- Each output pixel uses only the token of its own patch. Pixels at a patch edge do not use the pixels of the next patch.
- The model can show sharp lines at patch borders (patch artifacts).

### Fix

Use a 3×3 convolution with padding 1:

```python
self.conv = nn.Conv2d(final_dim, c_out, 3, padding=1)
```

Each output pixel now uses a 3×3 area, so pixels at a patch edge also use pixels from the next patch. This makes the patch borders smoother.

# Other Issues

## 3. The head order in `MHA` is not standard

MHA = Multi-Head Attention.

### My code (`MHA`)

```python
q = q.reshape(b, n, h, d//h).permute(0, 2, 1, 3)    # split: feature order (h, d//h)
...
x = x.reshape(b, h, n, d//h).permute(0, 2, 3, 1).reshape(b, n, d)   # join: feature order (d//h, h)
```

### The problem

The code splits the features in the order `(h, d//h)`, but it joins them in the order `(d//h, h)`. Thus the features go to `self.ff` in a different order.

The output is still correct, because `self.ff` is a learned linear layer. It learns the permutation. But:

- The code is hard to read and check.
- The weights are not compatible with `nn.MultiheadAttention`.

### Fix

Use the same order to join as to split:

```python
x = x.reshape(b, h, n, d//h).permute(0, 2, 1, 3).reshape(b, n, d)
```

## 4. One `LayerNorm` for both sublayers

### My code (`DiffusionTransformerLayer`)

```python
self.norm = nn.LayerNorm(dim)
...
x = self.norm(x)   # before attention
...
x = self.norm(x)   # before feedforward
```

### The problem

- The attention sublayer and the feedforward sublayer share one learned scale and shift.
- These learned values are not necessary. adaLN already gives a scale and a shift for each sample.

### Fix

Use two norms without learned values, as DiT does:

```python
self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
```

## 5. `x.clone()` for the residual is not necessary

### My code (`DiffusionTransformerLayer`)

```python
x_res = x.clone()
x = self.norm(x)
```

### The problem

`self.norm(x)` returns a new tensor. It does not change `x`. Thus `x_res` stays correct without a copy. The copy uses more memory and time.

### Fix

```python
x_res = x
x = self.norm1(x)
```
