"""Bloque Transformer en numpy puro: forward + backward a mano + comprobación numérica.

Sirve para entender (y verificar) las fórmulas de la guía sin autograd.
Uso:  python docs/transformer_numpy.py
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ── primitivas ───────────────────────────────────────────────────────────────
def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def layernorm_fwd(x, g, b, eps=1e-5):
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    xh = (x - mu) / np.sqrt(var + eps)
    return g * xh + b, (xh, np.sqrt(var + eps), g)


def layernorm_bwd(dy, cache):
    xh, sd, g = cache
    dg, db = (dy * xh).sum(0), dy.sum(0)
    dxh = dy * g
    dx = (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True)) / sd
    return dx, dg, db


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_grad(x):
    u = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    t = np.tanh(u)
    du = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * du


# ── atención (una secuencia, h cabezas) ──────────────────────────────────────
def mhsa_fwd(x, Wqkv, bqkv, Wo, bo, h):
    n, d = x.shape
    dh = d // h
    qkv = x @ Wqkv + bqkv                               # (n, 3d)
    q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(n, h, dh).transpose(1, 0, 2) for i in range(3))
    A = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dh))  # (h, n, n)
    o = (A @ v).transpose(1, 0, 2).reshape(n, d)         # concat de cabezas
    return o @ Wo + bo, (x, q, k, v, A, o, dh)


def mhsa_bwd(dy, cache, Wqkv, Wo, h):
    x, q, k, v, A, o, dh = cache
    n, d = x.shape
    dWo, dbo = o.T @ dy, dy.sum(0)
    do = (dy @ Wo.T).reshape(n, h, dh).transpose(1, 0, 2)
    dv = A.transpose(0, 2, 1) @ do                       # dV = Aᵀ dO
    dA = do @ v.transpose(0, 2, 1)                       # dA = dO Vᵀ
    dS = A * (dA - (dA * A).sum(-1, keepdims=True))      # Jacobiano del softmax
    dq = dS @ k / np.sqrt(dh)
    dk = dS.transpose(0, 2, 1) @ q / np.sqrt(dh)
    dqkv = np.concatenate([t.transpose(1, 0, 2).reshape(n, d) for t in (dq, dk, dv)], axis=1)
    return dqkv @ Wqkv.T, x.T @ dqkv, dqkv.sum(0), dWo, dbo


# ── bloque pre-LN completo ───────────────────────────────────────────────────
def init(d, h, r=4):
    s = 0.1
    return dict(g1=np.ones(d), b1=np.zeros(d), Wqkv=rng.normal(0, s, (d, 3 * d)), bqkv=rng.normal(0, s, 3 * d),
                Wo=rng.normal(0, s, (d, d)), bo=rng.normal(0, s, d), g2=np.ones(d), b2=np.zeros(d),
                W1=rng.normal(0, s, (d, r * d)), c1=rng.normal(0, s, r * d),
                W2=rng.normal(0, s, (r * d, d)), c2=rng.normal(0, s, d), h=h)


def block_fwd(x, P):
    xn, c1 = layernorm_fwd(x, P["g1"], P["b1"])
    a, ca = mhsa_fwd(xn, P["Wqkv"], P["bqkv"], P["Wo"], P["bo"], P["h"])
    x1 = x + a
    xn2, c2 = layernorm_fwd(x1, P["g2"], P["b2"])
    z = xn2 @ P["W1"] + P["c1"]
    m = gelu(z) @ P["W2"] + P["c2"]
    return x1 + m, (c1, ca, c2, xn2, z)


def block_bwd(dy, cache, P):
    c1, ca, c2, xn2, z = cache
    G = {}
    dm = dy
    G["W2"], G["c2"] = gelu(z).T @ dm, dm.sum(0)
    dz = (dm @ P["W2"].T) * gelu_grad(z)
    G["W1"], G["c1"] = xn2.T @ dz, dz.sum(0)
    dx1, G["g2"], G["b2"] = layernorm_bwd(dz @ P["W1"].T, c2)
    dx1 = dx1 + dy                                       # atajo residual
    dxn, G["Wqkv"], G["bqkv"], G["Wo"], G["bo"] = mhsa_bwd(dx1, ca, P["Wqkv"], P["Wo"], P["h"])
    dx, G["g1"], G["b1"] = layernorm_bwd(dxn, c1)
    return dx + dx1, G                                   # atajo residual


# ── comprobación numérica de gradientes ──────────────────────────────────────
def main():
    n, d, h = 7, 8, 2
    P, x = init(d, h), rng.normal(size=(n, d))
    W = rng.normal(size=(n, d))                          # L = sum(W * y), dL/dy = W
    y, cache = block_fwd(x, P)
    dx, G = block_bwd(W, cache, P)

    def loss(): return (block_fwd(x, P)[0] * W).sum()

    worst = 0.0
    eps = 1e-6
    for name in ["Wqkv", "Wo", "W1", "W2", "g1", "b2", "bqkv"]:
        num = np.zeros_like(P[name])
        for idx in np.ndindex(*P[name].shape):
            old = P[name][idx]
            P[name][idx] = old + eps; lp = loss()
            P[name][idx] = old - eps; lm = loss()
            P[name][idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        err = np.abs(num - G[name]).max() / (np.abs(num).max() + 1e-12)
        worst = max(worst, err)
        print(f"grad {name:5s}  error relativo máx = {err:.2e}")
    numx = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + eps; lp = loss()
        x[idx] = old - eps; lm = loss()
        x[idx] = old
        numx[idx] = (lp - lm) / (2 * eps)
    errx = np.abs(numx - dx).max() / np.abs(numx).max()
    print(f"grad x      error relativo máx = {errx:.2e}")
    assert max(worst, errx) < 1e-5, "los gradientes analíticos no coinciden"
    print("OK: backward analítico == diferencias finitas")


if __name__ == "__main__":
    main()
