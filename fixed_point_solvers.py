import jax.numpy as jnp


def anderson_solver(f, z_init, m=5, lam=1e-4, max_iter=25, tol=1e-2, beta=1.0):
    """Anderson acceleration for fixed points


    More information about the Anderson acceleration on:

    https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf

    """
    batch_size, d, h = z_init.shape
    X = jnp.zeros(shape=(batch_size, m, d * h), dtype=z_init.dtype)
    F = jnp.zeros(shape=(batch_size, m, d * h), dtype=z_init.dtype)
    X = X.at[:, 0].set(z_init.reshape(batch_size, -1))
    F = F.at[:, 0].set(f(z_init).reshape(batch_size, -1))

    X = X.at[:, 1].set(F[:, 0])
    F = F.at[:, 1].set(f(F[:, 0].reshape(batch_size, d, h)).reshape(batch_size, -1))

    H = jnp.zeros(shape=(batch_size, m + 1, m + 1), dtype=z_init.dtype)
    H = H.at[:, 0, 1:].set(1)
    H = H.at[:, 1:, 0].set(1)
    y = jnp.zeros(shape=(batch_size, m + 1, 1), dtype=z_init.dtype)
    y = y.at[:, 0].set(1)
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        GTG = jnp.matmul(G, jnp.transpose(G, (0, 2, 1))) + lam * jnp.eye(n, dtype=z_init.dtype)[None]
        H = H.at[:, 1:n + 1, 1:n + 1].set(GTG)
        alpha = jnp.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]
        X = X.at[:, k % m].set(
            beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0])
        F = F.at[:, k % m].set(f(X[:, k % m].reshape(batch_size, d, h)).reshape(batch_size, -1))
        res = jnp.linalg.norm(F[:, k % m] - X[:, k % m]).item() / (1e-5 + jnp.linalg.norm(F[:, k % m]).item())
        if res < tol:
            break

    return X[:, k % m].reshape(batch_size, d, h)


def fwd_solver(f, z_init):
    """Forward solver for fixed points"""
    z_prev, z = z_init, f(z_init)
    while jnp.linalg.norm(z_prev - z) < 1e-4:
        z_prev, z = z, f(z_init)
    return z
