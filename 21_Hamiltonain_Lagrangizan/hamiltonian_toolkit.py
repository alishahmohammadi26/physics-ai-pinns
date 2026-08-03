#!/usr/bin/env python3
"""
================================================================================
 HAMILTONIAN TOOLKIT  -  from classical mechanics to Hamiltonian Neural Networks
================================================================================

A single self-contained file. Only numpy + matplotlib are required
(scipy optional, used for one high-accuracy double-pendulum integration).

Run everything:            python hamiltonian_toolkit.py
Run one demo:              python hamiltonian_toolkit.py 4

  1  Hamilton's equations reproduce Newton              (symbolic sanity check)
  2  Integrator shootout: Euler vs symplectic vs RK4    (why structure > accuracy)
  3  Liouville's theorem: phase-space area is frozen
  4  Double pendulum: chaotic behaviour, conserved rules
  5  Hamiltonian Neural Network, hand-derived in NumPy  (the main event)
  6  Hamiltonian Monte Carlo in 40 lines                (same maths, no physics)

--------------------------------------------------------------------------------
THE ONE EQUATION EVERYTHING RESTS ON

    x = (q, p),      dx/dt = J grad H(x),      J = [[0, 1], [-1, 0]]

J is antisymmetric, so  dH/dt = grad(H) . J grad(H) = 0  for ANY H whatsoever.
Energy conservation is therefore a property of the STRUCTURE, not of the model.
That single fact is why HNNs work.
--------------------------------------------------------------------------------
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
J2 = np.array([[0.0, 1.0], [-1.0, 0.0]])          # the symplectic matrix

C = dict(bg="#0d1117", fg="#e6edf3", grid="#30363d",
         blue="#58a6ff", pink="#f778ba", green="#3fb950", amber="#d29922")
plt.rcParams.update({
    "figure.facecolor": C["bg"], "axes.facecolor": C["bg"],
    "savefig.facecolor": C["bg"], "text.color": C["fg"],
    "axes.labelcolor": C["fg"], "xtick.color": C["fg"], "ytick.color": C["fg"],
    "axes.edgecolor": C["grid"], "grid.color": C["grid"],
    "figure.dpi": 120, "font.size": 9,
})


# =============================================================================
# DEMO 1 - Hamilton's equations give back Newton
# =============================================================================
def demo1():
    """Differentiate H numerically and check it reproduces F = ma."""
    print("\n--- 1. Hamilton's equations vs Newton -------------------------")
    m, k = 1.3, 4.7

    def H(x):                       # mass-spring total energy
        q, p = x
        return p**2 / (2*m) + 0.5*k*q**2

    def grad(f, x, h=1e-6):
        g = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x); e[i] = h
            g[i] = (f(x+e) - f(x-e)) / (2*h)
        return g

    x = np.array([0.8, 0.35])
    qdot, pdot = J2 @ grad(H, x)
    print(f"  Hamilton:  qdot = {qdot:+.6f}   pdot = {pdot:+.6f}")
    print(f"  Newton  :  p/m  = {x[1]/m:+.6f}   -kq  = {-k*x[0]:+.6f}")
    print("  -> identical. Newton's laws are a COROLLARY of one scalar function.")


# =============================================================================
# DEMO 2 - Integrator shootout
# =============================================================================
def euler(x, h, dV):
    q, p = x
    return np.array([q + h*p, p - h*dV(q)])          # both use OLD values

def symplectic_euler(x, h, dV):
    q, p = x
    p = p - h*dV(q)                                   # kick first...
    return np.array([q + h*p, p])                     # ...drift with NEW p

def leapfrog(x, h, dV):
    q, p = x
    p = p - 0.5*h*dV(q)
    q = q + h*p
    p = p - 0.5*h*dV(q)
    return np.array([q, p])

def rk4(x, h, dV):
    f = lambda s: np.array([s[1], -dV(s[0])])
    k1 = f(x); k2 = f(x+h/2*k1); k3 = f(x+h/2*k2); k4 = f(x+h*k3)
    return x + h/6*(k1 + 2*k2 + 2*k3 + k4)


def demo2(h=0.22, n=1400):
    print("\n--- 2. Integrator shootout ------------------------------------")
    dV = lambda q: q                                   # H = (q^2 + p^2)/2
    E = lambda x: 0.5*(x[0]**2 + x[1]**2)
    methods = [("Euler", euler, C["pink"]), ("Symplectic Euler", symplectic_euler, C["green"]),
               ("Leapfrog", leapfrog, C["blue"]), ("RK4", rk4, C["amber"])]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, step, col in methods:
        x = np.array([1.0, 0.0]); tr = np.zeros((n, 2))
        for i in range(n):
            tr[i] = x; x = step(x, h, dV)
        e = 0.5*(tr[:, 0]**2 + tr[:, 1]**2)
        a1.plot(tr[:, 0], tr[:, 1], color=col, lw=1, label=name)
        a2.plot(np.arange(n)*h, e, color=col, lw=1.3, label=name)
        print(f"  {name:<18} energy error after {n} steps: "
              f"{abs(e[-1]-e[0])/e[0]*100:8.2f}%")

    a1.set_aspect("equal"); a1.set_xlabel("q"); a1.set_ylabel("p")
    a1.set_title("phase space"); a1.legend(fontsize=7)
    a2.set_xlabel("time"); a2.set_ylabel("energy"); a2.set_ylim(0, 1.5)
    a2.set_title("energy drift"); a2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("demo2_integrators.png"); plt.close(fig)
    print("  -> RK4 is the most ACCURATE per step yet still drifts.")
    print("     Symplectic methods are less accurate but stay BOUNDED forever.")
    print("  saved demo2_integrators.png")


# =============================================================================
# DEMO 3 - Liouville's theorem
# =============================================================================
def demo3():
    print("\n--- 3. Liouville: phase-space area is incompressible -----------")
    g = 9.81
    f = lambda s: np.stack([s[:, 1], -g*np.sin(s[:, 0])], 1)
    th = np.linspace(0, 2*np.pi, 400)
    s = np.stack([0.15*np.cos(th) - 1.6, 0.55*np.sin(th) + 1.0], 1)

    def area(c):                                       # shoelace formula
        x, y = c[:, 0], c[:, 1]
        return 0.5*abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    A0, h = area(s), 0.004
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for step in range(3001):
        if step % 750 == 0:
            ax.plot(np.append(s[:, 0], s[0, 0]), np.append(s[:, 1], s[0, 1]),
                    lw=1.4, label=f"t={step*h:.1f}s  area={area(s):.5f}")
            print(f"  t = {step*h:5.1f}s   area = {area(s):.6f}   "
                  f"(drift {abs(area(s)-A0)/A0*100:.4f}%)")
        k1 = f(s); k2 = f(s+h/2*k1); k3 = f(s+h/2*k2); k4 = f(s+h*k3)
        s = s + h/6*(k1 + 2*k2 + 2*k3 + k4)
    ax.set_xlabel("q"); ax.set_ylabel("p"); ax.legend(fontsize=7)
    ax.set_title("the blob deforms, the area does not")
    fig.tight_layout(); fig.savefig("demo3_liouville.png"); plt.close(fig)
    print("  -> This is why Hamiltonian systems can NEVER have attractors,")
    print("     and why HMC gets a Jacobian determinant of exactly 1 for free.")
    print("  saved demo3_liouville.png")


# =============================================================================
# DEMO 4 - Double pendulum: chaotic behaviour, conserved rules
# =============================================================================
def demo4():
    print("\n--- 4. Double pendulum: chaos WITH conservation ----------------")
    m1 = m2 = l1 = l2 = 1.0
    g = 9.81

    def rhs(y):
        t1, t2, p1, p2 = y
        d = t1 - t2
        den = l1*l2*(m1 + m2*np.sin(d)**2)
        dt1 = (l2*p1 - l1*p2*np.cos(d)) / (l1*den)
        dt2 = (l1*(m1+m2)*p2 - l2*m2*p1*np.cos(d)) / (l2*m2*den)
        C1 = p1*p2*np.sin(d)/den
        C2 = (l2**2*m2*p1**2 + l1**2*(m1+m2)*p2**2
              - 2*l1*l2*m2*p1*p2*np.cos(d)) * np.sin(2*d) / (2*den**2)
        return np.array([dt1, dt2,
                         -(m1+m2)*g*l1*np.sin(t1) - C1 + C2,
                         -m2*g*l2*np.sin(t2) + C1 - C2])

    def energy(y):
        t1, t2, p1, p2 = y
        d = t1 - t2
        den = l1*l2*(m1 + m2*np.sin(d)**2)
        dt1 = (l2*p1 - l1*p2*np.cos(d)) / (l1*den)
        dt2 = (l1*(m1+m2)*p2 - l2*m2*p1*np.cos(d)) / (l2*m2*den)
        T = 0.5*m1*(l1*dt1)**2 + 0.5*m2*((l1*dt1)**2 + (l2*dt2)**2
            + 2*l1*l2*dt1*dt2*np.cos(d))
        V = -(m1+m2)*g*l1*np.cos(t1) - m2*g*l2*np.cos(t2)
        return T + V

    h, n = 1e-4, 200000
    ys = [np.array([2.0, 2.2, 0.0, 0.0]),
          np.array([2.0, 2.2 + 1e-9, 0.0, 0.0])]      # differ by 1 nanoradian
    E0 = energy(ys[0])
    seps, Es = [], []
    for i in range(n):
        for j in (0, 1):
            y = ys[j]
            k1 = rhs(y); k2 = rhs(y+h/2*k1); k3 = rhs(y+h/2*k2); k4 = rhs(y+h*k3)
            ys[j] = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        if i % 300 == 0:
            seps.append(np.linalg.norm(ys[0]-ys[1])); Es.append(energy(ys[0]))
    seps, Es = np.array(seps), np.array(Es)
    t = np.arange(len(seps))*300*h

    print(f"  initial separation : 1.0e-09 rad")
    print(f"  final separation   : {seps[-1]:.3e} rad   "
          f"<- grew {seps[-1]/1e-9:.1e}x : chaos")
    print(f"  energy drift       : {abs(Es[-1]-E0):.3e} J  "
          f"(relative {abs(Es[-1]-E0)/abs(E0):.1e}) <- rules held")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.semilogy(t, seps, color=C["pink"])
    a1.set_xlabel("time (s)"); a1.set_ylabel("separation")
    a1.set_title("behaviour: diverges exponentially")
    a2.plot(t, Es - E0, color=C["green"])
    a2.set_xlabel("time (s)"); a2.set_ylabel("energy drift (J)")
    a2.set_title("rules: energy conserved")
    fig.tight_layout(); fig.savefig("demo4_double.png"); plt.close(fig)
    print("  -> Unpredictable and perfectly lawful, simultaneously.")
    print("     Chaos limits TRAJECTORY forecasting, never the CONSTRAINTS.")
    print("  saved demo4_double.png")


# =============================================================================
# DEMO 5 - Hamiltonian Neural Network, from scratch
# =============================================================================
"""
Architecture:   z = W1 x + b1,   h = tanh z,   H = w2 . h + b2   (SCALAR out)

    s_k       = w2_k * (1 - h_k^2)
    grad_x H  = W1^T s
    f_pred    = J grad_x H

Because the OUTPUT already contains a derivative, the weight gradients need
second derivatives. Derived by hand (no autodiff anywhere in this file):

    u   = J^T dL/df                          dL/d(grad H)
    c_k = sum_j u_j W1[k,j]                  dL/ds_k
    dz_k = c_k * w2_k * (-2 h_k (1 - h_k^2)) dL/dz_k    <- the 2nd-deriv term

    dL/dW1[k,i] = u_i s_k + dz_k x_i         (TWO paths into W1)
    dL/db1_k    = dz_k
    dL/dw2_k    = c_k (1 - h_k^2)
    dL/db2      = 0                          (energy offset is unobservable)
"""

def true_f(X):                                   # ideal mass-spring
    return np.stack([X[:, 1], -X[:, 0]], 1)


def make_data(n=1500, noise=0.05):
    r = rng.uniform(0.6, 2.1, n); th = rng.uniform(0, 2*np.pi, n)
    X = np.stack([r*np.cos(th), r*np.sin(th)], 1)
    return X, true_f(X) + rng.normal(0, noise, (n, 2))


def init(m=200, dout=1):
    return dict(W1=rng.normal(0, 1.0, (m, 2)), b1=np.zeros(m),
                W2=rng.normal(0, np.sqrt(1/m), (dout, m)), b2=np.zeros(dout))


def hnn_forward(P, X):
    Z = X @ P["W1"].T + P["b1"]
    Hh = np.tanh(Z); Hp = 1 - Hh**2
    w2 = P["W2"][0]
    S = Hp * w2
    G = S @ P["W1"]                              # grad_x H
    return G @ J2.T, dict(Hh=Hh, Hp=Hp, S=S, w2=w2)


def hnn_scalar(P, X):                            # the learned H itself
    return (np.tanh(X @ P["W1"].T + P["b1"]) @ P["W2"].T)[:, 0]


def hnn_grads(P, X, Y):
    F, c = hnn_forward(P, X)
    dF = 2*(F - Y) / (len(X)*2)
    U = dF @ J2
    Cc = U @ P["W1"].T
    Dz = Cc * c["w2"] * (-2*c["Hh"]*c["Hp"])
    return dict(W1=c["S"].T @ U + Dz.T @ X, b1=Dz.sum(0),
                W2=(Cc * c["Hp"]).sum(0)[None, :], b2=np.zeros(1)), \
           float(((F - Y)**2).mean())


def base_forward(P, X):
    Hh = np.tanh(X @ P["W1"].T + P["b1"])
    return Hh @ P["W2"].T + P["b2"], Hh


def base_grads(P, X, Y):
    F, Hh = base_forward(P, X)
    dF = 2*(F - Y) / (len(X)*2)
    Dz = (dF @ P["W2"]) * (1 - Hh**2)
    return dict(W1=Dz.T @ X, b1=Dz.sum(0), W2=dF.T @ Hh, b2=dF.sum(0)), \
           float(((F - Y)**2).mean())


def train(P, gfn, X, Y, steps=6000, lr=3e-3, bs=256):
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(w) for k, w in P.items()}
    hist = []
    for t in range(1, steps+1):
        i = rng.integers(0, len(X), bs)
        g, loss = gfn(P, X[i], Y[i])
        for k in P:
            m[k] = 0.9*m[k] + 0.1*g[k]
            v[k] = 0.999*v[k] + 0.001*g[k]**2
            P[k] -= lr*(m[k]/(1-0.9**t))/(np.sqrt(v[k]/(1-0.999**t)) + 1e-8)
        hist.append(loss)
    return P, np.array(hist)


def rollout(fn, x0, steps=1400, h=0.05):
    x = np.array(x0, float)[None, :]; out = np.zeros((steps, 2))
    for i in range(steps):
        out[i] = x[0]
        k1 = fn(x); k2 = fn(x+h/2*k1); k3 = fn(x+h/2*k2); k4 = fn(x+h*k3)
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return out


def check_gradients():
    """Finite-difference check of the hand-derived second-derivative gradients."""
    P = init(m=6); X, Y = make_data(20)
    g, _ = hnn_grads(P, X, Y)
    worst = 0.0
    for k in ("W1", "b1", "W2"):
        flat = P[k].ravel(); gf = g[k].ravel()
        for i in rng.integers(0, len(flat), min(8, len(flat))):
            o = flat[i]; e = 1e-6
            flat[i] = o + e; lp = ((hnn_forward(P, X)[0] - Y)**2).mean()
            flat[i] = o - e; lm = ((hnn_forward(P, X)[0] - Y)**2).mean()
            flat[i] = o
            worst = max(worst, abs((lp-lm)/(2*e) - gf[i]))
    return worst


def demo5():
    print("\n--- 5. Hamiltonian Neural Network (pure NumPy) -----------------")
    err = check_gradients()
    print(f"  gradient check vs finite differences: max abs error {err:.2e}")

    X, Y = make_data()
    Pb, hb = train(init(dout=2), base_grads, X, Y)
    Ph, hh = train(init(dout=1), hnn_grads, X, Y)
    print(f"  final train loss   baseline {hb[-500:].mean():.5f}"
          f"   HNN {hh[-500:].mean():.5f}   <- INDISTINGUISHABLE")

    Rt = rollout(true_f, [1.6, 0.0])
    Rb = rollout(lambda x: base_forward(Pb, x)[0], [1.6, 0.0])
    Rh = rollout(lambda x: hnn_forward(Ph, x)[0], [1.6, 0.0])
    E = lambda R: 0.5*(R[:, 0]**2 + R[:, 1]**2)
    print(f"  energy after long rollout:  baseline "
          f"{abs(E(Rb)[-1]-E(Rb)[0])/E(Rb)[0]*100:5.1f}% lost"
          f"   HNN {abs(E(Rh)[-1]-E(Rh)[0])/E(Rh)[0]*100:5.2f}%")
    print("  -> Same loss, completely different physics. The failure is")
    print("     INVISIBLE in-distribution: scaling data would not fix it.")

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    ax[0].plot(Rt[:, 0], Rt[:, 1], color=C["fg"], lw=3, alpha=.35)
    ax[0].plot(Rb[:, 0], Rb[:, 1], color=C["pink"], lw=1)
    ax[0].set_title("baseline rollout"); ax[0].set_aspect("equal")
    ax[1].plot(Rt[:, 0], Rt[:, 1], color=C["fg"], lw=3, alpha=.35)
    ax[1].plot(Rh[:, 0], Rh[:, 1], color=C["green"], lw=1)
    ax[1].set_title("HNN rollout"); ax[1].set_aspect("equal")
    t = np.arange(len(Rt))*0.05
    ax[2].plot(t, E(Rt), color=C["fg"], lw=2.5, alpha=.4, label="true")
    ax[2].plot(t, E(Rb), color=C["pink"], label="baseline")
    ax[2].plot(t, E(Rh), color=C["green"], label="HNN")
    ax[2].set_title("energy"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("demo5_hnn.png"); plt.close(fig)
    print("  saved demo5_hnn.png")


# =============================================================================
# DEMO 6 - Hamiltonian Monte Carlo: identical maths, zero physics
# =============================================================================
def demo6():
    print("\n--- 6. Hamiltonian Monte Carlo --------------------------------")
    # target: correlated 2-D Gaussian.  U(q) = -log pi(q)
    Sig = np.array([[1.0, 0.93], [0.93, 1.0]])
    Pr = np.linalg.inv(Sig)
    U = lambda q: 0.5 * q @ Pr @ q
    dU = lambda q: Pr @ q

    def hmc(q, eps=0.18, L=22):
        p = rng.normal(size=2)
        H0 = U(q) + 0.5*p @ p
        qn, pn = q.copy(), p.copy()
        pn -= 0.5*eps*dU(qn)                     # leapfrog: half kick
        for _ in range(L):
            qn += eps*pn                         # drift
            pn -= eps*dU(qn)                     # kick
        pn += 0.5*eps*dU(qn)                     # undo the trailing half kick
        pn = -pn                                 # reversibility
        H1 = U(qn) + 0.5*pn @ pn
        return (qn, True) if np.log(rng.uniform()) < H0 - H1 else (q, False)

    q = np.zeros(2); samples = []; acc = 0
    for _ in range(4000):
        q, a = hmc(q); acc += a; samples.append(q.copy())
    S = np.array(samples[500:])
    print(f"  acceptance rate      : {acc/4000:.3f}")
    print(f"  target correlation   : {Sig[0,1]:.3f}")
    print(f"  sampled correlation  : {np.corrcoef(S.T)[0,1]:.3f}")
    print("  -> No physical system anywhere here. -log(density) plays the role")
    print("     of potential energy; conservation is what makes long, high-")
    print("     acceptance proposals possible instead of a random walk.")

    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(S[:, 0], S[:, 1], ".", ms=1.6, color=C["blue"], alpha=.5)
    ax.set_title("HMC samples"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig("demo6_hmc.png"); plt.close(fig)
    print("  saved demo6_hmc.png")


DEMOS = {1: demo1, 2: demo2, 3: demo3, 4: demo4, 5: demo5, 6: demo6}

if __name__ == "__main__":
    print(__doc__)
    which = [int(a) for a in sys.argv[1:]] or sorted(DEMOS)
    for d in which:
        DEMOS[d]()
    print("\ndone.\n")
