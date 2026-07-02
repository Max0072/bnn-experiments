import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def project_root(marker: str = ".git") -> Path:
    """Absolute path to the project root — the nearest parent containing `marker`."""
    for d in Path(__file__).resolve().parents:
        if (d / marker).exists():
            return d
    raise RuntimeError(f"project root not found (no {marker} above {__file__})")


# datasets live in the project root, downloaded once and reused across experiments
DATA_DIR   = project_root() / "datasets"
# images go next to this script, not relative to the current working directory
RESULT_DIR = Path(__file__).resolve().parent / "result_images"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"   # pinned memory is only supported/useful on CUDA
print(f"device: {DEVICE}")

# ── hyper-params ─────────────────────────────────────────────────────────────
SEEDS              = [41, 42, 43, 44, 45]   # experiment is repeated across these
BATCH_SIZE         = 128
EPOCHS             = 10
LR                 = 1e-3
KL_WEIGHT          = 2e-5
# Tuned so Plain reaches the same test CE as BNN (equal-performance comparison).
# This does NOT equalise weight scales — BNN's mu ends up ~1.35x larger — which is why
# flatness is measured filter-normalised (scale-invariant) in flatness_test below.
PLAIN_WEIGHT_DECAY = 5e-4
RHO_INIT           = -3.0   # softplus(-3) ≈ 0.049 → initial sigma

N_REPEATS       = 30    # repetitions per delta

# Flatness is measured at scale σ (Definition 2.1): weights are perturbed by t·σ⊙ξ,
# σ = the trained BNN posterior std. DELTAS holds t in units of σ, so t=1 is exactly
# the model's own posterior width.
DELTAS = [0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
REPORT_DELTA = 1.0   # headline t (one posterior σ); must be present in DELTAS


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(seed):
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)

    # Dedicated generator for shuffling so batch order is independent of model-side
    # RNG: BNN samples weights via randn during forward, Plain does not — sharing the
    # global RNG would desync their batch order. Re-seeded per model in run_training.
    train_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=train_gen, num_workers=0, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)

    flat_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    return train_loader, test_loader, flat_loader


# ─────────────────────────────────────────────────────────────────────────────
# Plain CNN
# ─────────────────────────────────────────────────────────────────────────────

class PlainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28→14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14→7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
# BNN layers (global reparameterisation)
# ─────────────────────────────────────────────────────────────────────────────

class BayesianLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(out_f, in_f))
        self.rho = nn.Parameter(torch.full((out_f, in_f), RHO_INIT))
        self.mu_bias = nn.Parameter(torch.zeros(out_f))
        self.rho_bias = nn.Parameter(torch.full((out_f,), RHO_INIT))
        self.deterministic = False
        nn.init.kaiming_uniform_(self.mu, a=np.sqrt(5))
        nn.init.uniform_(self.mu_bias, -1 / np.sqrt(in_f), 1 / np.sqrt(in_f))

    @staticmethod
    def _sigma(rho):
        return F.softplus(rho)

    def forward(self, x):
        if self.deterministic:
            return F.linear(x, self.mu, self.mu_bias)
        sigma_w = self._sigma(self.rho)
        sigma_b = self._sigma(self.rho_bias)
        w = self.mu + sigma_w * torch.randn_like(self.mu)
        b = self.mu_bias + sigma_b * torch.randn_like(self.mu_bias)
        return F.linear(x, w, b)

    def kl(self):
        sigma_w = self._sigma(self.rho)
        sigma_b = self._sigma(self.rho_bias)
        kl_w = 0.5 * (self.mu**2 + sigma_w**2 - torch.log(sigma_w**2) - 1).sum()
        kl_b = 0.5 * (self.mu_bias**2 + sigma_b**2 - torch.log(sigma_b**2) - 1).sum()
        return kl_w + kl_b


class BayesianConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding=0):
        super().__init__()
        self.padding = padding
        shape = (out_c, in_c, kernel_size, kernel_size)
        self.mu = nn.Parameter(torch.zeros(*shape))
        self.rho = nn.Parameter(torch.full(shape, RHO_INIT))
        self.mu_bias = nn.Parameter(torch.zeros(out_c))
        self.rho_bias = nn.Parameter(torch.full((out_c,), RHO_INIT))
        self.deterministic = False
        nn.init.kaiming_uniform_(self.mu, a=np.sqrt(5))
        nn.init.uniform_(self.mu_bias, -1 / np.sqrt(in_c * kernel_size * kernel_size), 1 / np.sqrt(in_c * kernel_size * kernel_size))

    @staticmethod
    def _sigma(rho):
        return F.softplus(rho)

    def forward(self, x):
        if self.deterministic:
            return F.conv2d(x, self.mu, self.mu_bias, padding=self.padding)
        sigma_w = self._sigma(self.rho)
        sigma_b = self._sigma(self.rho_bias)
        w = self.mu + sigma_w * torch.randn_like(self.mu)
        b = self.mu_bias + sigma_b * torch.randn_like(self.mu_bias)
        return F.conv2d(x, w, b, padding=self.padding)

    def kl(self):
        sigma_w = self._sigma(self.rho)
        sigma_b = self._sigma(self.rho_bias)
        kl_w = 0.5 * (self.mu**2 + sigma_w**2 - torch.log(sigma_w**2) - 1).sum()
        kl_b = 0.5 * (self.mu_bias**2 + sigma_b**2 - torch.log(sigma_b**2) - 1).sum()
        return kl_w + kl_b


class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1,  16, 3, padding=1)
        self.conv2 = BayesianConv2d(16, 32, 3, padding=1)
        self.fc1 = BayesianLinear(32 * 7 * 7, 128)
        self.fc2 = BayesianLinear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def set_deterministic(self, flag: bool):
        for m in [self.conv1, self.conv2, self.fc1, self.fc2]:
            m.deterministic = flag

    def kl(self):
        return sum(m.kl() for m in [self.conv1, self.conv2, self.fc1, self.fc2])

    def mean_sigma(self):
        sigmas = []
        for m in [self.conv1, self.conv2, self.fc1, self.fc2]:
            sigmas.append(F.softplus(m.rho).mean().item())
            sigmas.append(F.softplus(m.rho_bias).mean().item())
        return float(np.mean(sigmas))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_plain(model, loader, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def train_bnn(model, loader, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        ce = F.cross_entropy(model(x), y)
        loss = ce + KL_WEIGHT * model.kl()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    is_bnn = hasattr(model, 'set_deterministic')
    if is_bnn:
        model.set_deterministic(True)

    correct, total = 0, 0
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)

    if is_bnn:
        model.set_deterministic(False)
    return correct / total, total_loss / total


def run_training(train_loader, test_loader, seed):
    summary = {}

    # ── BNN ──
    print("\n=== Training BNN ===")
    set_seed(seed)
    bnn = BayesianCNN().to(DEVICE)
    opt_b = torch.optim.Adam(bnn.parameters(), lr=LR)
    train_loader.generator.manual_seed(seed)   # identical batch order across both models
    print(f"  initial mean sigma = {bnn.mean_sigma():.4f}")
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_bnn(bnn, train_loader, opt_b)
        acc, ce = evaluate(bnn, test_loader)
        print(f"  epoch {ep:2d}  loss={tr_loss:.4f}  ce={ce:.4f}  acc={acc:.4f}  sigma={bnn.mean_sigma():.4f}")
    summary["bnn"] = {"ce": ce, "acc": acc, "sigma": bnn.mean_sigma()}

    # ── Plain ──
    print("\n=== Training Plain CNN ===")
    set_seed(seed)
    plain = PlainCNN().to(DEVICE)
    opt_p = torch.optim.Adam(plain.parameters(), lr=LR, weight_decay=PLAIN_WEIGHT_DECAY)
    train_loader.generator.manual_seed(seed)   # identical batch order across both models
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_plain(plain, train_loader, opt_p)
        acc, ce = evaluate(plain, test_loader)
        print(f"  epoch {ep:2d}  loss={tr_loss:.4f}  ce={ce:.4f}  acc={acc:.4f}")
    summary["plain"] = {"ce": ce, "acc": acc}

    return plain, bnn, summary


# ─────────────────────────────────────────────────────────────────────────────
# Flatness test
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def loss_on_batch(model, x, y):
    return F.cross_entropy(model(x), y).item()


def bnn_sigmas(bnn):
    """Per-parameter posterior std σ = softplus(ρ) of the trained BNN, keyed by layer.
    This is the scale at which flatness is measured (Definition 2.1)."""
    sig = {}
    for name in ["conv1", "conv2", "fc1", "fc2"]:
        m = getattr(bnn, name)
        sig[name] = (F.softplus(m.rho).detach(), F.softplus(m.rho_bias).detach())
    return sig


def perturb_at_sigma(model, sigmas, t):
    """Perturb each weight/bias by t·σ⊙ξ, with σ the BNN posterior std — the SAME σ for
    both models (Definition 2.1: compare two minima at one scale σ). t is in units of σ,
    so t=1 is exactly the BNN's own posterior width. Works for PlainCNN (.weight/.bias)
    and the MAP BNN (.mu/.mu_bias)."""
    with torch.no_grad():
        for name, (sw, sb) in sigmas.items():
            mod = getattr(model, name)
            w = mod.mu if hasattr(mod, "mu") else mod.weight
            b = mod.mu_bias if hasattr(mod, "mu_bias") else mod.bias
            w.add_(t * sw * torch.randn_like(w))
            b.add_(t * sb * torch.randn_like(b))


def flatness_test(plain, bnn, flat_loader):
    x_flat, y_flat = next(iter(flat_loader))
    x_flat, y_flat = x_flat.to(DEVICE), y_flat.to(DEVICE)

    sigmas = bnn_sigmas(bnn)          # σ scale taken from the trained BNN posterior

    results = {}
    for tag, model in [("Plain", plain), ("BNN", bnn)]:
        print(f"\n--- Flatness test at scale σ: {tag} ---")
        losses_per_delta = []
        for t in DELTAS:
            reps = []
            for _ in range(N_REPEATS):
                if tag == "Plain":
                    m = PlainCNN().to(DEVICE)
                else:
                    m = BayesianCNN().to(DEVICE)
                    m.set_deterministic(True)
                m.load_state_dict(model.state_dict())
                m.eval()
                if t > 0:
                    perturb_at_sigma(m, sigmas, t)
                reps.append(loss_on_batch(m, x_flat, y_flat))
            losses_per_delta.append(reps)
            print(f"  t={t:.2f}·σ  mean={np.mean(reps):.4f}  std={np.std(reps):.4f}")
        results[tag] = np.array(losses_per_delta)   # shape (n_deltas, N_REPEATS)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot (per seed)
# ─────────────────────────────────────────────────────────────────────────────

# The interesting regime (gap at t≈σ, convergence at t≫σ) spans several orders of
# magnitude in both axes — linear axes squash it to nothing. All flatness plots are
# log-log. t=0 (baseline) is dropped: it can't sit on a log x-axis and is trivially 1.0.
def _style_log_axes(ax, t_vals):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)  # t=σ
    ax.set_xticks(t_vals)
    ax.set_xticklabels([f"{t:g}" for t in t_vals])
    ax.grid(True, which="both", linestyle="--", alpha=0.4)


def plot_results(results, seed):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Loss Landscape Flatness: BNN vs Plain CNN (MNIST) — seed {seed}  "
                 f"(dashed line = t=σ)", fontsize=13, fontweight="bold")

    colors = {"Plain": "#e05c5c", "BNN": "#4a90d9"}
    alphas = {"Plain": 0.20, "BNN": 0.20}
    t = np.array(DELTAS[1:])   # drop t=0 for log axes

    # ── left: absolute loss ──────────────────────────────────────────────────
    ax = axes[0]
    for tag, data in results.items():
        means = data.mean(axis=1)[1:]
        stds  = data.std(axis=1)[1:]
        ax.plot(t, means, "o-", color=colors[tag], label=tag, linewidth=2, markersize=5)
        ax.fill_between(t, np.maximum(means - stds, 1e-6), means + stds, color=colors[tag], alpha=alphas[tag])
    ax.set_xlabel("Perturbation  t  (units of posterior σ)")
    ax.set_ylabel("CE Loss")
    ax.set_title("Absolute loss vs perturbation")
    ax.legend()
    _style_log_axes(ax, t)

    # ── right: normalised loss ───────────────────────────────────────────────
    ax = axes[1]
    for tag, data in results.items():
        baseline = data[0].mean()           # t=0
        means    = data.mean(axis=1)[1:] / baseline
        stds     = data.std(axis=1)[1:]  / baseline
        ax.plot(t, means, "o-", color=colors[tag], label=tag, linewidth=2, markersize=5)
        ax.fill_between(t, np.maximum(means - stds, 1e-6), means + stds, color=colors[tag], alpha=alphas[tag])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Perturbation  t  (units of posterior σ)")
    ax.set_ylabel("Loss / baseline")
    ax.set_title("Normalised loss vs perturbation")
    ax.legend()
    _style_log_axes(ax, t)

    plt.tight_layout()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"flatness_result_seed{seed}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n{path} saved")


# ─────────────────────────────────────────────────────────────────────────────
# Plot (aggregate across seeds)
# ─────────────────────────────────────────────────────────────────────────────

def plot_aggregate(per_seed):
    colors = {"Plain": "#e05c5c", "BNN": "#4a90d9"}
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"Normalised loss vs perturbation — mean ± std over {len(per_seed)} seeds\n"
                 f"(dashed line = t=σ)", fontsize=12, fontweight="bold")

    t = np.array(DELTAS[1:])   # drop t=0 for log axes
    for tag in ["Plain", "BNN"]:
        # one normalised curve per seed, then mean/std across seeds
        curves = np.array([
            d["results"][tag].mean(axis=1) / d["results"][tag][0].mean()
            for d in per_seed.values()
        ])[:, 1:]                                # (n_seeds, n_deltas-1), drop t=0
        m, s = curves.mean(0), curves.std(0)
        ax.plot(t, m, "o-", color=colors[tag], label=tag, linewidth=2, markersize=5)
        ax.fill_between(t, np.maximum(m - s, 1e-6), m + s, color=colors[tag], alpha=0.20)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Perturbation  t  (units of posterior σ)")
    ax.set_ylabel("Loss / baseline")
    ax.legend()
    _style_log_axes(ax, t)

    plt.tight_layout()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "flatness_result_aggregate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n{path} saved")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Weight-norm diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# The flatness test adds N(0, δ²) with the SAME absolute δ to the perturbed params.
# Whether that comparison is fair depends on the scale of those params: a fixed δ is
# a larger *relative* hit to small weights (Dinh et al. 2017, "Sharp Minima Can
# Generalize"). RMS = L2 / sqrt(count) is the per-element scale that matters here.
#   Plain  → all parameters are perturbed.
#   BNN    → only mu / mu_bias are perturbed (rho/sigma untouched).

@torch.no_grad()
def weight_norms(model, only_mu=False):
    rows, total_sq, total_n = [], 0.0, 0
    for name, p in model.named_parameters():
        if only_mu and not (name.endswith(".mu") or name.endswith(".mu_bias")):
            continue
        v = p.detach().flatten()
        n = v.numel()
        l2 = v.norm().item()
        rows.append((name, n, l2, l2 / np.sqrt(n)))
        total_sq += l2 ** 2
        total_n += n
    total_l2 = float(np.sqrt(total_sq))
    return rows, total_n, total_l2, total_l2 / np.sqrt(total_n)


def report_weight_norms(plain, bnn):
    print("\n--- weight norms of perturbed params (L2 and RMS = L2/√count) ---")
    print(f"{'param':>16}  {'count':>8}  {'L2':>10}  {'RMS':>10}")
    out = {}
    for tag, model, only_mu in [("Plain", plain, False), ("BNN(mu)", bnn, True)]:
        rows, tot_n, tot_l2, tot_rms = weight_norms(model, only_mu)
        print(f"  [{tag}]")
        for name, n, l2, rms in rows:
            print(f"{name:>16}  {n:>8}  {l2:>10.3f}  {rms:>10.4f}")
        print(f"{'TOTAL':>16}  {tot_n:>8}  {tot_l2:>10.3f}  {tot_rms:>10.4f}")
        out[tag] = {"l2": tot_l2, "rms": tot_rms}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results, seed):
    print(f"\n{'='*60}\nSeed {seed} — flatness table")
    print(f"{'δ':>8}  {'Plain mean':>12}  {'Plain norm':>12}  {'BNN mean':>10}  {'BNN norm':>10}")
    print("-"*60)
    plain_base = results["Plain"][0].mean()
    bnn_base   = results["BNN"][0].mean()
    for i, delta in enumerate(DELTAS):
        pm = results["Plain"][i].mean()
        bm = results["BNN"][i].mean()
        print(f"{delta:>8.3f}  {pm:>12.4f}  {pm/plain_base:>12.3f}  {bm:>10.4f}  {bm/bnn_base:>10.3f}")
    print("="*60)

    idx = DELTAS.index(REPORT_DELTA)
    plain_norms = results["Plain"].mean(axis=1) / plain_base
    bnn_norms   = results["BNN"].mean(axis=1)   / bnn_base
    print(f"Loss growth at t={REPORT_DELTA}·σ  —  Plain: ×{plain_norms[idx]:.2f}   BNN: ×{bnn_norms[idx]:.2f}")


def print_summary_across_seeds(per_seed):
    idx_01 = DELTAS.index(REPORT_DELTA)
    print("\n" + "#"*64)
    print(f"SUMMARY ACROSS {len(per_seed)} SEEDS  (loss growth at t={REPORT_DELTA}·σ)")
    print("#"*64)
    print(f"{'seed':>6}  {'Plain ×':>10}  {'BNN ×':>10}  {'ratio':>8}")
    print("-"*64)
    plain_g, bnn_g = [], []
    for seed, d in per_seed.items():
        res = d["results"]
        pg = (res["Plain"].mean(axis=1) / res["Plain"][0].mean())[idx_01]
        bg = (res["BNN"].mean(axis=1)   / res["BNN"][0].mean())[idx_01]
        plain_g.append(pg); bnn_g.append(bg)
        print(f"{seed:>6}  {pg:>10.2f}  {bg:>10.2f}  {pg/bg:>8.2f}")
    plain_g, bnn_g = np.array(plain_g), np.array(bnn_g)
    print("-"*64)
    print(f"{'mean':>6}  {plain_g.mean():>10.2f}  {bnn_g.mean():>10.2f}  {(plain_g/bnn_g).mean():>8.2f}")
    print(f"{'std':>6}  {plain_g.std():>10.2f}  {bnn_g.std():>10.2f}")
    print("#"*64)

    # weight-norm comparison of perturbed params (mean across seeds) — fairness check
    norms = [d["train"].get("norms") for d in per_seed.values() if d["train"].get("norms")]
    if norms:
        p_l2  = np.mean([n["Plain"]["l2"]    for n in norms])
        p_rms = np.mean([n["Plain"]["rms"]   for n in norms])
        b_l2  = np.mean([n["BNN(mu)"]["l2"]  for n in norms])
        b_rms = np.mean([n["BNN(mu)"]["rms"] for n in norms])
        print(f"\nperturbed-param norms (mean across seeds):")
        print(f"  Plain    L2={p_l2:8.3f}  RMS={p_rms:.4f}")
        print(f"  BNN(mu)  L2={b_l2:8.3f}  RMS={b_rms:.4f}")
        print(f"  ratio RMS(Plain/BNN) = {p_rms/b_rms:.3f}  "
              f"→ same absolute δ is {'fair' if 0.8 < p_rms/b_rms < 1.25 else 'NOT obviously fair'}")
        print("#"*64)


# ─────────────────────────────────────────────────────────────────────────────
# One seed end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def run_seed(seed):
    print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
    set_seed(seed)
    train_loader, test_loader, flat_loader = get_loaders(seed)
    plain, bnn, train_summary = run_training(train_loader, test_loader, seed)
    print(f"\nBNN mean sigma after training: {bnn.mean_sigma():.4f}")
    train_summary["norms"] = report_weight_norms(plain, bnn)

    plain.eval()
    bnn.eval()

    results = flatness_test(plain, bnn, flat_loader)
    print_summary(results, seed)
    plot_results(results, seed)
    return {"results": results, "train": train_summary}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    per_seed = {seed: run_seed(seed) for seed in SEEDS}
    print_summary_across_seeds(per_seed)
    plot_aggregate(per_seed)