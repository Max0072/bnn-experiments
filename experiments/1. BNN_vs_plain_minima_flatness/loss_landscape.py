import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {DEVICE}")

# ── hyper-params ─────────────────────────────────────────────────────────────
BATCH_SIZE         = 128
EPOCHS             = 10
LR                 = 1e-3
KL_WEIGHT          = 2e-5
PLAIN_WEIGHT_DECAY = 5e-4   # tune until Plain baseline CE ≈ BNN baseline CE
RHO_INIT           = -3.0   # softplus(-3) ≈ 0.049 → initial sigma

N_REPEATS       = 30    # repetitions per delta

DELTAS = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

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
# BNN layers (local reparameterisation)
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


def run_training(train_loader, test_loader):

    # ── BNN ──
    print("\n=== Training BNN ===")
    torch.manual_seed(SEED)
    bnn = BayesianCNN().to(DEVICE)
    opt_b = torch.optim.Adam(bnn.parameters(), lr=LR)
    print(f"  initial mean sigma = {bnn.mean_sigma():.4f}")
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_bnn(bnn, train_loader, opt_b)
        acc, ce = evaluate(bnn, test_loader)
        print(f"  epoch {ep:2d}  loss={tr_loss:.4f}  ce={ce:.4f}  acc={acc:.4f}  sigma={bnn.mean_sigma():.4f}")
    torch.save(bnn.state_dict(), "models/bnn_cnn.pt")
    print("bnn_cnn.pt saved")

    # ── Plain ──
    print("\n=== Training Plain CNN ===")
    torch.manual_seed(SEED)
    plain = PlainCNN().to(DEVICE)
    opt_p = torch.optim.Adam(plain.parameters(), lr=LR, weight_decay=PLAIN_WEIGHT_DECAY)
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_plain(plain, train_loader, opt_p)
        acc, ce = evaluate(plain, test_loader)
        print(f"  epoch {ep:2d}  loss={tr_loss:.4f}  ce={ce:.4f}  acc={acc:.4f}")
    torch.save(plain.state_dict(), "models/plain_cnn.pt")
    print("plain_cnn.pt saved")

    return plain, bnn


# ─────────────────────────────────────────────────────────────────────────────
# Flatness test
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def loss_on_batch(model, x, y):
    return F.cross_entropy(model(x), y).item()


def perturb_plain(model, delta):
    with torch.no_grad():
        for param in model.parameters():
                param.add_(torch.randn_like(param) * delta)


def perturb_bnn(model, delta):
    """Perturb only mu parameters; rho (and thus sigma) are untouched."""
    for name, param in model.named_parameters():
        if name.endswith(".mu") or name.endswith(".mu_bias"):
            param.data.add_(torch.randn_like(param) * delta)


def flatness_test(plain, bnn, flat_loader):
    x_flat, y_flat = next(iter(flat_loader))
    x_flat, y_flat = x_flat.to(DEVICE), y_flat.to(DEVICE)

    # BNN evaluated at w = mu (MAP estimate) — fair comparison with Plain
    bnn.set_deterministic(True)

    results = {}
    for tag, model, perturb_fn in [
        ("Plain", plain, perturb_plain),
        ("BNN",   bnn,   perturb_bnn),
    ]:
        print(f"\n--- Flatness test: {tag} ---")
        losses_per_delta = []
        for delta in DELTAS:
            reps = []
            for _ in range(N_REPEATS):
                if tag == "Plain":
                    m = PlainCNN().to(DEVICE)
                else:
                    m = BayesianCNN().to(DEVICE)
                    m.set_deterministic(True)
                m.load_state_dict(model.state_dict())
                m.eval()
                if delta > 0:
                    perturb_fn(m, delta)
                reps.append(loss_on_batch(m, x_flat, y_flat))
            losses_per_delta.append(reps)
            print(f"  delta={delta:.3f}  mean={np.mean(reps):.4f}  std={np.std(reps):.4f}")
        results[tag] = np.array(losses_per_delta)   # shape (n_deltas, N_REPEATS)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Loss Landscape Flatness: BNN vs Plain CNN (MNIST)", fontsize=13, fontweight="bold")

    colors = {"Plain": "#e05c5c", "BNN": "#4a90d9"}
    alphas = {"Plain": 0.20, "BNN": 0.20}

    # ── left: absolute loss ──────────────────────────────────────────────────
    ax = axes[0]
    for tag, data in results.items():
        means = data.mean(axis=1)
        stds  = data.std(axis=1)
        ax.plot(DELTAS, means, "o-", color=colors[tag], label=tag, linewidth=2, markersize=5)
        ax.fill_between(DELTAS,
                         means - stds, means + stds,
                         color=colors[tag], alpha=alphas[tag])
    ax.set_xlabel("Perturbation δ")
    ax.set_ylabel("CE Loss")
    ax.set_title("Absolute loss vs perturbation")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ── right: normalised loss ───────────────────────────────────────────────
    ax = axes[1]
    for tag, data in results.items():
        baseline = data[0].mean()           # delta=0
        means    = data.mean(axis=1) / baseline
        stds     = data.std(axis=1)  / baseline
        ax.plot(DELTAS, means, "o-", color=colors[tag], label=tag, linewidth=2, markersize=5)
        ax.fill_between(DELTAS,
                         means - stds, means + stds,
                         color=colors[tag], alpha=alphas[tag])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Perturbation δ")
    ax.set_ylabel("Loss / baseline")
    ax.set_title("Normalised loss vs perturbation")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.savefig("flatness_result.png", dpi=150, bbox_inches="tight")
    print("\nflatness_result.png saved")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "="*60)
    print(f"{'δ':>8}  {'Plain mean':>12}  {'Plain norm':>12}  {'BNN mean':>10}  {'BNN norm':>10}")
    print("-"*60)
    plain_base = results["Plain"][0].mean()
    bnn_base   = results["BNN"][0].mean()
    for i, delta in enumerate(DELTAS):
        pm = results["Plain"][i].mean()
        bm = results["BNN"][i].mean()
        print(f"{delta:>8.3f}  {pm:>12.4f}  {pm/plain_base:>12.3f}  {bm:>10.4f}  {bm/bnn_base:>10.3f}")
    print("="*60)

    # crossover point — require both models to have grown >2% above baseline
    # to filter out noise in the flat region near δ=0
    plain_norms = results["Plain"].mean(axis=1) / results["Plain"][0].mean()
    bnn_norms   = results["BNN"].mean(axis=1)   / results["BNN"][0].mean()
    CROSSOVER_THRESHOLD = 1.02
    crossover = None
    for i in range(1, len(DELTAS)):
        if plain_norms[i] > CROSSOVER_THRESHOLD and plain_norms[i] > bnn_norms[i]:
            crossover = DELTAS[i]
            break
    if crossover is not None:
        print(f"\nHypothesis check: meaningful divergence (Plain > BNN, both >+2%) at δ ≈ {crossover}")
    else:
        print("\nNote: Plain did not clearly overtake BNN in tested δ range.")

    idx_01 = DELTAS.index(0.1)
    print(f"Loss growth at δ=0.1  —  Plain: ×{plain_norms[idx_01]:.2f}   BNN: ×{bnn_norms[idx_01]:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # ── load or train ──
    plain = PlainCNN().to(DEVICE)
    bnn   = BayesianCNN().to(DEVICE)
    train_loader, test_loader, flat_loader = get_loaders()

    if os.path.exists("models/plain_cnn.pt") and os.path.exists("models/bnn_cnn.pt"):
        print("Found saved checkpoints — loading.")
        plain.load_state_dict(torch.load("models/plain_cnn.pt", map_location=DEVICE))
        bnn.load_state_dict(torch.load("models/bnn_cnn.pt",   map_location=DEVICE))
        print(f"BNN mean sigma after training: {bnn.mean_sigma():.4f}")
    else:
        plain, bnn = run_training(train_loader, test_loader)
        print(f"\nBNN mean sigma after training: {bnn.mean_sigma():.4f}")

    plain.eval()
    bnn.eval()

    # ── flatness test ──
    results = flatness_test(plain, bnn, flat_loader)

    # ── report ──
    print_summary(results)
    plot_results(results)
