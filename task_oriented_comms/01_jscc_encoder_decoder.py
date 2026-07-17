"""
Step 1 — an easy encoder/decoder for (task-oriented) communication.

This is the "hello world" you asked for, before touching HydraCollab-scale
collaborative perception. It builds ONE pipeline:

        input feature  ->  ENCODER  ->  [ noisy channel ]  ->  DECODER  ->  output

and trains it two ways so you can feel the difference that "task-oriented" makes:

  (A) Plain Deep-JSCC : the decoder tries to RECONSTRUCT the input.
  (B) Task-oriented   : the decoder tries to produce the TASK answer (the class).

Same encoder, same tiny channel, same bit budget. The only change is WHAT the
decoder is asked to recover. You will see that when the channel is small/noisy,
the task-oriented model keeps the task accurate while "reconstruct-then-classify"
collapses -- because it wastes its scarce channel capacity on class-irrelevant
detail. That single observation is the whole point of task-oriented comms.

Run:  python3 01_jscc_encoder_decoder.py
Needs only PyTorch (no torchvision, no downloads).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ----------------------------------------------------------------------------
# Config -- change these to build intuition.
# ----------------------------------------------------------------------------
INPUT_DIM    = 32     # size of the "scene feature" vector we want to send
NUM_CLASSES  = 4      # the downstream TASK: classify the feature
CHANNEL_DIM  = 4      # <-- the bottleneck: how many channel symbols we may send.
                      #     Smaller = harsher. This is the "bandwidth" knob.
TRAIN_SNR_DB = 10.0   # channel quality seen during training (signal-to-noise, dB)
N_TRAIN, N_TEST = 4000, 2000
EPOCHS, BATCH, LR = 40, 128, 2e-3


# ----------------------------------------------------------------------------
# Synthetic data: each sample has a CLASS (the task target) plus nuisance dims
# that are high-variance but carry NO class information. Reconstruction is
# tempted to spend channel capacity on the nuisance; the task doesn't care.
# ----------------------------------------------------------------------------
# Fixed, well-separated prototypes so the CLEAN task is easy (high ceiling).
# Class info lives ONLY in the first half of the vector; the second half is
# loud, class-irrelevant nuisance that MSE-reconstruction is tempted to chase.
_PROTOS = 2.2 * torch.randn(NUM_CLASSES, INPUT_DIM)
_PROTOS[:, INPUT_DIM // 2:] = 0.0

def make_dataset(n):
    y = torch.randint(0, NUM_CLASSES, (n,))
    x = _PROTOS[y] + 0.3 * torch.randn(n, INPUT_DIM)              # signal + small noise
    x[:, INPUT_DIM // 2:] += 2.5 * torch.randn(n, INPUT_DIM // 2)  # loud nuisance
    return x, y

Xtr, Ytr = make_dataset(N_TRAIN)
Xte, Yte = make_dataset(N_TEST)


# ----------------------------------------------------------------------------
# The channel. This is the ONE piece that separates comms from a plain autoencoder.
# We normalise the encoder output to unit average power per symbol, then add
# Gaussian noise whose variance is set by the SNR. Backprop flows straight through
# it (the noise has no parameters), so the encoder learns to be noise-robust.
# ----------------------------------------------------------------------------
def power_normalize(z):
    # scale each sample so mean power per symbol == 1
    rms = z.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    return z / rms

def awgn(z, snr_db):
    snr = 10 ** (snr_db / 10.0)
    noise_std = (1.0 / snr) ** 0.5          # signal power is 1 per symbol
    return z + noise_std * torch.randn_like(z)


# ----------------------------------------------------------------------------
# Encoder + two decoder heads.
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, CHANNEL_DIM),   # -> channel symbols
        )
    def forward(self, x):
        return power_normalize(self.net(x))

class ReconDecoder(nn.Module):   # plain JSCC: rebuild the input
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CHANNEL_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, INPUT_DIM),
        )
    def forward(self, z): return self.net(z)

class TaskDecoder(nn.Module):    # task-oriented: output the class directly
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CHANNEL_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, z): return self.net(z)


# A frozen "oracle" classifier trained on clean features. For the reconstruction
# model we send x -> encode -> channel -> reconstruct x_hat, then ask THIS
# classifier what x_hat is. That is the fair "reconstruct then use it" baseline.
class Oracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


def train(model_fn, loss_fn, tag):
    """Generic trainer. model_fn(x) -> prediction; loss_fn(pred, x, y) -> loss."""
    params = []
    for m in model_fn.modules_to_train:
        params += list(m.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i:i + BATCH]
            x, y = Xtr[idx], Ytr[idx]
            pred = model_fn(x)
            loss = loss_fn(pred, x, y)
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  [{tag}] trained.")


# ---- train the oracle on clean data (no channel) --------------------------
oracle = Oracle()
oo = torch.optim.Adam(oracle.parameters(), lr=LR)
for ep in range(30):
    perm = torch.randperm(N_TRAIN)
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i + BATCH]
        loss = F.cross_entropy(oracle(Xtr[idx]), Ytr[idx])
        oo.zero_grad(); loss.backward(); oo.step()
for p in oracle.parameters(): p.requires_grad_(False)
with torch.no_grad():
    clean_ceiling = (oracle(Xte).argmax(1) == Yte).float().mean().item()
print(f"  [oracle] clean task ceiling (no channel) = {clean_ceiling:.1%}")

# ---- model A: plain JSCC (reconstruct) ------------------------------------
encA, decA = Encoder(), ReconDecoder()
def forwardA(x): return decA(awgn(encA(x), TRAIN_SNR_DB))
forwardA.modules_to_train = [encA, decA]
train(forwardA, lambda pred, x, y: F.mse_loss(pred, x), "JSCC-reconstruct")

# ---- model B: task-oriented -----------------------------------------------
encB, decB = Encoder(), TaskDecoder()
def forwardB(x): return decB(awgn(encB(x), TRAIN_SNR_DB))
forwardB.modules_to_train = [encB, decB]
train(forwardB, lambda pred, x, y: F.cross_entropy(pred, y), "task-oriented")


# ----------------------------------------------------------------------------
# Evaluate: downstream TASK accuracy across a sweep of channel qualities.
# Both models were trained at TRAIN_SNR_DB; we test how they hold up elsewhere.
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_task_acc(snr_db):
    # A: reconstruct, then classify with the oracle
    xa = decA(awgn(encA(Xte), snr_db))
    accA = (oracle(xa).argmax(1) == Yte).float().mean().item()
    # B: read the task answer straight off the channel
    accB = (decB(awgn(encB(Xte), snr_db)).argmax(1) == Yte).float().mean().item()
    return accA, accB

print(f"\nChannel budget = {CHANNEL_DIM} symbols for a {INPUT_DIM}-dim feature "
      f"(compression {INPUT_DIM/CHANNEL_DIM:.0f}x). Trained @ {TRAIN_SNR_DB:.0f} dB.\n")
print(f"{'test SNR (dB)':>13} | {'reconstruct->classify':>22} | {'task-oriented':>14}")
print("-" * 56)
for snr in [-5, 0, 5, 10, 15, 20]:
    a, b = eval_task_acc(snr)
    print(f"{snr:>13} | {a:>21.1%} | {b:>13.1%}")

print("\nTakeaway: at the same tiny channel budget, the task-oriented decoder")
print("keeps the TASK accurate because the encoder only had to preserve what the")
print("task needs -- not the loud, useless nuisance dims. That is the seed idea")
print("you'll scale up to selective feature sharing + dual-link in HydraCollab.")
