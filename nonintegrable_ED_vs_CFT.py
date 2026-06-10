"""
Non-integrable Ising ED vs CFT comparison.

Model: H = -J sum_j sigma^z_j sigma^z_{j+1}  - Gamma sum_j sigma^x_j sigma^x_{j+1}  - h sum_j sigma^x_j
with bond-centered SSD deformation.

Case 1 (benchmark):  J=1/2, Gamma=0,    h=1/2    — free Ising, should match CFT exactly
Case 2 (interacting): J=1,   Gamma=0.25, h=0.6066 — interacting critical Ising, should be close to CFT
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.evolution import expm_multiply_parallel

# ============================================================
# Shared parameters
# ============================================================
L = 16
q = 2
cCFT = 0.5

T0 = -0.18
T1 = 0.18

s0_0, sp_0, sm_0 = 1.0, 0.0, 0.0
s0_1, sp_1, sm_1 = 1.0, 1.2, -0.4

Ncycle = 16
nlist = np.arange(Ncycle + 1)

hChiral = cCFT * (q**2 - 1) / (24.0 * q)
hAnti = hChiral

NA = L // q
startA = int(np.floor(L / 2 - NA / 2))
subsysA = list(range(startA, startA + NA))

dtype = np.complex128

# bond-centered envelopes
x_bond = np.arange(L) + 0.5
f0_bond = s0_0 + sp_0 * np.cos(2*np.pi*q*x_bond/L) + sm_0 * np.sin(2*np.pi*q*x_bond/L)
f1_bond = s0_1 + sp_1 * np.cos(2*np.pi*q*x_bond/L) + sm_1 * np.sin(2*np.pi*q*x_bond/L)

def fbond_to_hsite(f_bond):
    return 0.5 * (np.roll(f_bond, 1) + f_bond)

h0_site = fbond_to_hsite(f0_bond)
h1_site = fbond_to_hsite(f1_bond)

out_dir = os.path.dirname(os.path.abspath(__file__))


# ############################################################
#                         CFT part
# ############################################################

def su11_chiral(s0, sp, sm, T, L, q):
    ell = L / q
    c2 = -s0**2 + sp**2 + sm**2
    if abs(c2) < 1e-14:
        alpha = -1.0 - 1j * s0 * np.pi * T / ell
        beta  = -1j * (sp + 1j*sm) * np.pi * T / ell
    else:
        C = np.sqrt(abs(c2))
        arg = C * np.pi * T / ell
        if c2 < 0:
            alpha = -np.cos(arg) - 1j * (s0/C) * np.sin(arg)
            beta  = -1j * (sp + 1j*sm) / C * np.sin(arg)
        else:
            alpha = -np.cosh(arg) - 1j * (s0/C) * np.sinh(arg)
            beta  = -1j * (sp + 1j*sm) / C * np.sinh(arg)
    return np.array([[alpha, beta], [np.conj(beta), np.conj(alpha)]])

def su11_antichiral(s0, sp, sm, T, L, q):
    ell = L / q
    c2 = -s0**2 + sp**2 + sm**2
    if abs(c2) < 1e-14:
        alpha = -1.0 - 1j * s0 * np.pi * T / ell
        beta  = -1j * (sp - 1j*sm) * np.pi * T / ell
    else:
        C = np.sqrt(abs(c2))
        arg = C * np.pi * T / ell
        if c2 < 0:
            alpha = -np.cos(arg) - 1j * (s0/C) * np.sin(arg)
            beta  = -1j * (sp - 1j*sm) / C * np.sin(arg)
        else:
            alpha = -np.cosh(arg) - 1j * (s0/C) * np.sinh(arg)
            beta  = -1j * (sp - 1j*sm) / C * np.sinh(arg)
    return np.array([[alpha, beta], [np.conj(beta), np.conj(alpha)]])

def energy_cft(Pi_ch, Pi_anti):
    a, b = Pi_ch[0,0], Pi_ch[0,1]
    ap, bp = Pi_anti[0,0], Pi_anti[0,1]
    return (-(np.pi*cCFT/(6*L)) * q**2
            + (np.pi*cCFT/(12*L)) * (q**2-1)
              * ((abs(a)**2+abs(b)**2) + (abs(ap)**2+abs(bp)**2)))

def fidelity_cft(Pi_ch, Pi_anti):
    """Fidelity = |<psi0|psi>|^2 = |alpha|^{-4h} |alpha_bar|^{-4h_bar}."""
    aa  = abs(Pi_ch[0,0])
    aap = abs(Pi_anti[0,0])
    if aa < 1e-300 or aap < 1e-300:
        return 0.0
    return min(max(aa**(-4*hChiral) * aap**(-4*hAnti), 0), 1)

def entropy_cft(Pi_ch, Pi_anti):
    a, b = Pi_ch[0,0], Pi_ch[0,1]
    ap, bp = Pi_anti[0,0], Pi_anti[0,1]
    eps0 = 1e-300
    x1 = max(abs(a - b), eps0)
    x2 = max(abs(ap - bp), eps0)
    return (cCFT/3) * (np.log(x1) + np.log(x2))

# Build SU(1,1) step matrices
M0  = su11_chiral(s0_0, sp_0, sm_0, T0, L, q)
M1  = su11_chiral(s0_1, sp_1, sm_1, T1, L, q)
M0b = su11_antichiral(s0_0, sp_0, sm_0, T0, L, q)
M1b = su11_antichiral(s0_1, sp_1, sm_1, T1, L, q)

M0M1  = M0 @ M1
M0M1b = M0b @ M1b

# CFT time series
E_cft_full  = np.zeros(Ncycle+1)
E_cft_half  = np.zeros(Ncycle+1)
F_cft_full  = np.zeros(Ncycle+1)  # fidelity
F_cft_half  = np.zeros(Ncycle+1)
S_cft_full  = np.zeros(Ncycle+1)
S_cft_half  = np.zeros(Ncycle+1)

Mn  = np.eye(2, dtype=complex)
Mnb = np.eye(2, dtype=complex)

for n in range(Ncycle+1):
    if n > 0:
        Mn  = Mn @ M0M1
        Mnb = Mnb @ M0M1b
    E_cft_full[n] = energy_cft(Mn, Mnb)
    F_cft_full[n] = fidelity_cft(Mn, Mnb)
    S_cft_full[n] = entropy_cft(Mn, Mnb)
    Mn_h  = Mn @ M0;  Mnb_h = Mnb @ M0b
    E_cft_half[n] = energy_cft(Mn_h, Mnb_h)
    F_cft_half[n] = fidelity_cft(Mn_h, Mnb_h)
    S_cft_half[n] = entropy_cft(Mn_h, Mnb_h)

dE_cft_full = E_cft_full - E_cft_full[0]
dE_cft_half = E_cft_half - E_cft_half[0]
dS_cft_full = S_cft_full - S_cft_full[0]
dS_cft_half = S_cft_half - S_cft_half[0]

print("[CFT] done.")


# ############################################################
#         ED for a general Ising model
# ############################################################

def run_ed(J_val, Gamma_val, h_val, label):
    """Run ED for H = -J zz - Gamma xx - h x with SSD deformation. Returns dict of results."""
    print(f"\n{'='*60}")
    print(f"[ED] {label}: J={J_val}, Gamma={Gamma_val}, h={h_val}")
    print(f"{'='*60}")

    basis = spin_basis_1d(L, pauli=True)
    no_checks = dict(check_herm=False, check_pcon=False, check_symm=False)

    # Uniform H0
    Jzz = [[-J_val, j, (j+1)%L] for j in range(L)]
    Gxx = [[-Gamma_val, j, (j+1)%L] for j in range(L)]
    hx  = [[-h_val, j] for j in range(L)]
    H0 = hamiltonian([["zz", Jzz], ["xx", Gxx], ["x", hx]], [],
                      basis=basis, dtype=dtype, **no_checks)

    # Deformed H1
    Jzz1 = [[-J_val*float(f1_bond[j]), j, (j+1)%L] for j in range(L)]
    Gxx1 = [[-Gamma_val*float(f1_bond[j]), j, (j+1)%L] for j in range(L)]
    hx1  = [[-h_val*float(h1_site[j]), j] for j in range(L)]
    H1 = hamiltonian([["zz", Jzz1], ["xx", Gxx1], ["x", hx1]], [],
                      basis=basis, dtype=dtype, **no_checks)

    # Ground state
    E0_arr, psi0_mat = H0.eigsh(k=1, which="SA", maxiter=10**7)
    psi0 = psi0_mat[:, 0].copy()
    psi0 /= np.linalg.norm(psi0)
    psi = psi0.copy()
    print(f"[ED] E_gs = {E0_arr[0]:.10f}")

    def ent_vn(psi_vec):
        out = basis.ent_entropy(psi_vec, sub_sys_A=subsysA, density=False)
        return float(np.real(out["Sent_A"]))

    # Propagators
    U0 = expm_multiply_parallel(H0.tocsr(), a=-1j*T0, dtype=dtype)
    U1 = expm_multiply_parallel(H1.tocsr(), a=-1j*T1, dtype=dtype)
    dim = psi.shape[0]
    work = np.zeros(2*dim, dtype=dtype)

    E_full  = np.zeros(Ncycle+1)
    E_half  = np.zeros(Ncycle+1)
    F_full  = np.zeros(Ncycle+1)  # fidelity = |<psi0|psi>|^2
    F_half  = np.zeros(Ncycle+1)
    S_full  = np.zeros(Ncycle+1)
    S_half  = np.zeros(Ncycle+1)

    print("[ED] Floquet evolution...")
    for n in range(Ncycle+1):
        E_full[n] = float(np.real(H0.expt_value(psi)))
        F_full[n] = float(np.abs(np.vdot(psi0, psi))**2)
        S_full[n] = ent_vn(psi)

        psi_h = psi.copy()
        U0.dot(psi_h, work_array=work, overwrite_v=True)
        E_half[n] = float(np.real(H0.expt_value(psi_h)))
        F_half[n] = float(np.abs(np.vdot(psi0, psi_h))**2)
        S_half[n] = ent_vn(psi_h)

        if n < Ncycle:
            U0.dot(psi, work_array=work, overwrite_v=True)
            U1.dot(psi, work_array=work, overwrite_v=True)
            psi /= np.linalg.norm(psi)

        if n % 4 == 0:
            print(f"  cycle {n}/{Ncycle}")

    print("[ED] done.")
    return dict(
        E_full=E_full, E_half=E_half,
        F_full=F_full, F_half=F_half,
        S_full=S_full, S_half=S_half,
        dE_full=E_full - E_full[0], dE_half=E_half - E_half[0],
        dS_full=S_full - S_full[0], dS_half=S_half - S_half[0],
    )


# ############################################################
#         Run both cases
# ############################################################
res_free = run_ed(J_val=0.5, Gamma_val=0.0, h_val=0.5,
                  label="Free Ising (benchmark)")

res_int  = run_ed(J_val=1.0, Gamma_val=0.25, h_val=0.6066,
                  label="Interacting critical Ising")


# ############################################################
#         Plots
# ############################################################
param_base = f"L={L}, q={q}, T0={T0}, T1={T1}"

def make_comparison_plot(res, case_label, filename_tag):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"ED vs CFT — {case_label}  |  {param_base}", fontsize=13)

    # Energy FULL
    ax = axes[0, 0]
    ax.plot(nlist, res['dE_full'], 'b-o', ms=4, lw=1.2, label='ED')
    ax.plot(nlist, dE_cft_full, 'r--s', ms=5, lw=1.2, label='CFT')
    ax.set_xlabel('cycles n'); ax.set_ylabel(r'$\Delta\langle H_0\rangle$')
    ax.set_title('Energy FULL'); ax.legend(); ax.grid(True, alpha=0.3)

    # Energy HALF
    ax = axes[0, 1]
    ax.plot(nlist, res['dE_half'], 'b-o', ms=4, lw=1.2, label='ED')
    ax.plot(nlist, dE_cft_half, 'r--s', ms=5, lw=1.2, label='CFT')
    ax.set_xlabel('cycles n'); ax.set_ylabel(r'$\Delta\langle H_0\rangle$')
    ax.set_title('Energy HALF'); ax.legend(); ax.grid(True, alpha=0.3)

    # Fidelity FULL
    ax = axes[1, 0]
    ax.semilogy(nlist, res['F_full'], 'b-o', ms=4, lw=1.2, label='ED')
    ax.semilogy(nlist, F_cft_full, 'r--s', ms=5, lw=1.2, label='CFT')
    ax.set_xlabel('cycles n'); ax.set_ylabel(r'$|\langle\psi_0|\psi\rangle|^2$')
    ax.set_title('Fidelity FULL'); ax.legend(); ax.grid(True, alpha=0.3)

    # Fidelity HALF
    ax = axes[1, 1]
    ax.semilogy(nlist, res['F_half'], 'b-o', ms=4, lw=1.2, label='ED')
    ax.semilogy(nlist, F_cft_half, 'r--s', ms=5, lw=1.2, label='CFT')
    ax.set_xlabel('cycles n'); ax.set_ylabel(r'$|\langle\psi_0|\psi\rangle|^2$')
    ax.set_title('Fidelity HALF'); ax.legend(); ax.grid(True, alpha=0.3)

    # Entropy FULL
    ax = axes[2, 0]
    ax.plot(nlist, res['dS_full'], 'b-o', ms=4, lw=1.2, label='ED')
    ax.plot(nlist, dS_cft_full, 'r--s', ms=5, lw=1.2, label='CFT')
    ax.set_xlabel('cycles n'); ax.set_ylabel(r'$\Delta S_A$')
    ax.set_title('Entanglement FULL'); ax.legend(); ax.grid(True, alpha=0.3)

    # Entropy HALF
    ax = axes[2, 1]
    ax.plot(nlist, res['dS_half'], 'b-o', ms=4, lw=1.2, label='ED')
    ax.plot(nlist, dS_cft_half, 'r--s', ms=5, lw=1.2, label='CFT')
    ax.set_xlabel('cycles n'); ax.set_ylabel(r'$\Delta S_A$')
    ax.set_title('Entanglement HALF'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(out_dir, filename_tag + ".png")
    plt.savefig(fpath, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {fpath}")

    # Print errors
    print(f"\n  [{case_label}] Numerical comparison:")
    print(f"    max|dE_full| = {np.max(np.abs(res['dE_full'] - dE_cft_full)):.4e}")
    print(f"    max|dE_half| = {np.max(np.abs(res['dE_half'] - dE_cft_half)):.4e}")
    print(f"    max|F_full|  = {np.max(np.abs(res['F_full'] - F_cft_full)):.4e}")
    print(f"    max|F_half|  = {np.max(np.abs(res['F_half'] - F_cft_half)):.4e}")
    print(f"    max|dS_full| = {np.max(np.abs(res['dS_full'] - dS_cft_full)):.4e}")
    print(f"    max|dS_half| = {np.max(np.abs(res['dS_half'] - dS_cft_half)):.4e}")


make_comparison_plot(res_free, "Free Ising (J=0.5, Gamma=0, h=0.5)",
                     "benchmark_free_ising_vs_CFT")

make_comparison_plot(res_int, "Interacting (J=1, Gamma=0.25, h=0.6066)",
                     "interacting_ising_vs_CFT")


# ############################################################
#  Combined fidelity plot: both cases on same axes
# ############################################################
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Fidelity: Free vs Interacting vs CFT  |  {param_base}", fontsize=13)

for idx, (tag, res_f, res_i) in enumerate([
    ("FULL", res_free['F_full'], res_int['F_full']),
    ("HALF", res_free['F_half'], res_int['F_half']),
]):
    ax = axes[idx]
    F_cft = F_cft_full if tag == "FULL" else F_cft_half
    ax.semilogy(nlist, F_cft, 'k-', lw=2, label='CFT')
    ax.semilogy(nlist, res_f, 'b--o', ms=4, lw=1.2, label='Free Ising ED')
    ax.semilogy(nlist, res_i, 'r--^', ms=4, lw=1.2, label='Interacting ED')
    ax.set_xlabel('cycles n')
    ax.set_ylabel(r'$|\langle\psi_0|\psi\rangle|^2$')
    ax.set_title(f'Fidelity {tag}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fpath = os.path.join(out_dir, "fidelity_free_vs_interacting_vs_CFT.png")
plt.savefig(fpath, dpi=200)
plt.close()
print(f"[PLOT] Saved {fpath}")

print("\n[ALL DONE]")
