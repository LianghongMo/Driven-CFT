"""
Fidelity comparison: free Ising / interacting Ising / XX  vs  CFT,
with new parameters:
    H0 uniform,  H1 deformed with kappa_1 = (1.0, 1.2, -0.2),
    q = 2,  T1 = 0.3,  two cases  T0 = -0.3  and  T0 = +0.3.

Only the FULL fidelity is plotted (one figure per T0 sign).
Velocity-rescaled CFT is used for each model: T_CFT = v * T_lattice.
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
# Parameters (shared across both T0 signs)
# ============================================================
L = 16
q = 2

# New deformation parameters
s0_0, sp_0, sm_0 = 1.0, 0.0, 0.0
s0_1, sp_1, sm_1 = 1.0, 1.2, -0.2

Ncycle = 16
nlist = np.arange(Ncycle + 1)

NA = L // q
startA = int(np.floor(L / 2 - NA / 2))
subsysA = list(range(startA, startA + NA))

dtype = np.complex128

x_bond = np.arange(L) + 0.5
f0_bond = s0_0 + sp_0 * np.cos(2*np.pi*q*x_bond/L) + sm_0 * np.sin(2*np.pi*q*x_bond/L)
f1_bond = s0_1 + sp_1 * np.cos(2*np.pi*q*x_bond/L) + sm_1 * np.sin(2*np.pi*q*x_bond/L)

def fbond_to_hsite(f_bond):
    return 0.5 * (np.roll(f_bond, 1) + f_bond)

h1_site = fbond_to_hsite(f1_bond)

out_dir = os.path.dirname(os.path.abspath(__file__))


# ############################################################
# CFT (SU(1,1)) helpers
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

def fidelity_cft_full(T0, T1, cCFT):
    """Returns F_full[n] for n = 0..Ncycle, with M_cycle = M0 @ M1."""
    hCh = cCFT * (q**2 - 1) / (24.0 * q)
    hAn = hCh

    M0  = su11_chiral(s0_0, sp_0, sm_0, T0, L, q)
    M1  = su11_chiral(s0_1, sp_1, sm_1, T1, L, q)
    M0b = su11_antichiral(s0_0, sp_0, sm_0, T0, L, q)
    M1b = su11_antichiral(s0_1, sp_1, sm_1, T1, L, q)

    M0M1  = M0 @ M1
    M0M1b = M0b @ M1b

    trC = abs(np.trace(M0M1))
    print(f"    [CFT c={cCFT}] |Tr(M_cycle)| = {trC:.6f}  "
          f"{'HEATING' if trC > 2 else 'non-heating'}")

    F = np.zeros(Ncycle+1)
    Mn  = np.eye(2, dtype=complex)
    Mnb = np.eye(2, dtype=complex)
    for n in range(Ncycle+1):
        if n > 0:
            Mn  = Mn @ M0M1
            Mnb = Mnb @ M0M1b
        aa  = abs(Mn[0,0])
        aap = abs(Mnb[0,0])
        if aa > 1e-300 and aap > 1e-300:
            F[n] = min(max(aa**(-4*hCh) * aap**(-4*hAn), 0), 1)
    return F


# ############################################################
# ED (QuSpin) helpers
# ############################################################

def run_ed_ising(J_val, Gamma_val, h_val, T0_latt, T1_latt, label):
    """General Ising ED:  H = -J zz - Gamma xx - h x   with Pauli matrices."""
    print(f"\n  [{label}]  J={J_val}, Gamma={Gamma_val}, h={h_val}, "
          f"T0={T0_latt:.3f}, T1={T1_latt:.3f}")
    basis = spin_basis_1d(L, pauli=True)
    no_checks = dict(check_herm=False, check_pcon=False, check_symm=False)

    h_site_1 = fbond_to_hsite(f1_bond)

    Jzz = [[-J_val, j, (j+1)%L] for j in range(L)]
    Gxx = [[-Gamma_val, j, (j+1)%L] for j in range(L)]
    hx  = [[-h_val, j] for j in range(L)]
    H0 = hamiltonian([["zz", Jzz], ["xx", Gxx], ["x", hx]], [],
                      basis=basis, dtype=dtype, **no_checks)
    Jzz1 = [[-J_val*float(f1_bond[j]), j, (j+1)%L] for j in range(L)]
    Gxx1 = [[-Gamma_val*float(f1_bond[j]), j, (j+1)%L] for j in range(L)]
    hx1  = [[-h_val*float(h_site_1[j]), j] for j in range(L)]
    H1 = hamiltonian([["zz", Jzz1], ["xx", Gxx1], ["x", hx1]], [],
                      basis=basis, dtype=dtype, **no_checks)

    # velocity from gap (Delta_sigma = 1/8)
    E_arr, _ = H0.eigsh(k=4, which="SA", maxiter=10**7)
    E_arr = np.sort(E_arr)
    gap = E_arr[1] - E_arr[0]
    v = L * gap / (2 * np.pi * (1.0/8.0))

    psi0_arr = H0.eigsh(k=1, which="SA", maxiter=10**7)[1][:, 0]
    psi0 = psi0_arr / np.linalg.norm(psi0_arr)
    psi = psi0.copy()

    U0 = expm_multiply_parallel(H0.tocsr(), a=-1j*T0_latt, dtype=dtype)
    U1 = expm_multiply_parallel(H1.tocsr(), a=-1j*T1_latt, dtype=dtype)
    work = np.zeros(2*psi.shape[0], dtype=dtype)

    # Full-cycle (n) and half-cycle (n + 0.5, after U0) fidelities.
    F      = np.zeros(Ncycle + 1)
    F_half = np.zeros(Ncycle + 1)
    for n in range(Ncycle + 1):
        F[n] = float(np.abs(np.vdot(psi0, psi))**2)
        psi_h = psi.copy()
        U0.dot(psi_h, work_array=work, overwrite_v=True)
        F_half[n] = float(np.abs(np.vdot(psi0, psi_h))**2)
        if n < Ncycle:
            U0.dot(psi, work_array=work, overwrite_v=True)
            U1.dot(psi, work_array=work, overwrite_v=True)
            psi /= np.linalg.norm(psi)
    print(f"    gap={gap:.6f}, v={v:.4f}")
    return v, F, F_half

def run_ed_xx(J_xx, T0_latt, T1_latt):
    """XX:  H = -J_xx (xx + yy)."""
    print(f"\n  [XX] J_xx={J_xx},  T0={T0_latt:.3f}, T1={T1_latt:.3f}")
    basis = spin_basis_1d(L, pauli=True)
    no_checks = dict(check_herm=False, check_pcon=False, check_symm=False)

    xx0 = [[-J_xx, j, (j+1)%L] for j in range(L)]
    yy0 = [[-J_xx, j, (j+1)%L] for j in range(L)]
    H0 = hamiltonian([["xx", xx0], ["yy", yy0]], [],
                      basis=basis, dtype=dtype, **no_checks)
    xx1 = [[-J_xx*float(f1_bond[j]), j, (j+1)%L] for j in range(L)]
    yy1 = [[-J_xx*float(f1_bond[j]), j, (j+1)%L] for j in range(L)]
    H1 = hamiltonian([["xx", xx1], ["yy", yy1]], [],
                      basis=basis, dtype=dtype, **no_checks)

    # XX analytical velocity:  v_F = 4*J_xx
    v = 4 * J_xx

    psi0_arr = H0.eigsh(k=1, which="SA", maxiter=10**7)[1][:, 0]
    psi0 = psi0_arr / np.linalg.norm(psi0_arr)
    psi = psi0.copy()

    U0 = expm_multiply_parallel(H0.tocsr(), a=-1j*T0_latt, dtype=dtype)
    U1 = expm_multiply_parallel(H1.tocsr(), a=-1j*T1_latt, dtype=dtype)
    work = np.zeros(2*psi.shape[0], dtype=dtype)

    F      = np.zeros(Ncycle + 1)
    F_half = np.zeros(Ncycle + 1)
    for n in range(Ncycle + 1):
        F[n] = float(np.abs(np.vdot(psi0, psi))**2)
        psi_h = psi.copy()
        U0.dot(psi_h, work_array=work, overwrite_v=True)
        F_half[n] = float(np.abs(np.vdot(psi0, psi_h))**2)
        if n < Ncycle:
            U0.dot(psi, work_array=work, overwrite_v=True)
            U1.dot(psi, work_array=work, overwrite_v=True)
            psi /= np.linalg.norm(psi)
    print(f"    v={v:.4f} (analytical 4*J_xx)")
    return v, F, F_half


# ############################################################
# Run for both T0 signs and plot
# ############################################################

def run_case(T0_latt, T1_latt, label):
    print(f"\n{'='*60}")
    print(f"{label}:  T0 = {T0_latt:+.2f},  T1 = {T1_latt:+.2f}")
    print(f"{'='*60}")
    v_free, F_free, F_free_h = run_ed_ising(0.5, 0.0,  0.5,    T0_latt, T1_latt, "Free Ising")
    v_int,  F_int,  F_int_h  = run_ed_ising(1.0, 0.25, 0.6066, T0_latt, T1_latt, "Interacting Ising")
    v_xx,   F_xx,   F_xx_h   = run_ed_xx(0.5, T0_latt, T1_latt)
    print("\n  [CFT runs]")
    F_cft_free = fidelity_cft_full(v_free * T0_latt, v_free * T1_latt, cCFT=0.5)
    F_cft_int  = fidelity_cft_full(v_int  * T0_latt, v_int  * T1_latt, cCFT=0.5)
    F_cft_xx   = fidelity_cft_full(v_xx   * T0_latt, v_xx   * T1_latt, cCFT=1.0)
    return dict(v_free=v_free, v_int=v_int, v_xx=v_xx,
                F_free=F_free,     F_int=F_int,     F_xx=F_xx,
                F_free_h=F_free_h, F_int_h=F_int_h, F_xx_h=F_xx_h,
                F_cft_free=F_cft_free, F_cft_int=F_cft_int, F_cft_xx=F_cft_xx)


T1_latt = 0.3
heat = run_case(-0.3, T1_latt, "HEATING")
non  = run_case(+0.3, T1_latt, "NON-HEATING")

# ----- save all curves to npz for reproducibility / data availability -----
npz_path = os.path.join(out_dir, "fidelity_3panels_a_b_c_data.npz")
np.savez(
    npz_path,
    # metadata
    L=L, q=q, Ncycle=Ncycle,
    kappa1=np.array([s0_1, sp_1, sm_1]),
    T1=T1_latt, T0_heating=-0.3, T0_nonheating=+0.3,
    n_cycles=nlist,
    # velocities (extracted from finite-size gap / analytical for XX)
    v_free=heat['v_free'], v_int=heat['v_int'], v_xx=heat['v_xx'],
    # ---- heating (T0=-0.3) ----
    F_free_heat=heat['F_free'],   F_cft_free_heat=heat['F_cft_free'],
    F_int_heat=heat['F_int'],     F_cft_int_heat=heat['F_cft_int'],
    F_xx_heat=heat['F_xx'],       F_cft_xx_heat=heat['F_cft_xx'],
    # ---- non-heating (T0=+0.3) ----
    F_free_non=non['F_free'],     F_cft_free_non=non['F_cft_free'],
    F_int_non=non['F_int'],       F_cft_int_non=non['F_cft_int'],
    F_xx_non=non['F_xx'],         F_cft_xx_non=non['F_cft_xx'],
)
print(f"\n[DATA] saved {npz_path}")


def last_above(F, threshold):
    """Largest cycle index n* with F[n*] > threshold (strictly)."""
    above = np.where(np.asarray(F) > threshold)[0]
    return int(above[-1]) if len(above) > 0 else 0


# Colours from the reference figure: red-orange diamond (heating),
# muted-blue square (non-heating)
RED  = '#C0392B'   # heating
BLUE = '#3F6A8F'   # non-heating

panels = [
    # (tag, title, F_heat, F_heat_half, F_heat_cft,
    #  F_non, F_non_half, F_non_cft, n_xmax, marker_step, cft_floor)
    ('a', 'Free Ising',
        heat['F_free'], heat['F_free_h'], heat['F_cft_free'],
        non['F_free'],  non['F_free_h'],  non['F_cft_free'],  16, 2, 0.57),
    ('b', 'Interacting Ising',
        heat['F_int'],  heat['F_int_h'],  heat['F_cft_int'],
        non['F_int'],   non['F_int_h'],   non['F_cft_int'],    8, 1, 0.50),
    ('c', 'XX model',
        heat['F_xx'],   heat['F_xx_h'],   heat['F_cft_xx'],
        non['F_xx'],    non['F_xx_h'],    non['F_cft_xx'],     6, 1, 0.57),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 6.4), sharey=True)
ED_FLOOR  = 0.55   # show ED markers down to this value
CFT_FLOOR = 0.57   # show CFT line  down to this value
for ax, (tag, title,
         Fh_ed, Fh_ed_h, Fh_cft,
         Fn_ed, Fn_ed_h, Fn_cft,
         n_xmax, marker_step, cft_floor) in zip(axes, panels):
    # ED markers clipped at ED_FLOOR (global), CFT line at panel-specific
    # `cft_floor`, both capped at n_xmax.
    n_h_ed   = min(last_above(Fh_ed,  ED_FLOOR),  n_xmax)
    n_h_cft  = min(last_above(Fh_cft, cft_floor), n_xmax)
    n_n_ed   = min(last_above(Fn_ed,  ED_FLOOR),  n_xmax)
    n_n_cft  = min(last_above(Fn_cft, cft_floor), n_xmax)
    n_max    = n_xmax

    # CFT lines
    ax.plot(np.arange(n_h_cft+1), Fh_cft[:n_h_cft+1], color=RED, lw=3.5,
            zorder=2, solid_capstyle='butt')
    ax.plot(np.arange(n_n_cft+1), Fn_cft[:n_n_cft+1], color=BLUE, lw=3.5,
            zorder=2, solid_capstyle='butt')

    # ED markers at integer cycles, every `marker_step` cycle
    h_idx = np.arange(0, n_h_ed + 1, marker_step)
    n_idx = np.arange(0, n_n_ed + 1, marker_step)
    ax.plot(h_idx, Fh_ed[h_idx], color=RED, marker='D',
            lw=0, ms=15, mec='black', mew=1.0, zorder=3, label='heating')
    ax.plot(n_idx, Fn_ed[n_idx], color=BLUE, marker='s',
            lw=0, ms=15, mec='black', mew=1.0, zorder=3, label='non-heating')

    ax.set_xlabel(r'Floquet cycles $n$', fontsize=24)
    # leave generous margin so the line endpoints don't touch the axis frame
    ax.set_xlim(-0.7, n_max + 0.7)
    ax.set_ylim(0.55, 1.0)
    # Sparser tick labels with bigger font
    from matplotlib.ticker import MaxNLocator, FixedLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(FixedLocator([0.6, 0.8, 1.0]))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.text(0.04, 0.88, tag, transform=ax.transAxes, fontsize=33,
            fontweight='bold', va='top', ha='left')
    ax.text(0.5, 0.04, title, transform=ax.transAxes, fontsize=26,
            fontweight='bold', ha='center', va='bottom')
    if tag == 'a':
        ax.set_ylabel(r'Loschmidt echo  $|\langle\psi_0|\psi(n)\rangle|^2$',
                      fontsize=24)
    if tag == 'c':
        # place legend in upper-right so it does not overlap the model title
        ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(0.0, 0.45))

plt.tight_layout()
for ext in ('png', 'pdf'):
    fpath = os.path.join(out_dir, f"fidelity_3panels_a_b_c.{ext}")
    plt.savefig(fpath, dpi=200)
    print(f"\n  [PLOT] {fpath}")
plt.close()

print("\n[ALL DONE]")
