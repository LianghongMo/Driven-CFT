"""
Order-parameter heatmap |<sigma^z>(D, h)| for the model
    H = -sum_i [J sz sz + h sx + D sx sx]
on a uniform (D, h) grid with a small Z2-breaking pinning field eps_z.

This is independent of phasediagram.py (which only locates h_c via xi_chi peaks).
The critical line from phase_diagram_iDMRG_data.npz is overlaid on the heatmap.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import tenpy
tenpy.tools.misc.setup_logging(to_stdout="WARNING")
from tenpy.networks.mps import MPS
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite

out_dir = os.path.dirname(os.path.abspath(__file__))


class IsingGammaModel(CouplingMPOModel):
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        return SpinHalfSite(conserve=None)

    def init_terms(self, model_params):
        J = model_params.get('J', 1.0)
        h = model_params.get('h', 0.0)
        D = model_params.get('D', 0.0)
        eps_z = model_params.get('eps_z', 0.0)
        self.add_coupling(-4.0 * J, 0, 'Sz', 0, 'Sz', 1)
        self.add_coupling(-4.0 * D, 0, 'Sx', 0, 'Sx', 1)
        self.add_onsite(-2.0 * h, 0, 'Sx')
        if eps_z != 0.0:
            self.add_onsite(-2.0 * eps_z, 0, 'Sz')


def run_idmrg(J, h, D, chi, eps_z, max_sweeps=40, psi_init=None):
    pars = {'J': J, 'h': h, 'D': D, 'eps_z': eps_z, 'L': 2, 'bc_MPS': 'infinite'}
    model = IsingGammaModel(pars)
    if psi_init is None:
        psi = MPS.from_lat_product_state(model.lat, [['up'], ['up']])
    else:
        psi = psi_init.copy()
    dmrg_params = {
        'mixer': True,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-10},
        'max_E_err': 1e-9,
        'max_S_err': 1e-7,
        'max_sweeps': max_sweeps,
        'min_sweeps': 5,
    }
    eng = TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = eng.run()
    psi.canonical_form()
    sz = float(np.mean(psi.expectation_value('Sz'))) * 2.0
    return float(E), abs(sz), psi


# ============================================================
# Uniform (D, h) grid for the heatmap
# ============================================================
J = 1.0
# Uniform dense grid (step 0.025 in both axes).
D_grid = np.linspace(0.0, 1.0, 41)
h_grid = np.linspace(0.0, 1.1, 45)
CHI = 32
EPS_Z = 1e-3
MAX_SWEEPS = 40

print(f"[heatmap] {len(D_grid)} D x {len(h_grid)} h, chi={CHI}, eps_z={EPS_Z}")

mag_grid = np.full((len(D_grid), len(h_grid)), np.nan)

for i, D in enumerate(D_grid):
    print(f"\n  D={D:.3f}  (h: high -> low)")
    psi = None
    # reverse-sweep h: PM -> FM, warm-start picks the FM cat via eps_z
    for j in range(len(h_grid) - 1, -1, -1):
        h = h_grid[j]
        try:
            _, mag, psi = run_idmrg(J, h, D, chi=CHI, eps_z=EPS_Z,
                                     max_sweeps=MAX_SWEEPS, psi_init=psi)
        except Exception as e:
            mag = np.nan
            print(f"    h={h:.3f} FAILED: {e}")
        mag_grid[i, j] = mag
        print(f"    h={h:.3f}  |m|={mag:.3f}")

np.savez(os.path.join(out_dir, "order_param_heatmap_data.npz"),
         D_grid=D_grid, h_grid=h_grid, mag_grid=mag_grid, CHI=CHI, EPS_Z=EPS_Z)


# ============================================================
# Plot
# ============================================================
def edges_from_centers(c):
    c = np.asarray(c, dtype=float); e = np.zeros(len(c) + 1)
    e[1:-1] = 0.5 * (c[1:] + c[:-1])
    e[0]  = c[0]  - 0.5 * (c[1]  - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e

D_edges = np.clip(edges_from_centers(D_grid), 0, None)
h_edges = np.clip(edges_from_centers(h_grid), 0, None)
DD_e, HH_e = np.meshgrid(D_edges, h_edges, indexing='ij')

fig, ax = plt.subplots(figsize=(9, 7.5))
pcm = ax.pcolormesh(DD_e, HH_e, mag_grid, cmap='RdBu_r', vmin=0, vmax=1)
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r'$|\langle\sigma^z\rangle|$', fontsize=24)
cbar.ax.tick_params(labelsize=20)

# Overlay the critical line from the main phase-diagram run, if available
pd_path = os.path.join(out_dir, "phase_diagram_iDMRG_data.npz")
if os.path.exists(pd_path):
    pd = np.load(pd_path)
    D_scan = pd['D_scan']; h_critical = pd['h_critical']
    valid = np.isfinite(h_critical)
    D_dense = np.linspace(0, 1, 200)
    h_dense = interp1d(D_scan[valid], h_critical[valid], kind='cubic')(D_dense)
    ax.plot(D_dense, h_dense, 'k-', lw=2.5, zorder=4, label=r'$g_c(\Gamma)$ from $\xi_\chi$ peak')
    ax.plot(D_scan, h_critical, 'kD', ms=7, zorder=5)

ax.scatter(0.0, 1.0,    s=220, c='blue', marker='o', edgecolors='white',
           linewidths=2, zorder=10, label='Free Ising')
ax.scatter(0.25, 0.607, s=220, c='lime', marker='s', edgecolors='white',
           linewidths=2, zorder=10, label='Interacting Ising')
ax.scatter(1.0, 0.0,    s=220, c='red',  marker='^', edgecolors='white',
           linewidths=2, zorder=10, label='XX model')

ax.set_xlabel(r'$\Gamma / J$', fontsize=24)
ax.set_ylabel(r'$g / J$', fontsize=24)
ax.tick_params(axis='both', labelsize=20)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.15)
ax.legend(loc='upper right', fontsize=15, framealpha=0.95)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fpath = os.path.join(out_dir, f"order_param_heatmap.{ext}")
    plt.savefig(fpath, dpi=200)
    print(f"[PLOT] {fpath}")
plt.close()
