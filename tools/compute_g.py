import argparse
import numpy as np
from pathlib import Path


def read_hopping_list(path):
    """Read real-space Hamiltonian hopping list.

    Expected columns:
      i j R1_frac R2_frac R3_frac Re Im
    Lines starting with # are ignored.
    """
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            i = int(parts[0])
            j = int(parts[1])
            r = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
            val = float(parts[5]) + 1j * float(parts[6])
            data.append((i, j, r, val))
    if not data:
        raise ValueError(f'No hopping entries found in {path}')
    return data


def infer_nions(hops):
    return max(max(i, j) for i, j, _, _ in hops)


def build_hk(hops, k_frac, nions=None):
    """Reconstruct H(k) from hopping list.

    k_frac must be in fractional reciprocal coordinates, consistent with the
    Fortran phase convention exp(i 2π k·R_frac).
    """
    if nions is None:
        nions = infer_nions(hops)
    hk = np.zeros((nions, nions), dtype=np.complex128)
    for i, j, r_frac, val in hops:
        phase = np.exp(2j * np.pi * np.dot(k_frac, r_frac))
        hk[i - 1, j - 1] += val * phase
    return hk


def parse_wavefunction_file(path, k_index):
    """Parse one k-point block from tb_wavef.dat-like file.

    Returns
    -------
    bands : list[int]
        Global band indices stored in the file.
    psi : np.ndarray
        Shape (nions, nbands) complex coefficients.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    nions = None
    for line in lines:
        if line.startswith('# Wavefunctions:'):
            # format: # Wavefunctions: nkpts= ... nions= ... bands= istart iend
            parts = line.replace('#', '').replace('=', ' ').split()
            if 'nions' in parts:
                nions = int(parts[parts.index('nions') + 1])
            break

    blocks = []
    idx = 0
    total = len(lines)
    while idx < total:
        line = lines[idx].strip()
        if line.startswith('# kpoint'):
            toks = line.split()
            # '# kpoint <ik> <ib>'
            if len(toks) >= 4:
                try:
                    ik = int(toks[2])
                    ib = int(toks[3])
                except ValueError:
                    idx += 1
                    continue
                if ik == k_index:
                    coeffs = []
                    j = idx + 1
                    while j < total:
                        ln = lines[j].strip()
                        if not ln:
                            break
                        if ln.startswith('#'):
                            break
                        parts = ln.split()
                        if len(parts) >= 2:
                            try:
                                coeffs.append(float(parts[0]) + 1j * float(parts[1]))
                            except ValueError:
                                pass
                        j += 1
                    blocks.append((ib, np.array(coeffs, dtype=np.complex128)))
        idx += 1

    if not blocks:
        raise ValueError(f'k_index={k_index} not found in {path}')

    blocks.sort(key=lambda x: x[0])
    bands = [b for b, _ in blocks]
    nions_found = len(blocks[0][1])
    if nions is not None and nions != nions_found:
        print(f'[warn] header nions={nions}, parsed nions={nions_found}')

    psi = np.column_stack([vec for _, vec in blocks])
    return bands, psi


def parse_kpoints_explicit(path):
    """Parse explicit reciprocal KPOINTS list.

    Expected format:
      line1 comment
      line2 number of k-points
      line3 Reciprocal
      remaining lines: kx ky kz [weight]
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 4:
        raise ValueError('KPOINTS file too short')
    nk = int(lines[1].split()[0])
    mode = lines[2].lower()
    if not mode.startswith('reciprocal'):
        raise ValueError('This helper currently expects Reciprocal explicit KPOINTS')
    kpts = []
    for line in lines[3:3 + nk]:
        parts = line.split()
        if len(parts) < 3:
            continue
        kpts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if len(kpts) != nk:
        raise ValueError(f'Expected {nk} k-points, parsed {len(kpts)}')
    return np.array(kpts, dtype=float)


def compute_g(hr_list, hf_list, k_index, kq_index, k_frac,wf_r=None, wf_f=None, use_frozen_left=False):
    """Compute g matrix.

    By default (use_frozen_left=False) both bra and ket use relaxed-state
    wavefunctions: bra = psi_r(k+q), ket = psi_r(k). This follows the
    standard first-order electron-phonon matrix element definition.
    If `use_frozen_left=True`, the bra is taken from the frozen-state
    wavefunction file `wf_f` at kq (legacy option).
    """
    hops_r = read_hopping_list(hr_list)
    hops_f = read_hopping_list(hf_list)

    # right state: relaxed at k
    bands_r_k, psi_r_k = parse_wavefunction_file(wf_r, k_index)
    # left state: by default relaxed at k+q
    bands_r_kq, psi_r_kq = parse_wavefunction_file(wf_r, kq_index)

    if use_frozen_left:
        # legacy: read frozen-state wavefunction for left vector
        bands_f_kq, psi_f_kq = parse_wavefunction_file(wf_f, kq_index)
        left_bands = bands_f_kq
        psi_left = psi_f_kq
    else:
        left_bands = bands_r_kq
        psi_left = psi_r_kq

    # ensure dimensions
    nions = max(psi_r_k.shape[0], psi_left.shape[0])
    hr_k = build_hk(hops_r, k_frac, nions=nions)
    hf_k = build_hk(hops_f, k_frac, nions=nions)
    delta_h = hf_k - hr_k

    g = psi_left.conj().T @ delta_h @ psi_r_k
    return left_bands, bands_r_k, g


def save_g_matrix(path, bands_f, bands_r, g):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# columns: m_band_f n_band_r Re(g) Im(g) Abs(g)\n')
        for i, mb in enumerate(bands_f):
            for j, nb in enumerate(bands_r):
                val = g[i, j]
                f.write(f'{mb:8d} {nb:8d} {val.real:20.12e} {val.imag:20.12e} {abs(val):20.12e}\n')


def main():
    p = argparse.ArgumentParser(description='Compute g_mn = <phi_f|Hf-Hr|phi_r> from hopping lists and wavefunctions')
    p.add_argument('--hr', required=True, help='Relaxed-state real-space Hamiltonian list')
    p.add_argument('--hf', required=True, help='Frozen-phonon real-space Hamiltonian list')
    p.add_argument('--wf-r', required=True, help='Relaxed-state wavefunction file')
    p.add_argument('--wf-f', required=True, help='Frozen-state wavefunction file')
    p.add_argument('--kpoints', help='Explicit reciprocal KPOINTS file used for the calculation')
    p.add_argument('--k-index', type=int, help='1-based k-point index to use')
    p.add_argument('--kvec', nargs=3, type=float, help='Fractional k vector if no KPOINTS file is provided')
    p.add_argument('--kq-index', type=int, help='1-based k+q point index when using --kpoints')
    p.add_argument('--use-frozen-left', action='store_true', help='Use frozen-state wavefunction for left vector (legacy)')
    p.add_argument('--out', default='g_matrix.dat', help='Output matrix file')
    args = p.parse_args()

    if args.kpoints:
        kpts = parse_kpoints_explicit(args.kpoints)
        if args.k_index is None:
            raise SystemExit('Provide --k-index when using --kpoints')
        k_index = args.k_index
        if not (1 <= k_index <= len(kpts)):
            raise SystemExit(f'k-index out of range: 1..{len(kpts)}')
        k_frac = kpts[k_index - 1]
        if args.kq_index is None:
            kq_index = k_index
        else:
            kq_index = args.kq_index
            if not (1 <= kq_index <= len(kpts)):
                raise SystemExit(f'kq-index out of range: 1..{len(kpts)}')
    else:
        if args.kvec is None:
            raise SystemExit('Provide either --kpoints + --k-index, or --kvec')
        k_index = args.k_index if args.k_index is not None else 1
        k_frac = np.array(args.kvec, dtype=float)
        # when user provides explicit kvec, use same index for k+q unless provided
        kq_index = args.kq_index if args.kq_index is not None else k_index

    bands_f, bands_r, g = compute_g(args.hr, args.hf, k_index, kq_index, k_frac, wf_r=args.wf_r, wf_f=args.wf_f, use_frozen_left=args.use_frozen_left)
    save_g_matrix(args.out, bands_f, bands_r, g)
    print(f'Wrote {args.out} with shape {g.shape}')


if __name__ == '__main__':
    main()
