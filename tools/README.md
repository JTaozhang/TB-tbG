# g-matrix post-processing

This folder contains a small helper for computing

$$g_{mn} = \langle \phi^f_{m,k+q} | (H^f - H^r) | \phi^r_{n,k} \rangle$$

from real-space hopping lists and wavefunction outputs.

## 1. Export real-space Hamiltonian lists from TB-tbG

In the input file, enable:

```text
WHR T
FHR tb_hr.dat
```

Run once for the relaxed structure and once for the frozen-phonon structure.

## 2. Compute g

```bash
python tools/compute_g.py \
  --hr relaxed/tb_hr.dat \
  --hf frozen/tb_hr.dat \
  --wf-r relaxed/tb_wavef.dat \
  --wf-f frozen/tb_wavef.dat \
  --kpoints KPOINTS \
  --k-index 1 \
  --out g_matrix.dat
```

If you already know the k-vector, you can skip `--kpoints` and use:

```bash
python tools/compute_g.py \
  --hr relaxed/tb_hr.dat \
  --hf frozen/tb_hr.dat \
  --wf-r relaxed/tb_wavef.dat \
  --wf-f frozen/tb_wavef.dat \
  --kvec 0.0 0.0 0.0 \
  --out g_matrix.dat
```

## Output

`g_matrix.dat` contains one row per band pair:

- frozen-phonon band index
- relaxed-state band index
- Re(g)
- Im(g)
- |g|
