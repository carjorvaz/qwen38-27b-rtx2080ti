# patches-turing/

The Turing (SM75) series: 21 patches applied after `patches/`, in the order
listed in `series`, on the same vLLM 0.29.0 pin. Same install recipe, one
generation of hardware older: a 22 GB RTX 2080 Ti running the serving setup
this repo is built for.

Regenerated against `upstream/main` at `1cf8665` (the vLLM 0.29.0 pin plus
upstream's series hygiene), with the full `patches/` series already applied.
The stop-by-stop record of the 0.28.0 -> 0.29.0 rebase is in
[docs/turing-0.29-port.md](../docs/turing-0.29-port.md).

`series` is explicit rather than glob order, because the two directories are
applied one after the other and the file names here do not have to sort after
`patches/`'s; comments and blank lines are ignored.

To check the whole stack against a pristine checkout (this is what CI runs):

```bash
git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm /tmp/vllm
bash patches-turing/check_vllm_series.sh /tmp/vllm/vllm
```

To install it, boot a vLLM 0.29.0 environment and apply both directories, the
3090 series first, both at `--fuzz 0`:

```bash
SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
while read -r p; do
  case "$p" in ''|'#'*) continue ;; esac
  [ "$p" = dflash2-backport.patch ] && continue   # native since vLLM 0.28.0
  patch -p1 --fuzz 0 -d "$SP" < "patches/$p"
done < patches/series
while read -r p; do
  case "$p" in ''|'#'*) continue ;; esac
  patch -p1 --fuzz 0 -d "$SP" < "patches-turing/$p"
done < patches-turing/series
```

## Retired at the 0.29 port

Four 0.28-era patches are gone because upstream now carries them; do not
re-add them:

| retired | now provided by |
| --- | --- |
| `mamba-block-retirement.patch` | `patches/mamba-align-retire-null-gaps.patch` (the same vLLM #55450 backport, byte-identical hunks) |
| `vllm-engine-completion-log.patch` | `patches/engine-completion-log.patch` |
| `vllm-engine-stall-sentinel.patch` | `patches/engine-stall-sentinel.patch` |
| `vllm-sse-keep-alive.patch` | native in vLLM 0.29.0 (`entrypoints/openai/sse_keep_alive.py`) |

The tail patch `envs-knobs.patch` registers the Turing series' knobs in
`envs.py` and reads them through `envs`, matching upstream's
`patches/speed-knobs-envs.patch`. One of them (`VLLM_TURING_SPEC_NSEG`) was a
raw `os.environ` read in a file that 0.29 no longer imports `os` in, so this
patch also removes a crash on the ported tree.

`docs/turing-2080ti.md` explains what each patch is for and what the result
measures, including the parts that were measured and rejected. The two optional
prefill follow-ups, their quality trade-offs and GPU regression commands are in
`docs/turing-prefill.md`. Their environment flags default off.
