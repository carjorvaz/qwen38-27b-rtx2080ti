# patches-turing/

The Turing (SM75) series: 21 patches applied after `patches/`, in the order
listed in `series`. Same vLLM 0.28.0, same install recipe, one generation of
hardware older: a 22 GB RTX 2080 Ti running the serving setup this repo is built
for.

Generated against upstream at `1834917` with the full `patches/` series already
applied, which is the pin the numbers in `docs/turing-2080ti.md` were measured
on.

`series` is explicit rather than glob order, because the two directories are
applied one after the other and the file names here do not have to sort after
`patches/`'s.

To check the whole stack against a pristine checkout (this is what CI runs):

```bash
git clone --depth 1 --branch v0.28.0 https://github.com/vllm-project/vllm /tmp/vllm
bash patches-turing/check_vllm_series.sh /tmp/vllm/vllm
```

To install it, boot a vLLM 0.28.0 environment and apply both directories, the
3090 series first:

```bash
SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
for p in patches/*.patch; do
  [ "$p" = patches/dflash2-backport.patch ] && continue   # native in 0.28.0
  patch -p1 -d "$SP" < "$p"
done
while read -r p; do patch -p1 -d "$SP" < "patches-turing/$p"; done < patches-turing/series
```

`docs/turing-2080ti.md` explains what each patch is for and what the result
measures, including the parts that were measured and rejected. The two optional
prefill follow-ups, their quality trade-offs and GPU regression commands are in
`docs/turing-prefill.md`. Their environment flags default off.
