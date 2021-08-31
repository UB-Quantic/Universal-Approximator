# One qubit as a Universal Approximant

In this repository one can find all the code related to the paper *One 
qubit as a Universal Approximant*, [arXiv:2102.04032](https://arxiv.org/abs/2102.04032) 

- The files `main*.py` summarize all the code needed for performing optimizations as
explained in the paper. 
- The inner engine of the simulations is written in 
`ApproximantNN.py`
- The file `chi2.py` corresponds to a code for painting final results on $\chi^2$
- The file `painter_experimental.py` paints all the different approximations of
different functions
- All results are stored in the results folder, and `summary.csv` is an index for them

If you find our work interesting, please cite us as:

```
@article{PerezSalinas2021,
  doi = {10.1103/physreva.104.012405},
  url = {https://doi.org/10.1103/physreva.104.012405},
  year = {2021},
  month = jul,
  publisher = {American Physical Society ({APS})},
  volume = {104},
  number = {1},
  author = {Adri{\'{a}}n P{\'{e}}rez-Salinas and David L{\'{o}}pez-N{\'{u}}{\~{n}}ez and Artur Garc{\'{\i}}a-S{\'{a}}ez and P. Forn-D{\'{\i}}az and Jos{\'{e}} I. Latorre},
  title = {One qubit as a universal approximant},
  journal = {Physical Review A}
}
```

