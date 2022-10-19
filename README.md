### What is ANNS-ALT

In ANNS-ALT, the dataset consists of data vectors, say *x*, and each query is a vector *y* and a matrix (linear transform) *M*. The goal is to find the nearest neighbor of *y* in L2 distance after all data vectors are transformed by *M*, i.e. `argmin||Mx - y||2`.

ANNS-ALT is a very general mother problem. ANNS problems under large number of interesting metrics, including Mahalanobis distance, point-to-subspace distance, and weighted L2 distance, can be reduced to ANNS-ALT.

### How to make ONIAK

Simply run ```make``` from the `FALCONN` folder. Alternatively, you can specify the exact solution as follows.

> `falconn-axequaltoy` for the ONIAK LSH solution.

> `jlt-filter` for the JLT-based filter.

### How to run ONIAK

`./falconn-axequaltoy -cf [config file path]`

The config file requires the following attributes:

* `dimension`: the data dimension **PLUS TWO**. (We will add two dummy dimensions in our implementation.)

* `training size`: number of data vectors.

* `testing size`: number of queries.

* `number of query rows`: number of rows in each query matrix.

* `number of neighbors`: number of nearest neighbors to return.

* `eigenvalue filepath`: folder of precomputed eigenvalues (the dimension must be the next power of two after `dimension`).

* `data filename`, `query filename`, and `kernel filename`: path to data vectors (*x*), query vectors (*y*), and linear transformations (*M*). All in .fvecs format.

* `result filename`: path for output.

* `hash table parameters`: a list of parameters, one for each *partition*. 

    * `k`: number of hash functions in each table

    * `l`: number of hash tables.
    * `bucket width`: width of each hash bucket.
    * `upper` and `lower`: upper and lower bound (in **squared** l2 norm) of each partition.

The following attributes are used by the JLT-based filter:

> `number of filters`: the target (after reduction) dimension of JLT.

> `ratio of finalists`: the ratio of  candidates returned by JLT.

> `candidate filename`: if JLT is used in combination with ONIAK, it is a path to an .ivecs candidate file. Otherwise, it is "linear scan".


**DO NOT** change the other attributes (They are experimental).


### Authors

The ONIAK library is developed by Jingfan Meng, Huayi Wang, and Jun Xu. Mitsunori Ogihara also contributed to the ONIAK project.

### License

ONIAK is available under the [MIT License](https://opensource.org/licenses/MIT) (see LICENSE.txt).
Note that the third-party libraries in the `external/` folder are distributed under other open source licenses.
The Eigen library is licensed under the [MPL2](https://www.mozilla.org/en-US/MPL/2.0/).
The googletest and googlemock libraries are licensed under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
The pybind11 library is licensed under a [BSD-style license](https://github.com/pybind/pybind11/blob/master/LICENSE).

### Related Works

The *multi-probe LSH* implementation in ONIAK is based on 
> Lv, Qin, et al. ["Multi-probe LSH: efficient indexing for high-dimensional similarity search."](https://dl.acm.org/doi/abs/10.5555/1325851.1325958) Proceedings of the 33rd international conference on Very large data bases. 2007.

The asymmetric mapping pair (AMP) for reducing ANNS-ALT queries to ANNS-L2 is based on 
> Basri, Ronen, Tal Hassner, and Lihi Zelnik-Manor. ["Approximate nearest subspace search."](https://ieeexplore.ieee.org/abstract/document/5477422) IEEE transactions on pattern analysis and machine intelligence 33.2 (2010): 266-278.

### Acknowledgement

We would like to thank the authors of the [FALCONN](https://github.com/FALCONN-LIB/FALCONN) library, from which ONIAK is derived.

We have also drawn great inspirations from the HD3 pseudorandom rotation in the [research paper](https://proceedings.neurips.cc/paper/2015/hash/2823f4797102ce1a1aec05359cc16dd9-Abstract.html) of FALCONN.
