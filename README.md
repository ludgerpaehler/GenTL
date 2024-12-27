# Gen Inference and Learning Template Library (GenTL)
**Work-in-progress** template inference and learning library in C++ based on probabilistic programming principles

[![status](https://github.com/OpenGen/GenTL/actions/workflows/test.yml/badge.svg)](https://github.com/OpenGen/GenTL/actions?query=workflow/test)

[**Documentation (work-in-progress)**](https://opengen.github.io/gentl-docs/latest/)

This is currently a header-only library, and the source code consists of `.h` files under `include/gentl/`.

## Dependencies

### C++ compiler

Most of the library uses C++17 features, which are supported by recent clang++ and g++ versions.

Currently, `include/gentl/learning.h` uses C++20 features, which requires g++ version 11 or above, and clang++ version 11 or above.

### CMake

CMake 3.7 or above. See [Installing CMake](https://cmake.org/install/).

### Doxygen

This is only required for generating documentation.

See [Doxygen Installation](https://www.doxygen.nl/manual/install.html).

## Testing

Use CMake to build and test. Note that you can configure the C++ compiler to use with either the `CXX` environment variable or by setting the `CMAKE_CXX_COMPILER` variable, e.g.:
```
cmake -S . -B build -DCMAKE_CXX_COMPILER=g++-11
cmake --build build
cmake --build build --target test
```

## Generating documentation

```
cmake --build --target docs
```
Note that there is a Github Action that generates and deploys documentation to https://github.com/OpenGen/gentl-docs/ when there is a `push` to the `main` branch of this repository.

## References

```bibtex
@inproceedings{cusumano2019gen,
  title={Gen: a general-purpose probabilistic programming system with programmable inference},
  author={Cusumano-Towner, Marco F and Saad, Feras A and Lew, Alexander K and Mansinghka, Vikash K},
  booktitle={Proceedings of the 40th acm sigplan conference on programming language design and implementation},
  pages={221--236},
  year={2019}
}
```

Marco Cusumano-Towner's [PhD thesis](https://www.mct.dev).
