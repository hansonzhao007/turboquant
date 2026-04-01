load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "fast_hadamard",
    srcs = ["fast_hadamard.cc"],
    copts = ["-O3", "-mcpu=apple-m4"],
)

py_library(
    name = "turboquant_codebook",
    srcs = ["turboquant_codebook.py"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "turboquant_codebook_test",
    srcs = ["turboquant_codebook_test.py"],
    deps = [
        ":turboquant_codebook",
        requirement("numpy"),
    ],
)

py_library(
    name = "turboquant",
    srcs = ["turboquant.py"],
    deps = [
        requirement("numpy"),
        ":fast_hadamard",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "turboquant_test",
    srcs = ["turboquant_test.py"],
    deps = [
        ":turboquant",
        requirement("numpy"),
    ],
)

py_binary(
    name = "profile_tq",
    srcs = ["profile_tq.py"],
    deps = [
        ":turboquant",
        requirement("numpy"),
    ],
)
