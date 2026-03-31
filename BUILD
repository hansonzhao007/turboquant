load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

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
