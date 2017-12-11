package(default_visibility = ["//visibility:public"])

py_library(
    name = "model_util",
    srcs = ["model_util.py"],
    deps = [
        "//isl:tensorcheck",
        "//isl:util",
    ],
)

py_library(
    name = "fovea",
    srcs = ["fovea.py"],
    deps = [
        ":model_util",
        "//isl:tensorcheck",
        "//isl:util",
    ],
)

py_library(
    name = "concordance",
    srcs = ["concordance.py"],
    deps = [
        ":model_util",
        "//isl:tensorcheck",
        "//isl:util",
    ],
)

py_test(
    name = "model_test",
    size = "large",
    srcs = ["model_test.py"],
    deps = [
        ":fovea",
        ":model_util",
        "//isl:controller",
        "//isl:test_util",
    ],
)
