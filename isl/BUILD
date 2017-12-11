package(default_visibility = ["//visibility:public"])

py_library(
    name = "tensorcheck",
    srcs = ["tensorcheck.py"],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        ":tensorcheck",
    ],
)

py_library(
    name = "augment",
    srcs = [
        "augment.py",
    ],
    deps = [
        ":ops",
        ":util",
    ],
)

py_library(
    name = "visualize",
    srcs = [
        "visualize.py",
    ],
    deps = [
        ":ops",
        ":util",
    ],
)

py_library(
    name = "data_provider",
    srcs = [
        "data_provider.py",
    ],
    deps = [
        ":ops",
        ":tensorcheck",
        ":util",
    ],
)

py_library(
    name = "controller",
    srcs = [
        "controller.py",
    ],
    deps = [
        ":augment",
        ":data_provider",
        ":ops",
        ":tensorcheck",
        ":util",
        ":visualize",
        "//isl/models:model_util",
    ],
)

py_library(
    name = "infer",
    srcs = [
        "infer.py",
    ],
    deps = [
        ":augment",
        ":controller",
        ":data_provider",
        ":ops",
        ":tensorcheck",
        ":util",
        ":visualize",
        "//isl/models:model_util",
    ],
)

py_binary(
    name = "launch",
    srcs = [
        "launch.py",
    ],
    deps = [
        ":augment",
        ":controller",
        ":data_provider",
        ":infer",
        ":ops",
        ":util",
        "//isl/models:concordance",
        "//isl/models:model_util",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**/*.png",
        "testdata/*png",
    ]) + [
        "testdata/research_logo.jpg",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    data = [
        ":testdata",
    ],
    deps = [
        ":data_provider",
        ":ops",
        ":util",
    ],
)

py_test(
    name = "tensorcheck_test",
    srcs = ["tensorcheck_test.py"],
    # Enable the tensorcheck module.
    args = ["--tensorcheck_enable_checks"],
    deps = [
        ":tensorcheck",
        ":test_util",
    ],
)

py_test(
    name = "util_test",
    srcs = [
        "util_test.py",
    ],
    deps = [
        ":test_util",
        ":util",
    ],
)

py_test(
    name = "controller_test",
    size = "enormous",
    srcs = ["controller_test.py"],
    deps = [
        ":augment",
        ":controller",
        ":data_provider",
        ":test_util",
        ":util",
        "//isl/models:fovea",
    ],
)

py_test(
    name = "ops_test",
    size = "large",
    srcs = ["ops_test.py"],
    deps = [
        ":ops",
        ":test_util",
    ],
)

py_test(
    name = "augment_test",
    srcs = ["augment_test.py"],
    deps = [
        ":augment",
        ":test_util",
        ":util",
    ],
)

py_test(
    name = "visualize_test",
    size = "large",
    srcs = ["visualize_test.py"],
    deps = [
        ":data_provider",
        ":test_util",
        ":util",
        ":visualize",
    ],
)

py_test(
    name = "data_provider_test",
    srcs = ["data_provider_test.py"],
    deps = [
        ":data_provider",
        ":test_util",
        ":util",
    ],
)
