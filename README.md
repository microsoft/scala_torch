# scala-torch
JVM/Scala wrappers for LibTorch.

## State of this project

This project is mature enough to be used regularly in production code. The API exposed is fairly clean
and tries to follow PyTorch syntax as much as possible. The API is a mix of hand-written wrappings and a wrapper
around most of `Declarations.yaml`. 

That said, some internal documentation is not quite ready for public consumption yet, though there is enough
documentation that people who are already familiar with Scala and LibTorch can probably figure out what's going on. 
Code generation is accomplished through a combination of [Swig](https://www.swig.org) and a quick-and-dirty 
[Python script](swig/src/main/swig/bindgen.py) that reads in `Declarations.yaml`, which provides a language-independent 
API for a large part of LibTorch. This file is [deprecated](https://github.com/pytorch/pytorch/issues/69471) and in the 
future, we can hopefully replace `bindgen.py` using the forthcoming [torchgen](https://github.com/pytorch/pytorch/issues/69471#issuecomment-1273642655)
tool provided by PyTorch.

We have not yet published JARs for this project. These are coming soon. 

## Short tour

Scala-torch exposes an API that tries to mirror PyTorch as much as Scala syntax
allows. For example, taking some snippets from
[this tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html):

PyTorch:
```python
import torch

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

Scala-Torch:
```scala
import com.microsoft.scalatorch.torch
import com.microsoft.scalatorch.torch.syntax._

torch.ReferenceManager.forBlock { implicit rm =>
 val data = $($(1, 2), $(3, 4))
 val x_data = torch.tensor(data)
}
```


PyTorch:
```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

Scala-Torch:
```scala
val tensor = torch.ones($(4, 4))
println(s"First row: ${tensor(0)}")
println(s"First column: ${tensor(::, 0)}")
println(s"Last column: ${tensor(---, -1)}")
tensor(::, 1) = 0
println(tensor)
```

See [this file](scala-torch/src/test/scala/com/microsoft/scalatorch/torch/tutorial/PyTorchOrgTensorTutorialTest.scala) for 
a complete translation of the PyTorch tutorial into Scala-Torch.

### Memory management

One big difference between Scala-Torch and PyTorch is in memory management. Because Python and LibTorch both use 
reference counting, memory management is fairly transparent to users. However, since the JVM uses garbage collection
and [finalizers are not guaranteed to run](https://docs.oracle.com/javase/9/docs/api/java/lang/Object.html#finalize--),
it is not easy to make memory management transparent to the user. Scala-Torch elects to make memory management something
the user must control by providing [ReferenceManager](scala-torch/src/main/scala/com/microsoft/scalatorch/torch/ReferenceManager.scala)s 
that define the lifetime of any LibTorch-allocated object
that is added to it. All Scala-Torch methods that allocate objects from LibTorch take an `implicit` `ReferenceManager`,
so it is the responsibility of the caller to make sure there is a `ReferenceManager` in `implicit` scope (or passed
explicitly) and that that `ReferenceManager` will be `close()`ed when appropriate. See documentation and uses
of `ReferenceManager` for more examples.

## Handling of native dependencies

PyTorch provides pre-built binaries for the native code backing it [here](https://pytorch.org/get-started/locally/). 
We make use of the pre-built dynamic libraries by packaging them up in a jar, much like [TensorFlow Scala](http://platanios.org/tensorflow_scala/installation.html).
Downstream
projects have two options for handling the native dependencies: they can either 
1. Declare a dependency on the packaged native dependencies wrapped up with a jar using
```scala
val osClassifier = System.getProperty("os.name").toLowerCase match {
  case os if os.contains("mac") || os.contains("darwin") => "darwin"
  case os if os.contains("linux")                        => "linux"
  case os if os.contains("windows")                      => "windows"
  case os                                                => throw new sbt.MessageOnlyException(s"The OS $os is not a supported platform.")
}
libraryDependencies += ("com.microsoft.scalatorch" % "libtorch-jar" % "1.10.0").classifier(osClassifier + "_cpu")
```
2. Ensure that the libtorch dependencies are installed in the OS-dependent way, for example, in `/usr/lib` or in `LD_LIBRARY_PATH` on Linux,
or in `PATH` on windows. Note that on recent version of MacOS, [System Integrity Protected](https://developer.apple.com/library/archive/documentation/Security/Conceptual/System_Integrity_Protection_Guide/RuntimeProtections/RuntimeProtections.html)
resets `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` when working processes, so it is very hard to use that approach on MacOS. 

The native binaries for the JNI bindings for all three supported OSes are published in `scala-torch-swig.jar`, so there
is no need for OS-specific treatment of those libraries.

Approach 1 is convenient because sbt will handle the libtorch native dependency for you and users won't need install
libtorch or set any environment variables. This is the ideal approach for local development. 

There are several downsides of approach 1:
* it may unnecessarily duplicate installation of libtorch if, for example, pytorch is already installed
* jars for GPU builds of libtorch are not provided, so approach 2 is the only option if GPU support is required
* care must be taken when publishing any library that depends on Scala-Torch to not publish the dependency
 on the `libtorch-jar`, since that would force the consumer of that library to depend on whatever OS-specific
 version of the jar was used at building time. See the use of `pomPostProcess` in [build.sbt](build.sbt) for
 how we handle that. Note that another option is for downstream libraries to exclude the `libtorch-jar`
 using something like 
```scala
libraryDependencies += ("com.microsoft" % "scala-torch" % "0.1.0").exclude("com.microsoft.scalatorch", "libtorch-jar")
```

Approach 2 is the better option for CI, remote jobs, production, etc. 

### Local Development (MacOS)

You will need to have SWIG installed, which you can
install using `brew install swig`.

```
git submodule update --init --recursive
cd pytorch
python3 -m tools.codegen.gen -s aten/src/ATen -d torch/share/ATen
cd ..
curl https://download.pytorch.org/libtorch/cpu/libtorch-macos-$(pytorchVersion).zip -o libtorch.zip
unzip libtorch.zip
rm -f libtorch.zip
conda env create --name scala-torch --file environment.yml
conda activate scala-torch
export TORCH_DIR=$PWD/libtorch
sbt test
```

A similar setup should work for Linux and Windows. 

#### Troubleshooting

If you are using Clang 11.0.3 you may run into an error 
when compiling the `SobolEngineOps` file. This is most 
likely due to an issue with the compiler and it has already 
been reported [here](https://github.com/pytorch/pytorch/issues/35478).
A temporary workaround is to install another version of 
Clang (e.g., by executing `brew install llvm`). Another option
is to downgrade XCode to a version < 11.4.

### Upgrading the LibTorch version

To upgrade the underlying version of LibTorch:
* `cd pytorch; git checkout <commit>` with the `<commit>` of the desired release version, 
  best found [here](https://github.com/pytorch/pytorch/releases).
* Rerun the steps under **Local Development**.
* Change `TORCH_VERSION` in [run_tests.yml](.github/workflows/run_tests.yml).
* Address compilation errors when running `sbt compile`. Changes to [bindgen.py](swig/src/main/swig/bindgen.py) may
  be necessary.

# Contributors

Thanks to the following contributors to this project:

* [Adam Pauls](https://github.com/adampauls)
* [David Hall](https://github.com/dlwh)
* [Theo Lanman](https://github.com/theo-lanman)
* [Alex Kyte](https://github.com/alexanderkyte)
* [Hao Fang](https://github.com/hao-fang)
* [Anthony Platanios](https://github.com/eaplatanios)
* [Dmitrij Peters](https://github.com/Dpetters)

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.