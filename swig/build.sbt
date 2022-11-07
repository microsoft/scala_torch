name := "scala-torch-swig"

libraryDependencies += "com.github.fommil" % "jniloader" % "1.1"

// for some reason, SBT passes javacOptions to javadoc, but javadoc doesn't understand -target, so:
javacOptions in (Compile, doc) := Seq.empty

val packageName = "com.microsoft.scalatorch.torch.internal"

val torchDir = baseDirectory { d => Option(System.getenv("TORCH_DIR")).map(file).getOrElse(d / "../libtorch") }

val includeDirs = torchDir { d => Seq(d / "include", d / "include/torch/csrc/api/include/") }

// Swig stuff: generate into managed sources, try to be a good SBT citizen
val Swig = config("swig")

val generate = TaskKey[Seq[File]]("generate")

inConfig(Swig)(Defaults.paths)

// Output for this config is version-independent
Swig / crossTarget := (Swig / target).value

Swig / sourceDirectory := (sourceDirectory in Compile).value / "swig"

Compile / sourceManaged := (Swig / sourceManaged).value

Swig / generate := {
  val tgt = (Swig / sourceManaged).value
  tgt.mkdirs()

  // the cxx file will go here
  val native = tgt / "native"
  val include = includeDirs.value
  native.mkdirs()

  val out = streams.value
  val bindingGenScript = (sourceDirectory in Swig).value / "bindgen.py"
  val declarationsFile = bindingGenScript
    .getParentFile() / ".." / ".." / ".." / ".." / "pytorch" / "torch" / "share" / "ATen" / "Declarations.yaml"

  val cachedGen = FileFunction.cached(
    out.cacheDirectory / "bindgen",
    inStyle = FilesInfo.lastModified,
    outStyle = FilesInfo.exists,
  ) { (in: Set[File]) =>
    assert(in.contains(bindingGenScript), s"$bindingGenScript should be in ${in}")
    assert(in.contains(declarationsFile), s"$declarationsFile should be in ${in}")
    genSwigAndScala(bindingGenScript, declarationsFile, tgt, out.log)
  }
  val genned = cachedGen(((sourceDirectory in Swig).value ** ("*.py")).get.toSet + bindingGenScript + declarationsFile)

  val cachedRunSwig = FileFunction.cached(
    out.cacheDirectory / "swig",
    inStyle = FilesInfo.lastModified,
    outStyle = FilesInfo.exists,
  ) { (in: Set[File]) =>
    val theMainFile = in.find(_.getName == "torch_swig.i").get
    runSwig(theMainFile, tgt, native, include, out.log)
  }
  cachedRunSwig(genned).toSeq
}

/** This is pretty convoluted right now. This code calls bindgen.py, which generates swig declarations, which are turned
  * by the `swig` command into both a .cxx file and Java bindings. This code also generates
  * Scala wrappers for those Java bindings. Unfortunately, those Scala wrappers are defined
  * in a downstream project (scala-torch). In an ideal world, we would separately generate
  * the scala bindings in the scala-torch project, and also put the generate the scala files
  * under the [[sourceManaged]] directory. For now, out of laziness, we directly
  * read in .scala.in files from the downstream project and produce .scala files in that same project. We
  * have checked in the generated Scala files for now so the API is easy to see, but we proably shouldn't do that
  * in the long-run.
  */
def genSwigAndScala(
    bindingGenScript: File,
    declarationsFile: File,
    target: File,
    logger: Logger,
): Set[File] = {
  import scala.sys.process._

  val realTarget = target / packageName.replace(".", "/")
  realTarget.mkdirs()

  val scalaDir =
    bindingGenScript
      .getParentFile() / ".." / ".." / ".." / ".." / "scala-torch" / "src" / "main" / "scala" / "com" / "microsoft" / "scalatorch" / "torch"

  val bindingGenCmd =
    s"""python3 $bindingGenScript $declarationsFile ${bindingGenScript.getParentFile()} $scalaDir"""

  logger.info("Generating auto-generated swig bindings")
  logger.info(bindingGenCmd)

  val pyErrorCode = bindingGenCmd ! logger
  if (pyErrorCode != 0) {
    sys.error(s"aborting generation of swig files because $bindingGenScript failed")
  }

  (bindingGenScript.getParentFile() ** ("*.i")).get.toSet
}

def runSwig(
    swigFile: File,
    target: File,
    nativeTarget: File,
    includeDirs: Seq[File],
    logger: Logger,
): Set[File] = {
  import scala.sys.process._
  def stripExtension(file: File) = ext.matcher(file.getName).replaceAll("")

  val realTarget = target / packageName.replace(".", "/")
  realTarget.mkdirs()

  val cxx = s"${nativeTarget}/${stripExtension(swigFile)}.cxx"
  val totalInclude = includeDirs
  val includeI = totalInclude.mkString("-I", " -I", "")

  val cmd =
    s"""swig -DSWIGWORDSIZE64 -v -c++ -java -package $packageName $includeI -o $cxx -outdir $realTarget $swigFile"""

  logger.info(s"generating SWIG: ${swigFile}")
  logger.info(cmd)

  val errorCode = cmd ! logger
  if (errorCode != 0) {
    sys.error(s"aborting generation for $swigFile because swig was unhappy")
  }

  (target ** "*.java").get.toSet
}

unmanagedSourceDirectories in Compile += (sourceDirectory in Swig).value
sourceGenerators in Compile += (generate in Swig).taskValue
cleanFiles += (target in Swig).value

import java.util.regex.Pattern

val ext = Pattern.compile("(?<=.)\\.[^.]+$")

// native compilation should wait on swig, since swig makes the cxx file
nativeCompile := nativeCompile.dependsOn(generate in Swig).value

// this is a bit of a hack to make the naming consistent with JniLoader
resourceGenerators in Compile += Def.task {
  val libraries: Seq[(File, String)] = (nativeLibraries in Compile).value
  val resources: Seq[File] = for ((file, _) <- libraries) yield {

    val newName = file.getParentFile.getParentFile.getName match {
      case "x86_64-darwin" => "libtorch_swig.pred-osx-x86_64.jnilib"
      case "x86_64-linux"  => "libtorch_swig.pred-linux-x86_64.so"
      // TODO: flesh this out
    }
    val resource = (resourceManaged in Compile).value / newName

    // copy native library to a managed resource, so that it is always available
    // on the classpath, even when not packaged as a jar
    IO.copyFile(file, resource)
    resource
  }
  resources
}.taskValue
