lazy val Scala2_13Version = "2.13.10"
lazy val Scala2_12Version = "2.12.17"
val commonSettings = Seq(
  organization := "com.microsoft",
  scalaVersion := Scala2_13Version,
  version := "0.0.1-SM-05-SNAPSHOT",
  publishMavenStyle := true,
)
lazy val swig = (project in file("swig"))
  .settings(commonSettings: _*)
  .enablePlugins(JniNative)
  .settings(
    name := "swig",
    crossScalaVersions := Seq(Scala2_12Version, Scala2_13Version),
    crossPaths := true,
  )

lazy val `scala-torch` = (project in file("scala-torch"))
  .settings(commonSettings: _*)
  .settings(
    name := "scala-torch",
    crossScalaVersions := Seq(Scala2_12Version, Scala2_13Version),
  )
  .dependsOn(swig)

lazy val root = (project in file("."))
  .settings(commonSettings: _*)
  .settings(
    name := "scala-torch-parent",
  )
  .dependsOn(swig, `scala-torch`)
  .aggregate(swig, `scala-torch`)
