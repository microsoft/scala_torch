scalacOptions += "-target:jvm-1.8"
javacOptions ++= Seq("-target", "1.8", "-source", "1.8")

libraryDependencies += "org.scalatest" %% "scalatest" % "3.1.4" % "test"
libraryDependencies += "com.michaelpollmeier" %% "scala-arm" % "2.1"
libraryDependencies += "com.lihaoyi" %% "sourcecode" % "0.1.9"
libraryDependencies += "org.scala-lang.modules" %% "scala-collection-compat" % "2.5.0"

unmanagedSourceDirectories in Compile += {
  val sourceDir = (sourceDirectory in Compile).value
  CrossVersion.partialVersion(scalaVersion.value) match {
    case Some((2, n)) if n >= 13 =>
      sourceDir / "scala-2.13+"
    case _ =>
      sourceDir / "scala-2.13-"
  }
}

fork := true
