import scala.util.Try

import lmcoursier.CoursierConfiguration
import lmcoursier.definitions.Authentication
import sbt.{ File, Logger }

object Util {
  def osCudaClassifier: String = {
    val osString =
      Option(System.getenv("LIBTORCH_TARGET_OS")).getOrElse(System.getProperty("os.name")).toLowerCase match {
        case os if os.contains("mac") || os.contains("darwin") => "darwin"
        case os if os.contains("linux")                        => "linux"
        case os if os.contains("windows")                      => "windows"
        case os                                                => throw new sbt.MessageOnlyException(s"The OS $os is not a supported platform.")
      }
    val cudaString = Option(System.getenv("LIBTORCH_TARGET_CPU")).getOrElse("cpu")
    s"${osString}_$cudaString"
  }

  private val acceptHeader = "Accept" -> "application/octet-stream, application/json, application/xml, */*"
}
