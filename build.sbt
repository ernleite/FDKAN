import com.typesafe.sbt.SbtMultiJvm.multiJvmSettings
import com.typesafe.sbt.SbtMultiJvm.MultiJvmKeys.MultiJvm

val akkaVersion = "2.9.4"

lazy val `akka-sample-cluster-scala` = project
  .in(file("."))
  .settings(multiJvmSettings: _*)
  .settings(
    resolvers += "Akka library repository".at("https://repo.akka.io/maven"),
    organization := "DeepGenAI LLC",
    scalaVersion := "2.13.12",
    Compile / scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Xlog-reflective-calls", "-Xlint"),
    Compile / javacOptions ++= Seq("-Xlint:unchecked", "-Xlint:deprecation"),
    run / javaOptions ++= Seq("-Xms16G", "-Xmx16G", "-XX:+UseG1GC","-Djava.library.path=./target/native"),
    libraryDependencies ++= Seq(
      "com.typesafe.akka" %% "akka-actor-typed"           % akkaVersion,
      "com.typesafe.akka" %% "akka-actor" % akkaVersion,
      "com.typesafe.akka" %% "akka-cluster-typed"         % akkaVersion,
      "com.typesafe.akka" %% "akka-serialization-jackson" % akkaVersion,
      "ch.qos.logback"    %  "logback-classic"             % "1.4.7",
      "org.scalatest"     %% "scalatest"                  % "3.2.15"     % Test,
      "com.typesafe.akka" %% "akka-stream" % akkaVersion,
      "org.scalanlp" %% "breeze" % "2.1.0",
      "org.scalanlp" %% "breeze-natives" % "2.1.0",
      "org.scalanlp" %% "breeze-viz" % "2.1.0",
      "com.amazonaws" % "aws-java-sdk-s3" % "1.12.761",
      "javax.xml.bind" % "jaxb-api" % "2.3.1",
      "dev.ludovic.netlib" % "blas" % "3.0.3",
      "dev.ludovic.netlib" % "lapack" % "3.0.3",
      "dev.ludovic.netlib" % "arpack" % "3.0.3",
      "org.slf4j" % "slf4j-simple" % "2.0.9",
      "ai.djl" % "api" % "0.28.0",
      "ai.djl.pytorch" % "pytorch-engine" % "0.28.0",
      "commons-cli"%"commons-cli" %"1.5.0",
      "org.bytedeco" % "openblas" % "0.3.26-1.5.10",
      "dev.ludovic.netlib" % "parent" % "3.0.3" pomOnly(),
      "org.scala-lang.modules" %% "scala-swing" % "3.0.0"
    ),
    run / fork := true,
    Global / cancelable := false,
    parallelExecution in Global := true,
    // disable parallel tests
    Test / parallelExecution := false,
    licenses := Seq(("CC0", url("http://creativecommons.org/publicdomain/zero/1.0")))

  )
  .configs (MultiJvm)

