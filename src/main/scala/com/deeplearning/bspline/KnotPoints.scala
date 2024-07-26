package com.deeplearning.bspline

import breeze.linalg._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.stats.distributions._
import com.deeplearning.Network

import scala.util.Random

object KnotPoints {

  def initializeKnotPoints2D(controlPoints: Int, degree: Int): DenseVector[Double] = {
    val numKnots = controlPoints+degree
    // Generate the knot vector
    val knots = DenseVector.zeros[Double](controlPoints+degree)

    // First k+1 knots are 0
    for (i <- 0 until degree-1) {
      knots(i) = 0.0
    }
    // Middle knots are uniformly spaced
    val numMiddleKnots = numKnots - 2 * (degree)
    val ecart = (1.0/(numMiddleKnots+1))
    var middle = 0.0
    for (i <- degree until degree+numMiddleKnots) {
      middle += ecart
      knots(i) = middle
    }

    // Last k+1 knots are 1
    for (i <- degree+numMiddleKnots until numKnots)  {
      knots(i) = 1.0
    }

    knots
  }


}