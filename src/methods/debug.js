module.exports = PosenetPointer => {
  /**
   * Loops through each pose and draws their keypoints/skeletons
   * - Draws skeletons and keypoints
   */
  PosenetPointer.prototype.debugPoses = function() {
    const context = this.canvas.getContext("2d");

    this.poses.forEach(({ score, keypoints }) => {
      if (score >= this.options.posenet.minPoseConfidence) {
        const adjacentKeypoints = Posenet.getAdjacentKeyPoints(
          keypoints,
          this.options.posenet.minPartConfidence,
          context
        );

        context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawSkeleton(adjacentKeypoints, context);
        this.drawKeypoints(
          keypoints,
          this.options.posenet.minPartConfidence,
          context
        );
      }
    });
  };

  /**
   * Draw each tracked keypoint
   * - Draws keypoints only when they are "visible"
   *
   * @see https://github.com/tensorflow/tfjs-models/tree/master/posenet
   *
   * @param {ARR} keypoints The list of all keypoints
   * @param {NUM} minConfidence The minimum keypoint score needed to track
   * @param {OBJ} context The canvas context to draw into
   */
  PosenetPointer.prototype.drawKeypoints = function(
    keypoints,
    minConfidence,
    context
  ) {
    keypoints.forEach(({ position, score }) => {
      if (score > minConfidence) {
        context.beginPath();
        context.arc(position.x, position.y, 15, 0, 2 * Math.PI);
        context.fillStyle = "#00ff00";
        context.fill();
      }
    });
  };

  /**
   * Draw each tracked skeleton
   * @see https://github.com/tensorflow/tfjs-models/tree/master/posenet
   *
   * - Draws all visible segments captured with PoseNet.getAdjacentKeyPoints
   *
   * @param {ARR} adjacentPoints The list of all keypoints and their relationships
   * @param {OBJ} context The canvas context to draw into
   */
  PosenetPointer.prototype.drawSkeleton = function(adjacentPoints, context) {
    adjacentPoints.forEach(keypoints => {
      this.drawSegment(
        this.toTuple(keypoints[0].position),
        this.toTuple(keypoints[1].position),
        context
      );
    });
  };

  /**
   * Converts a position to a tuple
   * - Essentially converts an {x, y} object into a [y, x] array
   *
   * @param {OBJ} position {x, y}
   */
  PosenetPointer.prototype.toTuple = function({ x, y }) {
    return [y, x];
  };

  /**
   * Draws the skeleton segment
   * - A segment is a straight line between two tuples
   *
   * @param {OBJ} fromTuple [ay, ax] The starting point
   * @param {OBJ} toTuple [by, bx] The ending point
   * @param {HEX} color The color to draw in
   * @param {OBJ} context The canvas context to draw in
   */
  PosenetPointer.prototype.drawSegment = function([ay, ax], [by, bx], context) {
    const scale = 1;

    context.beginPath();
    context.moveTo(ax * scale, ay * scale);
    context.lineTo(bx * scale, by * scale);
    context.lineWidth = 10;
    context.strokeStyle = "#ff00ff";
    context.stroke();
  };
};
