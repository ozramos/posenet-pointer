import * as Posenet from "@tensorflow-models/posenet";
import { merge } from "lodash";

module.exports = class PoseNetPointer {
  constructor(opts = {}) {
    // Whether this device is supported
    this.isSupported = this.isWebGLSupported();
    // Whether we're running inference or not
    this._isRunning = false;
    // Plugins added with .use()
    this.plugins = {};
    // Stack of poses for smoothing
    this.poseStack = [];
    // Sanitize options and set defaults
    this.configure(opts);
  }

  /**
   * Starts running inference with PoseNet
   * @param {Function} cb A callback to call once PoseNet is ready
   */
  async start(cb) {
    await this.setupFeed();
    await this.initPosenet();
    this._isTracking = true;
    await this.trackPosesLoop(this);
    cb && cb();
  }

  /**
   * Recursive method for tracking poses on each animationFrame:
   * - This method is recursive, once called it continues until after
   *    pointer.stop() is called or until this._isTracking is false
   *
   * @param {PosenetPointer} context The this context, since we're in the
   *    constructor scope now
   */
  trackPosesLoop(context) {
    context.posenet && context.trackPoses();
    if (context.poses) {
      context.runCalculations();
      context.emitEvents();
      context.runPlugins();
    }

    context._isTracking &&
      requestAnimationFrame(() => this.trackPosesLoop(context));
  }

  /**
   * Either assigns passed poses or estimates new poses
   * - Automatically adjusts algorithm to match "single" or "multiple mode"
   * - If debug is on, displays the points and skeletons overlays on the webcam
   */
  async trackPoses() {
    let poses = [];

    // Get single pose
    if (this.options.posenet.maxUsers === 1) {
      let pose = await this.posenet.estimateSinglePose(
        this.video,
        this.options.posenet.imageScaleFactor,
        false,
        this.options.posenet.outputStride
      );
      poses = [pose];
      // Get multiple poses
    } else {
      poses = await this.posenet.estimateMultiplePoses(
        this.video,
        this.options.posenet.imageScaleFactor,
        false,
        this.options.posenet.outputStride,
        this.options.posenet.maxUsers,
        this.options.posenet.scoreThreshold,
        this.options.posenet.nmsRadius
      );
    }

    // Publicly set poses
    this.poses = poses;

    // Only draw when debug is on
    this.options.debug && poses && this.debugPoses();
  }

  /**
   * Runs all plugins
   */
  runPlugins() {
    Object.keys(this.plugins).forEach(key => {
      this.plugins[key].onFrame.call(this, this.poses);
    });
  }

  /**
   * @TODO Emits events
   * - Emits onSeeClarkePoseUpdates with (this.poses, seeclarke)
   */
  emitEvents() {
    window.dispatchEvent(
      new CustomEvent("posenetPointerUpdated", {
        detail: {
          context: this
        }
      })
    );
  }

  /**
   * Adds a plugin with `name` that calles `cb` on every frame
   * - Repeated calls to same `name` overwrites previous
   */
  use(name, cb) {
    this.plugins[name] = {
      onFrame: cb
    };
  }

  /**
   * Loops through each pose and draws their keypoints/skeletons
   * - Draws skeletons and keypoints
   */
  debugPoses() {
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
  }

  /**
   * Sets up the webcam and stream:
   * - This creates its own video/canvas elements, allowing you to have
   *    multiple instances going (for example, to use front/back cameras
   *    simultaneously)
   * - Recreates the video feed to reassign srcObject once it's been stopped
   */
  async setupFeed() {
    const isMobile = this.isMobile();

    this.canvas.width = this.video.width = 600;
    this.canvas.height = this.video.height = 500;

    // Set window size
    this.cache.window.area = window.outerWidth * window.outerHeight;
    window.addEventListener("resize", () => {
      this.cache.window.area = window.outerWidth * window.outerHeight;
    });

    this.video.srcObject = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user" },
      width: isMobile ? null : this.video.width,
      height: isMobile ? null : this.video.height
    });

    this.video.play();
  }

  /**
   * Init PoseNet
   */
  async initPosenet() {
    if (!this.posenet) {
      this.posenet = await Posenet.load({
        architecture: this.options.posenet.architecture,
        multiplier: this.options.posenet.multiplier,
        outputStride: this.options.posenet.outputStride
      });
    }
  }

  /**
   * Sets defaults to the missing constructor options:
   * - Sets defaults
   * - Creates a default debug container
   * - Creates a default video and canvas
   *
   * @param {Object} opts The options passed into the constructor. Pass null to use all defaults
   */
  configure(opts) {
    // Fallback for default target
    if (!opts.target) {
      opts.target = document.getElementById("posenet-pointer-debug");
      if (!opts.target) {
        opts.target = document.createElement("p");
        opts.target.style.position = "relative";
        document.body.appendChild(opts.target);
      }
    }

    const $video = opts.video || this.createDefaultVideo(opts.target);
    const $canvas = opts.canvas || this.createDefaultCanvas(opts.target);
    this.video = $video;
    this.canvas = $canvas;

    // Setup defaults
    this.options = merge(
      {
        autostart: false,
        canvas: $canvas,
        video: $video,
        debug: false,
        facingMode: "user",
        poseStackSize: 8,
        posenet: {
          architecture: "MobileNetV1",
          multiplier: 0.75,
          maxUsers: 1,
          minPoseConfidence: 0.1,
          minPartConfidence: 0.5,
          outputStride: 16,
          inputResolution: 257,
          nmsRadius: 20,
          scoreThreshold: 0.5
        },
        target: opts.target
      },
      opts
    );

    // Chache
    this.cache = {
      window: {
        // @SEE this.setupFeed
        area: 0
      }
    };
    opts.target.style.display = this.options.debug ? "initial" : "none";
  }

  /**
   * Creates a default (flipped) video and adds it to the DOM:
   * - The video is absolutely positioned within the $wrap
   *
   * @param {HTMLElement} $wrap A container to embed the video into
   *
   * @return {HTMLVideoElement} A hidden video used for inference with PoseNet
   */
  createDefaultVideo($wrap) {
    const $video = document.createElement("video");

    $wrap.classList.add("posenet-pointer-debug");
    $video.setAttribute("playsinline", true);
    $video.style.transform = "scale(-1, 1)";
    $video.style.position = "absolute";
    $wrap.appendChild($video);

    return $video;
  }

  /**
   * Creates a default (flipped) canvas and adds it to the DOM
   * - The canvas is added to the $wrap (along with the video) relatively
   *
   * @param {Element} $wrap The wrapping element to inject the canvas into
   *
   * @return {HTMLCanvasElement} A hidden canvas used for debugging with PoseNet
   */
  createDefaultCanvas($wrap) {
    const $canvas = document.createElement("canvas");
    $canvas.style.transform = "scale(-1, 1)";
    $canvas.style.position = "relative";
    $canvas.style.top = 0;
    $canvas.style.left = 0;

    $wrap.appendChild($canvas);

    return $canvas;
  }

  /**
   * Helpers for checking if we're on mobile
   * - Checks if we're on mobile
   * - Checks if we're on android
   * - Checks if we're on iOS
   */
  isMobile() {
    return this.isAndroid() || this.isIOS();
  }
  isAndroid() {
    return /Android/i.test(navigator.userAgent);
  }
  isIOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
  }

  /**
   * Checks if WebGL is supported. Depending on your deployment needs,
   * you can first check if WebGL is supported with this method, and then either
   * display a message or start the tracker.
   *
   * - This will automatically fail if canvas is not supported!
   * - Checks for webgl and experimental-webgl
   *
   * @see https://stackoverflow.com/a/22953053
   *
   * @param {Boolean} forceThrow (Optional) Whether to force throw an error. This is mostly for unit testing to test failures
   * @return {Boolean} Is WebGL supported?
   */
  isWebGLSupported() {
    try {
      let canvas = document.createElement("canvas");
      let isSupported = true;
      if (
        !canvas.getContext("webgl") ||
        !canvas.getContext("experimental-webgl")
      )
        isSupported = false;
      canvas.remove();

      return !!isSupported;
    } catch (e) {
      console.error(e);
      return false;
    }
  }

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
  drawKeypoints(keypoints, minConfidence, context) {
    keypoints.forEach(({ position, score }) => {
      if (score > minConfidence) {
        context.beginPath();
        context.arc(position.x, position.y, 15, 0, 2 * Math.PI);
        context.fillStyle = "#00ff00";
        context.fill();
      }
    });
  }

  /**
   * Draw each tracked skeleton
   * @see https://github.com/tensorflow/tfjs-models/tree/master/posenet
   *
   * - Draws all visible segments captured with PoseNet.getAdjacentKeyPoints
   *
   * @param {ARR} adjacentPoints The list of all keypoints and their relationships
   * @param {OBJ} context The canvas context to draw into
   */
  drawSkeleton(adjacentPoints, context) {
    adjacentPoints.forEach(keypoints => {
      this.drawSegment(
        this.toTuple(keypoints[0].position),
        this.toTuple(keypoints[1].position),
        context
      );
    });
  }

  /**
   * Converts a position to a tuple
   * - Essentially converts an {x, y} object into a [y, x] array
   *
   * @param {OBJ} position {x, y}
   */
  toTuple({ x, y }) {
    return [y, x];
  }

  /**
   * Draws the skeleton segment
   * - A segment is a straight line between two tuples
   *
   * @param {OBJ} fromTuple [ay, ax] The starting point
   * @param {OBJ} toTuple [by, bx] The ending point
   * @param {HEX} color The color to draw in
   * @param {OBJ} context The canvas context to draw in
   */
  drawSegment([ay, ax], [by, bx], context) {
    const scale = 1;

    context.beginPath();
    context.moveTo(ax * scale, ay * scale);
    context.lineTo(bx * scale, by * scale);
    context.lineWidth = 10;
    context.strokeStyle = "#ff00ff";
    context.stroke();
  }

  /**
   * Run Calculations
   */
  runCalculations() {
    this.calculateXY();
    this.calculateZ();
    this.emitEvents();
  }

  /**
   * Entry point for our hacky calculations
   * - Calculates "pointedAt" for each pose
   *
   * @NOTE The following pose keypoint indexes mean the following: (@SEE https://github.com/tensorflow/tfjs-models/tree/master/posenet#keypoints)
   * 0   nose
   * 1   leftEye
   * 2   rightEye
   * 3   leftEar
   * 4   rightEar
   * 5   leftShoulder
   * 6   rightShoulder
   * 7   leftElbow
   * 8   rightElbow
   * 9   leftWrist
   * 10	rightWrist
   * 11	leftHip
   * 12	rightHip
   * 13	leftKnee
   * 14	rightKnee
   * 15	leftAnkle
   * 16	rightAnkle
   *
   */
  calculateXY() {
    this.poses &&
      this.poses.forEach((pose, index) => {
        const nose = pose.keypoints[0];
        const envWidth = window.outerWidth;
        const envHeight = window.outerHeight;
        let poseAverages = 0;

        // Helps map a point on the.canvas to a point on the window
        const ratio = {
          width: envWidth / this.canvas.width,
          height: envHeight / this.canvas.height
        };

        // First, let's get where on the screen we are if looking dead ahead
        // The canvas is mirrored, so left needs to be flipped
        let x = -nose.position.x * ratio.width + envWidth;
        let y = nose.position.y * ratio.height;

        // @FIXME Now let's adjust for rotation
        let yaw = this.calculateHeadYaw(pose);
        let pitch = this.calculateHeadPitch(pose);
        x += (yaw * window.outerWidth) / 2;
        y += (pitch * window.outerHeight) / 2;

        // Let's add it to the stack
        this.poseStack[index] = this.poseStack[index] || [];
        this.poseStack[index].push({ x, y });
        if (this.poseStack[index].length > this.options.poseStackSize)
          this.poseStack[index].shift();

        // Finally let's get the average
        poseAverages = this.averagePoseStack(this.poseStack[index]);
        x = poseAverages.x;
        y = poseAverages.y;

        // Assign values
        pose.pointedAt = { x, y };
        pose.angles = { pitch, yaw };
        this.poses[index] = pose;
      });
  }

  /**
   * @FIXME Get the head's Yaw (looking left/right)
   * 👻 Let's unit test this AFTER we agree on a solid algorithm
   * 🧙 CAUTION HERO, FOR HERE BE 🐉 DRAGONS 🐉
   *
   * - 0* is you looking straight ahead
   * - 90* would be your head turned to the right
   * - -90* would be you looking to the left
   *
   * My basic algorithm is:
   *  1. What is the x distance from the nose to each eye?
   *
   *  2. The difference between these distances determines the angle
   *    - For this algorithm, angles are between -90 and 90 (looking left and right)
   *
   * Problems with this aglorithm:
   * - All of it
   */
  calculateHeadYaw(pose) {
    const points = pose.keypoints;
    let yaw = 0;
    let distanceRatio;
    let sideLookingAt;

    // 1. What is the x distance from the nose to each eye?
    let eyeNoseDistance = {
      left: Math.abs(points[1].position.x - points[0].position.x),
      right: Math.abs(points[2].position.x - points[0].position.x)
    };

    // 2. The difference between these distances determines the angle
    if (eyeNoseDistance.left > eyeNoseDistance.right) {
      distanceRatio = 1 - eyeNoseDistance.right / eyeNoseDistance.left;
      sideLookingAt = 1;
    } else {
      distanceRatio = 1 - eyeNoseDistance.left / eyeNoseDistance.right;
      sideLookingAt = -1;
    }

    // Try to tame this beast into a radian
    yaw = (distanceRatio * 90 * sideLookingAt * Math.PI) / 180;

    return yaw;
  }

  /**
   * @FIXME Get the head's Pitch (looking up/down)
   * 👻 Let's unit test this AFTER we agree on a solid algorithm
   * 🧙 CAUTION HERO, FOR HERE BE 🐉 DRAGONS 🐉
   *
   * - 0* is you looking straight ahead
   * - 90* would be your head turned upwards
   * - -90* would be you head turned downwards
   *
   * My basic algorithm is:
   *  1. Calculate the average Y's for both ears (or whichever is visible)
   *  2. Calculate the distance the eyes are apart
   *  3. Calculate the distance between the nose and the averaged ear Y
   */
  calculateHeadPitch(pose) {
    let yEarAverage = 0;
    let numEarsFound = 0;
    let eyeDistance = 0;
    let distanceRatio = 0;
    let points = pose.keypoints;

    // 1. Calculate the average Y's for both ears (or whichever is visible)
    if (points[3].score >= this.options.posenet.minPartConfidence) {
      numEarsFound++;
      yEarAverage += points[3].position.y;
    }
    if (points[4].score >= this.options.posenet.minPartConfidence) {
      numEarsFound++;
      yEarAverage += points[4].position.y;
    }
    yEarAverage = yEarAverage / numEarsFound;

    // 2. Calculate the distance the eyes are apart
    // - I am literally making this up as I go
    eyeDistance = points[1].position.x - points[2].position.x;
    distanceRatio = (points[0].position.y - yEarAverage) / eyeDistance;

    return (90 * distanceRatio * Math.PI) / 180;
  }

  /**
   * @FIXME Averages the pose stacks to reduce "wobble"
   *
   * @param {Object} poseStack The posestack to average out
   *
   * @return {Object} The averaged {x, y}
   */
  averagePoseStack(poseStack) {
    let x = 0;
    let y = 0;

    poseStack.forEach(pose => {
      x += pose.x;
      y += pose.y;
    });

    x = x / poseStack.length;
    y = y / poseStack.length;

    return { x, y };
  }

  /**
   * Entry point for calculating the depth (distance away from camera)
   * @SEE https://github.com/labofoz/SeeClarke.js/issues/1
   *
   * - [ ] Calculates area of triangle between eyes and nose
   * - [ ] Use this value as the "depth"
   */
  calculateZ() {
    this.poses &&
      this.poses.forEach((pose, index) => {
        const nose = pose.keypoints[0];
        const eyeL = pose.keypoints[1];
        const eyeR = pose.keypoints[2];
        const x = [nose.position.x, eyeL.position.x, eyeR.position.x];
        const y = [nose.position.y, eyeL.position.y, eyeR.position.y];
        let distance;

        // First calculate the area between
        let area =
          0.5 *
          Math.abs(
            x[0] * y[1] +
              x[1] * y[2] +
              x[2] * y[0] -
              x[1] * y[0] -
              x[2] * y[1] -
              x[0] * y[2]
          );

        // Next divide it by the size of the window so that it's consistent across
        // devices and screen resolutions
        distance = this.cache.window.area / area;

        // Assign values
        pose.pointedAt.z = distance;
        this.poses[index] = pose;
      });
  }
};
